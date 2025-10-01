# src/ibdqlib/train.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import KernelCenterer
from sklearn.svm import SVC as QSVC  # alias to keep QSVC naming

__all__ = ["train_qsvc_from_duckdb"]


def _fetch_embeddings(db_path: str, table: str = "embeddings") -> Tuple[np.ndarray, list[dict]]:
    """
    Load embeddings from DuckDB.

    Table schema expected:
      seq_id TEXT, length INTEGER, backend TEXT, dim INTEGER, embedding FLOAT[]

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Stack of embedding vectors.
    meta : list[dict]
        Per-row metadata (seq_id, length, backend, dim).
    """
    con = duckdb.connect(db_path)
    rows = con.execute(
        f"SELECT seq_id, length, backend, dim, embedding FROM {table}"
    ).fetchall()
    con.close()

    if not rows:
        raise RuntimeError(f"No rows found in table '{table}' of DB '{db_path}'.")

    meta = [{"seq_id": r[0], "length": r[1], "backend": r[2], "dim": r[3]} for r in rows]
    # DuckDB returns Python lists for FLOAT[]; convert each to np.array and stack.
    X = np.stack([np.asarray(r[4], dtype=np.float32) for r in rows])
    return X, meta


def _labels_from_rule(meta: list[dict], rule: str) -> np.ndarray:
    """
    Derive integer labels from metadata using a simple rule.
    Currently supports:
      - 'median-length': 1 if length > median(length), else 0
    """
    lengths = np.array([m["length"] for m in meta], dtype=float)

    if rule == "median-length":
        thresh = float(np.median(lengths))
        y = (lengths > thresh).astype(int)
    else:
        raise ValueError(f"Unknown rule: {rule}")

    return y


def train_qsvc_from_duckdb(
    db_path: str,
    table: str = "embeddings",
    labels_csv: Optional[str] = None,
    label_col: str = "label",
    rule: str = "median-length",
    test_size: float = 0.25,
    random_state: int = 42,
    feature_cap: Optional[int] = None,
    return_model: bool = False,
) -> Dict[str, Any] | Tuple[Dict[str, Any], QSVC, np.ndarray]:
    """
    Train an SVC with a precomputed kernel (X @ X.T) on the stored embeddings.

    If there aren't enough samples to stratify-split (n<4 or single class),
    we train on all data and report train accuracy instead.

    When return_model=True, returns (metrics, clf, Xref), where Xref are the
    training features that downstream prediction will use to build the
    precomputed kernel K = X_new @ Xref.T.
    """
    # ---- Load features
    X, meta = _fetch_embeddings(db_path, table)
    original_dim = int(meta[0]["dim"])

    # ---- Optional feature cap (to limit effective qubit count in a real quantum kernel)
    if feature_cap is not None and feature_cap > 0 and feature_cap < X.shape[1]:
        X = X[:, :feature_cap]
        used_dim = int(feature_cap)
    else:
        used_dim = int(X.shape[1])

    # ---- Labels
    if labels_csv:
        df = pd.read_csv(labels_csv)
        id_col = next(
            (c for c in df.columns if c.lower() in ["seq_id", "id", "name", "accession"]),
            None,
        )
        if not id_col:
            raise RuntimeError(
                "labels CSV must have a seq_id/id/name/accession column for joining."
            )
        lab_map = dict(zip(df[id_col].astype(str), df[label_col].astype(int)))
        try:
            y = np.array([int(lab_map[m["seq_id"]]) for m in meta], dtype=int)
        except KeyError as e:
            missing = str(e).strip("'")
            raise RuntimeError(f"Label missing for seq_id '{missing}' in labels CSV.") from None
    else:
        y = _labels_from_rule(meta, rule)

    classes = sorted(list(set(int(v) for v in y)))
    metrics: Dict[str, Any] = {
        "classes": classes,
        "used_features": used_dim,
    }

    # ---- Decide whether we can do a proper split
    can_split = (len(X) >= 4) and (len(set(y.tolist())) >= 2)

    # Precomputed-kernel SVC; we pass Gram matrices to fit/predict.
    clf = QSVC(kernel="precomputed")

    if can_split:
        from sklearn.model_selection import train_test_split

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        Ktr = Xtr @ Xtr.T
        clf.fit(Ktr, ytr)

        Kte = Xte @ Xtr.T
        yhat = clf.predict(Kte)

        metrics["test_accuracy"] = float(accuracy_score(yte, yhat))
        metrics["n_train"], metrics["n_test"] = int(len(Xtr)), int(len(Xte))
        Xref = Xtr
    else:
        # Train on all data and report training accuracy.
        K = X @ X.T
        clf.fit(K, y)
        yhat = clf.predict(K)

        metrics["train_accuracy"] = float(accuracy_score(y, yhat))
        metrics["n_train"] = int(len(X))
        metrics["note"] = (
            f"trained on all data (too few samples for a split); "
            f"feature_cap={used_dim} of {original_dim} original dims"
        )
        Xref = X

    return (metrics, clf, Xref) if return_model else metrics


def _fetch_multiomics(
    db_path: str,
    table: str = "multiomics",
) -> Tuple[np.ndarray, np.ndarray, list[str], dict[int, str], list[str], list[str]]:
    """Load multi-omics feature matrix and labels from DuckDB."""
    con = duckdb.connect(db_path)
    try:
        df = con.execute(
            f"SELECT sample_id, label, label_index, feature_names, features FROM {table}"
        ).fetch_df()
    except duckdb.Error as exc:  # pragma: no cover - surfaced to CLI caller
        raise RuntimeError(f"Failed to read table '{table}' from {db_path}: {exc}") from exc
    finally:
        con.close()

    if df.empty:
        raise RuntimeError(f"No rows found in table '{table}' of DB '{db_path}'.")

    raw_feature_names = df.loc[0, "feature_names"]
    if isinstance(raw_feature_names, tuple):
        raw_feature_names = raw_feature_names[0]
    if isinstance(raw_feature_names, np.ndarray):
        raw_feature_names = raw_feature_names.tolist()
    feature_names = list(raw_feature_names)
    if not isinstance(feature_names, list):
        raise RuntimeError("feature_names column must contain list values. Re-ingest dataset.")
    # Validate all rows share identical feature ordering
    def _normalize_names(names):
        if isinstance(names, tuple):
            names = names[0]
        if isinstance(names, np.ndarray):
            names = names.tolist()
        return list(names)

    if not df["feature_names"].apply(lambda names: _normalize_names(names) == feature_names).all():
        raise RuntimeError("Inconsistent feature ordering detected across rows; re-run ingest-omics.")

    try:
        X = np.vstack([
            np.asarray(row[0] if isinstance(row, tuple) else row, dtype=np.float32)
            for row in df["features"].tolist()
        ])
    except ValueError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to assemble feature matrix: {exc}") from exc

    labels_idx = df["label_index"].astype(int).to_numpy()
    label_map = {
        int(idx): str(lbl)
        for idx, lbl in zip(df["label_index"].astype(int), df["label"].astype(str))
    }
    sample_ids = df["sample_id"].astype(str).tolist()
    if "group_id" in df.columns:
        group_ids = df["group_id"].astype(str).tolist()
    else:
        group_ids = sample_ids

    return X, labels_idx, sample_ids, label_map, feature_names, group_ids


def _apply_clr(matrix: np.ndarray, indices: List[int], pseudocount: float = 1e-6) -> None:
    if not indices:
        return
    sub = matrix[:, indices]
    sub = np.log(sub + pseudocount)
    sub -= sub.mean(axis=1, keepdims=True)
    matrix[:, indices] = sub


def _resolve_cv_splits(y: np.ndarray, groups: List[str], desired: int) -> int:
    if desired <= 2:
        return max(2, desired)
    _, counts = np.unique(y, return_counts=True)
    min_class = counts.min()
    unique_groups = len(set(groups))
    max_possible = max(2, min(min_class, unique_groups))
    return min(desired, max_possible)


def train_multiomics_classifier(
    db_path: str,
    table: str = "multiomics",
    random_state: int = 42,
    C: float = 1.0,
    class_weight: Optional[str] = "balanced",
    cv_splits: int = 5,
    feature_k: Optional[int] = None,
    pseudocount: float = 1e-6,
    max_iter: int = 1000,
    return_model: bool = False,
):
    """Train a logistic-regression classifier on stored multi-omics features using grouped CV."""

    X_raw, y, sample_ids, label_map, feature_names, group_ids = _fetch_multiomics(db_path, table)

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least two distinct labels for supervised training")

    X = X_raw.astype(np.float64, copy=True)
    mgx_idx = [i for i, name in enumerate(feature_names) if name.startswith("mgx_")]
    mtx_idx = [i for i, name in enumerate(feature_names) if name.startswith("mtx_")]
    clr_indices = sorted(set(mgx_idx + mtx_idx))
    _apply_clr(X, clr_indices, pseudocount=pseudocount)

    class_indices = np.array(sorted(label_map.keys()))
    class_labels = [label_map[i] for i in class_indices]

    groups = np.array(group_ids)
    cv_actual = _resolve_cv_splits(y, groups.tolist(), cv_splits)
    splitter = StratifiedGroupKFold(n_splits=cv_actual, shuffle=True, random_state=random_state)

    all_true: List[int] = []
    all_pred: List[int] = []
    all_ids: List[str] = []
    all_folds: List[int] = []
    all_proba: List[np.ndarray] = []
    confusion_total = np.zeros((len(class_indices), len(class_indices)), dtype=int)
    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        selector = None
        if feature_k and 0 < feature_k < X_train_scaled.shape[1]:
            selector = SelectKBest(mutual_info_classif, k=feature_k)
            X_train_scaled = selector.fit_transform(X_train_scaled, y_train)
            X_test_scaled = selector.transform(X_test_scaled)

        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=None if class_weight in (None, "none") else class_weight,
            multi_class="auto",
            solver="lbfgs",
        )
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        proba_fold = clf.predict_proba(X_test_scaled) if hasattr(clf, "predict_proba") else None
        cm_fold = confusion_matrix(y_test, y_pred, labels=class_indices)
        confusion_total += cm_fold

        fold_metrics.append(
            {
                "fold": fold_idx,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "f1_macro": float(
                    precision_recall_fscore_support(
                        y_test, y_pred, average="macro", zero_division=0
                    )[2]
                ),
            }
        )

        clf_classes = clf.classes_
        if proba_fold is not None:
            aligned = np.zeros((proba_fold.shape[0], len(class_indices)))
            for col_idx, cls in enumerate(class_indices):
                if cls in clf_classes:
                    aligned[:, col_idx] = proba_fold[:, np.where(clf_classes == cls)[0][0]]
            proba_fold = aligned

        all_ids.extend(np.array(sample_ids)[test_idx])
        all_folds.extend([fold_idx] * len(test_idx))
        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())
        if proba_fold is not None:
            all_proba.extend(list(proba_fold))

    y_true = np.array(all_true, dtype=int)
    y_pred = np.array(all_pred, dtype=int)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    proba_array = np.vstack(all_proba) if all_proba else None
    roc_auc = None
    avg_precision = None
    if proba_array is not None and len(class_indices) > 1:
        try:
            y_bin = label_binarize(y_true, classes=class_indices)
            roc_auc = float(roc_auc_score(y_bin, proba_array, average="macro", multi_class="ovr"))
            avg_precision = float(average_precision_score(y_bin, proba_array, average="macro"))
        except ValueError:
            roc_auc = None

    report = classification_report(
        y_true,
        y_pred,
        labels=class_indices,
        target_names=class_labels,
        zero_division=0,
        output_dict=True,
    )

    scaler_full = StandardScaler().fit(X)
    X_full_scaled = scaler_full.transform(X)
    selector_full = None
    if feature_k and 0 < feature_k < X_full_scaled.shape[1]:
        selector_full = SelectKBest(mutual_info_classif, k=feature_k).fit(X_full_scaled, y)
        X_full_selected = selector_full.transform(X_full_scaled)
        selected_names = [feature_names[i] for i in selector_full.get_support(indices=True)]
    else:
        X_full_selected = X_full_scaled
        selected_names = feature_names

    clf_full = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=None if class_weight in (None, "none") else class_weight,
        multi_class="auto",
        solver="lbfgs",
    ).fit(X_full_selected, y)

    coef = clf_full.coef_
    if coef.ndim == 1:
        coef = coef[np.newaxis, :]
    if coef.shape[0] == 1 and len(class_indices) == 2:
        coef_matrix = np.vstack([coef[0], -coef[0]])
    else:
        coef_matrix = coef

    top_features: Dict[str, List[Dict[str, float]]] = {}
    for idx, cls in enumerate(clf_full.classes_):
        weights = coef_matrix[min(idx, coef_matrix.shape[0] - 1)]
        order = np.argsort(weights)[::-1]
        top_features[label_map[int(cls)]] = [
            {"feature": selected_names[j], "weight": float(weights[j])}
            for j in order[: min(10, len(selected_names))]
        ]

    metrics: Dict[str, Any] = {
        "classes": class_labels,
        "n_samples": int(len(X)),
        "cv_splits": cv_actual,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "roc_auc_ovr": roc_auc,
        "average_precision": avg_precision,
        "confusion_matrix": {
            "labels": class_labels,
            "matrix": confusion_total.astype(int).tolist(),
        },
        "classification_report": report,
        "fold_metrics": fold_metrics,
        "top_features": top_features,
    }

    if not return_model:
        return metrics

    pred_records: List[Dict[str, Any]] = []
    if proba_array is not None:
        prob_columns = [f"proba_{label_map[int(cls)]}" for cls in class_indices]
        for sid, fold, true, pred, proba_row in zip(all_ids, all_folds, y_true, y_pred, proba_array):
            row = {
                "sample_id": sid,
                "fold": int(fold),
                "label_index": int(true),
                "label": label_map[int(true)],
                "prediction_index": int(pred),
                "prediction": label_map[int(pred)],
            }
            for col_name, value in zip(prob_columns, proba_row):
                row[col_name] = float(value)
            pred_records.append(row)
    else:
        for sid, fold, true, pred in zip(all_ids, all_folds, y_true, y_pred):
            pred_records.append(
                {
                    "sample_id": sid,
                    "fold": int(fold),
                    "label_index": int(true),
                    "label": label_map[int(true)],
                    "prediction_index": int(pred),
                    "prediction": label_map[int(pred)],
                }
            )

    eval_df = pd.DataFrame(pred_records)

    bundle = {
        "scaler": scaler_full,
        "selector": selector_full,
        "clf": clf_full,
        "feature_names": feature_names,
        "selected_feature_names": selected_names,
        "label_map": label_map,
        "class_indices": class_indices.tolist(),
        "clr_indices": clr_indices,
        "pseudocount": pseudocount,
    }

    return metrics, bundle, eval_df

