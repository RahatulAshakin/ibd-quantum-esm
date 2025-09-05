# src/ibdqlib/train.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import duckdb
import numpy as np
from sklearn.metrics import accuracy_score
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
    import pandas as pd

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
