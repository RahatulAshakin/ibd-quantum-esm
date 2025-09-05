from __future__ import annotations
from pathlib import Path
import json
from typing import Optional, Tuple
import numpy as np
import duckdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .quantum import make_qsvc


def load_embeddings(db_path: str, table: str = "embeddings") -> Tuple[np.ndarray, list[dict]]:
    con = duckdb.connect(db_path)
    rows = con.execute(f"select seq_id, length, backend, dim, embedding from {table}").fetchall()
    con.close()
    if not rows:
        raise RuntimeError(f"No rows found in {db_path}:{table}")
    X = np.array([np.array(r[4], dtype=np.float32) for r in rows], dtype=np.float32)
    meta = [{"seq_id": r[0], "length": int(r[1]), "backend": r[2], "dim": int(r[3])} for r in rows]
    return X, meta


def build_labels(meta: list[dict], rule: str = "median-length", labels_csv: Optional[str] = None, label_col: str = "label") -> np.ndarray:
    import pandas as pd
    if labels_csv:
        df = pd.read_csv(labels_csv)
        if "seq_id" not in df.columns or label_col not in df.columns:
            raise ValueError("labels CSV must contain columns: seq_id and your label column (default: 'label')")
        m = {str(r["seq_id"]): r[label_col] for _, r in df.iterrows()}
        y = np.array([m.get(d["seq_id"]) for d in meta])
        if any(v is None for v in y):
            raise ValueError("Some seq_id from embeddings are missing in the labels CSV.")
        # Convert any non-binary labels to integers if possible
        if y.dtype.kind not in "biu":
            classes = {v: i for i, v in enumerate(sorted(set(y)))}
            y = np.array([classes[v] for v in y], dtype=np.int64)
        else:
            y = y.astype(np.int64)
        return y

    # rule-based label: 1 if length >= median else 0
    if rule == "median-length":
        lengths = np.array([d["length"] for d in meta], dtype=np.int64)
        med = float(np.median(lengths))
        y = (lengths >= med).astype(np.int64)
        return y

    raise ValueError(f"Unknown labeling rule: {rule}")


def train_qsvc_from_duckdb(
    db_path: str,
    table: str = "embeddings",
    labels_csv: Optional[str] = None,
    label_col: str = "label",
    rule: str = "median-length",
    test_size: float = 0.25,
    random_state: int = 42,
    feature_cap: int = 6,   # NEW: cap number of features (qubits)
) -> dict:
    X, meta = load_embeddings(db_path, table=table)
    y = build_labels(meta, rule=rule, labels_csv=labels_csv, label_col=label_col)

    # Reduce feature dimension for local simulation (avoid huge qubit counts).
    # Simple and safe: take the first K features.
    k = max(1, min(feature_cap, X.shape[1]))
    Xr = X[:, :k]

    clf = make_qsvc(feature_dim=Xr.shape[1], reps=2)

    metrics = {}
    unique_classes = np.unique(y)
    # Heuristic: split only if we have enough samples & at least 2 classes
    can_split = len(Xr) >= 6 and len(unique_classes) >= 2

    if can_split:
        Xtr, Xte, ytr, yte = train_test_split(Xr, y, test_size=test_size, random_state=random_state, stratify=y)
        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xte)
        metrics["test_accuracy"] = float(accuracy_score(yte, yhat))
        metrics["n_train"], metrics["n_test"] = int(len(Xtr)), int(len(Xte))
    else:
        clf.fit(Xr, y)
        yhat = clf.predict(Xr)
        metrics["train_accuracy"] = float(accuracy_score(y, yhat))
        metrics["n_train"] = int(len(Xr))
        metrics["note"] = (
            f"trained on all data (too few samples for a split); "
            f"feature_cap={k} of {X.shape[1]} original dims"
        )

    metrics["classes"] = [int(c) for c in unique_classes.tolist()]
    metrics["used_features"] = int(k)
    return metrics



def save_metrics(metrics: dict, out_path: str | Path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
