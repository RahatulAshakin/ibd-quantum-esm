"""Utilities for ingesting and storing multi-omics feature matrices."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import duckdb
import numpy as np
import pandas as pd


@dataclass
class MultiOmicsDataset:
    ids: List[str]
    X: np.ndarray
    y: np.ndarray
    y_original: pd.Series
    feature_names: List[str]
    dropped_features: List[str]
    groups: Optional[List[str]] = None

    @property
    def classes(self) -> List[str]:
        return sorted(pd.Series(self.y_original).astype(str).unique())


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def load_multiomics_csv(
    path: str,
    id_col: str,
    label_col: str,
    feature_cols: Optional[Iterable[str]] = None,
    min_variance: float = 1e-9,
    group_col: Optional[str] = None,
) -> MultiOmicsDataset:
    """Load a CSV containing patient-level multi-omics features.

    Parameters
    ----------
    path : str
        CSV file with at least an ID column, a label column, and one or more
        numeric feature columns.
    id_col : str
        Column used as patient/sample identifier.
    label_col : str
        Column with clinical labels (str or numeric).
    feature_cols : iterable[str], optional
        Explicit subset of feature columns. If omitted, all numeric columns
        except the ID/label columns are used.
    min_variance : float
        Features with variance <= min_variance are dropped to avoid degenerate
        dimensions.
    """
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"Missing id column '{id_col}' in {path}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in {path}")

    # Determine usable features
    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        features = list(feature_cols)
    else:
        numeric_cols = _numeric_columns(df)
        features = [c for c in numeric_cols if c not in {id_col, label_col}]
        if not features:
            raise ValueError("No numeric feature columns detected; specify --feature-col")

    initial_features = list(features)
    df_feats = df[features].apply(pd.to_numeric, errors="coerce")
    df_feats = df_feats.replace([np.inf, -np.inf], np.nan)
    # Impute with per-feature median
    df_feats = df_feats.fillna(df_feats.median())

    # Drop low variance features
    variances = df_feats.var(axis=0)
    keep_mask = variances > float(min_variance)
    kept = variances[keep_mask].index.tolist()
    if not kept:
        raise ValueError("All requested features have near-zero variance")
    dropped = [f for f in initial_features if f not in kept]
    if dropped:
        df_feats = df_feats[kept]
        features = kept

    ids = df[id_col].astype(str).tolist()
    groups = df[group_col].astype(str).tolist() if group_col and group_col in df.columns else None
    labels = df[label_col]

    if labels.isnull().any():
        raise ValueError("Labels contain null values; please clean your dataset")

    # Preserve original labels, but also encode numerically (for scikit-learn)
    y_original = labels.astype(str)
    y_codes, uniques = pd.factorize(y_original, sort=True)

    # Ensure deterministic ordering for classes
    # pd.factorize(sort=True) already sorts by unique labels
    X = df_feats.to_numpy(dtype=np.float32)
    y = y_codes.astype(int)

    return MultiOmicsDataset(
        ids=ids,
        X=X,
        y=y,
        y_original=y_original,
        feature_names=features,
        dropped_features=dropped,
        groups=groups,
    )


def append_multiomics_to_duckdb(
    dataset: MultiOmicsDataset,
    db_path: str,
    table: str = "multiomics",
    overwrite: bool = False,
) -> None:
    """Write the multi-omics dataset to DuckDB."""
    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db))
    try:
        if overwrite:
            con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
              sample_id TEXT,
              label TEXT,
              label_index INTEGER,
              feature_names TEXT[],
              features DOUBLE[],
              group_id TEXT
            )
            """
        )
        df = pd.DataFrame(
            {
                "sample_id": dataset.ids,
                "label": dataset.y_original.astype(str).tolist(),
                "label_index": dataset.y.tolist(),
                "feature_names": [dataset.feature_names] * len(dataset.ids),
                "features": [row.astype(float).tolist() for row in dataset.X],
                "group_id": dataset.groups if dataset.groups is not None else dataset.ids,
            }
        )
        con.register("df", df)
        con.execute(f"INSERT INTO {table} SELECT * FROM df")
    finally:
        con.close()
