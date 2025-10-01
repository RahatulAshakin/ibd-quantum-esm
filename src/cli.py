# src/cli.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Tuple

import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import KernelCenterer
import typer
from joblib import dump, load
from json import dumps


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .ibdqlib.qkernel import quantum_kernel_mats




import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-GUI backend so PNGs save fine on any machine
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



from .ibdqlib.runtime import list_backends
from .ibdqlib.esm_embed import get_embedder, read_fasta
from .ibdqlib.train import train_qsvc_from_duckdb, train_multiomics_classifier
from .ibdqlib.esmatlas import fold_sequence
from .ibdqlib.omics import load_multiomics_csv, append_multiomics_to_duckdb


from .ibdqlib.qesm import qesm_find_positions


app = typer.Typer(help="IBD Quantum + ESM pipeline")


# -----------------------
# Basic sanity commands
# -----------------------
@app.command("hello")
def hello_cmd() -> None:
    """Sanity check command."""
    print("IBD Quantum-ESM scaffold ready.")


@app.command("version")
def version_cmd() -> None:
    """Show the package version."""
    print("ibd-quantum-esm 0.0.1")


# -----------------------
# Embed sequences
# -----------------------
@app.command("embed")
def embed_cmd(
    input: str = typer.Argument(..., help="Path to FASTA (.fa/.fasta) or CSV file"),
    outdb: str = typer.Option("results/duckdb/embeddings.duckdb", "--outdb", help="DuckDB database file"),
    backend: str = typer.Option(
        "dummy",
        "--backend",
        help="Embedding backend: 'dummy' or 'esm2_t6_8M_UR50D'",
    ),
    batch: int = typer.Option(64, "--batch", help="Batch size for embedding"),
) -> None:
    """
    Embed protein sequences and store vectors in DuckDB (table: embeddings).
    CSV must contain a 'sequence' column; ID is taken from 'id'/'seq_id'/'name' if present,
    otherwise synthetic ids are created.
    """
    in_path = Path(input)
    if not in_path.exists():
        typer.secho(f"Input file not found: {in_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    # ---- read sequences ----
    if in_path.suffix.lower() in {".fa", ".fasta", ".fna"}:
        pairs = read_fasta(str(in_path))
        ids = [i for i, _ in pairs]
        seqs = [s for _, s in pairs]
    else:
        df_in = pd.read_csv(in_path)
        seq_col = next((c for c in df_in.columns if c.lower() in ["sequence", "seq", "aa_sequence", "protein_sequence"]), None)
        if not seq_col:
            raise RuntimeError("CSV must contain a 'sequence' column")
        id_col = next((c for c in df_in.columns if c.lower() in ["id", "seq_id", "name", "accession"]), None)
        ids = (df_in[id_col].astype(str).tolist()) if id_col else [f"seq_{i+1}" for i in range(len(df_in))]
        seqs = df_in[seq_col].astype(str).tolist()

    if not seqs:
        typer.secho("No sequences found.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    embed_fn, dim, backend_used = get_embedder(backend)

    # ---- batch embed ----
    embs: List[np.ndarray] = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        vecs = embed_fn(chunk)  # expected shape (len(chunk), dim)
        embs.append(np.asarray(vecs, dtype=np.float32))
    embs = np.vstack(embs)
    lengths = [len(s) for s in seqs]

    df = pd.DataFrame({
        "seq_id": ids,
        "length": lengths,
        "backend": backend_used,
        "dim": int(dim),
        "embedding": [row.astype(float).tolist() for row in embs],  # DuckDB LIST<FLOAT>
    })

    # ---- write to DuckDB ----
    out_path = Path(outdb)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(out_path))
    con.execute("""
        create table if not exists embeddings (
          seq_id text,
          length integer,
          backend text,
          dim integer,
          embedding float[]
        );
    """)
    con.register("df", df)
    con.execute("insert into embeddings select * from df")
    con.close()

    typer.secho(
        f"Inserted {len(df)} rows into {out_path} (table: embeddings) using backend='{backend_used}', dim={dim}.",
        fg=typer.colors.GREEN,
    )


@app.command("ingest-omics")
def ingest_omics_cmd(
    input: str = typer.Argument(..., help="CSV file with patient-level multi-omics features"),
    outdb: str = typer.Option("results/duckdb/embeddings.duckdb", "--outdb", help="DuckDB database file"),
    table: str = typer.Option("multiomics", "--table", help="Destination table name"),
    id_col: str = typer.Option("patient_id", "--id-col", help="Identifier column name"),
    label_col: str = typer.Option("clinical_status", "--label-col", help="Label column name"),
    feature_col: Optional[List[str]] = typer.Option(
        None,
        "--feature-col",
        help="Specify one or more feature columns (repeat --feature-col for each)",
        show_default=False,
    ),
    min_variance: float = typer.Option(1e-8, "--min-variance", help="Drop features with variance <= threshold"),
    overwrite: bool = typer.Option(False, "--overwrite/--append", help="Overwrite the DuckDB table before insert"),
    group_col: Optional[str] = typer.Option(None, "--group-col", help="Optional column with subject IDs for grouped CV"),
) -> None:
    """Ingest a multi-omics CSV and persist it to DuckDB for training."""

    in_path = Path(input)
    if not in_path.exists():
        typer.secho(f"Input file not found: {in_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    try:
        dataset = load_multiomics_csv(
            str(in_path),
            id_col=id_col,
            label_col=label_col,
            feature_cols=feature_col or None,
            min_variance=min_variance,
            group_col=group_col,
        )
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        typer.secho(f"Failed to load multi-omics data: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    append_multiomics_to_duckdb(dataset, db_path=outdb, table=table, overwrite=overwrite)

    class_counts = dataset.y_original.value_counts().to_dict()
    typer.secho(
        f"Inserted {len(dataset.ids)} samples with {len(dataset.feature_names)} feature(s) into {outdb} (table: {table}).",
        fg=typer.colors.GREEN,
    )
    typer.echo("Label distribution:")
    for label, count in sorted(class_counts.items()):
        typer.echo(f"  {label}: {count}")
    if dataset.dropped_features:
        typer.secho(
            "Dropped low-variance features: " + ", ".join(dataset.dropped_features),
            fg=typer.colors.YELLOW,
        )
    typer.echo("Feature columns: " + ", ".join(dataset.feature_names))


# -----------------------
# IBM Runtime smoke test
# -----------------------
@app.command("runtime-test")
def runtime_test_cmd(
    limit: int = typer.Option(10, "--limit", help="Number of backends to list")
) -> None:
    """
    Smoke test IBM Runtime credentials: lists a few backends.
    Does NOT submit any jobs.
    """
    try:
        rows = list_backends(limit=limit)
    except Exception as e:
        typer.secho(f"IBM Runtime auth failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.secho(f"Found {len(rows)} backend(s):", fg=typer.colors.GREEN)
    for name, n_qubits, is_sim in rows:
        typer.echo(f"- {name:20s}  qubits={n_qubits:>3}  simulator={is_sim}")


# -----------------------
# Query DB preview/stats
# -----------------------
@app.command("query")
def query_cmd(
    db: str = typer.Option("results/duckdb/embeddings.duckdb", "--db", help="DuckDB database file"),
    limit: int = typer.Option(5, "--limit", help="Preview N rows"),
    stats: bool = typer.Option(True, "--stats/--no-stats", help="Print summary stats"),
) -> None:
    """
    Print a small preview and stats from the embeddings table.
    """
    if not os.path.exists(db):
        typer.secho(f"Database not found: {db}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    con = duckdb.connect(db)
    try:
        total = con.execute("select count(*) from embeddings").fetchone()[0]
        typer.echo(f"rows={total}")

        rows = con.execute(
            "select seq_id, length, backend, dim from embeddings limit ?",
            [limit],
        ).fetchall()
        for r in rows:
            seq_id, length, backend, dim = r
            typer.echo(f"seq_id={seq_id} length={length} backend={backend} dim={dim}")

        if stats:
            agg = con.execute("""
                select backend, dim, count(*) as n, min(length) as min_len, max(length) as max_len
                from embeddings
                group by 1,2
                order by n desc
            """).fetchall()
            for backend, dim, n, mn, mx in agg:
                typer.echo(f"stat backend={backend} dim={dim} n={n} minlen={mn} maxlen={mx}")
    finally:
        con.close()


# -----------------------
# Train QSVC and save bundle
# -----------------------
@app.command("train-qsvc")
def train_qsvc_cmd(
    db: str = typer.Option("results/duckdb/embeddings.duckdb", "--db"),
    table: str = typer.Option("embeddings", "--table"),
    labels_csv: Optional[str] = typer.Option(None, "--labels-csv", help="Optional CSV with labels"),
    label_col: str = typer.Option("label", "--label-col"),
    rule: str = typer.Option("median-length", "--rule", help="Rule if no labels CSV given"),
    feature_cap: Optional[int] = typer.Option(None, "--feature-cap", help="Cap embedding dims for quantum kernel"),
    out: str = typer.Option("results/metrics/qsvc.json", "--out"),
    model_out: str = typer.Option("results/models/qsvc.joblib", "--model-out"),
) -> None:
    """
    Train a QSVC on embeddings and write metrics JSON (+ save model bundle).
    The bundle contains {"clf": fitted_svc, "Xref": training_features}.
    """
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)

    metrics, clf, Xref = train_qsvc_from_duckdb(
        db, table=table, labels_csv=labels_csv, label_col=label_col,
        rule=rule, feature_cap=feature_cap, return_model=True
    )

    # Save metrics
    Path(out).write_text(dumps(metrics, indent=2))
    # Save a bundle for prediction
    dump({"clf": clf, "Xref": Xref}, model_out)

    typer.secho(
        f"Wrote metrics to {out} and model to {model_out}: {metrics}",
        fg=typer.colors.GREEN,
    )



@app.command("train-omics")
def train_omics_cmd(
    db: str = typer.Option("results/duckdb/ibd.duckdb", "--db"),
    table: str = typer.Option("multiomics", "--table"),
    random_state: int = typer.Option(42, "--random-state"),
    test_size: Optional[float] = typer.Option(
        None,
        "--test-size",
        help="Compatibility shim; cross-validation already handles splitting so this value is ignored.",
        show_default=False,
    ),
    C: float = typer.Option(1.0, "--C", help="Inverse regularisation strength for logistic regression"),
    class_weight: str = typer.Option("balanced", "--class-weight", help="Class weight strategy: balanced or none"),
    cv_splits: int = typer.Option(5, "--cv-splits", help="Number of StratifiedGroupKFold splits"),
    feature_k: Optional[int] = typer.Option(None, "--feature-k", help="Select top-K features via mutual information inside each fold"),
    pseudocount: float = typer.Option(1e-6, "--pseudocount", help="Pseudocount for CLR transformation"),
    max_iter: int = typer.Option(1000, "--max-iter", help="Max iterations for logistic regression"),
    out: str = typer.Option("results/metrics/omics_classifier.json", "--out"),
    model_out: str = typer.Option("results/models/omics_classifier.joblib", "--model-out"),
    pred_out: str = typer.Option("results/predictions_omics.csv", "--pred-out"),
) -> None:
    """Train a logistic-regression classifier on stored multi-omics features."""

    class_weight_opt = class_weight.lower().strip() if class_weight else "balanced"
    if class_weight_opt not in {"balanced", "none"}:
        typer.secho("--class-weight must be 'balanced' or 'none'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    if test_size is not None:
        typer.secho("train-omics uses cross-validation; --test-size is accepted for compatibility but ignored.", fg=typer.colors.YELLOW)

    try:
        metrics, bundle, eval_df = train_multiomics_classifier(
            db_path=db,
            table=table,
            random_state=random_state,
            C=C,
            class_weight=class_weight_opt,
            cv_splits=cv_splits,
            feature_k=feature_k,
            pseudocount=pseudocount,
            max_iter=max_iter,
            return_model=True,
        )
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        typer.secho(f"Training failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(pred_out).parent.mkdir(parents=True, exist_ok=True)

    Path(out).write_text(dumps(metrics, indent=2))
    dump(bundle, model_out)
    eval_df.to_csv(pred_out, index=False)

    roc_display = f"{metrics['roc_auc_ovr']:.3f}" if metrics["roc_auc_ovr"] is not None else "n/a"
    typer.secho(
        (
            f"Multi-omics classifier metrics saved to {out}; model -> {model_out}; "
            f"predictions -> {pred_out}. F1_macro={metrics['f1_macro']:.3f}, ROC_AUC={roc_display}"
        ),
        fg=typer.colors.GREEN,
    )








def _center_kernel_train(Ktr: np.ndarray) -> np.ndarray:
    n = Ktr.shape[0]
    one = np.ones((n, n), dtype=Ktr.dtype) / n
    return Ktr - one @ Ktr - Ktr @ one + one @ Ktr @ one


def _center_kernel_test(Kte: np.ndarray, Ktr: np.ndarray) -> np.ndarray:
    # center test kernel against train statistics
    r_tr = Ktr.mean(axis=0, keepdims=True)   # (1, n)
    mu_tr = Ktr.mean()                        # scalar
    r_te = Kte.mean(axis=1, keepdims=True)    # (m, 1)
    return Kte - r_te - r_tr + mu_tr


@app.command("train-qsvc-quantum")
def train_qsvc_quantum_cmd(
    db: str = typer.Option("results/duckdb/uniprot.duckdb", "--db"),
    table: str = typer.Option("embeddings", "--table"),
    labels_csv: Optional[str] = typer.Option(None, "--labels-csv"),
    label_col: str = typer.Option("ec_top", "--label-col"),
    pca_components: int = typer.Option(12, "--pca-components", help="#qubits = this PCA dim"),
    angle_scale: float = typer.Option(1.0, "--angle-scale", help="Multiply PCA features before encoding"),
    angle_scale_grid: Optional[List[float]] = typer.Option(None, "--angle-scale-grid", help="Sweep of angle scales"),
    reps: int = typer.Option(1, "--reps", help="ZZ feature-map repetitions"),
    reps_grid: Optional[List[int]] = typer.Option(None, "--reps-grid", help="Sweep of repetitions"),
    entanglement: str = typer.Option("linear", "--entanglement", help="Entanglement pattern"),
    entanglement_grid: Optional[List[str]] = typer.Option(None, "--entanglement-grid", help="Sweep of entanglement patterns"),
    per_class_limit: Optional[int] = typer.Option(200, "--per-class-limit", help="Max train samples per class"),
    backend: str = typer.Option("statevector", "--backend", help="Shot-based backends are not yet wired; accepts 'statevector'"),
    shots: int = typer.Option(1024, "--shots", help="Shot count for sampling backends (ignored for statevector simulation)"),
    C: float = typer.Option(1.0, "--C"),
    class_weight: str = typer.Option("balanced", "--class-weight", help="balanced or none"),
    test_size: float = typer.Option(0.25, "--test-size"),
    random_state: int = typer.Option(42, "--random-state"),
    out: str = typer.Option("results/metrics/qsvc_quantum.json", "--out"),
    model_out: str = typer.Option("results/models/qsvc_quantum.joblib", "--model-out"),
):
    """Train a QSVC with fidelity quantum kernels on PCA-compressed embeddings."""
    from pathlib import Path
    import numpy as np
    import duckdb
    import pandas as pd
    from joblib import dump
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.preprocessing import KernelCenterer
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)

    class_weight_opt = class_weight.lower().strip() if class_weight else "balanced"
    if class_weight_opt not in {"balanced", "none"}:
        typer.secho("--class-weight must be 'balanced' or 'none'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    class_weight_kw = None if class_weight_opt == "none" else "balanced"

    backend_norm = backend.strip().lower() if backend else "statevector"
    if backend_norm != "statevector":
        typer.secho("Only 'statevector' backend is currently supported; defaulting to statevector simulator.", fg=typer.colors.YELLOW)
        backend_norm = "statevector"

    if shots <= 0:
        typer.secho("--shots must be a positive integer", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    angle_values = angle_scale_grid or [angle_scale]
    reps_values = reps_grid or [reps]
    entanglement_values = entanglement_grid or [entanglement]

    con = duckdb.connect(db)
    try:
        rows = con.execute(f"SELECT seq_id, embedding FROM {table}").fetchall()
    except duckdb.Error as exc:
        raise RuntimeError(f"Failed to read embeddings from table '{table}': {exc}") from exc
    finally:
        con.close()

    if not rows:
        raise RuntimeError("No embeddings available for training.")

    seq_ids = [r[0] for r in rows]
    X = np.stack([np.asarray(r[1], dtype=np.float64) for r in rows])

    if not labels_csv:
        typer.secho("labels_csv is required for supervised quantum training.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    labels_df = pd.read_csv(labels_csv)
    id_col = next((c for c in labels_df.columns if c.lower() in ["seq_id", "id", "name", "accession"]), None)
    if not id_col:
        raise RuntimeError("labels CSV must contain seq_id/id/name/accession column")
    label_map = dict(zip(labels_df[id_col].astype(str), labels_df[label_col]))
    try:
        y = np.array([label_map[sid] for sid in seq_ids])
    except KeyError as exc:
        missing = [sid for sid in seq_ids if sid not in label_map]
        raise RuntimeError(f"{len(missing)} seq_ids missing labels (e.g., {missing[:5]})") from exc

    if per_class_limit is not None and per_class_limit > 0:
        keep_idx = []
        df_idx = pd.DataFrame({"i": np.arange(len(y)), "y": y})
        for cls, grp in df_idx.groupby("y", sort=True):
            take = min(len(grp), per_class_limit)
            keep_idx.extend(grp.sample(n=take, random_state=random_state)["i"].tolist())
        keep_idx = sorted(keep_idx)
        X = X[keep_idx]
        y = y[keep_idx]
        seq_ids = [seq_ids[i] for i in keep_idx]

    if len(np.unique(y)) < 2:
        typer.secho("Need at least two classes for QSVC training", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    if pca_components > X.shape[1]:
        pca_components = X.shape[1]

    X_train, X_test, y_train, y_test, sid_train, sid_test = train_test_split(
        X,
        y,
        seq_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=pca_components, random_state=random_state).fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    baseline = SVC(C=C, kernel="rbf", class_weight=class_weight_kw, gamma="scale")
    baseline.fit(X_train_pca, y_train)
    baseline_pred = baseline.predict(X_test_pca)
    baseline_prec, baseline_rec, baseline_f1, _ = precision_recall_fscore_support(
        y_test, baseline_pred, average="macro", zero_division=0
    )
    baseline_acc = accuracy_score(y_test, baseline_pred)

    best = None
    for ang in angle_values:
        for reps_val in reps_values:
            for ent_val in entanglement_values:
                feature_map = ZZFeatureMap(feature_dimension=pca_components, reps=reps_val, entanglement=ent_val)
                kernel = FidelityQuantumKernel(feature_map=feature_map)
                try:
                    K_train = kernel.evaluate(x_vec=X_train_pca * ang)
                except Exception as exc:
                    typer.secho(f"Kernel evaluation failed for angle={ang}, reps={reps_val}, entanglement={ent_val}: {exc}", fg=typer.colors.YELLOW)
                    continue
                centerer = KernelCenterer().fit(K_train)
                K_train_centered = centerer.transform(K_train)
                svc = SVC(C=C, kernel="precomputed", class_weight=class_weight_kw)
                svc.fit(K_train_centered, y_train)
                K_test = kernel.evaluate(x_vec=X_test_pca * ang, y_vec=X_train_pca * ang)
                K_test_centered = centerer.transform(K_test)
                pred = svc.predict(K_test_centered)
                prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
                acc = accuracy_score(y_test, pred)
                bal_acc = balanced_accuracy_score(y_test, pred)
                result = {
                    "svc": svc,
                    "kernel": kernel,
                    "centerer": centerer,
                    "angle": ang,
                    "reps": reps_val,
                    "entanglement": ent_val,
                    "pred": pred,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                }
                if best is None or result["f1"] > best["f1"]:
                    best = result

    if best is None:
        raise RuntimeError("No valid quantum kernel configuration succeeded.")

    scaler_full = StandardScaler().fit(X)
    X_full_scaled = scaler_full.transform(X)
    pca_full = PCA(n_components=pca_components, random_state=random_state).fit(X_full_scaled)
    X_full_pca = pca_full.transform(X_full_scaled)
    X_full_processed = X_full_pca * best["angle"]
    feature_map_final = ZZFeatureMap(feature_dimension=pca_components, reps=best["reps"], entanglement=best["entanglement"])
    kernel_final = FidelityQuantumKernel(feature_map=feature_map_final)
    K_full = kernel_final.evaluate(x_vec=X_full_processed)
    centerer_full = KernelCenterer().fit(K_full)
    K_full_centered = centerer_full.transform(K_full)
    svc_final = SVC(C=C, kernel="precomputed", class_weight=class_weight_kw)
    svc_final.fit(K_full_centered, y)

    classes_sorted = sorted(set(y.tolist()))
    metrics = {
        "used_features": int(pca_components),
        "classes": classes_sorted,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "quantum_accuracy": float(best["accuracy"]),
        "quantum_precision_macro": float(best["precision"]),
        "quantum_recall_macro": float(best["recall"]),
        "quantum_f1_macro": float(best["f1"]),
        "quantum_balanced_accuracy": float(best["balanced_accuracy"]),
        "baseline_accuracy": float(baseline_acc),
        "baseline_precision_macro": float(baseline_prec),
        "baseline_recall_macro": float(baseline_rec),
        "baseline_f1_macro": float(baseline_f1),
        "accuracy_delta": float(best["accuracy"] - baseline_acc),
        "best_params": {
            "angle_scale": best["angle"],
            "reps": best["reps"],
            "entanglement": best["entanglement"],
        },
        "kernel_diagnostics": {
            "backend": backend_norm,
            "shots": int(shots),
            "angle_grid": [float(a) for a in angle_values],
            "reps_grid": [int(r) for r in reps_values],
            "entanglement_grid": [str(e) for e in entanglement_values],
            "candidates_tested": int(len(angle_values) * len(reps_values) * len(entanglement_values)),
        },
    }
    Path(out).write_text(dumps(metrics, indent=2))
    typer.secho(
        f"Best quantum F1={best['f1']:.3f} (acc={best['accuracy']:.3f}) vs baseline F1={baseline_f1:.3f} (acc={baseline_acc:.3f}).",
        fg=typer.colors.GREEN,
    )

    eval_df = pd.DataFrame({"sample_id": sid_test, "label": y_test, "prediction": best["pred"]})

    bundle = {
        "svc": svc_final,
        "scaler": scaler_full,
        "pca": pca_full,
        "angle_scale": best["angle"],
        "feature_map": {
            "type": "zz",
            "reps": best["reps"],
            "entanglement": best["entanglement"],
        },
        "centerer": centerer_full,
        "X_train_processed": X_full_processed.astype(np.float32),
        "class_labels": classes_sorted,
        "class_weight": class_weight_opt,
        "label_lookup": label_map,
    }
    dump(bundle, model_out)
    eval_df.to_csv(model_out + ".preds.csv", index=False)
@app.command("predict")
def predict_cmd(
    input: str = typer.Argument(..., help="FASTA (.fa/.fasta) or CSV with a 'sequence' column"),
    model: str = typer.Option("results/models/qsvc.joblib", "--model"),
    backend: str = typer.Option("esm2_t6_8M_UR50D", "--backend", help="Embedding backend to match training"),
    batch: int = typer.Option(64, "--batch"),
    out: str = typer.Option("results/predictions.csv", "--out"),
) -> None:
    """
    Load a trained QSVC bundle and predict labels for new sequences.
    """
    in_path = Path(input)
    if not in_path.exists():
        typer.secho(f"Input file not found: {in_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    # read sequences
    if in_path.suffix.lower() in {".fa", ".fasta", ".fna"}:
        pairs = read_fasta(str(in_path))
        ids = [i for i, _ in pairs]
        seqs = [s for _, s in pairs]
    else:
        df_in = pd.read_csv(in_path)
        seq_col = next((c for c in df_in.columns if c.lower() in ["sequence", "seq", "aa_sequence", "protein_sequence"]), None)
        if not seq_col:
            raise RuntimeError("CSV must contain a 'sequence' column")
        id_col = next((c for c in df_in.columns if c.lower() in ["id", "seq_id", "name", "accession"]), None)
        ids = (df_in[id_col].astype(str).tolist()) if id_col else [f"seq_{i+1}" for i in range(len(df_in))]
        seqs = df_in[seq_col].astype(str).tolist()

    if not seqs:
        typer.secho("No sequences found.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    # embed with chosen backend (batched)
    embed_fn, dim, backend_used = get_embedder(backend)
    chunks: List[np.ndarray] = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        vecs = embed_fn(chunk)
        chunks.append(np.asarray(vecs, dtype=np.float32))
    X = np.vstack(chunks)

    # Load bundle saved by train: {"clf": clf, "Xref": Xref}
    bundle = load(model)
    if not isinstance(bundle, dict) or "clf" not in bundle or "Xref" not in bundle:
        typer.secho("Loaded model is not a bundle with {'clf','Xref'}. Re-train and save again.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    clf = bundle["clf"]
    Xref = np.asarray(bundle["Xref"], dtype=np.float32)

    # Ensure feature dims match training (handles --feature-cap used at train time)
    if X.shape[1] < Xref.shape[1]:
        typer.secho(
            f"Feature dimension ({X.shape[1]}) is smaller than training feature dim ({Xref.shape[1]}). "
            f"Use the same backend and feature_cap as training.",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(code=2)
    if X.shape[1] > Xref.shape[1]:
        X = X[:, :Xref.shape[1]]

    # Build precomputed kernel against the training features
    K = X @ Xref.T

    yhat = clf.predict(K)
    df_out = pd.DataFrame({"seq_id": ids, "prediction": yhat})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    typer.secho(
        f"Wrote predictions to {out} using model={model}, backend='{backend_used}', dim={dim}",
        fg=typer.colors.GREEN,
    )





@app.command("predict-qsvc-quantum")
def predict_qsvc_quantum_cmd(
    input: str = typer.Argument(..., help="FASTA or CSV with sequences"),
    model: str = typer.Option("results/models/qsvc_quantum.joblib", "--model"),
    batch: int = typer.Option(64, "--batch"),
    out: str = typer.Option("results/predictions_qsvc_quantum.csv", "--out"),
):
    """Predict with a trained quantum-kernel SVC bundle."""
    from joblib import load
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    import numpy as np

    in_path = Path(input)
    if not in_path.exists():
        typer.secho(f"Input file not found: {in_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    if in_path.suffix.lower() in {".fa", ".fasta", ".fna"}:
        pairs = read_fasta(str(in_path))
        ids = [i for i, _ in pairs]
        seqs = [s for _, s in pairs]
    else:
        df_in = pd.read_csv(in_path)
        seq_col = next((c for c in df_in.columns if c.lower() in ["sequence", "seq", "aa_sequence", "protein_sequence"]), None)
        if not seq_col:
            raise RuntimeError("CSV must contain a 'sequence' column")
        id_col = next((c for c in df_in.columns if c.lower() in ["id", "seq_id", "name", "accession"]), None)
        ids = (df_in[id_col].astype(str).tolist()) if id_col else [f"seq_{i+1}" for i in range(len(df_in))]
        seqs = df_in[seq_col].astype(str).tolist()

    embed_fn, dim, backend_used = get_embedder("esm2_t6_8M_UR50D")
    embeddings = []
    for i in range(0, len(seqs), batch):
        embeddings.append(embed_fn(seqs[i:i + batch]))
    X_new = np.vstack(embeddings).astype(np.float64)

    bundle = load(model)
    required_keys = {"svc", "scaler", "pca", "angle_scale", "feature_map", "centerer", "X_train_processed"}
    if not required_keys.issubset(bundle.keys()):
        missing = required_keys - set(bundle.keys())
        raise RuntimeError(f"Model bundle is missing keys: {missing}")

    svc = bundle["svc"]
    scaler = bundle["scaler"]
    pca = bundle["pca"]
    angle_scale = bundle["angle_scale"]
    feature_map_cfg = bundle["feature_map"]
    centerer = bundle["centerer"]
    X_train_processed = np.asarray(bundle["X_train_processed"], dtype=np.float64)

    X_new_scaled = scaler.transform(X_new)
    X_new_pca = pca.transform(X_new_scaled)
    X_new_processed = X_new_pca * angle_scale

    feature_map = ZZFeatureMap(
        feature_dimension=X_train_processed.shape[1],
        reps=feature_map_cfg["reps"],
        entanglement=feature_map_cfg["entanglement"],
    )
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    K_new = kernel.evaluate(x_vec=X_new_processed, y_vec=X_train_processed)
    K_new_centered = centerer.transform(K_new)
    yhat = svc.predict(K_new_centered)

    df_out = pd.DataFrame({"seq_id": ids, "prediction": yhat})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    typer.secho(f"Wrote predictions to {out} using quantum kernel bundle.", fg=typer.colors.GREEN)

@app.command("qesm-demo")
def qesm_demo_cmd(
    text: str = typer.Argument(..., help="Short text to search (e.g., ACGTAC)"),
    pattern: str = typer.Argument(..., help="Pattern to find (e.g., GTA)"),
    shots: int = typer.Option(1024, "--shots", help="Simulator shots"),
    max_iters: int = typer.Option(None, "--max-iters", help="Override Grover iterations"),
) -> None:
    """
    Quantum Exact String Matching (Grover) demo.
    Prints classical matches and quantum measurement results.
    """
    try:
        sols, counts, iters = qesm_find_positions(text, pattern, shots=shots, max_iters=max_iters)
    except RuntimeError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.secho(f"Text='{text}'  Pattern='{pattern}'", fg=typer.colors.CYAN)
    if not sols:
        typer.secho("No classical matches found.", fg=typer.colors.YELLOW)
        return

    typer.secho(f"Classical matches at indices: {sols}", fg=typer.colors.GREEN)
    typer.echo(f"Grover iterations: {iters}")
    # Pretty print top few bitstrings
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
    typer.echo("Top results (bitstring -> counts):")
    for bitstr, c in top:
        # bitstring is most-significant bit first; convert to int index
        idx_val = int(bitstr, 2)
        typer.echo(f"  {bitstr}  (idx={idx_val}) : {c}")


@app.command("fold")
def fold_cmd(
    input: str = typer.Argument(..., help="FASTA file with sequences to fold"),
    outdir: str = typer.Option("results/structures", "--outdir", help="Where to save PDBs"),
    limit: int = typer.Option(0, "--limit", help="Fold at most N sequences (0 = no limit)"),
    sanitize: bool = typer.Option(False, "--sanitize/--no-sanitize", help="Replace invalid tokens with 'X'"),
):
    """
    Fold sequences via ESM Atlas API. Writes PDB files named <seq_id>.pdb to --outdir.
    """
    in_path = Path(input)
    if not in_path.exists():
        typer.secho(f"Input file not found: {in_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    # read ids + sequences from FASTA
    pairs = read_fasta(str(in_path))
    if limit and limit > 0:
        pairs = pairs[:limit]

    ok_count = 0
    for seq_id, seq in pairs:
        ok, msg = fold_sequence(seq_id=seq_id, seq=seq, outdir=outdir, sanitize=sanitize)
        if ok:
            typer.secho(f"[ok]   {seq_id}: {msg}", fg=typer.colors.GREEN)
            ok_count += 1
        else:
            typer.secho(f"[fail] {seq_id}: {msg}", fg=typer.colors.RED)

    typer.echo(f"Folded {ok_count}/{len(pairs)} sequence(s) into {outdir}")




@app.command("report")
def report_cmd(
    db: str = typer.Option("results/duckdb/embeddings.duckdb", "--db"),
    pred: str = typer.Option("results/predictions.csv", "--pred", help="Predictions CSV for evaluation"),
    figdir: str = typer.Option("results/figures", "--figdir"),
    tabledir: str = typer.Option("results/tables", "--tabledir"),
) -> None:
    """Create figures/tables for the current dataset (multi-omics preferred)."""
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize

    figdir_path = Path(figdir)
    tabledir_path = Path(tabledir)
    figdir_path.mkdir(parents=True, exist_ok=True)
    tabledir_path.mkdir(parents=True, exist_ok=True)

    if not Path(db).exists():
        typer.secho(f"Database not found: {db}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    pred_path = Path(pred)
    emitted_figs: set[Path] = set()
    emitted_tables: set[Path] = set()

    def _report_embeddings(df: pd.DataFrame) -> None:
        if df.empty:
            raise RuntimeError("Embeddings table is empty; run embed first.")

        def _as_array(obj, dim):
            if isinstance(obj, (list, tuple, np.ndarray)):
                arr = np.asarray(obj, dtype=np.float32)
            else:
                arr = np.frombuffer(obj, dtype=np.float32)
            return arr[: int(dim)]

        X = np.vstack([_as_array(e, d) for e, d in zip(df["emb"], df["dim"])])
        lengths = df["length"].to_numpy()

        summary = (
            df.groupby(["backend", "dim"], as_index=False)
            .agg(n=("seq_id", "count"), min_len=("length", "min"), mean_len=("length", "mean"), max_len=("length", "max"))
        )
        summary_path = tabledir_path / "embeddings_summary.csv"
        summary.to_csv(summary_path, index=False)
        emitted_tables.add(summary_path)

        fig_hist = figdir_path / "length_hist.png"
        plt.figure()
        plt.hist(lengths, bins=min(30, max(5, int(len(lengths) ** 0.5))))
        plt.xlabel("Sequence length (aa)")
        plt.ylabel("Count")
        plt.title("Sequence Length Distribution")
        plt.tight_layout()
        plt.savefig(fig_hist, dpi=180)
        plt.close()
        emitted_figs.add(fig_hist)

        fig_kernel = figdir_path / "kernel_heatmap.png"
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        K = Xn @ Xn.T
        plt.figure()
        im = plt.imshow(K, vmin=-1, vmax=1, interpolation="nearest")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Embedding Cosine Similarity")
        plt.xlabel("Sequence index")
        plt.ylabel("Sequence index")
        plt.tight_layout()
        plt.savefig(fig_kernel, dpi=180)
        plt.close()
        emitted_figs.add(fig_kernel)

        if pred_path.exists():
            df_pred = pd.read_csv(pred_path)
            copy_path = tabledir_path / "predictions.csv"
            df_pred.to_csv(copy_path, index=False)
            emitted_tables.add(copy_path)

            if {"seq_id", "prediction"}.issubset(df_pred.columns):
                joined = pd.merge(
                    df[["seq_id", "length"]],
                    df_pred[["seq_id", "prediction"]],
                    on="seq_id",
                    how="inner",
                )
                y_pred = joined["prediction"].astype(int).to_numpy()
                lens = joined["length"].to_numpy()
                thresh = float(np.median(lens))
                y_true = (lens > thresh).astype(int)
                acc = accuracy_score(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                fig_cm = figdir_path / "confusion_matrix.png"
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
                disp.plot(values_format="d", cmap="Blues", colorbar=False)
                plt.title(f"Confusion Matrix (median-length heuristic) | Acc={acc:.3f}")
                plt.tight_layout()
                plt.savefig(fig_cm, dpi=180)
                plt.close()
                emitted_figs.add(fig_cm)
                return

        fig_cm = figdir_path / "confusion_matrix.png"
        y_true = (lengths > float(np.median(lengths))).astype(int)
        cm = confusion_matrix(y_true, y_true, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(values_format="d", cmap="Greys", colorbar=False)
        plt.title("Confusion Matrix (placeholder)")
        plt.tight_layout()
        plt.savefig(fig_cm, dpi=180)
        plt.close()
        emitted_figs.add(fig_cm)

    def _report_multiomics(df: pd.DataFrame) -> None:
        if df.empty:
            raise RuntimeError("Multi-omics table is empty; run ingest-omics first.")

        raw_names = df.loc[0, "feature_names"]
        if isinstance(raw_names, tuple):
            raw_names = raw_names[0]
        if isinstance(raw_names, np.ndarray):
            raw_names = raw_names.tolist()
        if not isinstance(raw_names, (list, tuple)):
            raise RuntimeError("feature_names column must contain list entries.")
        feature_names = list(raw_names)

        def _norm(names):
            if isinstance(names, tuple):
                names = names[0]
            if isinstance(names, np.ndarray):
                names = names.tolist()
            return list(names)

        if not df["feature_names"].apply(lambda names: _norm(names) == feature_names).all():
            raise RuntimeError("Inconsistent feature ordering detected across rows.")

        def _row_array(row):
            if isinstance(row, tuple):
                row = row[0]
            if isinstance(row, np.ndarray):
                row = row.tolist()
            return np.asarray(row, dtype=np.float32)

        X = np.vstack([_row_array(row) for row in df["features"]])
        labels = df["label"].astype(str)
        df_feat = pd.DataFrame(X, columns=feature_names)
        df_feat["label"] = labels

        counts_path = tabledir_path / "label_counts.csv"
        labels.value_counts().rename_axis("label").reset_index(name="count").to_csv(counts_path, index=False)
        emitted_tables.add(counts_path)

        means_path = tabledir_path / "feature_means.csv"
        df_feat.groupby("label")[feature_names].mean().to_csv(means_path)
        emitted_tables.add(means_path)

        stds_path = tabledir_path / "feature_stds.csv"
        df_feat.groupby("label")[feature_names].std(ddof=0).to_csv(stds_path)
        emitted_tables.add(stds_path)

        overall_stats_path = tabledir_path / "feature_overall_stats.csv"
        df_feat[feature_names].agg(["mean", "std", "min", "max"]).T.to_csv(overall_stats_path)
        emitted_tables.add(overall_stats_path)

        fig_corr = figdir_path / "feature_correlation.png"
        corr = np.corrcoef(df_feat[feature_names].to_numpy().T)
        plt.figure(figsize=(max(6, 0.6 * len(feature_names)), max(5, 0.6 * len(feature_names))))
        im = plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(feature_names)), feature_names, rotation=60, ha="right")
        plt.yticks(range(len(feature_names)), feature_names)
        plt.title("Feature Correlation")
        plt.tight_layout()
        plt.savefig(fig_corr, dpi=180)
        plt.close()
        emitted_figs.add(fig_corr)

        if len(feature_names) >= 2:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_feat[feature_names].to_numpy())
            n_components = min(2, X_scaled.shape[1])
            pca = PCA(n_components=n_components, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            fig_pca = figdir_path / "pca_scatter.png"
            plt.figure()
            uniq_labels = sorted(labels.unique())
            for lab in uniq_labels:
                mask = labels == lab
                plt.scatter(
                    X_pca[mask, 0],
                    X_pca[mask, 1] if n_components > 1 else np.zeros(sum(mask)),
                    label=lab,
                )
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
            if n_components > 1:
                plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
            else:
                plt.ylabel("PC2 (n/a)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_pca, dpi=180)
            plt.close()
            emitted_figs.add(fig_pca)

        if pred_path.exists():
            df_pred = pd.read_csv(pred_path)
            if "label" not in df_pred.columns:
                df_pred = df_pred.merge(df[["sample_id", "label"]], on="sample_id", how="left")

            if {"label", "prediction"}.issubset(df_pred.columns):
                y_true = df_pred["label"].astype(str)
                y_pred = df_pred["prediction"].astype(str)
                labels_sorted = sorted(set(y_true) | set(y_pred))

                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="macro", zero_division=0
                )
                precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="weighted", zero_division=0
                )
                acc = accuracy_score(y_true, y_pred)

                roc_metric = None
                pr_metric = None
                proba_cols = [f"proba_{lab}" for lab in labels_sorted]
                if all(col in df_pred.columns for col in proba_cols):
                    proba = df_pred[proba_cols].to_numpy()
                    try:
                        if len(labels_sorted) == 2:
                            pos_label = labels_sorted[1]
                            pos_idx = proba_cols.index(f"proba_{pos_label}")
                            y_binary = (y_true == pos_label).astype(int)
                            roc_metric = float(roc_auc_score(y_binary, proba[:, pos_idx]))
                            pr_metric = float(average_precision_score(y_binary, proba[:, pos_idx]))
                        else:
                            y_bin = label_binarize(y_true, classes=labels_sorted)
                            roc_metric = float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro"))
                    except ValueError:
                        roc_metric = None

                perf_path = tabledir_path / "performance_metrics.csv"
                perf_rows = [
                    ("accuracy", acc),
                    ("precision_macro", float(precision_macro)),
                    ("recall_macro", float(recall_macro)),
                    ("f1_macro", float(f1_macro)),
                    ("precision_weighted", float(precision_weighted)),
                    ("recall_weighted", float(recall_weighted)),
                    ("f1_weighted", float(f1_weighted)),
                ]
                if roc_metric is not None:
                    perf_rows.append(("roc_auc", roc_metric))
                if pr_metric is not None:
                    perf_rows.append(("average_precision", pr_metric))
                pd.DataFrame(perf_rows, columns=["metric", "value"]).to_csv(perf_path, index=False)
                emitted_tables.add(perf_path)

                cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
                cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
                cm_path = tabledir_path / "confusion_matrix.csv"
                cm_df.to_csv(cm_path)
                emitted_tables.add(cm_path)

                fig_cm = figdir_path / "confusion_matrix.png"
                fig, ax = plt.subplots(figsize=(4 + 0.5 * len(labels_sorted), 3 + 0.5 * len(labels_sorted)))
                disp = ConfusionMatrixDisplay(cm, display_labels=labels_sorted)
                disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
                plt.tight_layout()
                plt.savefig(fig_cm, dpi=180)
                plt.close(fig)
                emitted_figs.add(fig_cm)

                report = classification_report(
                    y_true,
                    y_pred,
                    labels=labels_sorted,
                    target_names=labels_sorted,
                    output_dict=True,
                    zero_division=0,
                )
                report_path = tabledir_path / "classification_report.csv"
                pd.DataFrame(report).to_csv(report_path)
                emitted_tables.add(report_path)

    con = duckdb.connect(db)
    try:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        if "multiomics" in tables:
            df = con.execute(
                "SELECT sample_id, label, label_index, feature_names, features FROM multiomics"
            ).fetch_df()
            _report_multiomics(df)
        elif "embeddings" in tables:
            cols = con.execute("PRAGMA table_info('embeddings')").fetchall()
            colnames = {c[1].lower() for c in cols}
            emb_col = "embedding" if "embedding" in colnames else ("vec" if "vec" in colnames else None)
            if emb_col is None:
                raise RuntimeError("No embedding/vec column found in embeddings table.")
            df = con.execute(
                f"SELECT seq_id, length, backend, dim, {emb_col} AS emb FROM embeddings"
            ).fetch_df()
            _report_embeddings(df)
        else:
            raise RuntimeError("Database lacks both 'multiomics' and 'embeddings' tables.")
    except Exception as exc:
        typer.secho(f"Report generation failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    finally:
        con.close()

    lines = ["Artifacts generated:"]
    for path in sorted(emitted_tables):
        lines.append(f" - {path}")
    for path in sorted(emitted_figs):
        lines.append(f" - {path}")
    if len(lines) == 1:
        lines.append(" - none")
    typer.echo("\n".join(lines))

if __name__ == "__main__":
    app()
