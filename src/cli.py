# src/cli.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Tuple

import duckdb
import numpy as np
import pandas as pd
import typer
from joblib import dump, load
from json import dumps

from .ibdqlib.runtime import list_backends
from .ibdqlib.esm_embed import get_embedder, read_fasta
from .ibdqlib.train import train_qsvc_from_duckdb

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


# -----------------------
# Predict with saved bundle
# -----------------------
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


if __name__ == "__main__":
    app()
