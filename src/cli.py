from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import typer
import pandas as pd
import duckdb

from .ibdqlib.esm_embed import get_embedder, read_fasta


app = typer.Typer(help="IBD Quantum + ESM pipeline")

@app.command("hello")
def hello_cmd() -> None:
    """Sanity check command."""
    print("IBD Quantum-ESM scaffold ready.")

@app.command("version")
def version_cmd() -> None:
    """Show the package version."""
    # lightweight manual version for now
    print("ibd-quantum-esm 0.0.1")

@app.command("embed")
def embed_cmd(
    input: str = typer.Argument(..., help="Path to FASTA (.fa/.fasta) or CSV file"),
    outdb: str = typer.Option("results/duckdb/embeddings.duckdb", "--outdb", help="DuckDB database file"),
    model: str = typer.Option("esm2_t6_8M_UR50D", "--model", help="ESM model name; falls back to dummy if not available"),
    batch: int = typer.Option(64, "--batch", help="Batch size for ESM embedding"),
):
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
    ids: list[str]
    seqs: list[str]
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

    embed_fn, dim, backend = get_embedder(model)

    # ---- batch embed ----
    import numpy as np
    embs = []
    for i in range(0, len(seqs), batch):
        embs.append(embed_fn(seqs[i:i+batch]))
    embs = np.vstack(embs)
    lengths = [len(s) for s in seqs]

    df = pd.DataFrame({
        "seq_id": ids,
        "length": lengths,
        "backend": backend,
        "dim": dim,
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
        f"Inserted {len(df)} rows into {out_path} (table: embeddings) using backend='{backend}', dim={dim}.",
        fg=typer.colors.GREEN,
    )

if __name__ == "__main__":
    app()









@app.command("query")
def query_cmd(
    db: str = typer.Option("results/duckdb/embeddings.duckdb", "--db", help="DuckDB database file"),
    limit: int = typer.Option(5, "--limit", help="Preview N rows"),
    stats: bool = typer.Option(True, "--stats/--no-stats", help="Print summary stats"),
):
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

