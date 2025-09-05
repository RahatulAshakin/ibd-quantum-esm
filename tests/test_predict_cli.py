import json
import os
import subprocess
from pathlib import Path

PY = "python"  # CI will resolve the env's python

def run(cmd, cwd=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)

def test_train_and_predict_dummy(tmp_path: Path):
    # 1) write a tiny FASTA
    data = tmp_path / "data"
    results = tmp_path / "results"
    (data).mkdir(parents=True, exist_ok=True)
    (results / "duckdb").mkdir(parents=True, exist_ok=True)
    (results / "models").mkdir(parents=True, exist_ok=True)

    fasta = data / "demo.fa"
    fasta.write_text(">s1\nMKTIIALSYIF\n>s2\nGVVDSGTT\n")

    # 2) embed with dummy backend
    r = run(f'{PY} -m src embed "{fasta}" --backend dummy --outdb "{results / "duckdb" / "e.duckdb"}" --batch 8')
    assert r.returncode == 0, r.stderr

    # 3) train with cap small
    metrics = results / "metrics.json"
    model = results / "models" / "qsvc.joblib"
    r = run(f'{PY} -m src train-qsvc --db "{results / "duckdb" / "e.duckdb"}" --table embeddings --rule median-length --feature-cap 4 --out "{metrics}" --model-out "{model}"')
    assert r.returncode == 0, r.stderr
    assert model.exists()

    # 4) predict with the same backend
    preds = results / "preds.csv"
    r = run(f'{PY} -m src predict "{fasta}" --model "{model}" --backend dummy --out "{preds}"')
    assert r.returncode == 0, r.stderr
    text = preds.read_text().strip().splitlines()
    assert len(text) == 3  # header + 2 rows
