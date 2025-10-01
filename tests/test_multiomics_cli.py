import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PY = f'"{sys.executable}"'


def run(cmd, cwd=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)


def test_multiomics_ingest_train_report(tmp_path: Path):
    data_src = Path("data/ibd_multiomics_demo.csv")
    data_copy = tmp_path / "demo.csv"
    data_copy.write_text(data_src.read_text())

    db_path = tmp_path / "ibd.duckdb"
    metrics_path = tmp_path / "metrics.json"
    model_path = tmp_path / "model.joblib"
    preds_path = tmp_path / "preds.csv"
    figdir = tmp_path / "figs"
    tabledir = tmp_path / "tables"

    r = run(
        f'{PY} -m src ingest-omics "{data_copy}" --outdb "{db_path}" --overwrite'
    )
    assert r.returncode == 0, r.stderr

    r = run(
        f'{PY} -m src train-omics --db "{db_path}" --table multiomics '
        f'--out "{metrics_path}" --model-out "{model_path}" --pred-out "{preds_path}" '
        '--test-size 0.33 --random-state 0'
    )
    assert r.returncode == 0, r.stderr
    assert metrics_path.exists()
    assert model_path.exists()
    assert preds_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert "f1_macro" in metrics
    assert metrics["classes"], "expected non-empty class list"

    r = run(
        f'{PY} -m src report --db "{db_path}" --pred "{preds_path}" '
        f'--figdir "{figdir}" --tabledir "{tabledir}"'
    )
    assert r.returncode == 0, r.stderr
    assert (tabledir / "performance_metrics.csv").exists()
    assert (figdir / "confusion_matrix.png").exists()
