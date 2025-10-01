import json
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import qiskit  # noqa: F401
    QISKIT_OK = True
except Exception:
    QISKIT_OK = False

pytestmark = pytest.mark.skipif(not QISKIT_OK, reason="qiskit not installed")

PY = f'"{sys.executable}"'


def run(cmd, cwd=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)


def test_quantum_training_includes_baseline(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    fasta = data_dir / "train.fa"
    fasta.write_text(
        ">seqA\nMKTIIALSYIF\n>seqB\nGVVDSGTT\n>seqC\nMTSAGAA\n>seqD\nGGGGGGGG\n>seqE\nMNVNSTH\n>seqF\nGGGTTTCC\n"
    )

    labels_csv = data_dir / "labels.csv"
    labels_csv.write_text(
        "seq_id,label\n"
        "seqA,flare\n"
        "seqB,remission\n"
        "seqC,flare\n"
        "seqD,remission\n"
        "seqE,flare\n"
        "seqF,remission\n"
    )

    db_path = tmp_path / "embeddings.duckdb"
    metrics_path = tmp_path / "metrics.json"
    model_path = tmp_path / "model.joblib"

    r = run(
        f'{PY} -m src embed "{fasta}" --backend dummy --outdb "{db_path}" --batch 4'
    )
    assert r.returncode == 0, r.stderr

    cmd = (
        f'{PY} -m src train-qsvc-quantum --db "{db_path}" --table embeddings '
        f'--labels-csv "{labels_csv}" --label-col label --pca-components 2 '
        "--per-class-limit 3 --reps 1 --shots 256 --backend statevector --C 1.0 "
        f'--test-size 0.5 --out "{metrics_path}" --model-out "{model_path}"'
    )
    r = run(cmd)
    assert r.returncode == 0, r.stderr
    metrics = json.loads(metrics_path.read_text())
    assert model_path.exists()

    for key in [
        "quantum_accuracy",
        "quantum_f1_macro",
        "baseline_accuracy",
        "baseline_f1_macro",
        "accuracy_delta",
        "kernel_diagnostics",
    ]:
        assert key in metrics
    assert isinstance(metrics["kernel_diagnostics"], dict)
