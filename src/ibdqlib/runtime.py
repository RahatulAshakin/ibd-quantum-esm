from __future__ import annotations
import os
from typing import List, Tuple
from dotenv import load_dotenv

from qiskit_ibm_runtime import QiskitRuntimeService

def get_service() -> QiskitRuntimeService:
    """Load credentials from .env and return a QiskitRuntimeService."""
    # Load once (safe if called multiple times)
    load_dotenv()
    channel = os.getenv("QISKIT_IBM_CHANNEL", "ibm_cloud")
    token = os.getenv("QISKIT_IBM_TOKEN")
    instance = os.getenv("QISKIT_IBM_INSTANCE")

    if not token:
        raise RuntimeError("Missing QISKIT_IBM_TOKEN in .env")
    if not instance:
        raise RuntimeError("Missing QISKIT_IBM_INSTANCE in .env")

    svc = QiskitRuntimeService(channel=channel, token=token, instance=instance)
    return svc

def list_backends(limit: int = 10) -> List[Tuple[str, int, bool]]:
    """
    Return a quick summary (name, num_qubits, is_simulator) for up to `limit` backends.
    """
    svc = get_service()
    backends = svc.backends()
    rows: List[Tuple[str, int, bool]] = []
    for b in backends[:limit]:
        # Some backends expose .num_qubits; fall back to getattr
        n_qubits = getattr(b.configuration(), "num_qubits", None)
        if n_qubits is None:
            n_qubits = getattr(b, "num_qubits", -1)
        is_sim = getattr(b.configuration(), "simulator", False)
        rows.append((b.name, int(n_qubits) if n_qubits is not None else -1, bool(is_sim)))
    return rows
