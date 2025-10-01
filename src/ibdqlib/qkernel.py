# src/ibdqlib/qkernel.py
from __future__ import annotations
from typing import Optional, Tuple
import re
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector


def build_feature_map(n_qubits: int, reps: int = 2, entanglement: str = "linear") -> ZZFeatureMap:
    return ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement=entanglement)


def _ordered_params(feature_map: ZZFeatureMap):
    params = list(feature_map.parameters)
    def idx_of(p):
        s = str(p)
        m = re.match(r"x\[(\d+)\]$", s) or re.match(r"x(\d+)$", s)
        return int(m.group(1)) if m else 10**6
    return sorted(params, key=lambda p: (0 if str(p).startswith("x") else 1, idx_of(p), str(p)))


def _assign_params(feature_map: ZZFeatureMap, x: np.ndarray):
    ps = _ordered_params(feature_map)
    if len(ps) != len(x):
        raise ValueError(f"Parameter count {len(ps)} != len(x) {len(x)}")
    mapping = {p: float(x[i]) for i, p in enumerate(ps)}
    return feature_map.assign_parameters(mapping, inplace=False)  # Qiskit 1.x


def _scale_to_angles(Xtr: np.ndarray, Xte: Optional[np.ndarray], target=np.pi/4, pctl=90.0, per_feature=True):
    """
    Scale so typical magnitudes map to [-target, +target].
    Uses |X| 90th percentile computed on TRAIN ONLY (per feature by default).
    """
    if per_feature:
        s = np.percentile(np.abs(Xtr), pctl, axis=0)
        s[s < 1e-12] = 1.0
        alpha = target / s
    else:
        s = float(np.percentile(np.abs(Xtr), pctl))
        alpha = (target / s) if s > 1e-12 else 1.0
    Xtr_s = Xtr * alpha
    Xte_s = (Xte * alpha) if Xte is not None else None
    return Xtr_s, Xte_s


def _statevectors(feature_map: ZZFeatureMap, X: np.ndarray) -> np.ndarray:
    svs = []
    for x in X:
        circ = _assign_params(feature_map, x)
        sv = Statevector.from_instruction(circ).data
        svs.append(sv)
    return np.asarray(svs, dtype=np.complex128)


def quantum_kernel_mats(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray],
    n_qubits: int,
    reps: int = 2,
    entanglement: str = "linear",
    shots: Optional[int] = None,
    backend: str = "statevector",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Exact (statevector) quantum kernel:
      K_ij = |⟨φ(x_i)|φ(x_j)⟩|^2
    Angles scaled to ±π/4 using 90th percentile of TRAIN magnitudes.
    """
    Xtr_s, Xte_s = _scale_to_angles(X_train, X_test, target=np.pi/4, pctl=90.0, per_feature=True)

    fmap = build_feature_map(n_qubits, reps=reps, entanglement=entanglement)
    Strain = _statevectors(fmap, Xtr_s)
    Ktr = np.abs(Strain @ Strain.conj().T) ** 2

    Kte = None
    if Xte_s is not None:
        Stest = _statevectors(fmap, Xte_s)
        Kte = np.abs(Stest @ Strain.conj().T) ** 2

    return Ktr, Kte
