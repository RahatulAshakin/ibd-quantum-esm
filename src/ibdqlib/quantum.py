from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler   # V2 sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

def make_qsvc(feature_dim: int, reps: int = 2) -> QSVC:
    """
    QSVC using FidelityQuantumKernel (current Qiskit ML API)
    with a V2 sampler (StatevectorSampler).
    """
    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=reps)
    sampler = StatevectorSampler()                 # <-- V2
    fidelity = ComputeUncompute(sampler=sampler)
    qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    return QSVC(quantum_kernel=qkernel)
