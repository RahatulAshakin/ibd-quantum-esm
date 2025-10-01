# src/ibdqlib/qesm.py
from __future__ import annotations
import math
from typing import List, Tuple, Dict

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.circuit.library import MCXGate


def _to_bits(i: int, n: int) -> List[int]:
    """Little-endian bit list of length n."""
    return [(i >> k) & 1 for k in range(n)]


def _oracle_on_indices(n_index: int, solutions: List[int]) -> QuantumCircuit:
    """
    Oracle that flips phase (via ancilla |->) iff index register equals any solution index.
    Qubits: [index (n_index), ancilla (1)]
    """
    idx = QuantumRegister(n_index, "idx")
    anc = QuantumRegister(1, "anc")
    qc = QuantumCircuit(idx, anc, name="Oracle")

    if not solutions:
        # No-op oracle (no matches)
        return qc

    mcx = MCXGate(num_ctrl_qubits=n_index)

    for s in solutions:
        bits = _to_bits(s, n_index)
        # Map |s> -> |11...1> by X on zero-bits
        for q, b in enumerate(bits):
            if b == 0:
                qc.x(idx[q])

        # Multi-controlled X on ancilla => phase flip because anc is in |->.
        qc.append(mcx, [*idx, anc[0]])

        # Undo mapping
        for q, b in enumerate(bits):
            if b == 0:
                qc.x(idx[q])

    return qc


def _diffuser(n_index: int) -> QuantumCircuit:
    """
    Standard Grover diffuser on index register using the |-> ancilla trick.
    Qubits: [index (n_index), ancilla (1)]
    """
    idx = QuantumRegister(n_index, "idx")
    anc = QuantumRegister(1, "anc")
    qc = QuantumCircuit(idx, anc, name="Diffuser")

    qc.h(idx)
    qc.x(idx)

    mcx = MCXGate(num_ctrl_qubits=n_index)
    qc.append(mcx, [*idx, anc[0]])  # phase flip about |0...0>

    qc.x(idx)
    qc.h(idx)
    return qc


def qesm_find_positions(
    text: str, pattern: str, shots: int = 1024, max_iters: int | None = None
) -> Tuple[List[int], Dict[str, int], int]:
    """
    Grover demo: find all start indices where `pattern` occurs in `text`.

    Returns:
      (solutions, counts, n_iters)
        solutions: classical ground-truth matches
        counts: measurement counts from the quantum search (bitstrings over index qubits)
        n_iters: number of Grover iterations used
    """
    N = len(text)
    M = len(pattern)
    if M == 0 or M > N:
        return [], {}, 0

    # Classical truth set of matching start indices
    n_pos = N - M + 1
    solutions = [i for i in range(n_pos) if text[i:i + M] == pattern]
    if not solutions:
        return [], {}, 0

    # Number of index states we'll search over is 2^n_index (pad if n_pos not power of 2)
    n_index = max(1, math.ceil(math.log2(n_pos)))

    # Build the Grover circuit
    idx = QuantumRegister(n_index, "idx")
    anc = QuantumRegister(1, "anc")
    creg = ClassicalRegister(n_index, "c")
    qc = QuantumCircuit(idx, anc, creg, name="QESM")

    # Uniform superposition over index register
    qc.h(idx)
    # Prepare ancilla in |->  (X|0>=|1>, H|1>=|->)
    qc.x(anc[0])
    qc.h(anc[0])

    oracle = _oracle_on_indices(n_index, solutions)
    diffuser = _diffuser(n_index)

    # Grover iterations
    n_solutions = max(1, len(solutions))
    default_iters = max(1, int(round((math.pi / 4) * math.sqrt((2 ** n_index) / n_solutions))))
    n_iters = max_iters if max_iters is not None else default_iters

    # IMPORTANT: inline subcircuits (compose) rather than add opaque custom gates
    for _ in range(n_iters):
        qc.compose(oracle, qubits=[*idx, anc[0]], inplace=True)
        qc.compose(diffuser, qubits=[*idx, anc[0]], inplace=True)

    # Measure only the index register
    qc.measure(idx, creg)

    # Run on Aer QASM simulator (transpile so MCX decomposes to basis)
    try:
        from qiskit_aer import Aer
        backend = Aer.get_backend("qasm_simulator")
        tqc = transpile(qc, backend=backend, optimization_level=1)
        job = backend.run(tqc, shots=shots)
        result = job.result()
        counts = result.get_counts()
    except Exception as e:
        raise RuntimeError(
            "QESM demo requires a QASM simulator. Please install qiskit-aer:\n"
            "  pip install qiskit-aer\n"
            f"Original error: {e}"
        )

    return solutions, counts, n_iters
