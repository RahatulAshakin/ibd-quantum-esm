# tests/test_qesm.py
import pytest

# Skip the whole module if Aer isn't available (e.g., in minimal CI envs)
try:
    import qiskit_aer  # noqa: F401
    AER_OK = True
except Exception:
    AER_OK = False

pytestmark = pytest.mark.skipif(not AER_OK, reason="qiskit-aer not installed")

from ibdqlib.qesm import qesm_find_positions


def _idx_to_candidates(idx: int, nbits: int):
    """Return both big- and little-endian bitstring forms for a safety check."""
    be = format(idx, f"0{nbits}b")        # e.g. 2 -> '10' for nbits=2
    le = be[::-1]                         # reversed
    return {be, le}


def test_single_match_small_shots():
    # Text='ACGTAC', Pattern='GTA' -> match at index 2
    solutions, counts, iters = qesm_find_positions("ACGTAC", "GTA", shots=512)
    assert solutions == [2]
    assert isinstance(counts, dict) and sum(counts.values()) == 512
    # Make sure the measured keys include something that could represent idx=2
    nbits = max(len(next(iter(counts))) if counts else 1, 1)
    candidates = _idx_to_candidates(2, nbits)
    assert any(k in counts for k in candidates)


def test_two_matches_small_shots():
    # Text='ABABA', Pattern='ABA' -> matches at indices [0, 2]
    solutions, counts, iters = qesm_find_positions("ABABA", "ABA", shots=512)
    assert solutions == [0, 2]
    assert isinstance(counts, dict) and sum(counts.values()) == 512

    nbits = max(len(next(iter(counts))) if counts else 1, 1)
    cand0 = _idx_to_candidates(0, nbits)
    cand2 = _idx_to_candidates(2, nbits)
    # At least see keys for both target indices (either endian)
    assert any(k in counts for k in cand0)
    assert any(k in counts for k in cand2)
