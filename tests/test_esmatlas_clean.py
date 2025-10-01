import sys
sys.path.append("src")
from ibdqlib.esmatlas import _clean, ALLOWED

def test_allowed_has_ambiguous_codes():
    for ch in "XBZJ":
        assert ch in ALLOWED

def test_clean_replaces_bad_tokens_with_X():
    seq = "ABZJ*U-!acgt??"
    cleaned, bad = _clean(seq)
    assert set(bad).issubset(set("U*?-!"))  # non-AA flagged as bad
    # cleaned should be only letters from ALLOWED or X
    assert all((c in ALLOWED) or (c == "X") for c in cleaned)
