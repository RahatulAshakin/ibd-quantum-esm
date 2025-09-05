from src.ibdqlib.quantum import make_qsvc

def test_qsvc_smoke():
    clf = make_qsvc(feature_dim=4)
    assert clf is not None
