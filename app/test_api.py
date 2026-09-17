from app.rag import severity_stage, build_prognosis

def test_stage():
    assert severity_stage(5)=='early'
    assert severity_stage(20)=='moderate'
    assert severity_stage(60)=='advanced'

def test_prognosis_bounds():
    p=build_prognosis(20, {'humidity_pct':80})
    assert len(p)==3
    assert all(0 <= x['severity_percent'] <= 100 for x in p)
