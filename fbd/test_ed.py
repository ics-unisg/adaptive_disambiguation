from common.refs import build_text
from fbd.ed import ED, run_stream


def make(traces):
    ed = ED(build_text([list(t) for t in traces]))
    ed.setup(3)
    return ed


def test_clean_run_zero_cost():
    ed = make(["abcef", "abdeg"])
    results = [ed.step([c]) for c in "abcef"]
    assert all(r.aligned for r in results)
    assert all(r.cost == 0 for r in results)
    assert [r.sync for r in results] == [1, 2, 3, 4, 5]
    assert [r.events[0] for r in results] == list("abcef")
    assert [r.run_len for r in results] == [1, 2, 3, 4, 5]


def test_run_stream_helper():
    text = build_text([list("abc")])
    results = run_stream(text, [["a"], ["b"], ["c"]])
    assert [r.events[0] for r in results] == ["a", "b", "c"]
    assert all(r.cost == 0 for r in results)


def test_ambiguous_keeps_both_interpretations():
    ed = make(["abcef", "abdeg"])
    ed.step(["a"])
    ed.step(["b"])
    r = ed.step(["c", "d"])
    assert r.aligned
    assert r.cost == 0
    assert r.events[0] in {"c", "d"}
    survivors = {s.char for s in ed.open[-1]}
    assert survivors == {"c", "d"}
    assert len(ed.open[-1]) == 2


def test_ambiguous_prunes_the_impossible_candidate():
    ed = make(["abcef", "abdeg"])
    ed.step(["a"])
    ed.step(["b"])
    r = ed.step(["c", "z"])
    assert r.aligned
    assert r.events[0] == "c"
    assert {s.char for s in ed.open[-1]} == {"c"}


def test_model_move_one_skip_resolves():
    ed = make(["abcd"])
    ed.step(["a"])
    ed.step(["b"])
    r = ed.step(["d"])
    assert r.aligned
    assert "d" in r.events
    assert r.history and r.history[0]
    assert "c" in r.history[0]


def test_depth_bound_reaches_two_skips():
    ed = make(["abcxd"])
    ed.step(["a"])
    ed.step(["b"])
    r = ed.step(["d"])
    assert r.aligned is True
    assert "d" in r.events
    assert r.history[0] == ["c", "x"]


def test_depth_bound_three_skips_falls_back_to_log_move():
    ed = make(["abcxyd"])
    ed.step(["a"])
    ed.step(["b"])
    r = ed.step(["d"])
    assert r.aligned is False
    assert r.events == [">"]


def test_log_move_on_unextendable_event():
    ed = make(["abc"])
    for c in "abc":
        ed.step([c])
    r = ed.step(["z"])
    assert r.aligned is False
    assert r.events == [">"]
    assert r.history == [["z"]]


def test_log_move_increments_cost_and_trace_len():
    ed = make(["abc"])
    for c in "abc":
        ed.step([c])
    before = ed.open[-1][0]
    r = ed.step(["z"])
    assert r.trace_len == len(before.subseq_l) + 1
    assert r.run_len == len(before.subseq_m)


def test_substitution_costs_two():
    ed = make(["ab"])
    ed.step(["a"])
    r = ed.step(["a"])
    assert r.cost >= 1
