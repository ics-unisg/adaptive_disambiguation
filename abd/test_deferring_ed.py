from abd.deferring_ed import DeferringED, run_deferring
from abd.ed import ED
from common.refs import build_text


def labels_at(emissions):
    return [(e.at_event, e.label) for e in emissions]


def test_unambiguous_stream_no_deferral():
    text = build_text([list("abcef"), list("abdeg")])
    emissions, reports = run_deferring(text, ["a", "b", "c", "e", "f"])
    assert reports == []
    assert [e.label for e in emissions] == ["a", "b", "c", "e", "f"]
    assert all(e.deferred_by == 0 for e in emissions)


def test_matches_ed_baseline_on_unambiguous():
    text = build_text([list("abcef"), list("abdeg")])
    stream = ["a", "b", "c", "e", "f"]
    emissions, _ = run_deferring(text, stream)
    ed = ED(text)
    ed.setup(3)
    baseline = [ed.step([c]).events[0] for c in stream]
    assert [e.label for e in emissions] == baseline


def test_horizon_two_deferred_and_resolved():
    text = build_text([list("abcef"), list("abdeg")])
    emissions, reports = run_deferring(text, ["a", "b", ["c", "d"], "e", "f"])
    verdict = next(e for e in emissions if e.at_event == 2)
    assert verdict.label == "c"
    assert verdict.deferred_by == 2
    assert [e.label for e in emissions] == ["a", "b", "c", "e", "f"]
    assert len(reports) == 1
    r = reports[0]
    assert r.candidates == ["c", "d"]
    assert r.onset_k == 2.0
    assert r.actual_horizon == 2
    assert r.verdict == "c"
    assert r.k_trace == [("e", 1.0), ("f", 0.0)]


def test_nothing_emitted_while_holding():
    text = build_text([list("abcef"), list("abdeg")])
    d = DeferringED(text)
    assert [e.label for e in d.feed("a")] == ["a"]
    assert [e.label for e in d.feed("b")] == ["b"]
    assert d.feed(["c", "d"]) == []
    assert d.feed("e") == []
    flush = d.feed("f")
    assert [e.label for e in flush] == ["c", "e", "f"]


def test_never_separable_random_picks_no_decline():
    text = build_text([list("abce"), list("abde")])
    emissions, reports = run_deferring(text, ["a", "b", ["c", "d"], "e"],
                                       give_up_bound=5)
    r = reports[0]
    assert r.onset_k == 0.0
    assert r.was_random is True
    assert r.actual_horizon == 0
    verdict = next(e for e in emissions if e.at_event == 2)
    assert verdict.label in {"c", "d"}
    assert verdict.label != "?"


def test_shrinking_k_resolves_early():
    text = build_text([list("axpqr"), list("aypqs"), list("azm")])
    emissions, reports = run_deferring(text, ["a", ["x", "y", "z"], "m"])
    r = reports[0]
    assert r.onset_k == 3.0
    assert r.verdict == "z"
    assert r.actual_horizon == 1
    verdict = next(e for e in emissions if e.at_event == 1)
    assert verdict.label == "z"
    assert verdict.deferred_by == 1


def test_intervening_ambiguity_still_resolves():
    """An ambiguous event observed at the discriminating position pushes the
    resolution of an earlier ambiguity further out. Deferral continues while the
    pair is still separable, so the earlier ambiguity is not settled at random.

    M = {abdxev, acdyew, acfghi}, stream a [b|c] d [x|y] e v. [b|c] opens with
    onset_k = 2; [x|y] is observed at b/c's discriminating position, so it does not
    separate them and the remaining horizon grows back to 2. The stream matches
    abdxev uniquely, so [b|c] must resolve to b (not a random pick) and the whole
    output must be abdxev."""
    text = build_text([list("abdxev"), list("acdyew"), list("acfghi")])
    emissions, reports = run_deferring(text, ["a", ["b", "c"], "d", ["x", "y"], "e", "v"])

    bc = next(r for r in reports if r.candidates == ["b", "c"])
    assert bc.onset_k == 2.0
    assert bc.was_random is False
    assert bc.verdict == "b"

    xy = next(r for r in reports if r.candidates == ["x", "y"])
    assert xy.was_random is False
    assert xy.verdict == "x"

    assert "".join(e.label for e in sorted(emissions, key=lambda e: e.at_event)) == "abdxev"


def test_impossible_candidate_resolves_immediately():
    text = build_text([list("abcef"), list("abdeg")])
    d = DeferringED(text)
    d.feed("a")
    d.feed("b")
    out = d.feed(["c", "z"])
    assert [e.label for e in out] == ["c"]
    assert d.reports == []


def test_unresolved_at_stream_end_random_picks():
    text = build_text([list("abcef"), list("abdeg")])
    emissions, reports = run_deferring(text, ["a", "b", ["c", "d"], "e"])
    r = reports[0]
    assert r.was_random is True
    assert r.verdict in {"c", "d"}
    verdict = next(e for e in emissions if e.at_event == 2)
    assert verdict.label in {"c", "d"}
    assert "a" in [e.label for e in emissions] and "b" in [e.label for e in emissions]


def test_deviation_is_a_log_move():
    text = build_text([list("abc")])
    d = DeferringED(text)
    for c in "abc":
        d.feed(c)
    out = d.feed("z")
    assert [e.label for e in out] == [">"]
