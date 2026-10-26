from __future__ import annotations

import random
from dataclasses import dataclass, field

from abd.ed import ED
from common.refs import COMMA, build_text

NO_BOUND = 0.0

_COINCIDE = object()


def on_demand_k(engine, alphabet, pattern_x: str, pattern_y: str) -> float:
    memo: dict = {}

    def deepest_shared(ax: str, ay: str):
        key = (ax, ay)
        if key in memo:
            return memo[key]

        # identical complete continuation -> never separable
        if engine.feasible(ax, COMMA) and engine.feasible(ay, COMMA):
            memo[key] = _COINCIDE
            return _COINCIDE

        best = 0.0
        for c in alphabet:
            if not engine.feasible(ax, c) or not engine.feasible(ay, c):
                continue
            sub = deepest_shared(ax + c, ay + c)
            if sub is _COINCIDE:
                memo[key] = _COINCIDE
                return _COINCIDE
            if 1 + sub > best:
                best = 1 + sub
        memo[key] = best
        return best

    d = deepest_shared(pattern_x, pattern_y)
    return NO_BOUND if d is _COINCIDE else float(d + 1)


def _separable_k(pairs_k: dict, survivors: set) -> float:
    live = [k for pair, k in pairs_k.items() if pair <= survivors]
    return max(live) if live else NO_BOUND


def _no_separable_pair(pairs_k: dict, survivors: set) -> bool:
    if len(survivors) < 2:
        return False
    return _separable_k(pairs_k, survivors) == NO_BOUND


@dataclass
class Emission:
    at_event: int
    label: str
    deferred_by: int = 0
    note: str = ""


@dataclass
class DeferralReport:
    onset_event: int
    candidates: list
    onset_k: float
    verdict: str = ""
    actual_horizon: int = 0
    k_trace: list = field(default_factory=list)
    was_random: bool = False       # verdict was a forced random pick, not disambiguated
    random_reason: str = ""
    pick_pool: list = field(default_factory=list)  # candidates the random draw chose among


class DeferringED:
    def __init__(self, text: list[str], give_up_bound: int = 100,
                 give_up_unbounded: bool = True, seed: int | None = 0,
                 rng: random.Random | None = None, engine=None):
        self.text = text
        self.ed = ED(text, engine=engine)
        self.ed.setup(3)
        self.alphabet = self.ed.alphabet
        self.engine = self.ed.engine
        self.give_up_bound = give_up_bound
        self.give_up_unbounded = give_up_unbounded
        self.rng = rng if rng is not None else random.Random(seed)
        self.event_idx = -1
        self.labels: dict[int, str] = {}     # event index -> final label once known
        self.meta: dict[int, tuple] = {}     # index -> (held, was_random, note)
        self.next_emit = 0                   # next index to emit
        self.open: list[dict] = []           # concurrently open ambiguity positions
        self.reports: list[DeferralReport] = []

    def _pairs_k(self, intervals_by_label: dict) -> dict:
        pairs: dict = {}
        labels = sorted(intervals_by_label)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                x, y = labels[i], labels[j]
                k = NO_BOUND
                coincide = False
                for ix in intervals_by_label[x]:
                    for iy in intervals_by_label[y]:
                        kk = on_demand_k(self.engine, self.alphabet, ix, iy)
                        if kk == NO_BOUND:
                            coincide = True
                            break
                        k = max(k, kk)
                    if coincide:
                        break
                pairs[frozenset((x, y))] = NO_BOUND if coincide else k
        return pairs

    @staticmethod
    def _by_label(frontier, amb_index) -> dict:
        groups: dict = {}
        for s in frontier:
            if len(s.subseq_m) > amb_index:
                groups.setdefault(s.subseq_m[amb_index], []).append(s.interval)
        return groups

    # ----------------------------------------------------------------------- #
    def feed(self, event) -> list[Emission]:
        self.event_idx += 1
        idx = self.event_idx
        chars = [event] if isinstance(event, str) else list(event)
        ev_label = event if isinstance(event, str) else "".join(chars)

        result = self.ed.step(chars)
        self.last_step = result
        frontier = self.ed.open[-1]

        # 1) advance every already-open ambiguity with this event
        self._advance_open(frontier, ev_label)

        # 2) classify the current event
        if not result.aligned:
            self.labels[idx] = ">"
            self.meta[idx] = (0, False, "log move (noise)")
        else:
            amb_index = len(frontier[0].subseq_m) - 1
            here = {s.subseq_m[amb_index] for s in frontier if len(s.subseq_m) > amb_index}
            if len(chars) > 1 and len(here) > 1:
                pairs_k = self._pairs_k(self._by_label(frontier, amb_index))
                rep = DeferralReport(idx, sorted(here), _separable_k(pairs_k, here))
                self.reports.append(rep)
                pos = dict(idx=idx, amb_index=amb_index,
                           candidates=sorted(here), pairs_k=pairs_k,
                           report=rep, observed=0, survivors=set(here))
                if self.give_up_unbounded and _no_separable_pair(pairs_k, here):
                    self._settle(pos, was_random=True, pool=sorted(here),
                                 reason="no separable pair (maxK = 0)")
                else:
                    self.open.append(pos)
            else:
                lab = result.events[0] if len(chars) == 1 else next(iter(here))
                self.labels[idx] = lab
                self.meta[idx] = (0, False, "")
        return self._flush()

    def _advance_open(self, frontier, ev_label) -> None:
        # Deferral continues while the pair is still separable. There is no
        # fixed-length window: a timeout that could fire while the recomputed bound
        # is still defined would cut off a pair that a later event still resolves
        # (e.g. when the discriminating event is itself observed ambiguously and
        # the horizon extends). An ambiguity is settled only when one candidate
        # remains, all candidates deviate, the recomputed bound becomes undefined
        # (the continuations coincide, so no further event can separate them), or
        # the patience limit is reached; the stream end is handled by finish().
        still = []
        for pos in self.open:
            pos["observed"] += 1
            amb = pos["amb_index"]
            surv = {s.subseq_m[amb] for s in frontier if len(s.subseq_m) > amb}
            pos["survivors"] = surv
            pairs_now = self._pairs_k(self._by_label(frontier, amb))
            pos["pairs_now"] = pairs_now
            k_now = _separable_k(pairs_now, surv)
            pos["report"].k_trace.append((ev_label, k_now))

            if len(surv) == 1:
                self._settle(pos, next(iter(surv)), was_random=False)
            elif len(surv) == 0:
                self._settle(pos, was_random=True, pool=pos["candidates"],
                             reason="all candidates deviated")
            elif self.give_up_unbounded and _no_separable_pair(pairs_now, surv):
                self._settle(pos, was_random=True, pool=sorted(surv),
                             reason="no separable pair remains")
            elif pos["observed"] >= self.give_up_bound:
                self._settle(pos, was_random=True, pool=sorted(surv),
                             reason="patience exceeded")
            else:
                still.append(pos)
        self.open = still

    def _settle(self, pos, verdict=None, was_random=False, pool=None, reason="") -> None:
        if was_random:
            pool = sorted(pool)
            verdict = self.rng.choice(pool)
        rep = pos["report"]
        rep.verdict = verdict
        rep.was_random = was_random
        rep.actual_horizon = pos["observed"]
        if was_random:
            rep.random_reason = reason
            rep.pick_pool = pool
        self.labels[pos["idx"]] = verdict
        note = (f"RANDOM ({reason})" if was_random else "deferred") \
            + f" {rep.candidates} -> {verdict}"
        self.meta[pos["idx"]] = (pos["observed"], was_random, note)

    def _flush(self) -> list[Emission]:
        out = []
        while self.next_emit in self.labels:
            i = self.next_emit
            held, _rand, note = self.meta.get(i, (0, False, ""))
            out.append(Emission(i, self.labels[i], deferred_by=held, note=note))
            self.next_emit += 1
        return out

    def finish(self) -> list[Emission]:
        for pos in self.open:
            pool = sorted(pos["survivors"]) or pos["candidates"]
            self._settle(pos, was_random=True, pool=pool,
                         reason="stream ended, still ambiguous")
        self.open = []
        return self._flush()


def _fmt(k: float) -> str:
    return str(int(k))


# --------------------------------------------------------------------------- #
def run_deferring(text: list[str], stream: list, give_up_bound: int = 100):
    d = DeferringED(text, give_up_bound=give_up_bound)
    emissions = []
    for ev in stream:
        emissions.extend(d.feed(ev))
    emissions.extend(d.finish())  # random-pick anything still held; never declines
    emissions.sort(key=lambda e: e.at_event)
    return emissions, d.reports


def _demo() -> None:
    print("model ,abcef,abdeg, -- stream a b [c,d] e f")
    text = build_text([list("abcef"), list("abdeg")])
    emissions, reports = run_deferring(text, ["a", "b", ["c", "d"], "e", "f"])
    for e in emissions:
        d = f" [held {e.deferred_by}]" if e.deferred_by else ""
        n = f"   # {e.note}" if e.note else ""
        print(f"  event {e.at_event} -> {e.label}{d}{n}")
    for r in reports:
        tr = " ".join(f"{ev}:k={_fmt(k)}" for ev, k in r.k_trace)
        print(f"  [report] onset@{r.onset_event} {r.candidates} onset_k={_fmt(r.onset_k)}"
              f" -> {r.verdict} (horizon {r.actual_horizon}); shrinking-k: {tr}")


if __name__ == "__main__":
    _demo()
