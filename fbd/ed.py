from __future__ import annotations

from dataclasses import dataclass

from common.fm_engine import FMEngine
from common.refs import COMMA, build_text

_INF = 1_000_000


def _sym(s: str, i: int) -> str:
    return s[i] if 0 <= i < len(s) else "\x00"


class Row:
    __slots__ = ("number", "previous", "values")

    def __init__(self, number: int, previous: "Row | None"):
        self.number = number
        self.previous = previous
        self.values: list[int] = []


@dataclass
class State:
    subseq_m: str          # model subsequence matched
    subseq_l: str          # log subsequence
    char: str              # the last symbol consumed (get<2>)
    interval: str          # the matched pattern (== subseq_m); the "SA interval"
    row: Row               # DP row
    col_num: int           # number of columns = len(row.values)
    min_value: int         # cost of this state (get<6>)
    history: list          # model-move history (get<7>)
    sync_count: int        # synchronous moves so far (get<8>)


@dataclass
class StepResult:
    aligned: bool          # get<0>: an alignment/verdict was produced this step
    events: list           # get<1>: resolved candidate char(s), or ['>'] for log move
    history: list          # get<2>: per-candidate model-move histories
    cost: int              # get<3>: mismatches / accumulated cost ("k")
    sync: int              # get<4>: synchronous-move count
    run_len: int           # get<5>: length of the matched model subsequence
    trace_len: int         # get<6>: length of the log subsequence


class ED:

    def __init__(self, text: list[str], engine: FMEngine | None = None):
        self.text = text
        self.alphabet = sorted({s for s in text if s != COMMA}, reverse=True)
        self.engine = engine if engine is not None else FMEngine("".join(text))
        self.open: list[list[State]] = []

    def _extend(self, pattern: str, c: str) -> tuple[bool, str]:
        return (self.engine.feasible(pattern, c), pattern + c)

    def setup(self, max_ed: int = 3) -> None:
        # max_ed = maxK: a synchronous move is reachable only within this many
        # model-move skips in the depth-bounded search (cf. _resolve_by_model_moves).
        self.max_ed = max_ed
        first_row = Row(0, None)
        first_row.values.append(0)
        initial = State(
            subseq_m="", subseq_l="", char="_",
            interval="", row=first_row,       # empty pattern = the full interval
            col_num=1, min_value=0, history=[], sync_count=0,
        )
        self.open = [[initial]]

    def step(self, chars: list[str]) -> StepResult:
        pushes: list[State] = []
        aligned: list[str] = []

        for st in self.open[-1]:
            for c in self.alphabet:
                ok, ic = self._extend(st.interval, c)
                if not ok:
                    continue
                is_sf = c in chars
                if st.col_num == 1 and not is_sf:
                    continue

                nrow = Row(st.row.number + 1, st.row)
                nrow.values.append(st.row.values[0] + 1)
                for i in range(1, st.col_num):
                    diag = (st.row.values[i - 1] if c == _sym(st.subseq_m, i - 1)
                            else st.row.values[i - 1] + 2)
                    left = nrow.values[i - 1] + 1
                    up = st.row.values[i] + 1
                    nrow.values.append(min(diag, left, up))

                mv = min(
                    (st.row.values[st.col_num - 1] if is_sf
                     else st.row.values[st.col_num - 1] + 2),
                    nrow.values[st.col_num - 1] + 1,
                )
                nrow.values.append(mv)

                nsync = st.sync_count + (1 if is_sf else 0)
                if is_sf:
                    aligned.append(c)
                pushes.append(State(
                    st.subseq_m + c, st.subseq_l + c, c, ic, nrow,
                    st.col_num + 1, mv, [], nsync,
                ))

        copy_frontier = sorted(pushes, key=lambda s: s.min_value)

        if not aligned:
            return self._resolve_by_model_moves(copy_frontier, chars)
        return self._resolve_direct(copy_frontier)

    def _resolve_direct(self, copy_frontier: list[State]) -> StepResult:
        best0 = copy_frontier[0]
        best_candidates = [t for t in copy_frontier if t.min_value == best0.min_value]
        self.open[-1] = best_candidates

        best = best_candidates[0]
        return StepResult(
            aligned=True,
            events=[best.subseq_m[-1]],
            history=[],
            cost=best.min_value,
            sync=best.sync_count,
            run_len=len(best.subseq_m),
            trace_len=len(best.subseq_l),
        )

    def _resolve_by_model_moves(
        self, copy_frontier: list[State], chars: list[str]
    ) -> StepResult:
        frontier2: list[State] = []
        local_min = _INF

        for current in copy_frontier:
            if current.col_num < 2:
                continue
            ref_depth = current.row.number
            openstack: list[State] = [State(
                current.subseq_m, current.subseq_l, current.char, current.interval,
                current.row, current.col_num, current.min_value,
                [current.char], current.sync_count,
            )]

            while openstack:
                st = openstack.pop()
                if st.row.number - ref_depth >= self.max_ed:
                    continue

                for c in self.alphabet:
                    ok, ic = self._extend(st.interval, c)
                    if not ok:
                        continue
                    is_sf = c in chars

                    nrow = Row(st.row.number + 1, st.row)
                    nrow.values.append(st.row.values[0] + 1)
                    for i in range(1, st.col_num - 1):
                        diag = (st.row.values[i - 1] if c == _sym(st.subseq_m, i - 1)
                                else st.row.values[i - 1] + 2)
                        left = nrow.values[i - 1] + 1
                        up = st.row.values[i] + 1
                        nrow.values.append(min(diag, left, up))

                    mv = min(
                        (st.row.values[st.col_num - 2] if is_sf
                         else st.row.values[st.col_num - 2] + 2),
                        nrow.values[st.col_num - 2] + 1,
                        st.row.values[st.col_num - 1] + 1,
                    )
                    nrow.values.append(mv)

                    nsync = st.sync_count + (1 if is_sf else 0)
                    if is_sf and mv < local_min:
                        local_min = mv
                        frontier2.append(State(
                            st.subseq_m + c, st.subseq_l, c, ic, nrow,
                            st.col_num, mv, st.history, nsync,
                        ))
                    elif mv < local_min:
                        local_min = mv
                        openstack.append(State(
                            st.subseq_m + c, st.subseq_l, c, ic, nrow,
                            st.col_num, mv, st.history + [c], nsync,
                        ))

        if frontier2:
            frontier2.sort(key=lambda s: s.min_value)
            best = frontier2[0]
            best_candidates = [t for t in frontier2 if t.min_value == best.min_value]
            self.open[-1] = best_candidates
            return StepResult(
                aligned=True,
                events=[t.char for t in best_candidates],
                history=[t.history for t in best_candidates],
                cost=best.min_value,
                sync=best.sync_count,
                run_len=len(best.subseq_m),
                trace_len=len(best.subseq_l),
            )

        frontier3: list[State] = []
        for st in self.open[-1]:
            for c in chars:
                nrow = Row(st.row.number, st.row.previous)
                nrow.values = list(st.row.values)
                nrow.values.append(st.row.values[-1] + 1)
                frontier3.append(State(
                    st.subseq_m, st.subseq_l + c, c, st.interval, nrow,
                    st.col_num + 1, st.min_value + 1, [], st.sync_count,
                ))
        self.open[-1] = frontier3
        best = frontier3[0]
        return StepResult(
            aligned=False,
            events=[">"],
            history=[[c] for c in chars],
            cost=best.min_value,
            sync=best.sync_count,
            run_len=len(best.subseq_m),
            trace_len=len(best.subseq_l),
        )


def run_stream(text: list[str], stream: list[list[str]]) -> list[StepResult]:
    ed = ED(text)
    ed.setup(3)
    return [ed.step(list(ev)) for ev in stream]


def _demo() -> None:
    text = build_text([list("abcef"), list("abdeg")])
    ed = ED(text)
    ed.setup(3)
    print("model:", "".join(text))
    for ev in [["a"], ["b"], ["c", "d"], ["e"], ["f"]]:
        r = ed.step(ev)
        print(f"  step {ev} -> aligned={r.aligned} events={r.events} "
              f"cost={r.cost} sync={r.sync} run_len={r.run_len} "
              f"trace_len={r.trace_len} history={r.history}")


if __name__ == "__main__":
    _demo()
