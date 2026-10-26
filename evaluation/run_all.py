"""Run the whole evaluation and write the results as Excel workbooks -- one per
algorithm per dataset, in the layout the original paper used. This is the single
entry point for the evaluation; it writes no CSV files.

Algorithms (one workbook each):
  maxk       the fixed depth-bound aligner (fbd.ed.ED), the paper's baseline Alg. 3
  adaptive   the adaptive-k extension (abd.deferring_ed.DeferringED)
  baseline   the null model B: each ambiguous event resolved to a uniformly random
             candidate (baseline.xes), then aligned with the same fixed aligner

For each dataset every algorithm is run exactly once over its input log; the gold
standard is run once and shared across the three workbooks. Each workbook has:
  Summary               headline numbers: all-events accuracy, mean fitness (gold
                        and this algorithm), mean fitness drop, mean runtime
  AllEvents             one row per event: Case GSTD, Case <algo>, Event GSTD,
                        Event <algo>, Are equal  (-> all-events accuracy)
  Results               per trace: total events, # mismatches, accuracy; plus a
                        #mismatches -> #traces histogram
  Fitness               per case: |Run|/|Trace|/Sync/Async/Fitness for gold and for
                        this algorithm, and the Fitness drop; plus a <0/=0/>0 count
  Runtime_<InputLog>    per-trace runtime (ms) of this algorithm on its input log:
                        one column per repeat + Average
  Runtime_GoldStandard  same, on the gold-standard log

Usage:
  python -m evaluation.run_all --data-dir ../disambiguation/adaptive_k/data \
      --out results --repeats 3
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import OrderedDict
from pathlib import Path

import openpyxl

from common.refs import CASE_KEY, build_text, read_reference_traces
from common.fm_engine import FMEngine
from fbd.ed import ED
from abd.deferring_ed import DeferringED
from evaluation import emit_output
from evaluation.run_dataset import load_streams

SHORT = {"blood_donation": "BD", "smart_home": "SH", "hr_recruitment": "HR",
         "purchase_to_pay": "PTP"}

# algorithm -> (input log basename, timing kind). "maxk" and "baseline" are the
# same fixed aligner on different logs; "adaptive" is the deferring extension.
ALGOS = OrderedDict([
    ("maxk", ("ambiguous", "ed")),
    ("adaptive", ("ambiguous", "deferring")),
    ("baseline", ("baseline", "ed")),
])


def _last_per_case(records):
    last = OrderedDict()
    for r in records:
        last[r["case"]] = r
    return last


def _records(log, algo, ddir, labels, ck, text, engine, limit, maxk):
    """Per-event records for one algorithm, via the same functions that write
    output/*.json: emit_adaptive for the extension, emit_baseline otherwise.
    `maxk` is the fixed-bound depth and only affects the ED-based algorithms."""
    xes = ddir / f"{log}.xes"
    if algo == "adaptive":
        return emit_output.emit_adaptive(xes, labels, ck, text, engine, limit)
    return emit_output.emit_baseline(xes, labels, ck, text, engine, limit, maxk)


def _process(stream, text, engine, kind, maxk):
    """Run one case's event stream once, for timing. `kind` selects the aligner
    the matching _records call used, so the runtime matches the recorded output."""
    if kind == "ed":
        ed = ED(text, engine=engine)
        ed.setup(maxk)
        for ev in stream:
            ed.step([ev] if isinstance(ev, str) else list(ev))
    else:
        d = DeferringED(text, engine=engine, rng=random.Random(0))
        for ev in stream:
            d.feed(ev)
        d.finish()


def _per_trace_runtime(streams, text, engine, kind, repeats, maxk):
    """{case_id: [ms per repeat]} after one unrecorded warm-up pass."""
    for stream in streams.values():                # warm-up (primes the engine)
        _process(stream, text, engine, kind, maxk)
    times = OrderedDict((cid, []) for cid in streams)
    for _ in range(repeats):
        for cid, stream in streams.items():
            t0 = time.perf_counter()
            _process(stream, text, engine, kind, maxk)
            times[cid].append((time.perf_counter() - t0) * 1e3)
    return times


def _runtime_sheet(wb, title, rt, repeats):
    ws = wb.create_sheet(title)
    ws.append(["Trace ID"] + [f"run {i + 1} (ms)" for i in range(repeats)] + ["Average (ms)"])
    for cid, ms in rt.items():
        ws.append([cid] + [round(m, 4) for m in ms] + [round(sum(ms) / len(ms), 4)])


def _workbook(dataset, algo, log, gold, alg, min_run,
              input_rt, gold_rt, repeats, n_events_input, maxk):
    wb = openpyxl.Workbook()
    n = min(len(gold), len(alg))
    hits = sum(1 for i in range(n) if gold[i]["event"] == alg[i]["event"])
    accuracy = hits / n if n else 0.0

    # ---- per-trace mismatch counts (also feed Results + Summary) ----
    per_events, per_mis = OrderedDict(), OrderedDict()
    for i in range(n):
        c = gold[i]["case"]
        per_events[c] = per_events.get(c, 0) + 1
        per_mis[c] = per_mis.get(c, 0) + (gold[i]["event"] != alg[i]["event"])

    # ---- per-case fitness ----
    gl, al = _last_per_case(gold), _last_per_case(alg)
    fit_rows, drops = [], {"<0": 0, "=0": 0, ">0": 0}
    sum_fg = sum_fa = 0.0
    for c in gl:
        if c not in al:
            continue
        rg, ra = gl[c], al[c]
        fg = 1 - rg["k"] / (rg["trace_len"] + min_run)
        fa = 1 - ra["k"] / (ra["trace_len"] + min_run)
        drop = fg - fa
        drops["<0" if drop < 0 else "=0" if drop == 0 else ">0"] += 1
        sum_fg += fg
        sum_fa += fa
        fit_rows.append((c, rg["run_len"], rg["trace_len"], rg["sync"], rg["k"], fg,
                         ra["run_len"], ra["trace_len"], ra["sync"], ra["k"], fa, drop))
    ncase = len(fit_rows)
    mean_fg = sum_fg / ncase if ncase else 0.0
    mean_fa = sum_fa / ncase if ncase else 0.0

    # mean runtime (ms/trace and ms/event) on the input log
    in_per_trace = [sum(ms) / len(ms) for ms in input_rt.values()]
    ms_per_trace = sum(in_per_trace) / len(in_per_trace) if in_per_trace else 0.0
    ms_per_event = (sum(in_per_trace) / n_events_input) if n_events_input else 0.0
    gold_per_trace = [sum(ms) / len(ms) for ms in gold_rt.values()]
    gold_ms_per_trace = sum(gold_per_trace) / len(gold_per_trace) if gold_per_trace else 0.0

    # ---- Summary ----
    ws = wb.active
    ws.title = "Summary"
    head = [("Dataset", dataset), ("Algorithm", algo), ("Input log", f"{log}.xes")]
    if algo == "maxk":
        head.append(("maxK", maxk))
    for k, v in head + [
        ("Events", n), ("Traces", len(per_events)),
        ("All-events accuracy", round(accuracy, 6)),
        ("Mean fitness (gold)", round(mean_fg, 6)),
        ("Mean fitness (algorithm)", round(mean_fa, 6)),
        ("Mean fitness drop", round(mean_fg - mean_fa, 6)),
        ("Cases fitness drop < 0", drops["<0"]),
        ("Cases fitness drop = 0", drops["=0"]),
        ("Cases fitness drop > 0", drops[">0"]),
        ("Runtime repeats", repeats),
        ("Mean runtime input (ms/trace)", round(ms_per_trace, 4)),
        ("Mean runtime input (ms/event)", round(ms_per_event, 4)),
        ("Mean runtime gold (ms/trace)", round(gold_ms_per_trace, 4)),
    ]:
        ws.append([k, v])

    # ---- AllEvents ----
    ws = wb.create_sheet("AllEvents")
    ws.append(["Case GSTD", f"Case {algo}", "Event GSTD", f"Event {algo}", "Are equal"])
    for i in range(n):
        g, a = gold[i], alg[i]
        ws.append([g["case"], a["case"], str(g["event"]), str(a["event"]),
                   g["event"] == a["event"]])

    # ---- Results ----
    ws = wb.create_sheet("Results")
    ws.append(["Trace ID", "Total number of events", "Number of mismatches",
               "Av. accuracy per trace"])
    for c in per_events:
        tot, mis = per_events[c], per_mis[c]
        ws.append([c, tot, mis, 1 - mis / tot])
    hist = {}
    for c in per_mis:
        hist[per_mis[c]] = hist.get(per_mis[c], 0) + 1
    ntr = len(per_mis)
    ws.append([])
    ws.append(["# mismatches", "# traces", "%"])
    for k in sorted(hist):
        ws.append([k, hist[k], hist[k] / ntr if ntr else 0.0])

    # ---- Fitness ----
    ws = wb.create_sheet("Fitness")
    ws.append(["CaseID", "|Run_gstd|", "|Trace_gstd|", "Sync_gstd", "Async_gstd",
               "Fitness_gstd", f"|Run_{algo}|", f"|Trace_{algo}|", f"Sync_{algo}",
               f"Async_{algo}", f"Fitness_{algo}", "Fitness drop"])
    for row in fit_rows:
        ws.append(list(row))
    ws.append([])
    ws.append(["Fitness drop", "# cases"])
    for k in ("<0", "=0", ">0"):
        ws.append([k, drops[k]])

    # ---- Runtime (per trace), input log and gold ----
    _runtime_sheet(wb, f"Runtime_{log.title()}", input_rt, repeats)
    _runtime_sheet(wb, "Runtime_GoldStandard", gold_rt, repeats)
    return wb, accuracy, n


def build(dataset, data_dir: Path, out_dir: Path, repeats: int, limit,
          maxk: int = 3, algos=("maxk", "adaptive", "baseline")):
    """Write one workbook per requested algorithm for one dataset. `maxk` is the
    fixed-bound depth (used by the maxk/baseline aligners); `algos` selects which
    workbooks to produce."""
    ddir = data_dir / dataset
    labels = json.loads((ddir / "labels.json").read_text(encoding="utf-8"))
    ck = CASE_KEY[dataset]
    traces = read_reference_traces(ddir / "runs.txt")
    text = build_text(traces)
    engine = FMEngine("".join(text))
    min_run = min(len(t) for t in traces)

    # gold standard: records + per-trace runtime, computed once and shared. The
    # gold alignment uses the same maxk so the comparison is depth-consistent.
    gold = emit_output.emit_baseline(ddir / "gold_standard.xes", labels, ck, text, engine, limit, maxk)
    gold_streams = load_streams(ddir / "gold_standard.xes", labels, ck, limit)
    gold_rt = _per_trace_runtime(gold_streams, text, engine, "ed", repeats, maxk)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n######## {dataset} (maxK={maxk}) ########")
    for algo in algos:
        log, kind = ALGOS[algo]
        alg = _records(log, algo, ddir, labels, ck, text, engine, limit, maxk)
        streams = load_streams(ddir / f"{log}.xes", labels, ck, limit)
        n_events_input = sum(len(s) for s in streams.values())
        input_rt = _per_trace_runtime(streams, text, engine, kind, repeats, maxk)

        wb, accuracy, n = _workbook(dataset, algo, log, gold, alg, min_run,
                                    input_rt, gold_rt, repeats, n_events_input, maxk)
        name = f"{SHORT.get(dataset, dataset)}_{algo}"
        if algo == "maxk":
            name += f"_{maxk}"           # e.g. BD_maxk_4.xlsx
        path = out_dir / f"{name}.xlsx"
        wb.save(path)
        print(f"  {path.name}: events={n} all-events-accuracy={accuracy:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--datasets", nargs="+",
                    default=["blood_donation", "smart_home", "hr_recruitment"])
    ap.add_argument("--algos", nargs="+", default=["maxk", "adaptive", "baseline"],
                    choices=["maxk", "adaptive", "baseline"],
                    help="which algorithms to produce workbooks for")
    ap.add_argument("--maxk", type=int, default=3,
                    help="fixed-bound depth for the maxk/baseline aligners")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    for ds in args.datasets:
        build(ds, args.data_dir, args.out, args.repeats, args.limit, args.maxk, args.algos)


if __name__ == "__main__":
    main()
