from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from common.refs import build_text, read_reference_traces, CASE_KEY, DATASETS
from abd.deferring_ed import DeferringED
from fbd.ed import ED
from common.fm_engine import FMEngine
from evaluation.run_dataset import label_to_char, parse_events

LOG_TYPES = ["gold_standard", "baseline", "ambiguous"]


def _record(case_id, r):
    return {
        "case": case_id,
        "event": r.events if r.aligned else "_",
        "history": r.history,
        "k": r.cost,
        "sync": r.sync,
        "run_len": r.run_len,
        "trace_len": r.trace_len,
    }


def emit_baseline(xes_path, labels, case_key, text, engine, limit, maxk=3):
    eds: dict = {}
    seen: list = []
    results = []
    for case_id, candidates in parse_events(xes_path, labels, case_key):
        if case_id not in eds:
            if limit is not None and len(seen) >= limit:
                break
            seen.append(case_id)
            eds[case_id] = ED(text, engine=engine)
            eds[case_id].setup(maxk)
        chars = candidates if len(candidates) > 1 else [candidates[0]]
        r = eds[case_id].step(chars)
        results.append(_record(case_id, r))
    return results


def emit_adaptive(xes_path, labels, case_key, text, engine, limit):
    rng = random.Random(0)
    ds: dict = {}
    order: dict = {}
    seen: list = []
    results = []

    def apply(case_id, emissions):
        for e in emissions:
            order[case_id][e.at_event]["event"] = [e.label] if e.label != ">" else "_"

    for case_id, candidates in parse_events(xes_path, labels, case_key):
        if case_id not in ds:
            if limit is not None and len(seen) >= limit:
                break
            seen.append(case_id)
            ds[case_id] = DeferringED(text, engine=engine, rng=rng)
            order[case_id] = []
        d = ds[case_id]
        emissions = d.feed(candidates if len(candidates) > 1 else candidates[0])
        r = d.last_step
        slot = {"case": case_id, "event": "_", "history": r.history,
                "k": r.cost, "sync": r.sync, "run_len": r.run_len,
                "trace_len": r.trace_len}
        order[case_id].append(slot)
        results.append(slot)
        apply(case_id, emissions)
    for case_id, d in ds.items():
        apply(case_id, d.finish())
    return results


def run(dataset, data_dir, log_types, algo, limit):
    ddir = data_dir / dataset
    labels = json.loads((ddir / "labels.json").read_text(encoding="utf-8"))
    case_key = CASE_KEY[dataset]
    text = build_text(read_reference_traces(ddir / "runs.txt"))
    engine = FMEngine("".join(text))
    outdir = ddir / "output"
    outdir.mkdir(exist_ok=True)

    for log_type in log_types:
        xes = ddir / f"{log_type}.xes"
        t = time.perf_counter()
        if algo == "adaptive":
            results = emit_adaptive(xes, labels, case_key, text, engine, limit)
        else:
            results = emit_baseline(xes, labels, case_key, text, engine, limit)
        suffix = "" if algo == "baseline" else f"_{algo}"
        out = outdir / f"{log_type}{suffix}.json"
        out.write_text(
            "[" + ",\n".join(json.dumps(r, sort_keys=True) for r in results) + "]",
            encoding="utf-8",
        )
        print(f"{dataset}/{log_type} ({algo}): {len(results)} records -> {out}  "
              f"({time.perf_counter() - t:.1f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--log", choices=LOG_TYPES, help="one log type (default: all three)")
    ap.add_argument("--algo", choices=["baseline", "adaptive"], default="baseline")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    logs = [args.log] if args.log else LOG_TYPES
    run(args.dataset, args.data_dir, logs, args.algo, args.limit)


if __name__ == "__main__":
    main()
