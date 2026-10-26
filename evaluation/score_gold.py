from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common.refs import build_text, read_reference_traces, CASE_KEY, DATASETS
from abd.deferring_ed import DeferringED
from fbd.ed import ED
from common.fm_engine import FMEngine
from evaluation.run_dataset import load_streams


def gold_labels(data_dir: Path, dataset: str, case_key: str, limit):
    labels = json.loads((data_dir / dataset / "labels.json").read_text(encoding="utf-8"))
    streams = load_streams(data_dir / dataset / "gold_standard.xes", labels, case_key, limit)
    return {cid: [ev if isinstance(ev, str) else ev[0] for ev in st]
            for cid, st in streams.items()}


def baseline_labels(text, stream, engine):
    ed = ED(text, engine=engine)
    ed.setup(3)
    out = {}
    for i, ev in enumerate(stream):
        r = ed.step([ev] if isinstance(ev, str) else list(ev))
        out[i] = r.events
    return out


def adaptive_labels(text, stream, give_up_bound, rng, engine):
    d = DeferringED(text, give_up_bound=give_up_bound, rng=rng, engine=engine)
    emitted = {}
    for ev in stream:
        for e in d.feed(ev):
            emitted[e.at_event] = e.label
    for e in d.finish():
        emitted[e.at_event] = e.label
    reports = {r.onset_event: r for r in d.reports}
    return emitted, reports


def run(dataset, data_dir, limit, give_up_bound, seeds=20):
    case_key = CASE_KEY[dataset]
    labels = json.loads((data_dir / dataset / "labels.json").read_text(encoding="utf-8"))

    traces = read_reference_traces(data_dir / dataset / "runs.txt")
    text = build_text(traces)
    engine = FMEngine("".join(text))   # built once, shared across all cases

    amb_streams = load_streams(data_dir / dataset / "ambiguous.xes", labels, case_key, limit)
    gold = gold_labels(data_dir, dataset, case_key, limit)

    b_correct = 0
    ad_correct = ad_wrong = 0
    random_pools = []
    n_amb = 0
    length_mismatch = 0

    run_rng = random.Random(0)
    for cid, stream in amb_streams.items():
        g = gold.get(cid)
        if g is None or len(g) != len(stream):
            length_mismatch += 1
            continue
        amb_idx = [i for i, ev in enumerate(stream) if not isinstance(ev, str)]
        if not amb_idx:
            continue

        base = baseline_labels(text, stream, engine)
        adap, areports = adaptive_labels(text, stream, give_up_bound, run_rng, engine)

        for i in amb_idx:
            n_amb += 1
            truth = g[i]
            if base.get(i) == [truth]:
                b_correct += 1
            r = areports.get(i)
            if r is not None and r.was_random:
                random_pools.append((r.pick_pool, truth))
            else:
                if adap.get(i) == truth:
                    ad_correct += 1
                else:
                    ad_wrong += 1

    print(f"dataset={dataset}  cases={len(amb_streams)}  ambiguous_events={n_amb}"
          + (f"  (skipped {length_mismatch} length-mismatch cases)" if length_mismatch else ""))
    if n_amb == 0:
        print("  no ambiguous events to score.")
        return

    print(f"\nbaseline (maxK stand-in, always guesses):  accuracy={b_correct / n_amb:.3f}")

    n_disambig = ad_correct + ad_wrong
    n_random = len(random_pools)
    exp_random = sum((1.0 / len(p)) if t in p else 0.0 for p, t in random_pools)

    overalls = []
    for s in range(seeds):
        rng = random.Random(s)
        hits = sum(1 for p, t in random_pools if rng.choice(p) == t)
        overalls.append((ad_correct + hits) / n_amb)
    mean = sum(overalls) / len(overalls)
    var = sum((o - mean) ** 2 for o in overalls) / len(overalls)
    std = var ** 0.5

    print("\nadaptive-k (never declines; random pick when unseparable):")
    print(f"  disambiguated: {n_disambig}  accuracy={ad_correct / n_disambig:.3f}"
          if n_disambig else "  disambiguated: 0")
    print(f"  random fallback: {n_random}  (rate={n_random / n_amb:.3f})")
    if n_random:
        print(f"    expected accuracy (analytical 1/k) = {exp_random / n_random:.3f}")
    print(f"  overall accuracy over {seeds} seeds: "
          f"mean={mean:.3f}  std={std:.3f}  min={min(overalls):.3f}  max={max(overalls):.3f}")
    print(f"  overall accuracy (analytical expected) = {(ad_correct + exp_random) / n_amb:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--give-up-bound", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=20,
                    help="number of random seeds to average the fallback over")
    args = ap.parse_args()
    run(args.dataset, args.data_dir, args.limit, args.give_up_bound, args.seeds)


if __name__ == "__main__":
    main()
