from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common.refs import COMMA, build_text, read_reference_traces, CASE_KEY, DATASETS
from abd.deferring_ed import DeferringED
from evaluation.run_dataset import load_streams
from evaluation.score_gold import gold_labels


def run_adaptive(text, stream, give_up_unbounded):
    d = DeferringED(text, give_up_unbounded=give_up_unbounded)
    for ev in stream:
        d.feed(ev)
    d.finish()
    reports = {r.onset_event: r for r in d.reports}
    return reports




def run(dataset, data_dir, limit):
    case_key = CASE_KEY[dataset]
    labels = json.loads((data_dir / dataset / "labels.json").read_text(encoding="utf-8"))
    text = build_text(read_reference_traces(data_dir / dataset / "runs.txt"))
    alphabet = sorted({s for s in text if s != COMMA}, reverse=True)

    amb_streams = load_streams(data_dir / dataset / "ambiguous.xes", labels, case_key, limit)
    gold = gold_labels(data_dir, dataset, case_key, limit)

    cat = Counter()
    onset_kind = Counter()
    witnesses = []
    n_declines = 0

    for cid, stream in amb_streams.items():
        g = gold.get(cid)
        if g is None or len(g) != len(stream):
            continue
        amb_idx = [i for i, ev in enumerate(stream) if not isinstance(ev, str)]
        if not amb_idx:
            continue

        rep_on = run_adaptive(text, stream, give_up_unbounded=True)
        rep_off = run_adaptive(text, stream, give_up_unbounded=False)

        for i in amb_idx:
            r = rep_on.get(i)
            if not (r and r.was_random):
                continue  # disambiguated by default policy; not a forced pick
            n_declines += 1
            onset_kind["maxK_C=0" if r.onset_k == 0 else "maxK_C>0"] += 1

            r_off = rep_off.get(i)
            truth = g[i]
            if r_off is None or r_off.was_random:
                cat["STUCK (still forced-random without the give-up)"] += 1
            elif r_off.verdict == truth:
                cat["PREMATURE (disambiguates correctly)"] += 1
            else:
                cat["PREMATURE (disambiguates wrong)"] += 1

            if r.onset_k == 0 and len(witnesses) < 6:
                witnesses.append((cid, i, r.candidates))

    print(f"dataset={dataset}  declines analysed={n_declines}")
    print("\nby onset_k:")
    for k, v in onset_kind.most_common():
        print(f"  {k:8s} {v}")
    print("\nwhat happens if we DON'T give up when unbounded (keep pruning):")
    for k, v in cat.most_common():
        print(f"  {k:32s} {v}")
    if witnesses:
        print("\nsample unbounded declines (candidate sets whose branches share a full continuation):")
        for cid, i, cands in witnesses:
            print(f"  case {cid} event {i}: candidates {cands}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="smart_home",
                    choices=DATASETS)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    run(args.dataset, args.data_dir, args.limit)


if __name__ == "__main__":
    main()
