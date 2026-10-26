from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common.refs import read_reference_traces, DATASETS


def all_events_accuracy(gold_json: Path, other_json: Path):
    g = json.loads(gold_json.read_text(encoding="utf-8"))
    o = json.loads(other_json.read_text(encoding="utf-8"))
    n = min(len(g), len(o))
    hits = sum(1 for i in range(n) if g[i]["event"] == o[i]["event"])
    return hits / n, n


def _last_per_case(records):
    last = {}
    for r in records:
        last[r["case"]] = r          # key by case id as-is (not all ids are ints)
    return last


def fitness_table(gold_json: Path, amb_json: Path, base_json: Path,
                  runs_file: Path, out_csv: Path, adp_json: Path = None):
    """Per-case fitness, 1 - async / (|trace| + |shortest run|). Columns: gold
    standard, null-model baseline, algorithm (maxK), and, when ambiguous_adaptive
    is present, the adaptive algorithm."""
    min_run = min(len(t) for t in read_reference_traces(runs_file))
    load = lambda p: _last_per_case(json.loads(p.read_text(encoding="utf-8")))
    cols = [("gstd", load(gold_json)), ("base", load(base_json)), ("amb", load(amb_json))]
    if adp_json is not None and adp_json.exists():
        cols.append(("adp", load(adp_json)))

    header = ["CaseID"]
    for tag, _ in cols:
        header += [f"|Run_{tag}|", f"|Trace_{tag}|", f"Sync_{tag}", f"Async_{tag}", f"Fitness_{tag}"]
    common = set.intersection(*(set(d) for _, d in cols))
    rows = []
    for case in sorted(common):
        cells = [case]
        for _, d in cols:
            rec = d[case]
            fit = 1 - rec["k"] / (rec["trace_len"] + min_run)
            cells += [rec["run_len"], rec["trace_len"], rec["sync"], rec["k"], fit]
        rows.append(cells)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return rows, header


def run(dataset, data_dir: Path):
    out = data_dir / dataset / "output"
    gold = out / "gold_standard.json"
    print(f"=== {dataset} ===")
    for name, f in [("maxK", "ambiguous.json"), ("adaptive-k", "ambiguous_adaptive.json")]:
        path = out / f
        if not path.exists():
            print(f"  {name:11s} (missing {f})")
            continue
        acc, n = all_events_accuracy(gold, path)
        print(f"  {name:11s} all-events accuracy = {acc:.3f}   ({n} events)")

    base = out / "baseline.json"
    if base.exists():
        rows, header = fitness_table(gold, out / "ambiguous.json", base,
                                     data_dir / dataset / "runs.txt",
                                     out / f"fitness_{dataset}.csv",
                                     adp_json=out / "ambiguous_adaptive.json")
        idx = {h: i for i, h in enumerate(header)}
        for col in ("Fitness_gstd", "Fitness_base", "Fitness_amb", "Fitness_adp"):
            if col not in idx:
                continue
            mean = sum(r[idx[col]] for r in rows) / len(rows)
            print(f"  mean {col} = {mean:.3f}")
    else:
        print("  (no baseline.json; skipping fitness)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--data-dir", default="data", type=Path)
    args = ap.parse_args()
    run(args.dataset, args.data_dir)


if __name__ == "__main__":
    main()
