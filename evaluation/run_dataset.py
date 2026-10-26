from __future__ import annotations

import argparse
import json
import random
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

from common.refs import build_text, read_reference_traces, CASE_KEY, DATASETS
from abd.deferring_ed import DeferringED, _fmt
from common.fm_engine import FMEngine


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def label_to_char(labels: dict, name: str) -> str:
    return chr(labels[name] + ord("A"))


def parse_events(xes_path: Path, labels: dict, case_key: str):
    for _event, elem in ET.iterparse(str(xes_path), events=("end",)):
        if _localname(elem.tag) != "event":
            continue

        case_id = None
        candidates: list[str] = []

        for child in elem:
            tag = _localname(child.tag)
            key = child.get("key")
            if tag == "string" and key == "concept:name":
                candidates = [label_to_char(labels, child.get("value"))]
            elif tag == "string" and key == case_key:
                case_id = child.get("value")
            elif tag == "container" and key == "uncertainty:discrete_weak":
                names = []
                for entry in child:                      # uncertainty:entry blocks
                    for e in entry:
                        if _localname(e.tag) == "string" and e.get("key") == "concept:name":
                            names.append(e.get("value"))
                candidates = [label_to_char(labels, n) for n in names]

        if case_id is not None and candidates:
            yield case_id, candidates
        elem.clear()


def load_streams(xes_path: Path, labels: dict, case_key: str, limit: int | None):
    streams: "OrderedDict[str, list]" = OrderedDict()
    for case_id, candidates in parse_events(xes_path, labels, case_key):
        if case_id not in streams:
            if limit is not None and len(streams) >= limit:
                break
            streams[case_id] = []
        streams[case_id].append(candidates if len(candidates) > 1 else candidates[0])
    return streams


def run(dataset: str, data_dir: Path, limit: int | None, give_up_bound: int):
    ddir = data_dir / dataset
    labels = json.loads((ddir / "labels.json").read_text(encoding="utf-8"))
    case_key = CASE_KEY[dataset]

    traces = read_reference_traces(ddir / "runs.txt")
    text = build_text(traces)
    engine = FMEngine("".join(text))   # built once, shared across all cases
    print(f"model: {dataset}  runs={len(traces)}  text_len={len(text)}")

    t0 = time.perf_counter()
    streams = load_streams(ddir / "ambiguous.xes", labels, case_key, limit)
    print(f"parsed {len(streams)} cases  ({time.perf_counter() - t0:.2f}s)")

    results = []
    n_amb = n_disambiguated = n_random = 0
    horizons = []
    rng = random.Random(0)
    t1 = time.perf_counter()
    for case_id, stream in streams.items():
        d = DeferringED(text, give_up_bound=give_up_bound, rng=rng, engine=engine)
        emissions = []
        for ev in stream:
            emissions.extend(d.feed(ev))
        emissions.extend(d.finish())

        for r in d.reports:
            n_amb += 1
            if r.was_random:
                n_random += 1
            else:
                n_disambiguated += 1
                horizons.append(r.actual_horizon)

        results.append({
            "case": case_id,
            "emissions": [
                {"at": e.at_event, "label": e.label, "held": e.deferred_by, "note": e.note}
                for e in emissions
            ],
            "ambiguities": [
                {"onset": r.onset_event, "candidates": r.candidates,
                 "onset_k": _fmt(r.onset_k), "verdict": r.verdict,
                 "random": r.was_random, "random_reason": r.random_reason,
                 "horizon": r.actual_horizon,
                 "k_trace": [(ev, _fmt(k)) for ev, k in r.k_trace]}
                for r in d.reports
            ],
        })
    dt = time.perf_counter() - t1

    print(f"processed {len(streams)} cases in {dt:.2f}s "
          f"({dt / max(1, len(streams)) * 1000:.1f} ms/case)")
    print(f"ambiguous events: {n_amb}  disambiguated by deferral: {n_disambiguated}  "
          f"random fallback (unseparable): {n_random}")
    if horizons:
        print(f"deferral horizon: mean={sum(horizons)/len(horizons):.2f}  "
              f"min={min(horizons)}  max={max(horizons)}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    choices=DATASETS)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most this many cases (recommended)")
    ap.add_argument("--give-up-bound", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None,
                    help="write per-case emissions/ambiguities as JSON")
    args = ap.parse_args()

    results = run(args.dataset, args.data_dir, args.limit, args.give_up_bound)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
