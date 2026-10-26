"""Convert the Haifa 'traffic-data-to-event-log' uncertain logs + their BPMN
ground-truth models into this prototype's dataset format.

Source (https://github.com/HaifaUniversityBPM/traffic-data-to-event-log):
  * an uncertain XES log whose events carry, per event, the manually tagged
    ground truth (real_activity_action = "<Type> Start|End", real_single_-
    activity_action = "Activity Start|End|NoAction", real_case_id) and a set of
    competing activity candidates;
  * a BPMN model of the executed process (ground truth for the control flow).

Produces, under data/<dataset>/:
  runs.txt          reference runs = BPMN play-out, EACH TASK DUPLICATED so a run
                    has one symbol per Start and one per End (matching the log,
                    which records every activity as a Start event and an End
                    event). Loops are bounded by --loop-bound.
  labels.json       {activity name: index}; char = chr(index + 'A').
  ambiguous.xes     per event: the candidate set, as a Pegoraro
                    uncertainty:discrete_weak container (>1 candidate) or a single
                    concept:name (1 candidate). NoAction events get the reserved
                    label "NoAction" (absent from every run -> a log move).
  gold_standard.xes per event: the single true label (real_activity_action type,
                    or "NoAction").

Candidate set per event (probabilities ignored, i.e. "all candidates"):
  * strategy "activities": the log's own `activities` field (HR log has it).
  * strategy "argmax": union of the top-1 label of each uncertainty distribution
    container (for logs without an `activities` field, e.g. ptp).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

NOISE = "NoAction"


def _ln(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


# --------------------------------------------------------------------------- #
# BPMN -> reference runs (each task duplicated; loops bounded)
# --------------------------------------------------------------------------- #
def bpmn_runs(bpmn_path: Path, loop_bound: int) -> list[list[str]]:
    root = ET.parse(bpmn_path).getroot()
    tasks: dict[str, str] = {}
    starts, ends, flows = [], set(), []
    for el in root.iter():
        t = _ln(el.tag)
        if t == "task":
            tasks[el.get("id")] = el.get("name")
        elif t == "startEvent":
            starts.append(el.get("id"))
        elif t == "endEvent":
            ends.add(el.get("id"))
        elif t == "sequenceFlow":
            flows.append((el.get("sourceRef"), el.get("targetRef")))
    succ: dict[str, list[str]] = {}
    for s, d in flows:
        succ.setdefault(s, []).append(d)

    runs: list[list[str]] = []

    def walk(node: str, acc: list[str], visits: dict[str, int]):
        if node in ends:
            runs.append(acc)
            return
        if visits.get(node, 0) > loop_bound:        # bound any cycle
            return
        visits = dict(visits)
        visits[node] = visits.get(node, 0) + 1
        nm = tasks.get(node)
        acc2 = acc + [nm, nm] if nm else acc        # duplicate each task
        nxts = succ.get(node, [])
        if not nxts:                                # dangling (treat as end)
            runs.append(acc2)
            return
        for nxt in nxts:
            walk(nxt, acc2, visits)

    for st in starts:
        walk(st, [], {})
    # dedup, keep deterministic order
    seen, out = set(), []
    for r in runs:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return sorted(out, key=lambda r: (len(r), r))


# --------------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------------- #
def _event_fields(ev):
    """Return (true_type, action, candidates, case_id) for one <event>."""
    true_type = action = case_id = None
    activities = None
    argmax_union: set[str] = set()
    for s in ev:
        # note: real_case_id is stored as a numeric attribute, not a string, so we
        # must not restrict to <string> here; only the discrete_weak containers are.
        k, v = s.get("key"), s.get("value")
        if k == "real_activity_action":
            true_type = v
        elif k == "real_single_activity_action":
            action = v
        elif k == "real_case_id":
            case_id = v
        elif k == "activities":
            activities = set(re.findall(r"'([^']+)'", v))
        elif v == "discrete_weak":                   # a distribution container
            pairs = [(c.get("key"), float(c.get("value"))) for c in s]
            if pairs:
                argmax_union.add(max(pairs, key=lambda x: x[1])[0])
    return true_type, action, activities, argmax_union, case_id


def log_cases(log_path: Path, strategy: str):
    """Yield (case_id, [(candidates:set, gold:str, is_noise:bool), ...])."""
    root = ET.parse(log_path).getroot()
    cases: "OrderedDict[str, list]" = OrderedDict()
    for tr in root.iter():
        if _ln(tr.tag) != "trace":
            continue
        for ev in tr:
            if _ln(ev.tag) != "event":
                continue
            true_type, action, activities, argmax_union, case_id = _event_fields(ev)
            if case_id is None:
                continue
            is_noise = (action == NOISE) or (true_type in (None, NOISE))
            if is_noise:
                gold = NOISE
                cands = {NOISE}
            else:
                gold = true_type.rsplit(" ", 1)[0]   # strip " Start"/" End"
                cands = set(activities) if strategy == "activities" and activities else set(argmax_union)
                cands = {c.rsplit(" ", 1)[0] if c.endswith((" Start", " End")) else c for c in cands}
                if not cands:
                    cands = {gold}
            cases.setdefault(case_id, []).append((cands, gold, is_noise))
    return cases


# --------------------------------------------------------------------------- #
# Emit the four dataset files
# --------------------------------------------------------------------------- #
def _xes_event(case_key, case_id, candidates):
    cs = sorted(candidates)
    lines = [f'\t\t<event>\n\t\t\t<string key="{case_key}" value="{case_id}" />']
    if len(cs) == 1:
        lines.append(f'\t\t\t<string key="concept:name" value="{cs[0]}" />')
    else:
        lines.append('\t\t\t<container key="uncertainty:discrete_weak">')
        for c in cs:
            lines.append('\t\t\t\t<container key="uncertainty:entry">'
                         f'<string key="concept:name" value="{c}" /></container>')
        lines.append('\t\t\t</container>')
    lines.append('\t\t</event>')
    return "\n".join(lines)


def _write_xes(path, cases, case_key, pick):
    """pick(entry) -> candidate set to emit for that event."""
    out = ['<?xml version="1.0" encoding="UTF-8" ?>',
           '<log xes.version="1849-2016" xmlns="http://www.xes-standard.org/">']
    for case_id, entries in cases.items():
        out.append("\t<trace>")
        for entry in entries:
            out.append(_xes_event(case_key, case_id, pick(entry)))
        out.append("\t</trace>")
    out.append("</log>\n")
    path.write_text("\n".join(out), encoding="utf-8")


def convert(bpmn_path, log_path, out_dir, strategy, case_key, loop_bound):
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = bpmn_runs(bpmn_path, loop_bound)
    cases = log_cases(log_path, strategy)

    # label alphabet: all activity names in the runs, plus NoAction if the log has it
    names = sorted({n for r in runs for n in r})
    has_noise = any(e[2] for ents in cases.values() for e in ents)
    if has_noise:
        names.append(NOISE)                          # NoAction last -> absent from runs
    labels = {n: i for i, n in enumerate(names)}
    char = lambda n: chr(labels[n] + ord("A"))

    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (out_dir / "runs.txt").write_text(
        "\n".join("".join(char(n) for n in r) for r in runs) + "\n", encoding="utf-8")
    _write_xes(out_dir / "ambiguous.xes", cases, case_key, pick=lambda e: e[0])
    _write_xes(out_dir / "gold_standard.xes", cases, case_key, pick=lambda e: {e[1]})
    # null-model baseline: each event resolved to a uniformly random candidate
    # (seeded for reproducibility), as in the paper's baseline B.
    rng = random.Random(0)
    _write_xes(out_dir / "baseline.xes", cases, case_key,
               pick=lambda e: {rng.choice(sorted(e[0]))})

    n_ev = sum(len(e) for e in cases.values())
    n_amb = sum(1 for ents in cases.values() for e in ents if len(e[0]) > 1)
    n_noise = sum(1 for ents in cases.values() for e in ents if e[2])
    print(f"  {out_dir.name}: runs={len(runs)} cases={len(cases)} events={n_ev} "
          f"ambiguous={n_amb} noise={n_noise} labels={len(labels)}")
    print(f"    runs: " + ", ".join("".join(char(n) for n in r) for r in runs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="the datasets/ folder holding UNCERTAIN_XES_LOGS_ and BPMN_GROUND_TRUTH")
    ap.add_argument("--out", type=Path, required=True, help="the data/ folder to write into")
    ap.add_argument("--loop-bound", type=int, default=2)
    args = ap.parse_args()
    xes = args.src / "UNCERTAIN_XES_LOGS_"
    bpmn = args.src / "BPMN_GROUND_TRUTH"
    print("converting:")
    convert(bpmn / "hrrp.xml",
            xes / "uncertainty_hr_data_anonymize_recognition_results.xes",
            args.out / "hr_recruitment", "activities", "case:id", args.loop_bound)
    convert(bpmn / "order-to-cash-bpmn.xml",
            xes / "uncertainty_ptp_data_anonymize_recognition_results.xes",
            args.out / "purchase_to_pay", "argmax", "case:id", args.loop_bound)


if __name__ == "__main__":
    main()
