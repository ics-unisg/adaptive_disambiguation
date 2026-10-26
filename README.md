# Adaptive Look-ahead for On-the-fly Event Disambiguation via Alignments

This repository contains the Python implementation accompanying the Process Science paper *Adaptive Look-ahead for On-the-fly Event Disambiguation via Alignments*, which extends the paper *On-the-fly Event Disambiguation via Alignments*, presented at BPMDS 2026.
This repository contains two algorithms: the original fixed-bound one (FBD) from the BPMDS paper and the adaptive-bound extension (ABD) introduced in the Process Science paper.

## Structure

```
common/      no algorithm logic
  fm_engine.py   FM-index backing; stands in for the C++ Bi_FM_Index
  refs.py        reading reference traces; the concatenated text ",run1,run2,...,"
fbd/         the Python translation of the original algorithm fron the BPMDS paper
  ed.py          line-by-line port of the C++ ED class (ED.cpp / ED.hpp)
abd/         the adaptive-bound algorithm introduced in the Process Science paper
  ed.py          as fbd/ed.py, but with the first-hit search (see below)
  deferring_ed.py  the extension itself
evaluation/  the code to run the evaluation described in the paper
```

Note that `fbd/ed.py` and `abd/ed.py` differ in method `_resolve_by_model_moves`: the former bounds the model-move search by the fixed `maxK`, the latter explores one model move at a time and stops at the first depth at which a candidate becomes consumable, terminating at the end of a reference trace.

## Running

```bash
pip install -r requirements.txt
python -m pytest -q                                   # 37 tests
python -m evaluation.score_gold  --dataset smart_home --limit 200 --seeds 50
python -m evaluation.emit_output --dataset smart_home --log ambiguous --algo adaptive
python -m evaluation.evaluate_outputs --dataset smart_home
```

The datasets are not included. The evaluation code expects

```
data/<dataset>/{runs.txt, ambiguous.xes, gold_standard.xes, labels.json}
```

with `<dataset>` one of `blood_donation`, `smart_home`; `runs.txt` holds one reference run per line over single-character labels. Please note that the XES logs encode ambiguity as `uncertainty:discrete_weak` containers.
