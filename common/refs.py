from __future__ import annotations

from pathlib import Path

COMMA = ","


def read_reference_traces(file_path, delimiter: str | None = None) -> list[list[str]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference-trace file not found: {path}")

    traces: list[list[str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if delimiter is not None:
                trace = [t.strip() for t in line.split(delimiter) if t.strip()]
            elif any(ch.isspace() for ch in line):
                trace = line.split()
            else:
                trace = list(line)
            if not trace:
                raise ValueError(f"Empty trace on line {line_number}: {raw_line!r}")
            if COMMA in trace:
                raise ValueError(
                    f"Reserved separator {COMMA!r} occurs on line {line_number}."
                )
            traces.append(trace)

    if not traces:
        raise ValueError("The input file contains no non-empty traces.")
    return traces


def build_text(traces: list[list[str]]) -> list[str]:
    text: list[str] = []
    for trace in traces:
        text.append(COMMA)
        text.extend(trace)
    text.append(COMMA)
    return text

# Per-dataset case-id attribute key and the registered datasets.
CASE_KEY = {
    "smart_home": "routine_id",
    "blood_donation": "donation_id",
    "hr_recruitment": "case:id",
    "purchase_to_pay": "case:id",
}
DATASETS = list(CASE_KEY)
