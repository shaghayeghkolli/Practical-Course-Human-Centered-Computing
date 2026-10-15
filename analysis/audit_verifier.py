#!/usr/bin/env python3
"""Compare automatic verifier decisions with the released human audit.

The annotation tally contains one record per (post_id, perturbation) pair.
For A4_stt_artifacts, the experiment defines meaning as neutral, so the human
reference label depends only on whether the transformation was applied. For
all other types it requires both application and the expected meaning outcome.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--verified-csv", type=Path, required=True)
    parser.add_argument(
        "--export-matched",
        type=Path,
        help="Optionally export the matched rows with human reference columns.",
    )
    return parser.parse_args()


def as_bool(value: str) -> bool:
    value = value.strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"Expected True/False, got {value!r}")
    return value == "true"


def human_label(row: dict) -> bool | None:
    applied = row.get("a")
    if applied is None:
        return None
    if applied != "Yes":
        return False
    if row["pert"] == "A4_stt_artifacts":
        return True
    meaning = row.get("m")
    if meaning is None:
        return None
    return meaning == "Yes"


def effective_auto_label(row: dict) -> bool:
    """Return the retention decision used by the reported retrieval runs.

    The uploaded CSVs predate the neutral-STT recomputation and store every
    A4 row as verified=False. The result grids nevertheless include exactly
    the A4 rows with perturbation_applied=True, which is the rule documented
    by the paper and implemented by the current verifier.
    """
    if row["perturbation_name"] == "A4_stt_artifacts":
        return as_bool(row["perturbation_applied"])
    return as_bool(row["verified"])


def scores(pairs: list[tuple[bool, bool | None]]) -> dict[str, float | int]:
    counts = Counter((auto, human) for auto, human in pairs if human is not None)
    tp = counts[(True, True)]
    fp = counts[(True, False)]
    tn = counts[(False, False)]
    fn = counts[(False, True)]
    n = tp + fp + tn + fn

    def ratio(num: int, den: int) -> float:
        return num / den if den else math.nan

    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": ratio(tp + tn, n),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def format_scores(label: str, result: dict[str, float | int]) -> str:
    def pct(value: float | int) -> str:
        return "NA" if isinstance(value, float) and math.isnan(value) else f"{100*float(value):.1f}%"

    return (
        f"{label:25} n={result['n']:3d}  "
        f"TP={result['tp']:2d} FP={result['fp']:2d} "
        f"TN={result['tn']:2d} FN={result['fn']:2d}  "
        f"acc={pct(result['accuracy'])} precision={pct(result['precision'])} "
        f"recall={pct(result['recall'])} F1={pct(result['f1'])}"
    )


def main() -> None:
    args = parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    wanted: dict[tuple[str, str], tuple[str, dict]] = {}
    for language, rows in annotations.items():
        for row in rows:
            key = (str(row["post"]), row["pert"])
            if key in wanted:
                raise ValueError(f"Duplicate annotation key: {key}")
            wanted[key] = (language, row)

    matched: dict[tuple[str, str], dict] = {}
    with args.verified_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "post_id", "perturbation_name", "perturbation_applied", "verified"
        }
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")
        for row in reader:
            key = (str(row["post_id"]), row["perturbation_name"])
            if key not in wanted:
                continue
            if key in matched:
                raise ValueError(f"Duplicate CSV key: {key}")
            matched[key] = row

    missing = sorted(set(wanted).difference(matched))
    if missing:
        raise ValueError(f"Missing {len(missing)} annotated rows; first: {missing[:5]}")

    records: list[tuple[str, str, bool, bool | None]] = []
    for key, (language, annotation) in wanted.items():
        records.append(
            (
                language,
                annotation["pert"],
                effective_auto_label(matched[key]),
                human_label(annotation),
            )
        )

    if args.export_matched:
        first_row = next(iter(matched.values()))
        extra_fields = [
            "human_language",
            "human_applied",
            "human_naturalness",
            "human_meaning_ok",
            "human_verified",
        ]
        args.export_matched.parent.mkdir(parents=True, exist_ok=True)
        with args.export_matched.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(first_row) + extra_fields)
            writer.writeheader()
            for key, (language, annotation) in wanted.items():
                row = dict(matched[key])
                reference = human_label(annotation)
                row.update(
                    {
                        "human_language": language,
                        "human_applied": annotation.get("a"),
                        "human_naturalness": annotation.get("n"),
                        "human_meaning_ok": annotation.get("m"),
                        "human_verified": "" if reference is None else reference,
                    }
                )
                writer.writerow(row)
        print(f"Exported matched audit sample to {args.export_matched}")

    print(f"Matched {len(records)} annotation rows; {sum(r[3] is not None for r in records)} have complete reference labels.\n")
    print(format_scores("OVERALL", scores([(a, h) for _, _, a, h in records])))

    print("\nBy language")
    for language in annotations:
        pairs = [(a, h) for lang, _, a, h in records if lang == language]
        print(format_scores(language, scores(pairs)))

    print("\nBy perturbation")
    grouped: dict[str, list[tuple[bool, bool | None]]] = defaultdict(list)
    for _, perturbation, auto, human in records:
        grouped[perturbation].append((auto, human))
    for perturbation in sorted(grouped):
        print(format_scores(perturbation, scores(grouped[perturbation])))


if __name__ == "__main__":
    main()
