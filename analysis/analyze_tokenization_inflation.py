#!/usr/bin/env python3
"""Join tokenizer inflation cells to retrieval degradation and summarize association.

Correlations are descriptive mechanism checks, not causal estimates. In addition
to raw Pearson/Spearman correlations, the script reports a within-cell residual
correlation after iteratively removing additive model, dataset, language, and
perturbation means.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


NON_LATIN = {"AR", "HI", "ML", "ZH"}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return math.nan
    mx, my = mean(x), mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    )
    return numerator / denominator if denominator else math.nan


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for position in ordered[start:end]:
            output[position] = average_rank
        start = end
    return output


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(ranks(x), ranks(y))


def residualize(rows: list[dict], value: str, dimensions: tuple[str, ...]) -> list[float]:
    residuals = [float(row[value]) for row in rows]
    # Alternating projections handle the unbalanced factorial design.
    for _ in range(30):
        largest_adjustment = 0.0
        for dimension in dimensions:
            groups: dict[str, list[int]] = defaultdict(list)
            for index, row in enumerate(rows):
                groups[row[dimension]].append(index)
            for indices in groups.values():
                adjustment = mean([residuals[index] for index in indices])
                largest_adjustment = max(largest_adjustment, abs(adjustment))
                for index in indices:
                    residuals[index] -= adjustment
        if largest_adjustment < 1e-12:
            break
    return residuals


def statistics(rows: list[dict]) -> dict[str, float | int]:
    inflation = [float(row["token_inflation"]) for row in rows]
    degradation = [float(row["relative_map10_degradation"]) for row in rows]
    dimensions = ("model", "dataset", "language", "perturbation_name")
    inflation_residual = residualize(rows, "token_inflation", dimensions)
    degradation_residual = residualize(rows, "relative_map10_degradation", dimensions)
    return {
        "n": len(rows),
        "mean_token_inflation": mean(inflation),
        "mean_relative_map10_degradation": mean(degradation),
        "pearson": pearson(inflation, degradation),
        "spearman": spearman(inflation, degradation),
        "additive_fixed_effect_residual_pearson": pearson(
            inflation_residual, degradation_residual
        ),
    }


def load_token_cells(path: Path) -> dict[tuple[str, str, str, str], dict]:
    cells = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["model"],
                row["dataset"],
                row["language"],
                row["perturbation_name"],
            )
            if key in cells:
                raise ValueError(f"Duplicate tokenizer cell: {key}")
            cells[key] = row
    return cells


def load_retrieval_cells(paths: list[Path]) -> dict[tuple[str, str, str, str], dict]:
    cells = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["perturbation_name"] == "ALL" or row["language"] == "ALL":
                    continue
                key = (
                    row["model"],
                    row["dataset"],
                    row["language"],
                    row["perturbation_name"],
                )
                if key in cells:
                    raise ValueError(f"Duplicate retrieval cell: {key}")
                cells[key] = row
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-cells", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, nargs="+", required=True)
    parser.add_argument("--joined-output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    token_cells = load_token_cells(args.token_cells)
    retrieval_cells = load_retrieval_cells(args.retrieval_results)
    joined = []
    for key in sorted(set(token_cells).intersection(retrieval_cells)):
        token = token_cells[key]
        retrieval = retrieval_cells[key]
        original_map = float(retrieval["MAP@10_orig"])
        delta = float(retrieval["delta_MAP@10"])
        perturbation = key[3]
        joined.append(
            {
                "model": key[0],
                "dataset": key[1],
                "language": key[2],
                "script_group": "non_latin" if key[2] in NON_LATIN else "latin",
                "perturbation_name": perturbation,
                "family": perturbation[0],
                "n_token_sample": token["n"],
                "mean_tokens_original": token["mean_tokens_original"],
                "mean_tokens_perturbed": token["mean_tokens_perturbed"],
                "token_inflation": float(token["mean_paired_inflation_ratio"]) - 1.0,
                "map10_original": original_map,
                "map10_perturbed": float(retrieval["MAP@10_pert"]),
                "absolute_map10_degradation": delta,
                "relative_map10_degradation": delta / original_map if original_map else 0.0,
            }
        )

    args.joined_output.parent.mkdir(parents=True, exist_ok=True)
    with args.joined_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)

    subsets = {
        "all_cells": joined,
        "character_family_C": [row for row in joined if row["family"] == "C"],
        "latin_script": [row for row in joined if row["script_group"] == "latin"],
        "non_latin_script": [row for row in joined if row["script_group"] == "non_latin"],
        "character_family_C_latin": [
            row for row in joined if row["family"] == "C" and row["script_group"] == "latin"
        ],
        "character_family_C_non_latin": [
            row for row in joined if row["family"] == "C" and row["script_group"] == "non_latin"
        ],
    }
    report = {
        "token_cells": len(token_cells),
        "retrieval_cells": len(retrieval_cells),
        "joined_cells": len(joined),
        "unmatched_token_cells": len(set(token_cells).difference(retrieval_cells)),
        "unmatched_retrieval_cells": len(set(retrieval_cells).difference(token_cells)),
        "subsets": {name: statistics(rows) for name, rows in subsets.items()},
        "by_model": {
            model: statistics([row for row in joined if row["model"] == model])
            for model in sorted({row["model"] for row in joined})
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Joined cells: {args.joined_output}")
    print(f"Report      : {args.report_json}")


if __name__ == "__main__":
    main()
