#!/usr/bin/env python3
"""Report whether the files needed for retrieval sensitivity reruns exist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def first_existing(candidates: list[Path]) -> Path | None:
    return next((path for path in candidates if path.exists()), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "--external-root",
        type=Path,
        help="Optional directory containing datasets stored outside the repository.",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    external = args.external_root.resolve() if args.external_root else None

    def external_candidates(*parts: str) -> list[Path]:
        return [external.joinpath(*parts)] if external else []

    requirements: dict[str, list[Path]] = {
        "task_a_corpus": [
            project / "data/raw/MultiClaim.csv",
            *external_candidates("MultiClaim.csv"),
        ],
        "task_a_baseline": [
            project / "data/processed/multiclaimnet_preprocessed.csv",
            project / "data/processed/multiclaimnet_preprocessed",
        ],
        "task_a_generated": [
            project / "data/processed/multiclaimnet_verified.csv",
            *external_candidates("multiclaimnet_verified.csv"),
        ],
        "task_b_fact_checks": [
            project / "data/raw/fact_checks.csv",
            *external_candidates("fact_checks.csv"),
        ],
        "task_b_mapping": [
            project / "data/raw/fact_check_post_mapping.csv",
            *external_candidates("fact_check_post_mapping.csv"),
        ],
        "task_b_baseline": [project / "data/processed/multiclaimv2_preprocessed.csv"],
        "task_b_generated": [
            project / "data/processed/multiclaimv2_verified.csv",
            *external_candidates("multiclaimv2_verified.csv"),
        ],
        "embedding_cache": [
            project / ".embedding_cache",
            *external_candidates(".embedding_cache"),
            Path.cwd() / ".embedding_cache",
        ],
        "task_a_aggregate_results": [
            project / "results/multiclaimnet/multiclaimnet_results.csv",
            *external_candidates("results", "multiclaimnet", "multiclaimnet_results.csv"),
        ],
        "task_b_aggregate_results": [
            project / "results/multiclaimv2/multiclaimv2_results.csv",
            *external_candidates("results", "multiclaimv2", "multiclaimv2_results.csv"),
        ],
    }
    report = {}
    for label, candidates in requirements.items():
        found = first_existing(candidates)
        report[label] = {
            "available": found is not None,
            "path": str(found) if found else None,
            "searched": [str(path) for path in candidates],
        }

    core = (
        "task_a_corpus",
        "task_a_baseline",
        "task_b_fact_checks",
        "task_b_mapping",
        "task_b_baseline",
    )
    report["retrieval_rerun_ready"] = all(report[key]["available"] for key in core)
    report["cached_per_query_reconstruction_ready"] = report["embedding_cache"]["available"]

    for label, value in report.items():
        if not isinstance(value, dict):
            continue
        marker = "OK" if value["available"] else "MISSING"
        print(f"{marker:7} {label:32} {value['path'] or ''}")
    print(f"\nRetrieval rerun ready: {report['retrieval_rerun_ready']}")
    print(
        "Cached per-query reconstruction ready: "
        f"{report['cached_per_query_reconstruction_ready']}"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
