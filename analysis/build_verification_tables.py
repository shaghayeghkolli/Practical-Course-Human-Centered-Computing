#!/usr/bin/env python3
"""Build compact CSV summaries from the four-model consensus files."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


DATASETS = ("multiclaimnet", "multiclaimv2")
MODELS = ("qwen", "gemma", "llama", "mistral")
INDEPENDENT_MODELS = ("gemma", "llama", "mistral")


def parse_bool(value: str | None) -> bool | None:
    normalized = (value or "").strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def percentage(numerator: int, denominator: int) -> str:
    return "" if denominator == 0 else f"{100 * numerator / denominator:.2f}"


def is_source_error(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    return normalized.startswith("skipped — perturbation error:") or normalized.startswith(
        "skipped - perturbation error:"
    )


def update_scope(
    scope: str,
    row: dict[str, str],
    overall: dict[str, Counter],
    model_stats: dict[tuple[str, str], Counter],
    qwen_retained: dict[tuple[str, str], Counter],
    perturbation_stats: dict[tuple[str, str], Counter],
    language_stats: dict[tuple[str, str], Counter],
    perturbation_families: dict[str, str],
) -> None:
    tier = row["verification_tier"]
    votes = {model: parse_bool(row[f"{model}_verified"]) for model in MODELS}
    stats = overall[scope]
    stats["rows"] += 1
    stats[f"tier_{tier}"] += 1
    stats[f"deterministic_{row['deterministic_check']}"] += 1
    if tier != "incomplete":
        stats["complete"] += 1
    if tier in {"unanimous_4", "majority_3", "independent_3_qwen_reject"}:
        stats["majority_3plus"] += 1
    if tier in {"unanimous_4", "independent_3_qwen_reject"}:
        stats["independent_3_accept"] += 1
    if parse_bool(row["hybrid_unanimous_4"]):
        stats["strict_hybrid"] += 1
    if parse_bool(row["hybrid_independent_consensus"]):
        stats["hybrid_independent"] += 1

    for model, vote in votes.items():
        item = model_stats[(scope, model)]
        item["rows"] += 1
        error = row[f"{model}_verify_error"]
        if vote is None:
            item["unavailable"] += 1
            item["source_error" if is_source_error(error) else "verifier_error"] += 1
        elif vote:
            item["accepted"] += 1
        else:
            item["rejected"] += 1

        qwen_vote = votes["qwen"]
        if model != "qwen" and qwen_vote is not None and vote is not None:
            item["qwen_pair_available"] += 1
            item["qwen_agree"] += int(qwen_vote == vote)
            item["qwen_accept_model_reject"] += int(qwen_vote and not vote)
            item["qwen_reject_model_accept"] += int(not qwen_vote and vote)
            item["qwen_yes"] += int(qwen_vote)
            item["model_yes_in_pair"] += int(vote)

    if votes["qwen"] is True:
        for model in INDEPENDENT_MODELS:
            item = qwen_retained[(scope, model)]
            item["qwen_accepted"] += 1
            vote = votes[model]
            if vote is None:
                item["unavailable"] += 1
            elif vote:
                item["supported"] += 1
            else:
                item["rejected"] += 1

        independent_votes = [votes[model] for model in INDEPENDENT_MODELS]
        for label, threshold in (("all_three_independent", 3), ("at_least_two_of_three", 2)):
            item = qwen_retained[(scope, label)]
            item["qwen_accepted"] += 1
            if any(vote is None for vote in independent_votes):
                item["unavailable"] += 1
            elif sum(bool(vote) for vote in independent_votes) >= threshold:
                item["supported"] += 1
            else:
                item["rejected"] += 1

    perturbation = row["perturbation_name"] or "UNKNOWN"
    perturbation_families.setdefault(perturbation, row.get("family", "") or "UNKNOWN")
    language = row.get("pipeline_lang") or row.get("Language") or "UNKNOWN"
    for key, collection in (
        ((scope, perturbation), perturbation_stats),
        ((scope, language), language_stats),
    ):
        item = collection[key]
        item["rows"] += 1
        item[f"tier_{tier}"] += 1
        if tier != "incomplete":
            item["complete"] += 1
        if tier in {"unanimous_4", "majority_3", "independent_3_qwen_reject"}:
            item["majority_3plus"] += 1
        if tier in {"unanimous_4", "independent_3_qwen_reject"}:
            item["independent_3_accept"] += 1
        if parse_bool(row["hybrid_unanimous_4"]):
            item["strict_hybrid"] += 1


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("ensemble_verification"),
        help="Directory containing four_model_<dataset>_all.csv files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis_outputs/verification_tables"),
        help="Directory for the five generated summary tables.",
    )
    args = parser.parse_args()
    input_root = args.input_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    overall: dict[str, Counter] = defaultdict(Counter)
    model_stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    qwen_retained: dict[tuple[str, str], Counter] = defaultdict(Counter)
    perturbation_stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    language_stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    perturbation_families: dict[str, str] = {}

    for dataset in DATASETS:
        path = input_root / f"four_model_{dataset}_all.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                update_scope(
                    dataset,
                    row,
                    overall,
                    model_stats,
                    qwen_retained,
                    perturbation_stats,
                    language_stats,
                    perturbation_families,
                )
                update_scope(
                    "combined",
                    row,
                    overall,
                    model_stats,
                    qwen_retained,
                    perturbation_stats,
                    language_stats,
                    perturbation_families,
                )

    scope_order = {name: index for index, name in enumerate((*DATASETS, "combined"))}
    overall_rows = []
    for scope in (*DATASETS, "combined"):
        item = overall[scope]
        complete = item["complete"]
        overall_rows.append(
            {
                "scope": scope,
                "total_rows": item["rows"],
                "complete_four_model_rows": complete,
                "incomplete_rows": item["tier_incomplete"],
                "unanimous_4_llm": item["tier_unanimous_4"],
                "unanimous_4_llm_pct_complete": percentage(item["tier_unanimous_4"], complete),
                "majority_3_or_4_llm": item["majority_3plus"],
                "majority_3_or_4_llm_pct_complete": percentage(item["majority_3plus"], complete),
                "all_3_independent_accept": item["independent_3_accept"],
                "all_3_independent_accept_pct_complete": percentage(item["independent_3_accept"], complete),
                "split_2_2": item["tier_split_2"],
                "low_0_or_1_accept": item["tier_low_0_1"],
                "hybrid_unanimous_release_rows": item["strict_hybrid"],
                "hybrid_unanimous_release_pct_total": percentage(item["strict_hybrid"], item["rows"]),
                "hybrid_independent_release_rows": item["hybrid_independent"],
                "deterministic_fail": item["deterministic_fail"],
                "generation_error": item["deterministic_generation_error"],
            }
        )
    write_csv(output_root / "01_overall_consensus.csv", list(overall_rows[0]), overall_rows)

    model_rows = []
    for (scope, model), item in sorted(
        model_stats.items(), key=lambda pair: (scope_order[pair[0][0]], MODELS.index(pair[0][1]))
    ):
        available = item["accepted"] + item["rejected"]
        pair_n = item["qwen_pair_available"]
        agreement = item["qwen_agree"]
        kappa = ""
        if model != "qwen" and pair_n:
            observed = agreement / pair_n
            q_yes = item["qwen_yes"] / pair_n
            m_yes = item["model_yes_in_pair"] / pair_n
            expected = q_yes * m_yes + (1 - q_yes) * (1 - m_yes)
            if expected != 1:
                kappa = f"{(observed - expected) / (1 - expected):.4f}"
        model_rows.append(
            {
                "scope": scope,
                "model": model,
                "total_rows": item["rows"],
                "available_verdicts": available,
                "accepted": item["accepted"],
                "rejected": item["rejected"],
                "acceptance_pct_available": percentage(item["accepted"], available),
                "source_generation_errors": item["source_error"],
                "actual_verifier_errors": item["verifier_error"],
                "qwen_pair_available": pair_n if model != "qwen" else "",
                "agreement_with_qwen": agreement if model != "qwen" else "",
                "agreement_with_qwen_pct": percentage(agreement, pair_n) if model != "qwen" else "",
                "cohen_kappa_with_qwen": kappa,
                "qwen_accept_model_reject": item["qwen_accept_model_reject"] if model != "qwen" else "",
                "qwen_reject_model_accept": item["qwen_reject_model_accept"] if model != "qwen" else "",
            }
        )
    write_csv(
        output_root / "02_model_verdicts_and_qwen_agreement.csv", list(model_rows[0]), model_rows
    )

    retained_rows = []
    verifier_order = (*INDEPENDENT_MODELS, "all_three_independent", "at_least_two_of_three")
    for (scope, verifier), item in sorted(
        qwen_retained.items(),
        key=lambda pair: (scope_order[pair[0][0]], verifier_order.index(pair[0][1])),
    ):
        available = item["supported"] + item["rejected"]
        retained_rows.append(
            {
                "scope": scope,
                "independent_check": verifier,
                "qwen_accepted_rows": item["qwen_accepted"],
                "available_independent_judgments": available,
                "independently_supported": item["supported"],
                "independently_rejected": item["rejected"],
                "unavailable": item["unavailable"],
                "support_pct_available": percentage(item["supported"], available),
                "support_pct_all_qwen_accepted": percentage(item["supported"], item["qwen_accepted"]),
            }
        )
    write_csv(output_root / "03_qwen_retained_revalidation.csv", list(retained_rows[0]), retained_rows)

    def grouped_rows(collection: dict[tuple[str, str], Counter], dimension: str) -> list[dict[str, object]]:
        rows = []
        for (scope, value), item in sorted(
            collection.items(), key=lambda pair: (scope_order[pair[0][0]], pair[0][1])
        ):
            complete = item["complete"]
            row: dict[str, object] = {"scope": scope, dimension: value}
            if dimension == "perturbation":
                row["family"] = perturbation_families[value]
            row.update(
                {
                    "total_rows": item["rows"],
                    "complete_four_model_rows": complete,
                    "incomplete_rows": item["tier_incomplete"],
                    "unanimous_4_llm": item["tier_unanimous_4"],
                    "unanimous_4_llm_pct_complete": percentage(item["tier_unanimous_4"], complete),
                    "majority_3_or_4_llm": item["majority_3plus"],
                    "majority_3_or_4_llm_pct_complete": percentage(item["majority_3plus"], complete),
                    "all_3_independent_accept": item["independent_3_accept"],
                    "hybrid_unanimous_release_rows": item["strict_hybrid"],
                    "hybrid_unanimous_release_pct_total": percentage(item["strict_hybrid"], item["rows"]),
                    "split_2_2": item["tier_split_2"],
                    "low_0_or_1_accept": item["tier_low_0_1"],
                }
            )
            rows.append(row)
        return rows

    perturbation_rows = grouped_rows(perturbation_stats, "perturbation")
    language_rows = grouped_rows(language_stats, "language")
    write_csv(output_root / "04_consensus_by_perturbation.csv", list(perturbation_rows[0]), perturbation_rows)
    write_csv(output_root / "05_consensus_by_language.csv", list(language_rows[0]), language_rows)

    print(f"Wrote five CSV tables to {output_root}")


if __name__ == "__main__":
    main()
