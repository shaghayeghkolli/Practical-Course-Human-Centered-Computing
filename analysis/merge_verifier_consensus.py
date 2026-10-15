#!/usr/bin/env python3
"""Merge Qwen, Gemma, Llama, and Mistral verifier outputs.

The full verifier writes rows in completion order, so line-wise CSV merging is
unsafe. This script builds a temporary disk-backed SQLite index keyed by the
dataset identifier and perturbation name, then exports rows in the original
Qwen/source order.

Qwen's uploaded files predate the neutral A4 speech-to-text fix. For every
model, the effective A4 decision is therefore recomputed from
``perturbation_applied``; other perturbations use the model's ``verified``
column. A non-empty verifier error makes that model's vote unavailable.
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from deterministic_validate import validate_row


MODELS = ("qwen", "gemma", "llama", "mistral")
INDEPENDENT_MODELS = ("gemma", "llama", "mistral")
VERDICT_COLUMNS = (
    "perturbation_applied",
    "meaning_preserved",
    "verified",
    "verify_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True, help="Original Qwen-verified CSV")
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--mistral", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Merged all-row CSV")
    parser.add_argument(
        "--strict-output",
        type=Path,
        help="Optional smaller CSV containing only unanimous 4/4 rows.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional JSON summary. Default: <output>.summary.json",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Export incomplete model joins instead of stopping.",
    )
    return parser.parse_args()


def detect_id(fieldnames: list[str]) -> str:
    if "NID" in fieldnames:
        return "NID"
    if "post_id" in fieldnames:
        return "post_id"
    raise ValueError("Expected an NID or post_id column")


def parse_bool(value: str | None) -> bool | None:
    normalized = (value or "").strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def effective_vote(
    perturbation: str,
    applied: str | None,
    verified: str | None,
    error: str | None,
) -> bool | None:
    if (error or "").strip():
        return None
    value = applied if perturbation == "A4_stt_artifacts" else verified
    return parse_bool(value)


def bool_text(value: bool | None) -> str:
    return "" if value is None else str(value)


def validate_columns(fieldnames: list[str], id_col: str, path: Path) -> None:
    required = {id_col, "perturbation_name", *VERDICT_COLUMNS}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def create_database(connection: sqlite3.Connection) -> None:
    fields = []
    for model in MODELS:
        fields.extend(
            [
                f"{model}_applied TEXT",
                f"{model}_meaning TEXT",
                f"{model}_verified_raw TEXT",
                f"{model}_error TEXT",
                f"{model}_present INTEGER NOT NULL DEFAULT 0",
            ]
        )
    connection.execute(
        "CREATE TABLE verdicts ("
        "item_id TEXT NOT NULL, perturbation TEXT NOT NULL, "
        + ", ".join(fields)
        + ", PRIMARY KEY (item_id, perturbation)) WITHOUT ROWID"
    )


def ingest_qwen(connection: sqlite3.Connection, path: Path) -> tuple[str, list[str], int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        id_col = detect_id(fieldnames)
        validate_columns(fieldnames, id_col, path)
        sql = (
            "INSERT INTO verdicts (item_id, perturbation, qwen_applied, "
            "qwen_meaning, qwen_verified_raw, qwen_error, qwen_present) "
            "VALUES (?, ?, ?, ?, ?, ?, 1)"
        )
        count = 0
        for row in reader:
            try:
                connection.execute(
                    sql,
                    (
                        row[id_col],
                        row["perturbation_name"],
                        row["perturbation_applied"],
                        row["meaning_preserved"],
                        row["verified"],
                        row["verify_error"],
                    ),
                )
            except sqlite3.IntegrityError as exc:
                key = (row[id_col], row["perturbation_name"])
                raise ValueError(f"Duplicate Qwen/source key {key}") from exc
            count += 1
            if count % 10_000 == 0:
                connection.commit()
        connection.commit()
    return id_col, fieldnames, count


def ingest_model(
    connection: sqlite3.Connection,
    path: Path,
    model: str,
    expected_id_col: str,
) -> int:
    sql = (
        f"UPDATE verdicts SET {model}_applied=?, {model}_meaning=?, "
        f"{model}_verified_raw=?, {model}_error=?, {model}_present=1 "
        f"WHERE item_id=? AND perturbation=? AND {model}_present=0"
    )
    count = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        id_col = detect_id(fieldnames)
        if id_col != expected_id_col:
            raise ValueError(f"{path} uses {id_col}, expected {expected_id_col}")
        validate_columns(fieldnames, id_col, path)
        for row in reader:
            cursor = connection.execute(
                sql,
                (
                    row["perturbation_applied"],
                    row["meaning_preserved"],
                    row["verified"],
                    row["verify_error"],
                    row[id_col],
                    row["perturbation_name"],
                ),
            )
            if cursor.rowcount != 1:
                key = (row[id_col], row["perturbation_name"])
                exists = connection.execute(
                    "SELECT 1 FROM verdicts WHERE item_id=? AND perturbation=?",
                    key,
                ).fetchone()
                problem = "duplicate model row" if exists else "key absent from Qwen/source"
                raise ValueError(f"{path}: {problem}: {key}")
            count += 1
            if count % 10_000 == 0:
                connection.commit()
        connection.commit()
    return count


def model_fields() -> list[str]:
    output = []
    for model in MODELS:
        output.extend(
            [
                f"{model}_perturbation_applied",
                f"{model}_meaning_preserved",
                f"{model}_verified_raw",
                f"{model}_verified",
                f"{model}_verify_error",
            ]
        )
    return output


def consensus_fields() -> list[str]:
    return [
        "independent_vote_count",
        "all_model_vote_count",
        "independent_consensus",
        "majority_3_of_4",
        "unanimous_4",
        "verification_tier",
        "deterministic_check",
        "deterministic_reason",
        "deterministic_metrics",
        "hybrid_independent_consensus",
        "hybrid_unanimous_4",
    ]


def classify(votes: dict[str, bool | None]) -> tuple[dict[str, str], str]:
    if any(value is None for value in votes.values()):
        fields = {
            "independent_vote_count": "",
            "all_model_vote_count": "",
            "independent_consensus": "",
            "majority_3_of_4": "",
            "unanimous_4": "",
            "verification_tier": "incomplete",
        }
        return fields, "incomplete"

    independent_count = sum(bool(votes[name]) for name in INDEPENDENT_MODELS)
    all_count = sum(bool(votes[name]) for name in MODELS)
    independent_consensus = independent_count == 3
    majority = all_count >= 3
    unanimous = all_count == 4
    if unanimous:
        tier = "unanimous_4"
    elif independent_consensus:
        tier = "independent_3_qwen_reject"
    elif majority:
        tier = "majority_3"
    elif all_count == 2:
        tier = "split_2"
    else:
        tier = "low_0_1"
    fields = {
        "independent_vote_count": str(independent_count),
        "all_model_vote_count": str(all_count),
        "independent_consensus": str(independent_consensus),
        "majority_3_of_4": str(majority),
        "unanimous_4": str(unanimous),
        "verification_tier": tier,
    }
    return fields, tier


def export_rows(
    connection: sqlite3.Connection,
    qwen_path: Path,
    output_path: Path,
    strict_path: Path | None,
    id_col: str,
    source_fields: list[str],
) -> dict:
    base_fields = [field for field in source_fields if field not in VERDICT_COLUMNS]
    output_fields = base_fields + model_fields() + consensus_fields()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if strict_path:
        strict_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "rows": 0,
        "tiers": Counter(),
        "deterministic_checks": Counter(),
        "hybrid_independent_consensus": Counter(),
        "hybrid_unanimous_4": Counter(),
        "by_language": defaultdict(Counter),
        "by_perturbation": defaultdict(Counter),
    }
    select = (
        "SELECT "
        + ", ".join(
            f"{model}_{suffix}"
            for model in MODELS
            for suffix in ("applied", "meaning", "verified_raw", "error", "present")
        )
        + " FROM verdicts WHERE item_id=? AND perturbation=?"
    )

    strict_handle = None
    try:
        with qwen_path.open(newline="", encoding="utf-8-sig") as source, output_path.open(
            "w", newline="", encoding="utf-8"
        ) as destination:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(destination, fieldnames=output_fields)
            writer.writeheader()
            strict_writer = None
            if strict_path:
                strict_handle = strict_path.open("w", newline="", encoding="utf-8")
                strict_writer = csv.DictWriter(strict_handle, fieldnames=output_fields)
                strict_writer.writeheader()

            for row in reader:
                key = (row[id_col], row["perturbation_name"])
                values = connection.execute(select, key).fetchone()
                if values is None:
                    raise RuntimeError(f"Internal join failure for {key}")
                output_row = {field: row.get(field, "") for field in base_fields}
                votes: dict[str, bool | None] = {}
                offset = 0
                for model in MODELS:
                    applied, meaning, verified_raw, error, present = values[offset : offset + 5]
                    offset += 5
                    if not present:
                        applied = meaning = verified_raw = ""
                        error = "missing model row"
                    vote = effective_vote(
                        row["perturbation_name"], applied, verified_raw, error
                    )
                    votes[model] = vote
                    output_row.update(
                        {
                            f"{model}_perturbation_applied": applied or "",
                            f"{model}_meaning_preserved": meaning or "",
                            f"{model}_verified_raw": verified_raw or "",
                            f"{model}_verified": bool_text(vote),
                            f"{model}_verify_error": error or "",
                        }
                    )

                consensus, tier = classify(votes)
                output_row.update(consensus)
                original = row.get("original_text") or row.get("Claim") or row.get("post_body") or ""
                deterministic = validate_row(row, original)
                independent_value = parse_bool(consensus["independent_consensus"])
                unanimous_value = parse_bool(consensus["unanimous_4"])
                hybrid_independent = (
                    None
                    if independent_value is None
                    else independent_value
                    and deterministic.status not in {"fail", "generation_error"}
                )
                hybrid_unanimous = (
                    None
                    if unanimous_value is None
                    else unanimous_value
                    and deterministic.status not in {"fail", "generation_error"}
                )
                output_row.update(
                    {
                        "deterministic_check": deterministic.status,
                        "deterministic_reason": deterministic.reason,
                        "deterministic_metrics": json.dumps(
                            deterministic.metrics,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "hybrid_independent_consensus": bool_text(hybrid_independent),
                        "hybrid_unanimous_4": bool_text(hybrid_unanimous),
                    }
                )
                writer.writerow(output_row)
                if strict_writer and hybrid_unanimous:
                    strict_writer.writerow(output_row)

                summary["rows"] += 1
                summary["tiers"][tier] += 1
                summary["deterministic_checks"][deterministic.status] += 1
                summary["hybrid_independent_consensus"][bool_text(hybrid_independent)] += 1
                summary["hybrid_unanimous_4"][bool_text(hybrid_unanimous)] += 1
                language = row.get("pipeline_lang") or row.get("language") or "UNKNOWN"
                summary["by_language"][language][tier] += 1
                summary["by_perturbation"][row["perturbation_name"]][tier] += 1
                if summary["rows"] % 50_000 == 0:
                    print(f"  exported {summary['rows']:,} rows", flush=True)
    finally:
        if strict_handle:
            strict_handle.close()

    return {
        "rows": summary["rows"],
        "tiers": dict(summary["tiers"]),
        "deterministic_checks": dict(summary["deterministic_checks"]),
        "hybrid_independent_consensus": dict(summary["hybrid_independent_consensus"]),
        "hybrid_unanimous_4": dict(summary["hybrid_unanimous_4"]),
        "by_language": {key: dict(value) for key, value in summary["by_language"].items()},
        "by_perturbation": {
            key: dict(value) for key, value in summary["by_perturbation"].items()
        },
    }


def main() -> None:
    args = parse_args()
    for path in (args.qwen, args.gemma, args.llama, args.mistral):
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")

    with tempfile.NamedTemporaryFile(prefix="verifier_join_", suffix=".sqlite") as temp:
        connection = sqlite3.connect(temp.name)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        create_database(connection)

        id_col, source_fields, source_count = ingest_qwen(connection, args.qwen)
        print(f"Qwen/source: {source_count:,} rows")
        paths = {"gemma": args.gemma, "llama": args.llama, "mistral": args.mistral}
        for model, path in paths.items():
            count = ingest_model(connection, path, model, id_col)
            print(f"{model:7}: {count:,} rows")

        missing = {}
        for model in INDEPENDENT_MODELS:
            count = connection.execute(
                f"SELECT COUNT(*) FROM verdicts WHERE {model}_present=0"
            ).fetchone()[0]
            if count:
                missing[model] = count
        if missing and not args.allow_incomplete:
            raise SystemExit(
                f"Incomplete model outputs: {missing}. Finish/retry them or pass --allow-incomplete."
            )

        summary = export_rows(
            connection,
            args.qwen,
            args.output,
            args.strict_output,
            id_col,
            source_fields,
        )
        connection.close()

    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Merged output : {args.output}")
    if args.strict_output:
        print(f"Strict output : {args.strict_output}")
    print(f"Summary       : {summary_path}")
    print("Tiers:", summary["tiers"])


if __name__ == "__main__":
    main()
