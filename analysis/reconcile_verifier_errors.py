#!/usr/bin/env python3
"""Extract failed verifier rows and replace them after a retry pass.

Workflow:
  1. ``extract`` copies rows with non-empty ``verify_error`` to a small CSV.
  2. Run perturbations/verify.py on that CSV with the same model.
  3. ``apply`` replaces the failed rows with successful retry decisions while
     preserving the original full-output row order.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def detect_id(fieldnames: list[str]) -> str:
    if "NID" in fieldnames:
        return "NID"
    if "post_id" in fieldnames:
        return "post_id"
    raise ValueError("Expected NID or post_id")


def key(row: dict[str, str], id_col: str) -> tuple[str, str]:
    return row[id_col], row["perturbation_name"]


def is_retryable_error(message: str) -> bool:
    message = message.strip()
    return bool(message) and not message.startswith("skipped — perturbation error:")


def extract(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    source_errors = 0
    with input_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        if "verify_error" not in fields:
            raise ValueError("Input has no verify_error column")
        with output_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                message = row["verify_error"].strip()
                if is_retryable_error(message):
                    writer.writerow(row)
                    count += 1
                elif message:
                    source_errors += 1
    print(f"Extracted {count:,} retryable verifier failures to {output_path}")
    print(f"Left {source_errors:,} source-generation failures unchanged")


def apply(input_path: Path, retry_path: Path, output_path: Path) -> None:
    with retry_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        retry_fields = list(reader.fieldnames or [])
        id_col = detect_id(retry_fields)
        replacements: dict[tuple[str, str], dict[str, str]] = {}
        for row in reader:
            row_key = key(row, id_col)
            if row_key in replacements:
                raise ValueError(f"Duplicate retry key: {row_key}")
            if row.get("verify_error", "").strip():
                raise ValueError(f"Retry still failed for {row_key}: {row['verify_error']}")
            replacements[row_key] = row

    output_path.parent.mkdir(parents=True, exist_ok=True)
    replaced = 0
    original_errors = 0
    with input_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        source_id = detect_id(fields)
        if source_id != id_col:
            raise ValueError(f"Retry uses {id_col}; source uses {source_id}")
        with output_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                row_key = key(row, id_col)
                if is_retryable_error(row.get("verify_error", "")):
                    original_errors += 1
                    replacement = replacements.pop(row_key, None)
                    if replacement is not None:
                        writer.writerow(replacement)
                        replaced += 1
                        continue
                writer.writerow(row)
    if replacements:
        first = list(replacements)[:5]
        raise ValueError(f"{len(replacements)} retry rows were absent from source; first: {first}")
    if replaced != original_errors:
        raise ValueError(f"Replaced {replaced} of {original_errors} failed source rows")
    print(f"Replaced all {replaced:,} failed rows; wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--input", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--input", type=Path, required=True)
    apply_parser.add_argument("--retry", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.input, args.output)
    else:
        apply(args.input, args.retry, args.output)


if __name__ == "__main__":
    main()
