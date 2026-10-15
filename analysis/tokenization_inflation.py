#!/usr/bin/env python3
"""Measure perturbation-induced token-count inflation for retrieval models.

The script first builds a reproducible reservoir sample stratified by dataset,
language, and perturbation. The sample can be cached as JSONL, allowing all 13
tokenizers to be evaluated without repeatedly scanning the 1.2 GB source CSVs.

Output is one row per (model, dataset, language, perturbation) cell with mean
original/perturbed lengths, mean paired inflation, and the fraction of queries
whose token count increased.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


TASK_DESCRIPTION = "Given a claim, retrieve fact-checks that verify or refute it"


@dataclass(frozen=True)
class TokenizerSpec:
    hf_id: str
    query_prefix: str = ""
    trust_remote_code: bool = False
    max_length: int | None = None


MODEL_SPECS: dict[str, TokenizerSpec] = {
    "labse": TokenizerSpec("sentence-transformers/LaBSE"),
    "me5": TokenizerSpec("intfloat/multilingual-e5-large", "query: "),
    "mpnet-multi": TokenizerSpec("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    "minilm-multi": TokenizerSpec("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    "bge-m3": TokenizerSpec("BAAI/bge-m3"),
    "all-mpnet-base-v2": TokenizerSpec("sentence-transformers/all-mpnet-base-v2"),
    "all-minilm-l12": TokenizerSpec("sentence-transformers/all-MiniLM-L12-v2"),
    "all-distilroberta": TokenizerSpec("sentence-transformers/all-distilroberta-v1"),
    "sentence-t5-large": TokenizerSpec("sentence-transformers/sentence-t5-large"),
    "instructor-large": TokenizerSpec(
        "hkunlp/instructor-large",
        "Represent the claim for retrieving relevant fact-checks: ",
    ),
    "sfr-embedding-mistral": TokenizerSpec(
        "Salesforce/SFR-Embedding-Mistral",
        f"Instruct: {TASK_DESCRIPTION}\nQuery: ",
        max_length=1024,
    ),
    "qwen3-embedding-8b": TokenizerSpec(
        "Qwen/Qwen3-Embedding-8B",
        f"Instruct: {TASK_DESCRIPTION}\nQuery: ",
        max_length=1024,
    ),
    "llama-embed-nemotron-8b": TokenizerSpec(
        "nvidia/llama-embed-nemotron-8b",
        f"Instruct: {TASK_DESCRIPTION}\nQuery: ",
        trust_remote_code=True,
        max_length=1024,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", help="Verified/generated CSV files")
    parser.add_argument("--sample-cache", type=Path, required=True, help="Reusable sampled JSONL")
    parser.add_argument("--sample-per-cell", type=int, default=250)
    parser.add_argument(
        "--only-effective-verified",
        action="store_true",
        help="Sample only Qwen-retained rows, with A4 acceptance recomputed from applied.",
    )
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--models", default=",".join(MODEL_SPECS))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def dataset_name(path: Path) -> str:
    return path.stem.removesuffix("_verified")


def text_column(fieldnames: list[str]) -> str:
    if "Claim" in fieldnames:
        return "Claim"
    if "post_body" in fieldnames:
        return "post_body"
    raise ValueError("Expected Claim or post_body column")


def collect_sample(
    paths: list[Path], sample_per_cell: int, seed: int, only_effective_verified: bool
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    reservoirs: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    seen: dict[tuple[str, str, str], int] = defaultdict(int)
    total = 0
    for path in paths:
        name = dataset_name(path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            original_col = text_column(fields)
            required = {original_col, "perturbed_text", "perturbation_name", "pipeline_lang"}
            missing = required.difference(fields)
            if missing:
                raise ValueError(f"{path} missing {sorted(missing)}")
            for row in reader:
                if only_effective_verified:
                    decision_field = (
                        "perturbation_applied"
                        if row["perturbation_name"] == "A4_stt_artifacts"
                        else "verified"
                    )
                    if row.get(decision_field, "").strip().lower() != "true":
                        continue
                total += 1
                key = (name, row["pipeline_lang"] or "UNKNOWN", row["perturbation_name"])
                seen[key] += 1
                record = {
                    "dataset": name,
                    "language": key[1],
                    "perturbation_name": key[2],
                    "original_text": row[original_col],
                    "perturbed_text": row["perturbed_text"],
                }
                bucket = reservoirs[key]
                if len(bucket) < sample_per_cell:
                    bucket.append(record)
                else:
                    replacement = rng.randrange(seen[key])
                    if replacement < sample_per_cell:
                        bucket[replacement] = record
        print(f"  sampled {path.name}", flush=True)
    sample = [record for key in sorted(reservoirs) for record in reservoirs[key]]
    print(
        f"Scanned {total:,} rows; retained {len(sample):,} rows "
        f"across {len(reservoirs):,} cells."
    )
    return sample


def write_sample(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_sample(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def batched(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def tokenize_lengths(tokenizer, texts: list[str], batch_size: int) -> list[int]:
    lengths: list[int] = []
    for _, batch in batched(texts, batch_size):
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_length=True,
        )
        if "length" in encoded:
            lengths.extend(int(value) for value in encoded["length"])
        else:
            lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def summarize_model(
    model_key: str,
    spec: TokenizerSpec,
    sample: list[dict[str, str]],
    batch_size: int,
    local_files_only: bool,
) -> list[dict[str, str | int | float]]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required for tokenization. Install analysis/requirements.txt"
        ) from exc

    print(f"Loading {model_key}: {spec.hf_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id,
        trust_remote_code=spec.trust_remote_code,
        local_files_only=local_files_only,
    )
    original_texts = [spec.query_prefix + row["original_text"] for row in sample]
    perturbed_texts = [spec.query_prefix + row["perturbed_text"] for row in sample]
    original_lengths = tokenize_lengths(tokenizer, original_texts, batch_size)
    perturbed_lengths = tokenize_lengths(tokenizer, perturbed_texts, batch_size)

    accumulators: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row, original, perturbed in zip(sample, original_lengths, perturbed_lengths):
        key = (row["dataset"], row["language"], row["perturbation_name"])
        acc = accumulators[key]
        acc["n"] += 1
        acc["original_sum"] += original
        acc["perturbed_sum"] += perturbed
        acc["delta_sum"] += perturbed - original
        acc["ratio_sum"] += perturbed / max(original, 1)
        acc["increased"] += perturbed > original
        if spec.max_length is not None:
            acc["original_over_limit"] += original > spec.max_length
            acc["perturbed_over_limit"] += perturbed > spec.max_length

    output = []
    for (dataset, language, perturbation), acc in sorted(accumulators.items()):
        n = int(acc["n"])
        output.append(
            {
                "model": model_key,
                "hf_id": spec.hf_id,
                "dataset": dataset,
                "language": language,
                "perturbation_name": perturbation,
                "n": n,
                "mean_tokens_original": acc["original_sum"] / n,
                "mean_tokens_perturbed": acc["perturbed_sum"] / n,
                "mean_token_delta": acc["delta_sum"] / n,
                "mean_paired_inflation_ratio": acc["ratio_sum"] / n,
                "fraction_token_count_increased": acc["increased"] / n,
                "configured_max_length": spec.max_length or "",
                "fraction_original_over_limit": (
                    acc["original_over_limit"] / n if spec.max_length else ""
                ),
                "fraction_perturbed_over_limit": (
                    acc["perturbed_over_limit"] / n if spec.max_length else ""
                ),
            }
        )
    del tokenizer
    return output


def main() -> None:
    args = parse_args()
    if args.sample_cache.exists():
        sample = read_sample(args.sample_cache)
        print(f"Loaded {len(sample):,} sampled rows from {args.sample_cache}")
    else:
        if not args.inputs:
            raise SystemExit("--inputs are required when --sample-cache does not exist")
        sample = collect_sample(
            args.inputs,
            args.sample_per_cell,
            args.seed,
            args.only_effective_verified,
        )
        write_sample(args.sample_cache, sample)
        print(f"Wrote sample cache: {args.sample_cache}")

    if args.collect_only:
        cells = {(r["dataset"], r["language"], r["perturbation_name"]) for r in sample}
        print(f"Collection complete: {len(sample):,} rows in {len(cells):,} cells")
        return
    if args.output is None:
        raise SystemExit("--output is required unless --collect-only is used")

    requested = [name for name in args.models.split(",") if name]
    unknown = sorted(set(requested).difference(MODEL_SPECS))
    if unknown:
        raise SystemExit(f"Unknown model keys: {unknown}")
    rows = []
    for model in requested:
        rows.extend(
            summarize_model(
                model,
                MODEL_SPECS[model],
                sample,
                args.batch_size,
                args.local_files_only,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} tokenizer/cell rows to {args.output}")


if __name__ == "__main__":
    main()
