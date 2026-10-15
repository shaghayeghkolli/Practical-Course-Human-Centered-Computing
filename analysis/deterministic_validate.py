#!/usr/bin/env python3
"""Apply conservative, perturbation-specific structural validators.

These checks complement rather than replace semantic LLM verification. ``pass``
and ``fail`` are emitted only when the implementation contract can be checked
from the original and perturbed strings. ``partial_pass`` means a necessary
condition was observed but semantic or linguistic judgment is still required.
``not_applicable`` means the perturbation is entirely semantic for this tool.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from perturbations.perturbations import (  # noqa: E402
    _HOMOGLYPH_MAP,
    _LEET_MAP,
    _OCR_MAP,
)


@dataclass(frozen=True)
class CheckResult:
    status: str
    reason: str
    metrics: dict[str, int | float | str | bool]


EMOJI_RANGES = (
    (0x1F1E6, 0x1F1FF),  # flags
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F700, 0x1F77F),
    (0x1F780, 0x1F7FF),
    (0x1F800, 0x1F8FF),
    (0x1F900, 0x1F9FF),
    (0x1FA00, 0x1FAFF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
)
VARIATION_SELECTORS = {0xFE0E, 0xFE0F}
SKIN_TONES = range(0x1F3FB, 0x1F400)
ZWJ = 0x200D
KEYCAP = 0x20E3


def _is_emoji_base(char: str) -> bool:
    value = ord(char)
    return any(start <= value <= end for start, end in EMOJI_RANGES)


def emoji_spans(text: str) -> list[tuple[int, int]]:
    """Return approximate Unicode emoji-cluster spans without external regex."""
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        start = i
        code = ord(text[i])
        if text[i] in "#*0123456789":
            j = i + 1
            if j < len(text) and ord(text[j]) in VARIATION_SELECTORS:
                j += 1
            if j < len(text) and ord(text[j]) == KEYCAP:
                spans.append((start, j + 1))
                i = j + 1
                continue
        if not _is_emoji_base(text[i]):
            i += 1
            continue
        i += 1
        # Two regional indicators form one flag.
        if 0x1F1E6 <= code <= 0x1F1FF and i < len(text):
            if 0x1F1E6 <= ord(text[i]) <= 0x1F1FF:
                i += 1
        while i < len(text) and (
            ord(text[i]) in VARIATION_SELECTORS or ord(text[i]) in SKIN_TONES
        ):
            i += 1
        while i + 1 < len(text) and ord(text[i]) == ZWJ and _is_emoji_base(text[i + 1]):
            i += 2
            while i < len(text) and (
                ord(text[i]) in VARIATION_SELECTORS or ord(text[i]) in SKIN_TONES
            ):
                i += 1
        spans.append((start, i))
    return spans


def trailing_emoji_info(text: str) -> tuple[int, str]:
    spans = emoji_spans(text)
    cursor = len(text.rstrip())
    count = 0
    suffix_start = cursor
    for start, end in reversed(spans):
        if end > cursor:
            continue
        if text[end:cursor].strip():
            break
        count += 1
        suffix_start = start
        cursor = start
    return count, text[:suffix_start].rstrip()


def remove_emojis(text: str) -> str:
    spans = emoji_spans(text)
    if not spans:
        return text
    output = []
    cursor = 0
    for start, end in spans:
        output.append(text[cursor:start])
        cursor = end
    output.append(text[cursor:])
    return "".join(output)


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def mapped_substitution_check(
    original: str,
    perturbed: str,
    mapping: dict[str, str],
    *,
    lower_keys: bool = False,
    max_changes: int = 3,
) -> CheckResult:
    if len(original) != len(perturbed):
        return CheckResult("fail", "length changed", {"length_delta": len(perturbed) - len(original)})
    differences = []
    invalid = []
    for index, (source, target) in enumerate(zip(original, perturbed)):
        if source == target:
            continue
        key = source.lower() if lower_keys else source
        differences.append(index)
        if mapping.get(key) != target:
            invalid.append(index)
    metrics = {"changes": len(differences), "invalid_changes": len(invalid)}
    if invalid:
        return CheckResult("fail", "contains changes outside the allowed substitution map", metrics)
    if not differences:
        return CheckResult("fail", "no mapped substitutions found", metrics)
    if len(differences) > max_changes:
        return CheckResult("fail", f"more than {max_changes} substitutions", metrics)
    return CheckResult("pass", "all differences are allowed mapped substitutions", metrics)


def validate_emoji_relevant(original: str, perturbed: str) -> CheckResult:
    count, base = trailing_emoji_info(perturbed)
    metrics = {"trailing_emoji_clusters": count, "original_unchanged": base == original.rstrip()}
    if count != 2:
        return CheckResult("fail", "expected exactly two trailing emojis", metrics)
    if base != original.rstrip():
        return CheckResult("fail", "text before trailing emojis differs from original", metrics)
    return CheckResult("pass", "exactly two emojis appended with original text unchanged", metrics)


def validate_emoji_disruptive(original: str, perturbed: str) -> CheckResult:
    spans = emoji_spans(perturbed)
    restored = normalize_space(remove_emojis(perturbed))
    internal = any(end < len(perturbed.rstrip()) for _, end in spans)
    metrics = {
        "emoji_clusters": len(spans),
        "has_internal_emoji": internal,
        "restored_original": restored == normalize_space(original),
    }
    if not spans or not internal:
        return CheckResult("fail", "no internal emoji found", metrics)
    if restored != normalize_space(original):
        return CheckResult("fail", "removing emojis does not restore original text", metrics)
    return CheckResult("pass", "internal emoji insertion is the only normalized change", metrics)


def validate_hashtags(original: str, perturbed: str) -> CheckResult:
    starts_original = perturbed.startswith(original)
    suffix = perturbed[len(original) :].strip() if starts_original else ""
    tokens = suffix.split()
    hashtags = [token for token in tokens if re.fullmatch(r"#[^\W_][\w]*", token, re.UNICODE)]
    metrics = {
        "starts_with_exact_original": starts_original,
        "suffix_tokens": len(tokens),
        "hashtags": len(hashtags),
    }
    if not starts_original:
        return CheckResult("fail", "perturbed text does not start with exact original", metrics)
    if len(tokens) != 6 or len(hashtags) != 6:
        return CheckResult("fail", "expected exactly six appended hashtag tokens", metrics)
    return CheckResult("pass", "exact original plus six hashtag tokens", metrics)


def validate_stt(original: str, perturbed: str) -> CheckResult:
    original_tokens = original.split()
    perturbed_tokens = perturbed.split()
    changes = sum(a != b for a, b in zip(original_tokens, perturbed_tokens))
    changes += abs(len(original_tokens) - len(perturbed_tokens))
    metrics = {
        "original_tokens": len(original_tokens),
        "perturbed_tokens": len(perturbed_tokens),
        "approx_changed_tokens": changes,
    }
    if not 1 <= changes <= 3:
        return CheckResult("fail", "expected approximately one to three changed tokens", metrics)
    return CheckResult(
        "partial_pass",
        "change count is plausible; phonetic validity still requires language judgment",
        metrics,
    )


YEAR_PATTERN = re.compile(r"(?<!\d)(1\d{3}|20\d{2}|21\d{2})(?!\d)")


def validate_temporal(original: str, perturbed: str) -> CheckResult:
    original_years = [int(value) for value in YEAR_PATTERN.findall(original)]
    perturbed_years = [int(value) for value in YEAR_PATTERN.findall(perturbed)]
    metrics = {
        "original_years": len(original_years),
        "perturbed_years": len(perturbed_years),
    }
    if not original_years:
        starts_original = perturbed.startswith(original)
        metrics["starts_with_exact_original"] = starts_original
        metrics["appended_characters"] = max(len(perturbed) - len(original), 0)
        if not starts_original or len(perturbed) <= len(original):
            return CheckResult("fail", "no-year case must append an anchor to exact original", metrics)
        return CheckResult("partial_pass", "anchor appended in no-year case", metrics)
    shifted = len(original_years) == len(perturbed_years) and all(
        target == source + 3 for source, target in zip(original_years, perturbed_years)
    )
    metrics["all_years_shifted_by_3"] = shifted
    if not shifted:
        return CheckResult("fail", "not every detected year shifted forward by exactly three", metrics)
    return CheckResult(
        "partial_pass",
        "year shifts are correct; anchor and other factual preservation still need judgment",
        metrics,
    )


def validate_word_splitting(original: str, perturbed: str) -> CheckResult:
    original_compact = "".join(char for char in original if not char.isspace())
    perturbed_compact = "".join(char for char in perturbed if not char.isspace())
    original_spaces = sum(char.isspace() for char in original)
    perturbed_spaces = sum(char.isspace() for char in perturbed)
    added = perturbed_spaces - original_spaces
    metrics = {"added_whitespace_characters": added, "nonspace_text_unchanged": original_compact == perturbed_compact}
    if original_compact != perturbed_compact:
        return CheckResult("fail", "non-whitespace characters changed", metrics)
    if added != 2:
        return CheckResult("fail", "expected exactly two inserted whitespace characters", metrics)
    return CheckResult("pass", "exactly two spaces inserted and no other characters changed", metrics)


WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


def validate_clickbait(original: str, perturbed: str) -> CheckResult:
    if not perturbed.casefold().endswith(original.casefold()):
        return CheckResult(
            "fail",
            "case-insensitive original body is not an exact suffix",
            {"body_is_suffix": False},
        )
    body = perturbed[len(perturbed) - len(original) :]
    prefix = perturbed[: len(perturbed) - len(original)].strip()
    original_words = WORD_PATTERN.findall(original)
    body_words = WORD_PATTERN.findall(body)
    aligned = len(original_words) == len(body_words) and all(
        source.casefold() == target.casefold()
        for source, target in zip(original_words, body_words)
    )
    changed_to_caps = sum(
        source != target and target.isupper()
        for source, target in zip(original_words, body_words)
    ) if aligned else -1
    metrics = {
        "has_hook_prefix": bool(prefix),
        "body_case_only_aligned": aligned,
        "words_changed_to_caps": changed_to_caps,
    }
    if not prefix or not aligned or changed_to_caps != 3:
        return CheckResult("fail", "expected a hook plus exactly three capitalization changes", metrics)
    return CheckResult("pass", "hook prepended and exactly three body words capitalized", metrics)


def validate_presupposition(original: str, perturbed: str) -> CheckResult:
    ends_original = perturbed.endswith(original)
    prefix = perturbed[: len(perturbed) - len(original)].strip() if ends_original else ""
    metrics = {"ends_with_exact_original": ends_original, "prefix_characters": len(prefix)}
    if not ends_original or not prefix:
        return CheckResult("fail", "expected a non-empty frame before exact original text", metrics)
    return CheckResult("pass", "non-empty frame prepended to exact original", metrics)


NEGATION_PATTERN = re.compile(r"\b(?:not|never|no|false|n't)\b", re.IGNORECASE)


def validate_negation(original: str, perturbed: str, high: bool) -> CheckResult:
    original_count = len(NEGATION_PATTERN.findall(original))
    perturbed_count = len(NEGATION_PATTERN.findall(perturbed))
    added = perturbed_count - original_count
    metrics = {"additional_negation_terms": added}
    minimum = 2 if high else 1
    if added < minimum:
        return CheckResult("fail", f"expected at least {minimum} additional negation terms", metrics)
    return CheckResult(
        "partial_pass",
        "negation markers were added; logical reversal still requires semantic judgment",
        metrics,
    )


def validate_pair(name: str, original: str, perturbed: str) -> CheckResult:
    if not perturbed:
        return CheckResult("fail", "perturbed text is empty", {})
    if original == perturbed:
        return CheckResult("fail", "original and perturbed texts are identical", {})

    validators = {
        "A1_emoji_relevant": lambda: validate_emoji_relevant(original, perturbed),
        "A1_emoji_disruptive": lambda: validate_emoji_disruptive(original, perturbed),
        "A2_hashtagification": lambda: validate_hashtags(original, perturbed),
        "A3_ocr_artifacts": lambda: mapped_substitution_check(original, perturbed, _OCR_MAP),
        "A4_stt_artifacts": lambda: validate_stt(original, perturbed),
        "B2_temporal_drift": lambda: validate_temporal(original, perturbed),
        "C1_homoglyphs": lambda: mapped_substitution_check(
            original, perturbed, _HOMOGLYPH_MAP, lower_keys=True
        ),
        "C2_leetspeak": lambda: mapped_substitution_check(
            original, perturbed, _LEET_MAP, lower_keys=True
        ),
        "C3_word_splitting": lambda: validate_word_splitting(original, perturbed),
        "D2_clickbait_llm": lambda: validate_clickbait(original, perturbed),
        "E2_presupposition": lambda: validate_presupposition(original, perturbed),
        "P_negation_low": lambda: validate_negation(original, perturbed, False),
        "P_negation_high": lambda: validate_negation(original, perturbed, True),
    }
    validator = validators.get(name)
    if validator is None:
        return CheckResult(
            "not_applicable",
            "requires semantic, linguistic, or translation judgment",
            {},
        )
    return validator()


def validate_row(row: dict[str, str], original: str) -> CheckResult:
    if row.get("error", "").strip():
        return CheckResult(
            "generation_error",
            "perturbation pipeline reported a generation error",
            {},
        )
    return validate_pair(row["perturbation_name"], original, row.get("perturbed_text", ""))


def detect_text_column(fieldnames: list[str]) -> str:
    if "Claim" in fieldnames:
        return "Claim"
    if "post_body" in fieldnames:
        return "post_body"
    if "original_text" in fieldnames:
        return "original_text"
    raise ValueError("Expected Claim, post_body, or original_text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summary_only and args.output is None:
        raise SystemExit("Provide --output or use --summary-only")

    counts = Counter()
    by_perturbation: dict[str, Counter] = defaultdict(Counter)
    output_handle = None
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        text_col = detect_text_column(fieldnames)
        required = {"perturbation_name", "perturbed_text", text_col}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        writer = None
        if not args.summary_only:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output_handle = args.output.open("w", newline="", encoding="utf-8")
            extra = ["deterministic_check", "deterministic_reason", "deterministic_metrics"]
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames + extra)
            writer.writeheader()

        rows = itertools.islice(reader, args.limit) if args.limit else reader
        for index, row in enumerate(rows, 1):
            result = validate_row(row, row.get("original_text") or row[text_col])
            counts[result.status] += 1
            by_perturbation[row["perturbation_name"]][result.status] += 1
            if writer:
                output = dict(row)
                output.update(
                    {
                        "deterministic_check": result.status,
                        "deterministic_reason": result.reason,
                        "deterministic_metrics": json.dumps(result.metrics, ensure_ascii=False, separators=(",", ":")),
                    }
                )
                writer.writerow(output)
            if index % 100_000 == 0:
                print(f"  checked {index:,} rows", flush=True)

    if output_handle:
        output_handle.close()
    summary = {
        "input": str(args.input),
        "counts": dict(counts),
        "by_perturbation": {key: dict(value) for key, value in sorted(by_perturbation.items())},
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
