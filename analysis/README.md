# Validation and analysis utilities

These scripts operate on generated CSV files kept outside version control. All
paths are supplied on the command line; no machine-specific directory layout is
required.

## Deterministic validation

`deterministic_validate.py` checks transformations with structural contracts,
including emoji and hashtag insertion, mapped character substitutions, word
splitting, clickbait casing, and presupposition prefixes. Semantic edits return
`partial_pass` or `not_applicable` and still require model or human review.

```bash
python analysis/deterministic_validate.py \
  --input data/processed/multiclaimnet_verified.csv \
  --output analysis_outputs/deterministic_rows.csv \
  --summary-json analysis_outputs/deterministic_summary.json
```

## Verifier error reconciliation

Extract failed rows, retry them with `perturbations/verify.py`, and merge the
new judgments into the original output:

```bash
python analysis/reconcile_verifier_errors.py extract \
  --input MODEL_OUTPUT.csv --output MODEL_ERRORS.csv

python perturbations/verify.py \
  --input MODEL_ERRORS.csv --output MODEL_RETRIES.csv \
  --model MODEL_ID --base-url http://localhost:8100/v1

python analysis/reconcile_verifier_errors.py apply \
  --input MODEL_OUTPUT.csv --retry MODEL_RETRIES.csv --output MODEL_CLEAN.csv
```

## Multi-model consensus

`merge_verifier_consensus.py` combines the original decision with three
independent verifier files. It retains individual judgments and adds majority,
unanimous, and conservative hybrid decisions. A temporary SQLite index avoids
assuming that parallel outputs have the same row order.

```bash
python analysis/merge_verifier_consensus.py \
  --qwen QWEN.csv --gemma GEMMA.csv --llama LLAMA.csv --mistral MISTRAL.csv \
  --output analysis_outputs/four_model_all.csv \
  --strict-output analysis_outputs/four_model_unanimous.csv \
  --summary-json analysis_outputs/four_model_summary.json
```

Place `four_model_multiclaimnet_all.csv` and
`four_model_multiclaimv2_all.csv` in one directory to create compact summaries:

```bash
python analysis/build_verification_tables.py \
  --input-root ensemble_verification \
  --output-root analysis_outputs/verification_tables
```

## Tokenization analysis

`tokenization_inflation.py` builds a reproducible reservoir sample by
model/language/perturbation cell and measures tokenizer-length changes.

```bash
python analysis/tokenization_inflation.py \
  --inputs data/processed/multiclaimnet_verified.csv \
           data/processed/multiclaimv2_verified.csv \
  --sample-cache analysis_outputs/tokenization_sample.jsonl \
  --sample-per-cell 250 \
  --output analysis_outputs/tokenization_cells.csv
```

Join tokenization cells to retrieval results with:

```bash
python analysis/analyze_tokenization_inflation.py \
  --token-cells analysis_outputs/tokenization_cells.csv \
  --retrieval-results results/multiclaimnet/multiclaimnet_results.csv \
                      results/multiclaimv2/multiclaimv2_results.csv \
  --joined-output analysis_outputs/tokenization_retrieval_joined.csv \
  --report-json analysis_outputs/tokenization_report.json
```

## Human audit comparison

```bash
python analysis/audit_verifier.py \
  --annotations HUMAN_ANNOTATIONS.csv \
  --verified-csv data/processed/multiclaimnet_verified.csv
```

## Rerun readiness

```bash
python analysis/check_reanalysis_readiness.py \
  --external-root /path/to/external/data \
  --json analysis_outputs/readiness.json
```

Omit `--external-root` to inspect only the repository-standard paths.
