# Data layout

Data is not committed to this repository. Obtain each source from its original
provider and comply with its license, access conditions, and privacy rules.

Expected layout:

```text
data/
├── raw/
│   ├── MultiClaim.csv
│   ├── posts.csv
│   ├── fact_checks.csv
│   └── fact_check_post_mapping.csv
└── processed/
    ├── preprocessed_multiclaim.csv
    ├── preprocessed_posts.csv
    ├── multiclaimnet_preprocessed.csv
    ├── multiclaimv2_preprocessed.csv
    ├── multiclaimnet_verified.csv
    └── multiclaimv2_verified.csv
```

## Required source columns

### `MultiClaim.csv`

The preprocessing and Task A retrieval code use at least:

- `NID`: unique claim identifier
- `Claim`: claim text
- `ClusterID`: relevance cluster
- `Language`: ISO 639-3 language code

### `posts.csv`

The post preprocessor uses at least:

- `post_id`: unique post identifier
- `post_body`: original post text
- `post_detected_language`: ISO 639-1 or compatible language tag

### `fact_checks.csv`

Task B uses:

- `fact_check_id`: unique fact-check identifier
- `claim`: fact-check claim text

### `fact_check_post_mapping.csv`

Task B relevance mappings use:

- `post_id`
- `fact_check_id`

## Supported languages

The pipeline normalizes English, German, Spanish, French, Polish,
Serbo-Croatian/Bosnian, Hindi, Arabic, Malayalam, and Chinese to the internal
tags defined in `perturbations/utils.py`.

## Generated schemas

Perturbation output preserves the input columns and adds `perturbation_name`,
`family`, `perturbed_text`, `changed`, and `error`. Verification adds
`perturbation_applied`, `meaning_preserved`, `verified`, and `verify_error`.

The entire `data/` tree except this guide is ignored by Git. Before publishing
any derived file, check whether it contains original post text or identifiers
that cannot be redistributed.
