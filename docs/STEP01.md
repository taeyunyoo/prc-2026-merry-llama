# STEP 01 — Metadata completion

This stage fills selected missing metadata from the fixed donor population.
If the 14-file stage has completed, continue to STEP 02 rather than rerunning it.

## Input and output

- Input: `data/raw/`, 12 monthly 2025 training files, `ranking.parquet`, and
  `final_ranking.parquet`. Submission templates are not movement inputs.
- Output: `data/interim/metadata_filled/`, the same 14 filenames plus
  `preprocessing_summary.json`.
- Existing values, targets, movement IDs/order, all original columns and both
  DEP/ARR phases are preserved.

## Source method

The two-pass v21 method fills only missing `CALLSIGN_flt`,
`MARKET_SEGMENT_flt`, `FLIGHT_TYPE_flt`, `WK_TBL_CAT_flt` and
`AIRCRAFT_OPERATOR_flt` on rows with missing AOBT.
The key is `FLIGHT_mvt` with surrounding whitespace removed; case is preserved.
Donors are AOBT-present rows of both phases in the 12 training files and the
original ranking. Final is a recipient only.

1. Use keys whose complete donors agree on all five fields. Incomplete donors
   do not veto this first pass; this preserves the source implementation.
2. For each field separately, use keys with no missing donor values and one
   distinct value across the full donor population.

## Execution from a fresh checkout

Activate the intended Python environment and install the local package once:

```bash
python -m pip install --no-deps -e .
python -u scripts/preproc.py --set all
```

`--root PATH` selects a project root explicitly. If this stage is already
complete in the current project, no rerun is required for model training.

Normal completion:

```text
COMPLETE: metadata preprocessing (14 files).
```

For an intentionally rebuilt or interrupted STEP 01, `--overwrite` regenerates
its derived outputs from raw inputs. This is not part of STEP 04 execution.
Continue with `docs/STEP02.md`, then `docs/STEP03.md`, then `docs/STEP04.md`.
