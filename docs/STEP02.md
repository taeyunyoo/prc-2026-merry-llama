# Step 02: core and external features

This stage builds core and external features from completed metadata outputs
and the prepared external resources described below.

## Installation and execution

For a fresh checkout, install the project from its root:

```bash
python -m pip install -e .
```

An existing editable installation needs no reinstall. The command below runs in
a fresh Python process.

After stage 01 completes for all 14 datasets, run from the project root:

```bash
python -u scripts/create_base_features.py
```

`--root /path/to/project` selects a different data root. Without it, the root is
the installed editable checkout, using the existing `ProjectPaths` definition.
No particular account name, environment name or operating system is required.

## Required inputs

`data/interim/metadata_filled/` must contain all 14 movement parquet files and
the successful all-datasets `preprocessing_summary.json`. This verifies the
handoff from the already completed preprocessing stage.

Keep these prepared resources in `data/external/`:

| Resource | Expected name | Matching rule |
| --- | --- | --- |
| Weather | `IEM_METAR_20250101_20260731_10_AIRPORTS.parquet` | Same airport, latest observation at/before movement time, no more than 60 minutes old |
| Reference | `taxiout_reference_features_10airports.csv` | Airport and UTC movement date |
| Distance | `airport_pairs_training_ranking.csv` | Ordered departure/destination pair |

Resources are selected by schema, not the newest filename. Exactly one candidate
for each type must be present. If multiple prepared versions are deliberately
retained, pass `--weather`, `--reference`, and/or `--distance` with the intended
path. Relative paths are resolved against the project root.

Weather must include `station`, `valid`, `tmpf`, `sknt`, `vsby`, `wxcodes`, `metar`
and the previously prepared columns `wx_gust_factor_kt`,
`wx_gust_reported_flag`, `wx_ceiling_ft`, `wx_no_ceiling_flag`.
`station` uses four-letter airport identifiers for all 10 airports. For the
existing convention, a no-ceiling flag of 1 requires a ceiling value of 60000 ft.
This stage consumes the corrected weather resource; it does not independently
reinterpret METAR gust/ceiling conventions.

Reference columns are `airport`, `date` (YYYY-MM-DD),
`avg_predeparture_delay_sec`, `avg_reference_taxiout_sec`, and
`avg_additional_taxiout_sec`. Valid airport/date keys must cover the required
training, ranking and final movements; missing reference coverage fails before
feature output is written. Reference values are finite seconds.

Distance requires `ADEP`, `ADES`, and `distance_km`. Missing distances remain
missing and are reported. Missing codes and `ZZZZ` are not matched. Resource
files are read-only. This step does not download or regenerate the resources;
the prepared resource bundle is a prerequisite and must be supplied separately
by the team alongside authorized competition inputs. Use the files from the
accepted run to preserve their gust/ceiling, reference and distance conventions.
The weather file contains IEM METAR observations; the reference CSV contains the
prepared airport/date delay and taxi-out statistics; the distance CSV contains
the prepared ordered airport-pair distances. Their schemas above are the input
contract, not a claim that this repository downloads those sources.

## Output contract

Each input produces a file of the same name in `data/interim/base_features/`.
The files retain every original column, dtype, row and row order. This includes
arrival movements and source-only columns that later traffic features need.
The 46 added columns are calculated for departure rows; their arrival entries
are null. Targets and timestamps are retained unchanged.

| Added group | Count |
| --- | ---: |
| Resolved ADEP, ADES, aircraft type and flight rule | 4 |
| Local calendar and cyclical time | 6 |
| Time differences | 15 |
| Previous airport departure counts | 3 |
| Previous same-runway heavy/super count | 1 |
| Weather | 13 |
| Reference statistics | 3 |
| Distance | 1 |
| Total | 46 |

These 46 columns plus eight retained raw categorical columns provide the 54
non-median inputs in the source method's 55-column base feature set.
`stand_runway_median_taxiout_sec` is deliberately reserved for fitting under the
correct training/validation scope. No target is used to calculate this stage's
46 columns. Later arrival/traffic/airside features extend these outputs.

Time is parsed in UTC with nanosecond normalization where needed. Local time uses
the configured airport timezone, including daylight saving changes. Traffic
windows exclude equal-time departures. Airport count windows crossing the known
start of data coverage are null, matching the existing definition.

Training airport counts use the 2025 context; ranking and final airport counts
use their own observation periods. The legacy heavy/super context is training
plus original ranking for those outputs, and training plus final for final
outputs. Original ranking is never concatenated with final. Duplicate departure
IDs within a context and training/evaluation ID overlaps are rejected.

## Completion and diagnostics

Successful completion prints:

```text
COMPLETE: base/external feature generation (14 files, 46 added columns).
```

The output folder also contains:

- `step02_file_summary.csv`: row counts and weather/reference/distance coverage.
- `step02_missingness.csv`: missing counts for every added feature.
- `step02_complete.json`: complete-stage contract and content fingerprints.

Weather or distance missingness is reported without inventing replacement values.
Absence of a complete marker means the stage is not ready for downstream use.
The runner writes each parquet through a temporary file and checks that source
columns and output values survive the parquet round trip.

Run the same command again to verify and reuse an unchanged complete stage.
To perform a read-only check explicitly:

```bash
python scripts/create_base_features.py --verify
```

If a run was interrupted or inputs intentionally changed, rebuild this stage only:

```bash
python -u scripts/create_base_features.py --overwrite
```

Do not use `--overwrite` routinely. It is unnecessary for a successful unchanged
stage. Neither command repeats metadata preprocessing.

## Verification delivered with this stage

```bash
python -m unittest discover -s tests -v
```

The small synthetic test runs actual metadata preprocessing then all 14 feature
builds, checks raw/ARR preservation, UTC precision, local daylight saving time,
traffic ties/coverage, ranking/final isolation, unmatched values, weather age
boundaries, verified reuse and changed-input rejection. It uses no private files
and no additional test framework.

In the development workspace, a separate comparison runs the original v21 core
and external implementations on the same synthetic training/ranking fixtures and
requires exact equality across their 13 files and 46 added columns. This checks
the port, not full-data v55 model reproduction. Full-data coverage is checked at runtime; matching a previous submission is
not required.
