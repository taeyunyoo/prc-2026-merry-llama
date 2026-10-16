![Team Merry Llama](merry-llama.png)

# PRC 2026 — Merry Llama

Taxi-out time prediction for the PRC Data Challenge 2026.

Departure taxi-out prediction (`TAXITIME_SEC_mvt`, seconds) using LightGBM,
CatBoost, operational features and the team's frozen v54/v55 schedule rules.
The pipeline prepares data, trains and saves models, and creates ranking or
final submissions from saved models.

## Installation

The completed training run used Python 3.10.20 on Windows. Use Python 3.10
for the same environment; the package supports Python 3.10 or newer. Install
from the repository root in the environment that will run the pipeline:

```bash
python -m pip install -r requirements-model.txt
python -m pip install --no-deps -e .
```

The model dependencies are pinned. To load an existing model bundle, use its
recorded Python and package versions. No command installs packages automatically.

## Repository structure

| Path | Purpose |
| --- | --- |
| `configs/` | Pipeline configuration |
| `data/raw/` | Competition datasets and blank submission templates |
| `data/external/` | Prepared weather, airport, and route resources |
| `models/` | Saved models and learned feature state |
| `notebooks/` | Training and prediction launcher |
| `scripts/` | Command-line entry points |
| `src/merry_llama/` | Pipeline implementation |
| `docs/` | Stage instructions and pipeline documentation |
| `tests/` | Implementation tests |

## Required data

Competition data requires authorized challenge access. Keep the original files
in `data/raw/`; all pipeline paths are relative to this repository.

| Input | Files / scope |
| --- | --- |
| Training | 12 monthly files from `training_2025-01-01_2025-02-01.parquet` through `training_2025-12-01_2026-01-01.parquet` |
| Ranking | `ranking.parquet`: January and July 2026 movements |
| Final | `final_ranking.parquet`: January, February, June and July 2026 movements |
| Blank ranking template | `submitting.parquet` |
| Blank final template | `final_submitting.parquet` |

Do not use a previous prediction file as a blank submission template.
The normal full workflow requires all 14 movement files and both templates.

Place the team's three prepared resources in `data/external/`:

- `IEM_METAR_20250101_20260731_10_AIRPORTS.parquet`
- `taxiout_reference_features_10airports.csv`
- `airport_pairs_training_ranking.csv`

The scripts read these prepared resources directly; they do not download or
regenerate them. The resources must cover the ranking and final months.
See [STEP02.md](docs/STEP02.md) for schemas and matching rules.

## Run from raw inputs

Run these commands **in order** from the repository root:

```bash
python -u scripts/preproc.py -s all
python -u scripts/create_base_features.py
python -u scripts/create_model_data.py
python -u scripts/train.py
python -u scripts/predict.py --dataset final
```

The first three commands prepare the shared datasets. `train.py` fits learned
features and models, saves them, and generates the ranking submission.
`predict.py` loads the saved run for final inference. It never trains models.
Full training took approximately 8 hours 11 minutes in the completed run;
runtime depends on hardware. Prediction is a separate, shorter operation.

For an existing completed STEP 01–04 run, execute only:

```bash
python -u scripts/predict.py --dataset final
```

To generate ranking predictions again without training:

```bash
python -u scripts/predict.py --dataset ranking
```

Nothing is uploaded to GitHub or the competition bucket by these commands.

## Jupyter

After completing the first three data preparation commands, open `notebooks/TrainModelAndSubmit.ipynb` with the configured Python kernel.
The single launcher cell has an `ACTION` setting:

| ACTION | Behavior |
| --- | --- |
| `"train"` | Fit once or reuse a completed matching run, then generate ranking predictions |
| `"ranking"` | Saved-model ranking inference only |
| `"final"` | Saved-model final inference only |

For the existing completed run, set `ACTION = "final"`. Notebook code delegates
to the same scripts; model logic is not duplicated in notebook cells.

## Outputs and reuse

| Location | Contents |
| --- | --- |
| `data/interim/` | Metadata and base-feature outputs |
| `data/model_dataset/` | Full and historical-validation departure datasets, movement context and fitted reference tables |
| `models/v55_reproduction/` | Saved predictors, learned feature state, category mappings, settings and completion manifests |
| `data/submission/merry-llama.parquet` | Ranking predictions |
| `data/submission/merry-llama_final.parquet` | Final predictions |
| `logs/` | Execution logs when using the notebook |

Existing matching data/model stages are reused. Model and upstream data
integrity checks remain active.
If an existing submission contains different predictions, a run-ID suffix is
used instead of overwriting it. The printed `output` is the authoritative path.
The final file uploaded to the challenge must be named `merry-llama_final.parquet`.

Keep competition datasets, generated data, trained models, logs and reports
out of Git. To distribute a saved run separately, retain the entire model directory
and the matching prepared project data, including completion manifests. Copying
only `.cbm` or `.txt` predictors is insufficient.

## Method and documentation

The model branches use airport-specific and pooled predictors, a separate LIRF
missing-AOBT model, learned route medians and airside estimates, and frozen
schedule corrections. See [ARCHITECTURE.md](ARCHITECTURE.md) and
[STEP04.md](docs/STEP04.md) for the method and saved-run behavior.

Stage instructions: [STEP01](docs/STEP01.md), [STEP02](docs/STEP02.md),
[STEP03](docs/STEP03.md), [STEP04](docs/STEP04.md).
[PIPELINE.md](docs/PIPELINE.md) describes stage boundaries.

To run the small implementation tests (not full model training):

```bash
python -m unittest discover -s tests -v
```


## Historical backtesting

An isolated, audit-gated January–October 2025 fit / November–December 2025
retrospective experiment is available through `scripts/backtest.py`. See
[the protocol and leakage audit](docs/BACKTEST.md) before supplying inputs. It
rebuilds training-only reference state, disables the published category rules,
and leaves the accepted full-year workflow unchanged. No historical RMSE has
been calculated; existing validation exports alone do not establish isolation.
