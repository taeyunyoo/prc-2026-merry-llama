# PRC 2026 architecture

The project predicts departure taxi-out time (`TAXITIME_SEC_mvt`, seconds).
Reusable algorithms live in `src`, command-line entry points in `scripts`, and
the official workflow launcher in `notebooks`. The model follows the team's
v52 ensemble with frozen v54/v55 schedule corrections.

## Package responsibilities

All module paths below are relative to `src/merry_llama/`.

| Module | Responsibility |
| --- | --- |
| `data.py` | Project-relative paths, file groups and metadata donor scope |
| `preprocessor.py` | Two-pass filling of five metadata fields |
| `features/` | Core, external, arrival and movement features |
| `make_dataset.py` | Shared departure datasets, movement context and reference state |
| `learned_features.py` | Fit/apply route medians and airside models |
| `model_config.py`, `model_utils.py` | Frozen settings and model input preparation |
| `model_store.py` | Native model persistence, categories and component checkpoints |
| `lirf_model.py` | LIRF mixture classifier and residual model |
| `modeling.py` | Airport/AOBT routing, LightGBM/CatBoost fitting and blending |
| `postprocess.py` | Frozen v54/v55 rules, clipping and rounding |
| `submission.py` | Blank template validation, ID alignment, schema-preserving output |
| `training.py` | Training and saved-model inference orchestration |

Scripts are thin wrappers around these functions. The official notebook selects
`train`, `ranking` or `final` and starts the corresponding script in a fresh
Python process. It contains no training algorithms or personal filesystem paths.

## Data and model state

| Location | Contents |
| --- | --- |
| `data/raw/` | Original movement inputs and blank templates; read-only |
| `data/external/` | Three separately supplied prepared external resources |
| `data/interim/` | Metadata and base features with movement context |
| `data/model_dataset/` | Shared departure features, reference tables and manifests |
| `data/metadata/` | Generated schemas, feature definitions and provenance |
| `configs/` | Tracked provenance and frozen schedule rules |
| `models/<run_id>/` | Predictors, categories, learned features, settings and manifests |
| `data/submission/` | Ranking/final predictions |
| `logs/` | Notebook execution logs |

The default run directory remains `models/v55_reproduction/` so the completed
model bundle can be reused. This historical name is not a matching requirement.
Model loading checks integrity and accepts only the current implementation or
the specifically supported earlier workflow release. Workflow cleanup does not
rewrite model files, run configuration, or completion manifests.

## Fitting and prediction

Observed-AOBT departures use airport-specific residual models blended equally
across LightGBM and two CatBoost depths. Missing-AOBT departures at the other nine
airports combine pooled predictors, airside estimates and a 25% local LightGBM
blend where eligible. LIRF uses its three-type classifier baseline and a 50%
residual correction. Route medians and airside estimates are fitted using their
specified training folds/scopes, then saved for inference.

The frozen v54 and v55 category rules are applied in that order after the model
prediction. Rule discovery is not run on ranking or final data. Clipping and
rounding preserve the existing prediction method.

STEP 03 data is shared by LightGBM and CatBoost; their different feature subsets
and category handling are applied during model preparation. Full inference state
includes the category vocabularies, learned feature models and reference tables,
not just the final predictors.

## Dataset boundaries

- Retain arrivals until traffic/context features are complete; submit departures
  aligned by `MVT_ID_mvt`.
- Metadata donors are the 12 training files and original ranking. Final is
  target-only for metadata imputation.
- Ranking covers January/July 2026; final covers January/February/June/July.
  Build each evaluation context independently without concatenating overlaps.
- Evaluation departure targets and block times are not inference inputs.
- Preserve template order, non-target columns and numeric types; reject missing
  predictions or incorrect ID coverage.

The accepted ranking score for the current pipeline is RMSE 265.7088 seconds on
344,841 flights. Numeric identity with a historical submission is not a release
criterion. The prediction method remains unchanged by workflow/documentation
cleanup. See `docs/PIPELINE.md` for commands and `docs/STEP04.md` for reuse.
