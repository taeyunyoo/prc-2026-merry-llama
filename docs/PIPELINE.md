# Pipeline execution

All entry points share the project-relative paths in `merry_llama.data` and call
package functions. No step requires an old experiment folder or a previous
submission. Authorized competition inputs and the three prepared external
resources described in README are prerequisites.

| Stage | Command | Output |
| --- | --- | --- |
| 01 Metadata | `python scripts/preproc.py -s all` | `data/interim/metadata_filled/` |
| 02 Base/external | `python scripts/create_base_features.py` | `data/interim/base_features/` |
| 03 Operational dataset | `python scripts/create_model_data.py` | `data/model_dataset/` |
| 04 Training and ranking | `python scripts/train.py` | Saved model bundle and `data/submission/merry-llama.parquet` |
| 05 Saved-model final inference | `python scripts/predict.py --dataset final` | `data/submission/merry-llama_final.parquet` |

Use the commands sequentially on a fresh project. With completed data and models,
run only saved-model inference. `predict.py --dataset ranking` produces ranking
predictions without fitting. `TrainModelAndSubmit.ipynb` exposes training,
ranking inference and final inference through its `ACTION` setting.

## Stage boundaries

Arrival and departure movements remain available through feature construction.
Only the shared model dataset is restricted to departures. Ranking and final
contexts are built independently so overlapping January/July flights are not
counted twice. Metadata donors are 2025 training plus the original ranking data;
final movements are metadata targets only.

STEP 03 saves arrival reference tables by their fitting scopes. STEP 04 copies
these references into the model bundle, fits route medians and airside features,
and records their full-year inference state. Learned feature columns are added
in memory without rewriting STEP 03 datasets. Final departure labels and block
times are not used by the inference code.

## Reuse and integrity

STEP 01 refuses to overwrite its outputs unless `--overwrite` is explicitly
selected. Do not rerun it on an already completed project. STEP 02 and STEP 03
validate and reuse unchanged completed outputs. Their completion manifests bind
upstream inputs, implementations, outputs and fitted reference tables.

The saved model run binds the STEP 03 manifest, model code, package versions,
settings and model artifacts. Training resumes completed components, and a fully
completed matching run is loaded without fitting. Final inference requires that
saved run and its matching prepared project data; it does not require a prior
ranking report. The model loader accepts the known earlier workflow release
whose fitting/prediction modules are unchanged, without modifying its manifests.

Template ID coverage, row order, types and finite predictions are validated when
writing each submission. These runtime checks protect the actual output; they do
not compare predictions to an earlier submission.

## Distribution

Commit source, entry points, notebook, settings, rules, requirements and docs.
Supply authorized competition data and prepared external resources separately.
Saved models may be distributed separately as the complete run directory;
otherwise the documented training command creates them. See README for the input
files and environment. No command uploads data or submissions automatically.
