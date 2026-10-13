# STEP 04 — Model training and submission

This stage reads completed STEP 03 datasets, fits the configured model branches
and learned features, saves all inference state, and creates a ranking submission.
Final inference separately loads that saved state. No previous submission is
required, and no prediction-comparison reports are generated.

## Environment

Install `requirements-model.txt` and the package as described in README.
The completed Windows run used Python 3.10.20, NumPy 2.2.6, pandas 2.3.3,
PyArrow 25.0.0, LightGBM 4.7.0, CatBoost 1.2.10 and scikit-learn 1.7.2.
An existing bundle must be loaded with its recorded Python/package versions.
`--allow-version-mismatch` is an explicit training experiment option; the
resulting environment is still recorded and checked when loading that run.

## Commands

After STEP 03, fit the models and create ranking predictions:

```bash
python -u scripts/train.py --root .
```

For a completed run, predict without fitting:

```bash
python -u scripts/predict.py --root . --dataset ranking
python -u scripts/predict.py --root . --dataset final
```

`predict.py` defaults to final inference. The notebook
`notebooks/TrainModelAndSubmit.ipynb` runs the same entry points; select the
operation with `ACTION`. Choose `"final"` for an already trained run.
No command uploads the output to the competition bucket.

## Saved run

The default run ID remains `v55_reproduction` to preserve the existing model
location. Keep `models/v55_reproduction/` intact. Its name is historical; it does
not enable or require comparison with v55. `--run-id NAME` selects a separate
run when deliberately changing training settings or data.

The workflow update accepts the exact earlier release of the orchestration and
submission modules. All seven fitting/feature/model modules must still match.
It does not rewrite any saved manifest. Arbitrary changes to model code, inputs,
package versions or model artifacts remain errors.

Completed component fits are reused after interruption. A fully completed
matching run skips fitting altogether. `RUNNING.lock` prevents concurrent writes
to one run. If a process was force-killed, confirm it stopped before removing
only its stale lock. Do not delete the model directory or its manifests.

## Model method

| Component | Behavior |
| --- | --- |
| Route median | Hierarchical median; five folds by shuffled UTC date, seed 42; saved full-year mapping |
| Airside | AOBT-present donors excluding the queried training month; 40% sample, seed 42, labels capped at 10,800 s; saved full-year predictor |
| Observed AOBT | Airport-specific residual LightGBM + CatBoost depth 8 + CatBoost depth 10, equal average, fixed tree counts |
| Missing AOBT, other nine airports | `(pooled LGB + airside + pooled CB)/3`, then 25% local LGB when at least 200 eligible training rows |
| Missing AOBT, LIRF | Three-type classifier baseline, forward OOF May–December residual LightGBM, 200 trees, weight 0.5 |
| Schedule corrections | Frozen v54 then v55 rules; no rule discovery on evaluation data |
| Rounding | Clip to zero and round half up |

Training uses CPU. The completed full-data run took approximately 490.8 minutes;
this is an observation, not a runtime guarantee. Saved-model inference does not
repeat this work. All model hyperparameters, feature calculations and blend
weights are unchanged by the workflow update.

## Output

| Path | Purpose |
| --- | --- |
| `models/v55_reproduction/` | Models, feature state, category vocabularies, frozen rules, settings and manifests |
| `data/submission/merry-llama.parquet` | Ranking predictions |
| `data/submission/merry-llama_final.parquet` | Final predictions |
| `logs/step04_run.log` | Notebook training log |
| `logs/predict_ranking.log`, `logs/predict_final.log` | Notebook inference logs |

STEP 04 prints a completion summary with status, dataset, rows, output path and
SHA-256. It does not create comparison CSVs, diagnostic ZIPs or report folders.
Use the printed `output` path: if a pre-existing submission has different values,
a run-ID suffix prevents silent replacement. The challenge's final uploaded
filename is `merry-llama_final.parquet`.

The older `merry-llama_v55_reproduced.parquet` output is not deleted or rewritten.
The historical reference under `data/reference/`, if still present, is unused.

## Runtime checks and result

Checks cover completed STEP 03 inputs, model artifacts, environment, frozen
rules and blank submission template IDs. Output writing validates template order,
types, non-target columns and finite predictions. These checks do not depend on
an older submission or a report file.

The current pipeline's ranking output, submitted as `merry-llama_v58.parquet`,
received `Succeeded`, 344,841 evaluated pairs and RMSE **265.7088 seconds**.
The final dataset's score is not established by this ranking result. The full-year
training command does not calculate an independent holdout RMSE.
