# Historical backtest: protocol, leakage audit and execution

This opt-in experiment fits January–October 2025 and evaluates November–December
2025. All boundaries use **movement timestamps in UTC**, with half-open intervals
`[2025-01-01, 2025-11-01)` and `[2025-11-01, 2026-01-01)`. Filenames alone do not
establish membership: every movement is checked, and misplaced or missing times
fail. Departure IDs must be unique across all twelve inputs.

No historical score has been calculated for this change. Authorized competition
inputs and trained artifacts are unavailable here. The accepted ranking result
(265.7088 seconds on 344,841 flight pairs) is separate evidence; STEP04's completed
full-year refit did not calculate independent holdout RMSE. Neither that ranking
result nor a score from the existing `validation/` files establishes an independent
historical test.

## Estimand and model selection

This is a **retrospective batch experiment**, with published fixed model settings
and retrained parameters. It is not pushback-time or online prediction. Actual
movement/takeoff times, AOBT proxies, future next-event gaps, arrival relationships
and within-holdout operational contexts are deliberately retained. Holdout labels
never enter fitting or feature construction, but holdout operational records do
enter the retrospective batch context. A defensible claim must state this
information set explicitly. Arrival taxi-in observations are operational inputs
in this protocol, not held-out departure targets.

The published trees, feature set, blend weights, special LIRF design and other
choices may have been selected with November–December historical outcomes or
ranking feedback. Their independence from this holdout is unestablished. Every
run and metric report is therefore marked `published_frozen_exploratory` and
**must not be described as independent validation**. The v54/v55 category rules
are disabled: they contain fitted selection statistics whose train-only
provenance is not established. This makes the experiment a rules-disabled
variant, not an exact reproduction of the accepted submission. Predictions are
clipped to nonnegative values and rounded half-up to seconds before primary RMSE;
unrounded predictions are also retained for inspection.

For a future confirmatory evaluation, preregister the information set, labels,
features, hyperparameters and postprocessing; select them using only earlier
training periods (for example an inner September–October validation with donors
ending August), then use a genuinely untouched period. Repeated tuning on this
holdout converts it to development data. Nothing here assumes a manuscript.

## Leakage audit

| Component | Existing scope / risk | Backtest handling / remaining requirement |
| --- | --- | --- |
| `training.py` | All twelve 2025 months | Separate entry point passes only ten rebuilt monthly files to all fit functions. No production code or artifacts are overwritten. |
| STEP01 metadata | `metadata_donor_names()` includes all 2025 and ranking; even target-free imputation can be transductive | Requires audited/rebuilt STEP02 inputs. Fit `preprocessor.fit_lookups` on January–October raw donors only and apply that saved lookup to both periods; do not reuse existing filled files as evidence. |
| STEP02 core counts | Training-wide operational context; heavy/super donor universe also includes ranking | Rebuild counts from training-only operational records for fitting rows, and a documented training-plus-holdout context for query rows. Ranking/final records must be excluded. Review source timestamps and inherited categorical values too. |
| External airport/date reference CSV | Matched by the same UTC date, including delay and taxi-out aggregates; preparation/provenance is not demonstrated by the schema | Block scoring until evidence establishes no held-out target contributions or unavailable same-day outcomes. Rebuild with pre-cutoff donors/as-of publication times, or create a separately specified feature ablation. A lag cannot be assumed safe without source lineage. |
| Weather | Backward match within 60 minutes is enforced, but corrected gust/ceiling fields are prepared externally | Review parsing, station coverage, report/correction availability and archive revisions against the retrospective information set. Backward valid-time matching alone does not prove release-time availability. |
| Route distances | File universe includes training and ranking | Verify distances are static geometry/authoritative route values, not target-derived, and document missing-route policy. Rebuild universe from authorized training/query routes if necessary. |
| v42 arrival references | Historical cutoff already exists, but existing context can mix scopes | Refit all eight reference tables from January–October context only. Reference donors require both landing and in-block before cutoff. Save and reuse for queries; no holdout refitting. |
| v50 movement stand baseline | Existing validation shares full-year arrival medians | Refit airport/stand taxi-in medians from pre-cutoff completed arrivals. Both contexts use these frozen tables. Existing STEP03 caches are not consumed. |
| Route medians | Target-derived UTC-day five-fold OOF | OOF and full saved maps use fitting rows only. Unknown routes fall back through runway/stand/airport to the training global median. Training-internal OOF is not a forward simulation. |
| Airside models | Leave-one-month-out features and full fitted model | Only January–October present-AOBT donors. Saved inference model and categories use that scope; no November/December OOF training. Internal median/airside nesting follows the source and is not claimed to be strictly nested or causal. |
| LIRF mixture and residual | Class labels use departure targets/block-schedule relationships | Mixture medians, classifier vocabulary and model fit use ten training files only; forward residual OOF is May–October. Query loading excludes target and block-time label columns. Insufficient type-B/OOF donors fail rather than invent predictions. |
| Model selection and category rules | Historical/leaderboard tuning may have informed published settings/rules | Fixed published settings are exploratory; no holdout tuning or early stopping is added. Rules disabled. Confirmatory selection requires a separate untouched design. |

The upstream preparation items are **prerequisites, not automatically solved by a
row split**. The accepted STEP01–04 implementation remains unchanged. This change
does not fabricate replacement daily references or declare them safe by deleting
labels from an already contaminated feature table.

## Audited input contract

Supply exactly the twelve `training_*.parquet` files, with the mixed arrival and
departure rows and STEP02 base columns. Neither `full/` nor `validation/` STEP03
files qualify; preexisting operational or learned columns are rejected.

Copy `configs/backtest_audit.example.json` to a private experiment file. All
approvals default to false. Name the reviewer, attach concrete evidence for each
check, and populate `base_sha256` with the twelve filenames mapped to SHA-256
content digests. Useful evidence includes raw/resource hashes, source periods,
publication/availability semantics, the train-only metadata lookup inventory,
preparation code revision and the batch-context definition. Evidence should
identify retained records, not merely say "reviewed". Keep source provenance with
the experiment. The audit binds those approved inputs to bytes, but an attestation
is a human judgment: code cannot verify the truth of its prose.

Example inventory helper (does not approve any checks):

```python
from pathlib import Path
from merry_llama.backtest import inventory
from merry_llama.data import training_names
base = Path('/authorized/experiment/base_features')
print(inventory([base/n for n in training_names()], base))
```

Do not approve the example for the accepted STEP02 exports: the metadata and
reference provenance concerns above remain unresolved. The synthetic fixture
uses fictional attestations solely to exercise software contracts.

## Run

In a dedicated environment, install the project and model dependencies:

```bash
python -m pip install -e .
python -m pip install -r requirements-model.txt
python scripts/backtest.py \
  --base /authorized/experiment/base_features \
  --audit /authorized/experiment/audit.json \
  --output /authorized/experiment/backtests/run01 --threads 4
```

`--prepare-only` rebuilds feature datasets without requiring model libraries or
writing metrics. It still requires the audit. `--allow-version-mismatch` records
an explicitly non-exact environment; missing model libraries still fail. Use a
new output directory for every run. No resume or overwrite option exists, so a
full-year model/cache cannot be silently reused. Output cannot overlap inputs
or the checkout's production data, models, configs or source trees.

The run creates:

- `audit.json` and `run_config.json`: exact approved base hashes, fitting/evaluation
  boundaries, Python/package versions, implementation fingerprints, model-selection
  caveat, thread setting and output inventory. Completion is written last.
- `dataset/fit/`, `dataset/evaluation/`, `dataset/_references/` and movement caches:
  independently rebuilt operational data. Fitting context never sees holdout
  movements. Query context includes training for boundary warm-up; it uses frozen
  references and target-masked holdout departures. Departures with invalid
  holdout labels remain prediction rows.
- `models/`: training-only route medians, airside, LIRF and ensemble artifacts.
- `predictions.parquet`: all holdout departure IDs, rounded and unrounded seconds.
- `metrics.json`: overall pair-weighted RMSE, every configured airport's RMSE,
  evaluated pair counts and invalid-target exclusions. Airports with no eligible
  targets have null RMSE, not zero.

The truth file is saved separately for scoring. Evaluation target values are
replaced by an eligibility sentinel **before** any STEP03 feature construction,
so even validity-based row selection cannot leak their values. LIRF inference
uses the target-free loader; the model query has its target column removed.
Scoring demands exact ID-set equality and one-to-one joins. Missing/extra or
duplicate predictions, nonfinite predictions and out-of-period truth fail.
Nonfinite/negative truth is counted and excluded; zero is valid. Overall RMSE
is over flight pairs, not an unweighted average of airport RMSEs.

Inspect `run_config.json`: only `status=complete` denotes a scored completed run.
A failed run may leave diagnostic files, which are not valid results. A prepared
run has `status=prepared` and no score. Keep input and code snapshots immutable
during execution; base hashes are rechecked after preparation and fitting.

## Synthetic verification and current limitations

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

New tests cover cutoff membership, audit refusal/hash binding, train/evaluation
ID isolation, holdout-label masking, arrival perturbation invariance of fitted
references and fitting features, LIRF fitting path scope, unknown-route median
fallback, exact prediction coverage, invalid labels and hand-calculated overall
and airport RMSE. They do not require private competition data or model fitting.
The full suite needs the project's normal Parquet dependencies.

In the implementation environment, syntax compilation, two metric tests and five audit-gate tests
passed. The complete synthetic suite could not import because `pyarrow` was not
installed, and the restricted package index provided no installable distribution.
The Parquet integration tests and an actual LightGBM/CatBoost backtest still need
to run in an authorized environment. No real overall or airport RMSE is reported.
Synthetic arithmetic checks are software tests, not evidence of predictive skill.
