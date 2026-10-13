# Step 03: shared operational feature datasets

## What this step does

This step reads the verified STEP 02 outputs and adds 69 operational features:
40 from the frozen v42 arrival engine and 29 movement/condition/stand features
used by the v50/v52 lineage. It creates departure datasets for both LightGBM and
CatBoost. It does not train either prediction estimator.

The remaining four fitted-feature columns are deliberately reserved for training:
`stand_runway_median_taxiout_sec`, `cf_recent_vs_usual_rwy`, `airside_est`, and
`airside_minus_proxy`. Their fitting and out-of-fold scope must match the source
method. This stage never invents placeholder values for them.

## Execution

From the installed project root:

```bash
python -u scripts/create_model_data.py
```

`--root /path/to/project` selects an explicit data root. The notebook launcher
may call this command, but feature algorithms live entirely inside the repository.
The currently installed preprocessing dependencies are sufficient; no new model
library is required for this step.

The command first verifies the STEP 02 completion marker and source/output
content hashes. It then builds training, ranking and final movement contexts,
fits/saves reference tables, and writes the scope-specific departure datasets.
Progress is printed for each airport, context and written file. This is more
work than STEP 02; reference matching and per-airport statistics may take time.

## Outputs

| Location under `data/model_dataset/` | Contents |
| --- | --- |
| `full/` | 12 training files plus `ranking.parquet` and `final_ranking.parquet` (14 files) |
| `validation/` | The 12 training files with historical validation-scope arrival references |
| `_references/full/` | Eight saved v42 full-scope reference tables and their inventory |
| `_references/validation/` | Eight saved v42 validation-scope reference tables and their inventory |
| `_references/movement/` | Saved v50 airport/stand taxi-in baselines and their inventory |
| `_movement/` | Reusable movement feature caches for train, ranking and final |
| `step03_file_summary.csv` | Output row counts, invalid training targets and movement join coverage |
| `step03_missingness.csv` | Missing counts for each feature on eligible departure rows |
| `step03_complete.json` | Stage contract, source/implementation fingerprints and output inventory |

The 26 output parquet datasets keep every departure row, its original columns,
column types and order within the source file. Arrival rows remain in upstream
`base_features/` for subsequent context calculations. They are not model
prediction rows. `final_ranking.parquet` in `full/` should have 670,790 departure
rows for the supplied competition input; the program derives and validates row
counts rather than hard-coding this number.

Training rows with nonfinite or negative taxi-out labels remain present, with
their 69 new features null, matching the legacy feature export's exclusions.
They are counted in the report and must be excluded from model fitting later.
Zero is a valid target under the source policy.

## Source semantics and scopes

| Component | Preserved policy |
| --- | --- |
| v42 full arrival references | Original arrival timestamp cutoff before 2026-01-01; full training departure turnaround references |
| v42 validation references | Original arrival cutoff before 2025-11-01; training departure movement months <= 10 |
| v50 movement stand baseline | All 2025 training arrivals, shared by full and validation, as in the original export |
| Final reference application | Reload full-scope reference tables; no reference fitting on final data |
| Traffic/condition context | Training, ranking and final processed separately for v50; no ranking/final concatenation |
| v42 final relationship context | Training plus final, with saved full references; original ranking excluded |

Arrival taxi-in reference values use the respective source definitions: v42
uses arrival `TAXITIME_SEC_mvt`, whereas v50 derives taxi-in from in-block minus
landing time. They are intentionally not made interchangeable. Departure target
values only determine the original valid-training-row population at this stage;
they are not used as predictive feature values.

The original batch definitions are preserved, including future next-departure /
next-stand gaps, window boundaries, tie handling, minimum rolling counts and
fallback rules. They should not be interpreted as an online prediction API.
The final context's arrival coverage intervals are extended for January–February
and June–July 2026. Windows crossing the beginning of observed coverage retain
the appropriate missing value instead of treating the unobserved gap as data.

The movement builder retains the legacy use of raw movement airport fields.
Departures omitted by that builder can still exist in the shared dataset with
null movement features; `movement_unmatched` reports this situation. No silent
airport-source substitution is introduced.

The historical validation donor scopes are retained. A different validation
design is a separate experiment. STEP 04 copies the saved reference tables into
the fitted model run bundle.

## Completion, reuse and recovery

Successful completion prints:

```text
COMPLETE: operational model datasets (26 files, 69 added columns).
```

The completion marker is written last. Repeating the same command verifies and
reuses a complete unchanged stage. A read-only verification is also available:

```bash
python scripts/create_model_data.py --verify
```

For an interrupted run or intentionally changed inputs, explicitly rebuild only
STEP 03 with `--overwrite`. This never modifies STEP 01/02 outputs. Do not use
that option routinely or treat a directory containing partial parquet files as
a completed stage.

## Validation scope

The delivered synthetic integration tests run STEP 01, STEP 02 and STEP 03 in
sequence, and check the 26-file handoff, original DEP columns/order, invalid-target
handling, full/validation movement consistency, separate final context, June
coverage boundary, saved-state application, content-verified reuse and corruption
detection. They need no source experiment folders or extra testing framework:

```bash
python -m unittest discover -s tests -v
```

Development comparisons also executed the original supplied v42 engine for both
reference scopes and the original v50 movement functions, requiring exact values
on shared synthetic inputs. Numeric storage types are standardized at the same
float32 export boundary used by the source. The full-data pipeline has also completed on the user inputs; the reported
ranking score is documented in README.
