"""Opt-in, audited retrospective batch backtest; never writes production assets."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data import training_names
from .features import arrival, movement, state
from .features.base import write_json
from .features.core import clean
from .features.definitions import ID, TIME, TARGET, ADDED, DEFERRED
from .make_dataset import (read_departures, read_arrivals, movement_features,
                           export_dataset, atomic_table)
from . import learned_features as learned
from .model_utils import assert_ids
from .backtest_metrics import score
from .backtest_protocol import CONTRACT, CHECKS, inventory, validate_audit

START = pd.Timestamp('2025-01-01', tz='UTC')
CUTOFF = pd.Timestamp('2025-11-01', tz='UTC')
END = pd.Timestamp('2026-01-01', tz='UTC')


def check_months(paths):
    """Do not trust filenames as the temporal split; reject misplaced rows."""
    seen = set()
    for month, path in enumerate(paths, 1):
        d = pq.read_table(path, columns=[ID, TIME, 'PHASE_mvt']).to_pandas()
        t = pd.to_datetime(d[TIME], utc=True, errors='raise', format='mixed')
        lo = pd.Timestamp(year=2025, month=month, day=1, tz='UTC')
        hi = END if month == 12 else pd.Timestamp(year=2025, month=month+1, day=1, tz='UTC')
        if t.isna().any() or not ((t >= lo) & (t < hi)).all():
            raise ValueError(f'{path.name}: movement time outside declared month')
        ids = d.loc[clean(d.PHASE_mvt).eq('DEP').fillna(False), ID]
        if clean(ids).isna().any() or ids.duplicated().any() or seen.intersection(ids):
            raise ValueError('Missing/duplicate/overlapping departure IDs')
        seen.update(ids)
        names = set(pq.read_schema(path).names)
        if names.intersection(ADDED + DEFERRED):
            raise ValueError('Backtest requires base features, not existing STEP 03/04 data')


def mask_evaluation(path, destination):
    """Keep truth out of feature construction, including validity-based row filters."""
    table = pq.read_table(path)
    dep = clean(table['PHASE_mvt'].to_pandas()).eq('DEP').fillna(False).to_numpy()
    labels = table.select([ID, 'ADEP', TIME, TARGET]).to_pandas().loc[dep].copy()
    target = pd.to_numeric(table[TARGET].to_pandas(), errors='raise').to_numpy(float).copy()
    target[dep] = 0.0  # eligibility sentinel, never a model-fitting label
    table = table.set_column(table.schema.get_field_index(TARGET), TARGET, pa.array(target))
    atomic_table(table, destination)
    return labels


def fit_movement_references(paths):
    mv = movement.load_movements(paths)
    cutoff = CUTOFF.timestamp()
    donors = mv.loc[mv.phase.eq('ARR') & (mv['T'] < cutoff) & (mv.block < cutoff)]
    stand, airport = movement.stand_baseline(movement.clean_arrivals(donors))
    if airport.empty:
        raise ValueError('No pre-cutoff arrival movement baseline donors')
    return {'stand': stand, 'airport': airport}


def prepare(base, output):
    """Rebuild fit context without holdout; apply its references to batch context."""
    paths = [base / name for name in training_names()]
    check_months(paths)
    train_paths = paths[:10]
    masked = output / '_masked'
    masked.mkdir(parents=True)
    labels = []
    eval_paths = []
    for path in paths[10:]:
        dest = masked / path.name
        labels.append(mask_evaluation(path, dest))
        eval_paths.append(dest)
    truth = pd.concat(labels, ignore_index=True)
    assert_ids(truth, 'holdout truth')
    # Fit references on training only, including arrival completion before cutoff.
    dep, arr = read_departures(train_paths), read_arrivals(train_paths)
    arr = arr.loc[(arr.LAND < CUTOFF) & (arr.INBLOCK < CUTOFF)].copy()
    enriched, refs = arrival.add_arrival_features(dep, arr, scope='validation')
    state.save_references(refs, output / '_references/arrival')
    movement_refs = fit_movement_references(train_paths)
    state.save_references(movement_refs, output / '_references/movement')
    train_mv, _ = movement_features(train_paths, movement_refs)
    atomic_table(pa.Table.from_pandas(train_mv, preserve_index=False), output / '_movement/train.parquet')
    for path in train_paths:
        export_dataset(path, enriched, train_mv, output / 'fit' / path.name, 'backtest-fit')
    # Warm-start holdout context from training, but never refit saved references.
    context = train_paths + eval_paths
    qdep, qarr = read_departures(context), read_arrivals(context)
    qdep.loc[qdep[TIME].ge(CUTOFF), '_ranking'] = True
    enriched, _ = arrival.add_arrival_features(qdep, qarr, scope='validation', references=refs)
    eval_mv, _ = movement_features(context, movement_refs)
    atomic_table(pa.Table.from_pandas(eval_mv, preserve_index=False), output / '_movement/evaluation.parquet')
    for path in eval_paths:
        export_dataset(path, enriched, eval_mv, output / 'evaluation' / path.name, 'backtest-evaluation')
    atomic_table(pa.Table.from_pandas(truth, preserve_index=False), output / 'truth.parquet')
    return [output/'fit'/p.name for p in train_paths], [output/'evaluation'/p.name for p in eval_paths]


def fit_learned(train, dataset, folder):
    """Training-internal OOF only; final state never sees November/December."""
    if not train[TIME].between(START, CUTOFF, inclusive='left').all():
        raise ValueError('Learned feature donor outside fitting interval')
    directory = folder / 'learned'
    learned.save_median(learned.RouteMedian().fit(train), directory / 'median_full')
    values = learned.oof_median(train)
    train = learned.with_median(train, values, dataset / '_movement/train.parquet')
    present = train.loc[~train._missing]
    estimates = np.full(len(train), np.nan)
    for month in sorted(train._month.unique()):
        mask = train._month.eq(month).to_numpy()
        donors = present.loc[present._month.ne(month)]
        if len(donors) < 3:
            raise ValueError('Insufficient training-internal airside donors')
        sub = directory / f'airside_oof_{month:02d}'
        learned.fit_airside(donors, sub)
        estimates[mask] = learned.predict_regressor(sub, train.loc[mask])
    if not np.isfinite(estimates).all():
        raise ValueError('Nonfinite airside OOF estimates')
    learned.fit_airside(present, directory / 'airside_full')
    return learned.main_precision(learned.set_estimate(train, estimates))




def run(base, audit_path, output, *, threads=-1, prepare_only=False, allow_version_mismatch=False):
    from .training import environment
    from .lirf_model import fit_lirf, read_lirf, predict_lirf
    from .modeling import fit_models, predict_models
    from .postprocess import rounded
    base, output = Path(base).resolve(), Path(output).resolve()
    audit = validate_audit(base, audit_path)  # no output or fit before audit passes
    paths = [base/n for n in training_names()]
    check_months(paths)
    if threads == 0 or threads < -1:
        raise ValueError('threads must be -1 or positive')
    # Avoid every production tree and all inputs, including indirect parent writes.
    project = Path(__file__).resolve().parents[2]
    for protected in [base, project/'data', project/'models', project/'configs', project/'src']:
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('Output overlaps input or production assets')
    env = None if prepare_only else environment(allow_version_mismatch)
    before = inventory(paths, base)
    output.mkdir(parents=True, exist_ok=False)  # fresh run; never resume foreign fitted state
    shutil.copyfile(audit_path, output/'audit.json')
    source = Path(__file__).parent
    cfg = {'contract': CONTRACT, 'status': 'running', 'base_sha256': before,
           'fit_interval': [str(START), str(CUTOFF)], 'evaluation_interval': [str(CUTOFF), str(END)],
           'implementation': inventory(list(source.rglob('*.py')), source),
           'model_selection': audit['model_selection'], 'threads': threads,
           'environment': env or {'python': platform.python_version(),
               'packages': {p: version(p) for p in ['numpy', 'pandas', 'pyarrow']}},
           'interpretation': 'retrospective batch; selection independence unestablished; rules disabled'}
    write_json(output/'run_config.json', cfg)
    dataset, models = output/'dataset', output/'models'
    fit_paths, eval_paths = prepare(base, dataset)
    if inventory(paths, base) != before:
        raise ValueError('Base inputs changed during preparation')
    if not prepare_only:
        fit_lirf(fit_paths, models)  # mixture, medians and May-October residual OOF only
        train = fit_learned(learned.read_inputs(fit_paths), dataset, models)
        fit_models(train, models, threads)
        query = learned.read_inputs(eval_paths).drop(columns=TARGET)
        query = learned.apply_learned(query, dataset/'_movement/evaluation.parquet', models)
        raw = read_lirf(eval_paths, training=False)
        raw = raw.set_index(ID).loc[query.loc[query._special, ID]].reset_index()
        lp, _ = predict_lirf(raw, models)
        raw_prediction = predict_models(query, models, lp)
        prediction = pd.DataFrame({ID: query[ID].to_numpy(), 'prediction_sec': rounded(raw_prediction),
                                   'raw_prediction_sec': raw_prediction})
        atomic_table(pa.Table.from_pandas(prediction, preserve_index=False), output/'predictions.parquet')
        metrics = score(pd.read_parquet(dataset/'truth.parquet'), prediction)
        metrics['interpretation'] = cfg['interpretation']
        write_json(output/'metrics.json', metrics)
    if inventory(paths, base) != before:
        raise ValueError('Base inputs changed during run')
    cfg['status'] = 'prepared' if prepare_only else 'complete'
    cfg['artifacts_sha256'] = inventory([p for p in output.rglob('*') if p.is_file()
                                        and p != output/'run_config.json'], output)
    write_json(output/'run_config.json', cfg)  # completion last; incomplete runs never reused
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True, help='audited/rebuilt STEP 02 directory')
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='new isolated run directory')
    p.add_argument('--threads', type=int, default=-1)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--allow-version-mismatch', action='store_true')
    args = p.parse_args()
    run(args.base, args.audit, args.output, threads=args.threads, prepare_only=args.prepare_only,
        allow_version_mismatch=args.allow_version_mismatch)


if __name__ == '__main__':
    main()
