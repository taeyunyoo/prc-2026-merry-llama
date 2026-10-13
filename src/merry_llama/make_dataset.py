"""Stage 03: shared departure datasets with frozen v42/v50 operational features.

Inputs are the verified, unmodified stage-02 files. Learned taxi-out medians and
airside estimates are fitted by the training stage, not by this module.
"""
from __future__ import annotations
import gc
import json
import os
import platform
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data import ProjectPaths, dataset_names, training_names
from .features import arrival, movement, definitions, state
from .features.core import clean, dep_mask, read_table, stamp
from .features.external import dates
from .features.base import digest, validate_base_outputs, write_json
from .features.definitions import (
    ID, TIME, TARGET, AOBT, PROXY, AIRPORTS, ADDED, ARRIVAL_FEATURES,
    MOVEMENT_FEATURES, CATEGORICAL_ADDED, DEFERRED,
)

CONTRACT = 'prc2026.model_dataset.v1'
MARKER = 'step03_complete.json'
DEP_COLUMNS = [ID, TIME, TARGET, AOBT, PROXY, 'PHASE_mvt', 'ADEP',
               'SCHED_TIME_UTC_mvt', 'RUNWAY_mvt', 'STAND_mvt',
               'AIRCRAFT_TYPE', 'FLIGHT_mvt']
ARR_COLUMNS = [ID, 'PHASE_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'AIRCRAFT_TYPE_mvt',
               'RUNWAY_mvt', 'STAND_mvt', TIME, 'BLOCK_TIME_UTC_mvt',
               'SCHED_TIME_UTC_mvt', TARGET]


def implementation_fingerprint():
    modules = [Path(__file__), *(Path(m.__file__) for m in (arrival, movement, definitions, state))]
    return {p.name: digest(p) for p in modules}


def atomic_table(table, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f'.{destination.name}.{uuid4().hex}.tmp')
    try:
        pq.write_table(table, temp, compression='snappy')
        if not read_table(temp).equals(table):
            raise RuntimeError(f'Parquet round-trip mismatch: {destination}')
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def check_ids(values, label):
    if clean(values).isna().any() or values.duplicated().any():
        raise ValueError(f'{label}: missing or duplicate departure IDs')


def read_departures(paths):
    frames = []
    for path in paths:
        d = read_table(path, DEP_COLUMNS).to_pandas()
        d = d.loc[dep_mask(d)].copy().reset_index(drop=True)
        check_ids(d[ID], path.name)
        d[TIME] = dates(d[TIME])
        if d[TIME].isna().any():
            raise ValueError(f'{path.name}: missing departure movement time')
        for name in ['ADEP', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE', 'FLIGHT_mvt']:
            d[name] = clean(d[name])
        if not d.ADEP.isin(AIRPORTS).all():
            raise ValueError(f'{path.name}: missing or unknown departure airport')
        d[PROXY] = pd.to_numeric(d[PROXY], errors='raise').astype(float)
        if not clean(d[AOBT]).isna().eq(d[PROXY].isna()).all():
            raise ValueError(f'{path.name}: AOBT/proxy missingness mismatch')
        evaluation = path.name in {'ranking.parquet', 'final_ranking.parquet'}
        d['_ranking'] = evaluation
        d['_month'] = d[TIME].dt.month
        if evaluation:
            d[TARGET] = np.nan
        else:
            y = pd.to_numeric(d[TARGET], errors='raise').to_numpy(float)
            d = d.loc[np.isfinite(y) & (y >= 0)].copy()
        frames.append(d)
    result = pd.concat(frames, ignore_index=True)
    check_ids(result[ID], 'Departure feature context')
    return result


def read_arrivals(paths):
    frames = []
    for path in paths:
        d = read_table(path, ARR_COLUMNS).to_pandas()
        frames.append(d.loc[clean(d.PHASE_mvt).eq('ARR').fillna(False)])
    a = pd.concat(frames, ignore_index=True).drop(columns='PHASE_mvt').rename(columns={
        'ADES_mvt': 'APT', 'FLIGHT_mvt': 'FLIGHT', 'AIRCRAFT_TYPE_mvt': 'TYPE',
        'RUNWAY_mvt': 'RUNWAY', 'STAND_mvt': 'STAND', TIME: 'LAND',
        'BLOCK_TIME_UTC_mvt': 'INBLOCK', 'SCHED_TIME_UTC_mvt': 'SCHED', TARGET: 'TAXI_IN'})
    for name in ['APT', 'FLIGHT', 'TYPE', 'RUNWAY', 'STAND']:
        a[name] = clean(a[name])
    for name in ['LAND', 'INBLOCK', 'SCHED']:
        a[name] = pd.to_datetime(a[name], utc=True, errors='coerce', format='mixed')
    a['TAXI_IN'] = pd.to_numeric(a.TAXI_IN, errors='coerce')
    if a.empty:
        raise ValueError('No ARR context. Stage 02 must retain the mixed movement rows.')
    return a


def movement_features(paths, references=None):
    """Build one independent movement context; never mix ranking and final."""
    mv = movement.load_movements(paths)
    dep_ids = mv.loc[mv.phase.eq('DEP'), ID]
    check_ids(dep_ids, 'Movement context')
    if references is None:
        stand, airport = movement.stand_baseline(movement.clean_arrivals(mv[mv.phase.eq('ARR')]))
        references = {'stand': stand, 'airport': airport}
    parts = []
    required = [c for c in MOVEMENT_FEATURES if c != 'cf_proxy_minus_recent']
    for ap in AIRPORTS:
        frame = mv.loc[mv.airport.eq(ap)]
        dep, arr = frame.loc[frame.phase.eq('DEP')], frame.loc[frame.phase.eq('ARR')]
        print(f'  Movement {ap}: DEP={len(dep):,}, ARR={len(arr):,}', flush=True)
        if dep.empty:
            continue
        tables = [movement.traffic(dep, arr), movement.condition(dep, arr),
                  movement.arrival(dep, arr, references['stand'], references['airport'])]
        merged = tables[0]
        for table in tables[1:]:
            merged = merged.merge(table, on=ID, how='left', validate='one_to_one', sort=False)
        merged['cf_config'] = clean(merged['cf_config'])
        parts.append(merged[[ID] + required])
    if not parts:
        raise ValueError('No eligible departure movements')
    result = pd.concat(parts, ignore_index=True)
    check_ids(result[ID], 'Movement features')
    for name in required:
        if name != 'cf_config':
            # Cast when exporting; preserve float64 arithmetic until then.
            result[name] = pd.to_numeric(result[name], errors='raise')
    return result, references


def export_dataset(path, enriched, movement_data, destination, scope):
    """Keep every DEP row; invalid training labels remain and are documented."""
    source = read_table(path)
    is_dep = dep_mask(source.select(['PHASE_mvt']).to_pandas()).to_numpy(bool)
    source = source.filter(pa.array(is_dep))
    ids = source[ID].to_pandas().reset_index(drop=True)
    check_ids(ids, path.name)
    v42 = enriched.set_index(ID)
    eligible = ids.isin(v42.index)
    y = pd.to_numeric(source[TARGET].to_pandas(), errors='raise').to_numpy(float)
    invalid_target = (~np.isfinite(y) | (y < 0)) if path.name.startswith('training_') else np.zeros(len(ids), bool)
    if (~eligible.to_numpy() & ~invalid_target).any():
        raise ValueError(f'{path.name}: missing operational features for valid/evaluation departures')
    values = v42.reindex(ids)[ARRIVAL_FEATURES].reset_index(drop=True)
    mv = movement_data.set_index(ID)
    # The source movement builder filters missing/unknown raw movement airports.
    # Its original left join retains those departures with null movement features.
    unmatched_movements = int((eligible & ~ids.isin(mv.index)).sum())
    movement_values = mv.reindex(ids).reset_index(drop=True)
    proxy = pd.to_numeric(source[PROXY].to_pandas(), errors='raise').to_numpy(float)
    movement_values['cf_proxy_minus_recent'] = proxy - movement_values.cf_recent_taxi_rwy_60min.to_numpy(float)
    for name in MOVEMENT_FEATURES:
        values[name] = movement_values[name]
    values.loc[~eligible.to_numpy(), :] = np.nan
    table = source
    for name in ADDED:
        dtype = pa.string() if name in CATEGORICAL_ADDED else pa.float32()
        if name not in CATEGORICAL_ADDED:
            v = values[name].to_numpy(dtype=np.float32, na_value=np.nan)
            if np.isinf(v).any():
                raise ValueError(f'{scope}/{path.name}: infinite feature {name}')
            column = pa.array(v, type=dtype, from_pandas=True)
        else:
            column = pa.array(values[name], type=dtype, from_pandas=True)
        table = table.append_column(name, column)
    atomic_table(table, destination)
    missing = [{'scope': scope, 'file': path.name, 'feature': name,
                'eligible_departures': int(eligible.sum()),
                'missing_eligible': int(values.loc[eligible.to_numpy(), name].isna().sum())}
               for name in ADDED]
    record = {'path': f'{scope}/{path.name}', 'rows': source.num_rows,
              'eligible_departures': int(eligible.sum()),
              'invalid_training_targets': int(invalid_target.sum()),
              'movement_unmatched': unmatched_movements,
              'added_columns': len(ADDED), 'sha256': digest(destination)}
    print(f"  Saved {record['path']}: DEP={source.num_rows:,}, +{len(ADDED)} features, "
          f"invalid training targets={record['invalid_training_targets']:,}", flush=True)
    return record, missing


def validate_model_data(root=None):
    project = ProjectPaths.from_root(root)
    validate_base_outputs(project.root)
    marker = project.model_data_dir / MARKER
    report = json.loads(marker.read_text(encoding='utf-8'))
    if (report.get('contract') != CONTRACT or report.get('status') != 'complete'
            or report.get('added_features') != ADDED
            or report.get('implementation') != implementation_fingerprint()):
        raise ValueError('Model-data completion contract or implementation differs')
    base_marker = project.base_features_dir / 'step02_complete.json'
    if digest(base_marker) != report['base_completion_sha256']:
        raise ValueError('Upstream base-feature completion record changed')
    expected = {f'full/{n}' for n in dataset_names()} | {f'validation/{n}' for n in training_names()}
    if len(report['files']) != 26 or {r['path'] for r in report['files']} != expected:
        raise ValueError('Model-data inventory must contain 26 output datasets')
    for record in report['files'] + report['artifacts']:
        path = project.model_data_dir / record['path']
        if not path.is_file() or digest(path) != record['sha256']:
            raise ValueError(f'Model-data output changed or missing: {path}')
    return report


def run_model_data(root=None, *, overwrite=False):
    project = ProjectPaths.from_root(root)
    output = project.model_data_dir
    print('Verifying completed STEP 02 inputs...', flush=True)
    validate_base_outputs(project.root)
    paths = [project.base_features_dir / name for name in dataset_names()]
    marker = output / MARKER
    if output.exists() and any(output.iterdir()) and not overwrite:
        if not marker.is_file():
            raise FileExistsError(f'Incomplete/unverified output: {output}. Use --overwrite to rebuild STEP 03 only.')
        report = validate_model_data(project.root)
        print('UNCHANGED: verified STEP 03 reused; no files rewritten.', flush=True)
        print(f'COMPLETE: operational model datasets (26 files, {len(ADDED)} added columns).', flush=True)
        return report
    required = set(DEP_COLUMNS + ARR_COLUMNS + ['ADEP_mvt'])
    for path in paths:
        names = set(pq.read_schema(path).names)
        if required - names:
            raise ValueError(f'{path.name}: missing required context columns {sorted(required - names)}')
        if set(ADDED + DEFERRED) & names:
            raise ValueError(f'{path.name}: downstream features already present in base inputs')
    stamps = {p: stamp(p) for p in paths}
    base_marker = project.base_features_dir / 'step02_complete.json'
    base_hash = digest(base_marker)
    output.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    train_paths, rank_path, final_path = paths[:12], paths[12], paths[13]
    records, missing = [], []
    print('Building movement context: training (12 months)', flush=True)
    train_mv, movement_refs = movement_features(train_paths)
    state.save_references(movement_refs, output / '_references/movement')
    # Persist only reusable calculated movement columns, never overwrite upstream data.
    cache = output / '_movement'
    atomic_table(pa.Table.from_pandas(train_mv, preserve_index=False), cache / 'train.parquet')
    del train_mv
    gc.collect()
    print('Building movement context: ranking', flush=True)
    rank_mv, _ = movement_features([rank_path], movement_refs)
    atomic_table(pa.Table.from_pandas(rank_mv, preserve_index=False), cache / 'ranking.parquet')
    del rank_mv
    gc.collect()
    print('Building movement context: final (independent of ranking)', flush=True)
    final_mv, _ = movement_features([final_path], movement_refs)
    atomic_table(pa.Table.from_pandas(final_mv, preserve_index=False), cache / 'final_ranking.parquet')
    del final_mv, movement_refs
    gc.collect()

    for scope, context, query_paths in [
        ('full', 'ranking', train_paths + [rank_path]),
        ('validation', 'ranking', train_paths),
        ('full', 'final_ranking', train_paths + [final_path]),
    ]:
        label = 'final' if context == 'final_ranking' else scope
        print(f'Building v42 arrival relationships: {label}', flush=True)
        dep, arr = read_departures(query_paths), read_arrivals(query_paths)
        references = state.load_references(output / '_references/full') if label == 'final' else None
        enriched, refs = arrival.add_arrival_features(
            dep, arr, scope=scope, context=context, references=references)
        if label != 'final':
            state.save_references(refs, output / '_references' / scope)
        enriched = enriched[[ID] + ARRIVAL_FEATURES].copy()
        del dep, arr, refs, references
        gc.collect()
        to_write = [final_path] if label == 'final' else query_paths
        movement_name = None
        mv = None
        for path in to_write:
            desired = 'train' if path.name.startswith('training_') else path.stem
            if desired != movement_name:
                mv = pd.read_parquet(cache / f'{desired}.parquet')
                movement_name = desired
            record, coverage = export_dataset(path, enriched, mv, output / scope / path.name, scope)
            records.append(record)
            missing.extend(coverage)
            pd.DataFrame(records).drop(columns='sha256').to_csv(output / 'step03_file_summary.csv', index=False)
        del enriched, mv
        gc.collect()
    if any(stamp(p) != old for p, old in stamps.items()) or digest(base_marker) != base_hash:
        raise RuntimeError('Base-feature inputs changed during STEP 03')
    pd.DataFrame(missing).to_csv(output / 'step03_missingness.csv', index=False)
    artifacts = [{'path': p.relative_to(output).as_posix(), 'sha256': digest(p)}
                 for folder in [output / '_references', cache]
                 for p in sorted(folder.rglob('*')) if p.is_file()]
    report = {
        'contract': CONTRACT, 'status': 'complete', 'stage': 3,
        'added_features': ADDED, 'deferred_to_training': DEFERRED,
        'implementation': implementation_fingerprint(), 'base_completion_sha256': base_hash,
        'files': records, 'artifacts': artifacts,
        'reference_policy': {
            'v42_full': 'Original timestamp cutoff before 2026-01-01; full training turnaround references',
            'v42_validation': 'Original arrival cutoff before 2025-11-01; departure months <=10',
            'v50_movement': 'All 2025 arrival stand baselines for both full and validation (source policy)',
            'final': 'Saved full references; independent training+final arrival context and final-only movement context',
        },
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'pyarrow': pa.__version__},
    }
    write_json(marker, report)
    print('\n' + pd.DataFrame(records).drop(columns='sha256').to_string(index=False), flush=True)
    print(f'Output: {output}', flush=True)
    print(f'COMPLETE: operational model datasets (26 files, {len(ADDED)} added columns).', flush=True)
    return report
