"""Movement formulas from the original v50 export (Dylan v5 lineage).

The full-2025 arrival stand baseline is retained for both historical scopes.
No LightGBM/CatBoost import or fitting is needed for these features.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .definitions import ID, TIME, AOBT, AIRPORTS

EPOCH = pd.Timestamp('1970-01-01', tz='UTC')
CAP, DAY, MIN_N, BIN = 3600, 86400, 3, 1800

def secs(col):
    return (pd.to_datetime(col, utc=True, errors='coerce') - EPOCH).dt.total_seconds().to_numpy()

def txt(col):
    return col.astype('string').str.strip().str.upper().to_numpy(dtype=object, na_value=None)

def load_movements(files):
    frames = []
    for f in files:
        names = pq.read_schema(f).names
        adep = 'ADEP_mvt' if 'ADEP_mvt' in names else 'ADEP'
        cols = [ID, 'PHASE_mvt', adep, 'ADES_mvt', TIME, 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
                'STAND_mvt', 'RUNWAY_mvt', AOBT]
        d = pq.read_table(f, columns=cols).to_pandas()
        phase = txt(d['PHASE_mvt'])
        frames.append(pd.DataFrame({
            ID: d[ID].to_numpy(), 'phase': phase,
            'airport': np.where(phase == 'DEP', txt(d[adep]), txt(d['ADES_mvt'])),
            'runway': txt(d['RUNWAY_mvt']), 'stand': txt(d['STAND_mvt']),
            'T': secs(d[TIME]), 'block': secs(d['BLOCK_TIME_UTC_mvt']), 'sched': secs(d['SCHED_TIME_UTC_mvt']),
            'A': np.where(phase == 'DEP', secs(d[AOBT]), np.nan)}))
        print(f'LOAD movements {f.name}', flush=True)
    df = pd.concat(frames, ignore_index=True)
    return df[df.airport.isin(AIRPORTS) & np.isfinite(df['T']) & df.phase.isin(['DEP', 'ARR'])]

def count_between(sorted_t, lo, hi, lo_inclusive=True):
    left = np.searchsorted(sorted_t, lo, side='left' if lo_inclusive else 'right')
    return np.maximum(np.searchsorted(sorted_t, hi, side='left') - left, 0).astype(float)

def set_col(out, pos, name, values):
    out.iloc[pos, out.columns.get_loc(name)] = values

def rolling_stat(times, values, minutes, how='median', eval_times=None):
    if eval_times is None:
        s = pd.Series(values, index=pd.to_datetime(times, unit='s'))
        r = s.rolling(f'{minutes}min', closed='left', min_periods=MIN_N)
        return (r.median() if how == 'median' else r.mean()).to_numpy()
    t = np.concatenate([times, eval_times])
    v = np.concatenate([values, np.full(len(eval_times), np.nan)])
    order = np.argsort(t, kind='stable')
    s = pd.Series(v[order], index=pd.to_datetime(t[order], unit='s'))
    r = s.rolling(f'{minutes}min', closed='left', min_periods=MIN_N)
    res = np.empty(len(t))
    res[order] = (r.median() if how == 'median' else r.mean()).to_numpy()
    return res[len(times):]

def traffic(dep, arr):
    T, A = dep['T'].to_numpy(), dep['A'].to_numpy()
    good = np.isfinite(A) & (T - A > 0) & (T - A < 3 * 3600)
    out = pd.DataFrame({ID: dep[ID].to_numpy()}, index=dep.index)
    for c in ['tf_dep_rwy_prev_5min', 'tf_dep_rwy_prev_10min', 'tf_dep_rwy_prev_20min', 'tf_sec_since_prev_dep_rwy',
              'tf_sec_to_next_dep_rwy', 'tf_queue_during_taxi_rwy', 'tf_arr_rwy_prev_10min']:
        out[c] = np.nan
    arr_rw = {r: np.sort(g['T'].to_numpy()) for r, g in arr.groupby('runway') if r is not None}
    for rw, idx in dep.groupby('runway').groups.items():
        if rw is None:
            continue
        pos = dep.index.get_indexer(idx)
        t, a, ok = T[pos], A[pos], good[pos]
        times = np.sort(t)
        for w in (5, 10, 20):
            set_col(out, pos, f'tf_dep_rwy_prev_{w}min', count_between(times, t - 60 * w, t))
        i = np.searchsorted(times, t, side='left') - 1
        set_col(out, pos, 'tf_sec_since_prev_dep_rwy',
                np.minimum(np.where(i >= 0, t - times[np.clip(i, 0, None)], np.nan), CAP))
        j = np.searchsorted(times, t, side='right')
        set_col(out, pos, 'tf_sec_to_next_dep_rwy',
                np.minimum(np.where(j < len(times), times[np.clip(j, None, len(times) - 1)] - t, np.nan), CAP))
        set_col(out, pos, 'tf_queue_during_taxi_rwy', np.where(ok, count_between(times, np.where(ok, a, t), t, False), np.nan))
        set_col(out, pos, 'tf_arr_rwy_prev_10min', count_between(arr_rw[rw], t - 600, t) if rw in arr_rw else 0.0)
    all_t = np.sort(T)
    out['tf_queue_during_taxi_apt'] = np.where(good, count_between(all_t, np.where(good, A, T), T, False), np.nan)
    vA, vT = np.sort(A[good]), np.sort(T[good])
    taxiing = np.searchsorted(vA, A, side='right') - np.searchsorted(vT, A, side='right') - 1
    out['tf_taxiing_at_pushback'] = np.where(good, np.maximum(taxiing, 0), np.nan)
    at = np.sort(arr['T'].to_numpy())
    out['tf_arr_prev_10min'] = count_between(at, T - 600, T)
    out['tf_arr_prev_20min'] = count_between(at, T - 1200, T)
    out['tf_arr_during_taxi'] = np.where(good, count_between(at, np.where(good, A, T), T, False), np.nan)
    return out

def condition(dep, arr):
    dep = dep.sort_values('T')
    T, A = dep['T'].to_numpy(), dep['A'].to_numpy()
    proxy = T - A
    proxy = np.where(np.isfinite(proxy) & (proxy > 0) & (proxy < 3 * 3600), proxy, np.nan)
    out = pd.DataFrame({ID: dep[ID].to_numpy()}, index=dep.index)
    out['cf_recent_taxi_apt_30min'] = rolling_stat(T, proxy, 30)
    out['cf_recent_taxi_rwy_30min'], out['cf_recent_taxi_rwy_60min'] = np.nan, np.nan
    rw = dep['runway'].to_numpy(dtype=object)
    for r in pd.unique(rw):
        if r is None:
            continue
        m = rw == r
        out.loc[out.index[m], 'cf_recent_taxi_rwy_30min'] = rolling_stat(T[m], proxy[m], 30)
        out.loc[out.index[m], 'cf_recent_taxi_rwy_60min'] = rolling_stat(T[m], proxy[m], 60)

    def per_block(frame):
        f = frame.dropna(subset=['runway']).assign(block=(frame['T'] // BIN).astype('int64'))
        return f.groupby('block')['runway'].agg(lambda s: ','.join(sorted(set(s))))
    prev = (dep['T'] // BIN).astype('int64') - 1
    d, a = prev.map(per_block(dep)), prev.map(per_block(arr))
    out['cf_config'] = ('D:' + d.fillna('-') + '|A:' + a.fillna('-')).where(d.notna() | a.notna())
    out['cf_n_dep_runways'] = pd.to_numeric(d.str.count(','), errors='raise').add(1).fillna(0).to_numpy()
    out['cf_n_arr_runways'] = pd.to_numeric(a.str.count(','), errors='raise').add(1).fillna(0).to_numpy()
    return out

def clean_arrivals(arr):
    a = arr.copy()
    a['taxi_in'] = a['block'] - a['T']
    a = a[np.isfinite(a['taxi_in']) & (a['taxi_in'] > 0) & (a['taxi_in'] < 2 * 3600)]
    a['delay'] = (a['block'] - a['sched']).clip(-2 * 3600, 6 * 3600)
    return a

def stand_baseline(arr_clean):
    by_stand = arr_clean.groupby(['airport', 'stand'])['taxi_in'].agg(['median', 'size'])
    return by_stand[by_stand['size'] >= 20]['median'], arr_clean.groupby('airport')['taxi_in'].median()

def on_ground(land, inb, x):
    return (np.searchsorted(np.sort(land), x, side='right') - np.searchsorted(np.sort(inb), x, side='right')).astype(float)

def arrival(dep, arr, base_stand, base_airport):
    a = clean_arrivals(arr)
    T, A = dep['T'].to_numpy(), dep['A'].to_numpy()
    out = pd.DataFrame({ID: dep[ID].to_numpy()}, index=dep.index)
    land, inb = a['T'].to_numpy(), a['block'].to_numpy()
    out['ar_taxiing_at_takeoff'] = on_ground(land, inb, T)
    good = np.isfinite(A) & (T - A > 0) & (T - A < 3 * 3600)
    out['ar_taxiing_at_pushback'] = np.where(good, on_ground(land, inb, np.where(good, A, T)), np.nan)
    usual = np.array([base_stand.get((ap, st), np.nan) for ap, st in zip(a['airport'], a['stand'])], dtype=float)
    usual = np.where(np.isnan(usual), base_airport.get(dep['airport'].iloc[0], np.nan), usual)
    out['ar_taxiin_med_30min'] = rolling_stat(inb, a['taxi_in'].to_numpy(), 30, 'median', T)
    out['ar_taxiin_excess_30min'] = rolling_stat(inb, a['taxi_in'].to_numpy() - usual, 30, 'median', T)
    out['ar_delay_mean_60min'] = rolling_stat(inb, a['delay'].to_numpy(), 60, 'mean', T)
    out['ar_delayed15_share_60min'] = rolling_stat(inb, (a['delay'].to_numpy() >= 900).astype(float), 60, 'mean', T)
    for c in ['bk_last_arr_gap', 'bk_second_arr_gap', 'bk_last_arr_delay', 'bk_n_arr_stand_3h',
              'bk_prev_dep_stand_gap', 'bk_next_arr_gap']:
        out[c] = np.nan
    arr_st = {s: g.sort_values('block') for s, g in a.groupby('stand') if s is not None}
    dep_st = {s: np.sort(g['T'].to_numpy()) for s, g in dep.groupby('stand') if s is not None}
    for st, idx in dep.groupby('stand').groups.items():
        if st is None:
            continue
        pos = dep.index.get_indexer(idx)
        t = T[pos]
        if st in arr_st:
            ib, dl = arr_st[st]['block'].to_numpy(), arr_st[st]['delay'].to_numpy()
            i = np.searchsorted(ib, t, side='left') - 1
            last = np.where(i >= 0, t - ib[np.clip(i, 0, None)], np.nan)
            second = np.where(i >= 1, t - ib[np.clip(i - 1, 0, None)], np.nan)
            set_col(out, pos, 'bk_last_arr_gap', np.where(last <= DAY, last, np.nan))
            set_col(out, pos, 'bk_second_arr_gap', np.where(second <= DAY, second, np.nan))
            set_col(out, pos, 'bk_last_arr_delay', np.where(last <= DAY, dl[np.clip(i, 0, None)], np.nan))
            set_col(out, pos, 'bk_n_arr_stand_3h', count_between(ib, t - 3 * 3600, t))
            j = np.searchsorted(ib, t, side='left')
            nxt = np.where(j < len(ib), ib[np.clip(j, None, len(ib) - 1)] - t, np.nan)
            set_col(out, pos, 'bk_next_arr_gap', np.where(nxt <= DAY, nxt, np.nan))
        else:
            set_col(out, pos, 'bk_n_arr_stand_3h', 0.0)
        ds = dep_st[st]
        k = np.searchsorted(ds, t, side='left') - 1
        prev = np.where(k >= 0, t - ds[np.clip(k, 0, None)], np.nan)
        set_col(out, pos, 'bk_prev_dep_stand_gap', np.where(prev <= DAY, prev, np.nan))
    return out
