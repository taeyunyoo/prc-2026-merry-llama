"""Core feature definitions ported from v21_02_build_core_features.py.

The stage-2 runner supplies dataset-specific contexts and output paths.
"""
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TIMEZONES = {
    'EDDF': 'Europe/Berlin', 'EDDM': 'Europe/Berlin',
    'EGLL': 'Europe/London', 'EHAM': 'Europe/Amsterdam',
    'LEBL': 'Europe/Madrid', 'LEMD': 'Europe/Madrid',
    'LFPG': 'Europe/Paris', 'LIRF': 'Europe/Rome',
    'LSZH': 'Europe/Zurich', 'LTFM': 'Europe/Istanbul',
}
ID, TIME = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt'
PAIRS = {
    'ADEP': ('ADEP_mvt', 'ADEP_flt'),
    'ADES': ('ADES_mvt', 'ADES_flt'),
    'AIRCRAFT_TYPE': ('AIRCRAFT_TYPE_mvt', 'AIRCRAFT_TYPE_flt'),
    'FLIGHT_RULE': ('FLIGHT_RULE_flt', 'FLIGHT_RULE_mvt'),
}
DIRECT = [
    'RUNWAY_mvt', 'STAND_mvt', 'MARKET_SEGMENT_flt', 'FLIGHT_TYPE_flt',
    'WK_TBL_CAT_flt', 'AIRCRAFT_OPERATOR_flt', 'FLIGHT_mvt', 'CALLSIGN_flt',
]
T = dict(takeoff=TIME, schedule='SCHED_TIME_UTC_mvt',
         aobt3='AOBT_3_flt', eobt1='EOBT_1_flt',
         iobt='IOBT_flt', lobt='LOBT_flt')
DIFFERENCES = {
    'takeoff_minus_schedule_sec': ('takeoff', 'schedule'),
    'aobt3_taxi_proxy_sec': ('takeoff', 'aobt3'),
    'takeoff_minus_eobt1_sec': ('takeoff', 'eobt1'),
    'takeoff_minus_iobt_sec': ('takeoff', 'iobt'),
    'takeoff_minus_lobt_sec': ('takeoff', 'lobt'),
    'aobt3_minus_schedule_sec': ('aobt3', 'schedule'),
    'eobt1_minus_schedule_sec': ('eobt1', 'schedule'),
    'iobt_minus_schedule_sec': ('iobt', 'schedule'),
    'lobt_minus_schedule_sec': ('lobt', 'schedule'),
    'aobt3_minus_eobt1_sec': ('aobt3', 'eobt1'),
    'aobt3_minus_iobt_sec': ('aobt3', 'iobt'),
    'aobt3_minus_lobt_sec': ('aobt3', 'lobt'),
    'eobt1_minus_iobt_sec': ('eobt1', 'iobt'),
    'lobt_minus_eobt1_sec': ('lobt', 'eobt1'),
    'lobt_minus_iobt_sec': ('lobt', 'iobt'),
}
LOCAL = ['local_month', 'local_day_of_week', 'local_hour',
         'local_weekend_flag', 'local_hour_sin', 'local_hour_cos']
COUNTS = [f'dep_count_prev_{m}min' for m in (10, 30, 60)]
WAKE = 'hs_dep_same_runway_prev_15min'
ADDED = list(PAIRS) + LOCAL + list(DIFFERENCES) + COUNTS + [WAKE]
REQUIRED = list(dict.fromkeys(
    [ID, 'PHASE_mvt', 'TAXITIME_SEC_mvt', 'BLOCK_TIME_UTC_mvt']
    + list(T.values()) + DIRECT + [c for p in PAIRS.values() for c in p]
))
assert len(ADDED) == 29


def clean(s):
    return s.astype('string').str.strip().str.upper().replace(
        ['', 'NAN', 'NONE', 'NULL', '<NA>', 'NAT'], pd.NA)


def utc(s):
    return pd.to_datetime(s.mask(clean(s).isna()), utc=True,
                          format='mixed', errors='raise')


def stamp(p):
    s = p.stat()
    return s.st_size, s.st_mtime_ns


def read_table(p, columns=None):
    with pq.ParquetFile(p) as f:
        return f.read(columns=columns)


def dep_mask(d):
    return clean(d['PHASE_mvt']).eq('DEP').fillna(False)


def airport_counts(c, coverage):
    result = pd.DataFrame(np.nan, index=c.index, columns=COUNTS)
    for _, positions in c.groupby('ADEP', sort=False).indices.items():
        dates = c.iloc[positions][TIME].astype('datetime64[ns, UTC]').array.asi8
        donors = np.sort(dates)
        upper = np.searchsorted(donors, dates, side='left')
        for minutes, name in zip((10, 30, 60), COUNTS):
            window = minutes * 60 * 10**9
            values = (upper - np.searchsorted(donors, dates-window, side='left')).astype(float)
            covered = np.zeros(len(dates), dtype=bool)
            for start, end in coverage:
                covered |= (dates-window >= start.value) & (dates < end.value)
            values[~covered] = np.nan
            result.loc[positions, name] = values
    result.index = pd.Index(c[ID], name=ID)
    return result


def wake_counts(c):
    # Legacy v20 donor universe is supplied by the runner; no targets are used.
    # Window [time - 15 minutes, time); exclude ties; no coverage-edge masking.
    result = np.full(len(c), np.nan)
    times = c[TIME].astype('datetime64[ns, UTC]').array.asi8
    heavy = c['heavy'].to_numpy(dtype=bool)
    for _, pos in c.groupby(['ADEP', 'runway'], sort=False, dropna=True).indices.items():
        donors = np.sort(times[pos[heavy[pos]]])
        result[pos] = (np.searchsorted(donors, times[pos], side='left')
                       - np.searchsorted(donors, times[pos]-15*60*10**9, side='left'))
    return pd.Series(result, index=pd.Index(c[ID], name=ID), name=WAKE)


def derive(d, counts, wake):
    f = pd.DataFrame(index=d.index)
    for name, (preferred, fallback) in PAIRS.items():
        f[name] = clean(d[preferred]).fillna(clean(d[fallback]))
    dates = {key: utc(d[col]) for key, col in T.items()}
    for name, (end, start) in DIFFERENCES.items():
        f[name] = (dates[end] - dates[start]).dt.total_seconds()
    for name in LOCAL:
        f[name] = np.nan
    for airport, zone in TIMEZONES.items():
        mask = f['ADEP'].eq(airport).fillna(False)
        if not mask.any():
            continue
        local = dates['takeoff'].loc[mask].dt.tz_convert(zone)
        f.loc[mask, 'local_month'] = local.dt.month.to_numpy()
        f.loc[mask, 'local_day_of_week'] = local.dt.dayofweek.to_numpy()
        f.loc[mask, 'local_hour'] = local.dt.hour.to_numpy()
        f.loc[mask, 'local_weekend_flag'] = local.dt.dayofweek.ge(5).astype(float).to_numpy()
        minute = local.dt.hour*60 + local.dt.minute + local.dt.second/60.0
        angle = 2*np.pi*minute/1440.0
        f.loc[mask, 'local_hour_sin'] = np.sin(angle).to_numpy()
        f.loc[mask, 'local_hour_cos'] = np.cos(angle).to_numpy()
    for name in COUNTS:
        f[name] = d[ID].map(counts[name])
    f[WAKE] = d[ID].map(wake)
    if np.isinf(f[[c for c in ADDED if c not in PAIRS]].to_numpy(dtype=float)).any():
        raise ValueError('Infinite derived feature value')
    return f[ADDED]

