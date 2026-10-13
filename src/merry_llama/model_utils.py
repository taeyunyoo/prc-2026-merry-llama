"""Source-compatible numeric and category preparation."""
import numpy as np
import pandas as pd
from .model_config import *

def clean(series):
    return series.astype('string').str.strip().str.upper().replace(
        ['', 'NAN', 'NONE', 'NULL', '<NA>', 'NAT'], pd.NA)

def utc(series, errors='raise'):
    # Do not divide an unqualified int64 timestamp by 1e9: its unit may be us/ms.
    return pd.to_datetime(series, utc=True, format='mixed', errors=errors).astype('datetime64[ns, UTC]')

def assert_ids(frame, label):
    if frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError(f'{label}: missing or duplicate DEP IDs')

def model_input(frame, vocab, features, cats=MODEL_CATS):
    data = {}
    for c in features:
        if c in cats:
            data[c] = pd.Categorical(frame[c], categories=vocab[c])
        else:
            values = frame[c].to_numpy(dtype=np.float32, na_value=np.nan)
            if np.isinf(values).any():
                raise ValueError(f'Float32 overflow: {c}')
            data[c] = values
    return pd.DataFrame(data, index=frame.index, columns=features)

def vocab_of(frame, cats):
    return {c: sorted(frame[c].dropna().unique().tolist()) for c in cats}

def target(frame, residual):
    return frame[TARGET].to_numpy(float) - (frame[PROXY].to_numpy(float) if residual else 0)

def cb_frame(frame, feats, cats):
    data = {c: (frame[c].astype(object).where(frame[c].notna(), 'NA').astype(str) if c in cats
                else frame[c].to_numpy(dtype=np.float32, na_value=np.nan)) for c in feats}
    return pd.DataFrame(data, index=frame.index)

def lirf_clean(s):
    return s.astype('string').str.strip().str.upper()

def lirf_utc(s):
    return pd.to_datetime(s, utc=True, errors='coerce')

def lirf_frame_x(d):
    return pd.DataFrame(dict(ICAO=lirf_clean(d.CALLSIGN_flt).str[:3], AL=lirf_clean(d.FLIGHT_mvt).str[:2], TS=d[SCHEDULE].astype(float), hour=d.local_hour.astype(float), month=d.local_month.astype(float), dow=d.local_day_of_week.astype(float), STAND=lirf_clean(d.STAND_mvt), TYPE=lirf_clean(d.AIRCRAFT_TYPE), SEG=lirf_clean(d.MARKET_SEGMENT_flt), ADES=lirf_clean(d.ADES), dist=d.distance_km.astype(float), RWY=lirf_clean(d.RUNWAY_mvt)), index=d.index)

def lirf_labels(d):
    gap = (lirf_utc(d.BLOCK_TIME_UTC_mvt) - lirf_utc(d.SCHED_TIME_UTC_mvt)).dt.total_seconds() / 60
    k = np.round(gap / 60)
    is_a = (np.abs(gap - k * 60) <= 2) & k.isin([-2, -1, 0])
    return np.where(is_a, 0, np.where(d[TARGET].to_numpy(float) > 40000, 2, 1))

def lirf_residual_x(d, base):
    x = lirf_frame_x(d)
    for c in lirf_OPS:
        x[c] = pd.to_numeric(d[c], errors='coerce').to_numpy(float)
    for c in ['base', 'pA', 'pB', 'pC']:
        x[c] = base[c].to_numpy(float)
    x['base_minus_schedule'] = x['base'] - x['TS']
    return x.replace([np.inf, -np.inf], np.nan)
