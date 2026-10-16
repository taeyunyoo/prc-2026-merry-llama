"""Exact-coverage metrics for the fixed November–December 2025 holdout."""
import numpy as np
import pandas as pd
from .model_config import ID, TIME, TARGET, AIRPORTS

CUTOFF = pd.Timestamp('2025-11-01', tz='UTC')
END = pd.Timestamp('2026-01-01', tz='UTC')


def assert_ids(frame, label):
    ids = frame[ID].astype('string').str.strip()
    if ids.isna().any() or ids.eq('').any() or frame[ID].duplicated().any():
        raise ValueError(f'{label}: missing or duplicate departure IDs')


def score(truth, predictions):
    """Exact ID coverage before exclusions; count invalid labels explicitly."""
    assert_ids(truth, 'truth')
    assert_ids(predictions, 'predictions')
    if set(truth[ID]) != set(predictions[ID]):
        raise ValueError('Prediction IDs must exactly cover holdout departures')
    if not truth.ADEP.isin(AIRPORTS).all():
        raise ValueError('Unknown/missing holdout airport')
    t = pd.to_datetime(truth[TIME], utc=True, errors='raise', format='mixed')
    if t.isna().any() or not ((t >= CUTOFF) & (t < END)).all():
        raise ValueError('Truth outside evaluation interval')
    d = truth.merge(predictions[[ID, 'prediction_sec']], on=ID, validate='one_to_one')
    p = pd.to_numeric(d.prediction_sec, errors='raise').to_numpy(float)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite prediction; no silent pair dropping')
    y = pd.to_numeric(d[TARGET], errors='raise').to_numpy(float)
    valid = np.isfinite(y) & (y >= 0)
    def metric(mask):
        selected = mask & valid
        n = int(selected.sum())
        return {'rows': int(mask.sum()), 'evaluated_pairs': n,
                'excluded_invalid_targets': int((mask & ~valid).sum()),
                'rmse_sec': float(np.sqrt(np.mean((p[selected]-y[selected])**2))) if n else None}
    if not valid.any():
        raise ValueError('No valid evaluation labels')
    return {'overall': metric(np.ones(len(d), bool)),
            'airports': {a: metric(d.ADEP.eq(a).to_numpy()) for a in AIRPORTS}}
