"""External feature definitions ported from v21_03_match_external_features.py."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

AIRPORTS = {'EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM'}
CORRECTED = ['wx_gust_factor_kt','wx_gust_reported_flag','wx_ceiling_ft','wx_no_ceiling_flag']
WX = [
    'wx_temperature_f', 'wx_wind_speed_kt', 'wx_gust_factor_kt',
    'wx_gust_reported_flag', 'wx_visibility_sm', 'wx_ceiling_ft',
    'wx_no_ceiling_flag', 'wx_metar_available_flag', 'wx_precip_flag',
    'wx_snow_ice_flag', 'wx_freezing_precip_flag', 'wx_thunderstorm_flag',
    'wx_fog_mist_flag',
]
REF = ['avg_predeparture_delay_sec','avg_reference_taxiout_sec','avg_additional_taxiout_sec']
ADDED = WX + REF + ['distance_km']
KEYS = ['PHASE_mvt','ADEP','ADES','MVT_TIME_UTC_mvt']
assert len(ADDED) == 17


def clean(s):
    return s.astype('string').str.strip().str.upper().replace(
        ['', 'NAN', 'NONE', 'NULL', '<NA>', 'NAT'], pd.NA)


def dates(s):
    # UTC conversion alone preserves the source precision (us/ns).
    # merge_asof requires the same timezone AND precision on both sides.
    return pd.to_datetime(
        s.mask(clean(s).isna()), utc=True, errors='raise', format='mixed'
    ).astype('datetime64[ns, UTC]')


def read_table(path, columns=None):
    with pq.ParquetFile(path) as pf:
        return pf.read(columns=columns)


def load_weather(weather_path):
    names = pq.read_schema(weather_path).names
    required = ['station','valid','tmpf','sknt','vsby','wxcodes','metar'] + CORRECTED
    missing = set(required) - set(names)
    if missing:
        raise ValueError(f'Weather source missing required columns: {missing}. See docs/STEP02.md for the prepared-resource contract.')
    w = read_table(weather_path, required).to_pandas().reset_index(drop=True)
    w['station'] = clean(w['station'])
    w['valid'] = dates(w['valid'])
    if w[['station','valid']].isna().any().any() or w.duplicated(['station','valid']).any():
        raise ValueError('Missing or duplicate weather station/valid keys')
    if not AIRPORTS.issubset(set(w.station)):
        raise ValueError(f'Weather source missing airports: {AIRPORTS-set(w.station)}')
    out = w[['station','valid']].copy()
    for target, source in [('wx_temperature_f','tmpf'),('wx_wind_speed_kt','sknt'),('wx_visibility_sm','vsby')]:
        out[target] = pd.to_numeric(w[source], errors='raise')
    for col in CORRECTED:
        out[col] = pd.to_numeric(w[col], errors='raise')
    # Preserve v20 flags: prefer wxcodes, fall back to METAR text when empty.
    parsed = clean(w['wxcodes']).fillna('')
    raw = clean(w['metar']).fillna('')
    code_text = parsed.mask(parsed.eq(''), raw)
    patterns = {
        'wx_precip_flag': r'(?:^|\s)[+-]?(?:VC)?(?:MI|PR|BC|DR|BL|SH|TS|FZ)?(?:DZ|RA|SN|SG|PL|GR|GS|UP)(?:DZ|RA|SN|SG|PL|GR|GS|UP)*(?=\s|$)',
        'wx_snow_ice_flag': r'(?:^|\s)[+-]?(?:VC)?(?:MI|PR|BC|DR|BL|SH|TS|FZ)?(?:DZ|RA|GR|GS|UP)*(?:SN|SG|PL|IC)(?:DZ|RA|SN|SG|PL|IC|GR|GS|UP)*(?=\s|$)',
        'wx_freezing_precip_flag': r'(?:^|\s)[+-]?(?:VC)?FZ(?:RA|DZ)(?=\s|$)',
        'wx_thunderstorm_flag': r'(?:^|\s)[+-]?(?:VC)?TS(?:DZ|RA|SN|SG|PL|GR|GS|UP)*(?=\s|$)',
        'wx_fog_mist_flag': r'(?:^|\s)[+-]?(?:VC)?(?:MI|PR|BC|FZ)?(?:FG|BR)(?=\s|$)',
    }
    for col, pattern in patterns.items():
        out[col] = code_text.str.contains(pattern, regex=True, na=False).astype(float)
    out['wx_metar_available_flag'] = 1.0
    for col in WX:
        out[col] = out[col].astype('float64')
    if np.isinf(out[WX].to_numpy()).any():
        raise ValueError('Infinite weather feature value')
    for col in [c for c in WX if c.endswith('_flag')]:
        if not out[col].dropna().isin([0,1]).all():
            raise ValueError(f'Invalid weather flag: {col}')
    if out['wx_gust_factor_kt'].dropna().lt(0).any():
        raise ValueError('Negative gust factor')
    if not out.loc[out.wx_no_ceiling_flag.eq(1), 'wx_ceiling_ft'].eq(60000).all():
        raise ValueError('no_ceiling_flag=1 requires ceiling=60000')
    return out[['station','valid']+WX].sort_values('valid', kind='stable')


def load_reference(reference_path):
    r = pd.read_csv(reference_path, dtype='string', encoding='utf-8-sig')
    absent = set(['airport','date']+REF)-set(r.columns)
    if absent:
        raise ValueError(f'Reference source missing columns: {absent}')
    r['airport'] = clean(r.airport)
    r['date'] = pd.to_datetime(r.date, format='%Y-%m-%d', errors='raise').dt.strftime('%Y-%m-%d')
    if r[['airport','date']].isna().any().any() or r.duplicated(['airport','date']).any():
        raise ValueError('Missing or duplicate reference keys')
    for col in REF:
        r[col] = pd.to_numeric(r[col], errors='raise').astype(float)
    if not np.isfinite(r[REF].to_numpy()).all():
        raise ValueError('Missing or infinite reference value')
    return r.set_index(['airport','date'])[REF]


def load_distance(distance_path):
    r = pd.read_csv(distance_path, dtype='string', encoding='utf-8-sig')
    if not {'ADEP','ADES','distance_km'}.issubset(r.columns):
        raise ValueError('Distance CSV requires ADEP/ADES/distance_km')
    for col in ['ADEP','ADES']:
        r[col] = clean(r[col])
    # Exclude missing codes and ZZZZ from matching; leave source CSV unchanged.
    good = r[['ADEP','ADES']].notna().all(axis=1) & ~r[['ADEP','ADES']].isin(['ZZZZ']).any(axis=1)
    r = r.loc[good].copy()
    if r.duplicated(['ADEP','ADES']).any():
        raise ValueError('Duplicate distance route keys')
    r['distance_km'] = pd.to_numeric(clean(r.distance_km), errors='raise').astype(float)
    values = r.distance_km.dropna()
    if not np.isfinite(values).all() or not values.between(0,20016).all():
        raise ValueError('Invalid distance value')
    return r.set_index(['ADEP','ADES'])['distance_km']


def reference_values(d, reference):
    key = pd.MultiIndex.from_arrays([d.ADEP, d.time.dt.strftime('%Y-%m-%d')], names=['airport','date'])
    match = reference.reindex(key)
    if match.isna().any().any():
        raise ValueError('Unmatched UTC reference keys:\n'+str(match.loc[match.isna().any(axis=1)].head(20)))
    return match.to_numpy(dtype=float)


def match_external(d, weather, reference, distance):
    f = pd.DataFrame(np.nan, index=d.index, columns=ADDED)
    for airport, group in d.groupby('ADEP', sort=False):
        left = pd.DataFrame({'position':group.index, 'time':group.time.to_numpy()}).sort_values('time', kind='stable')
        right = weather.loc[weather.station.eq(airport), ['valid']+WX].copy()
        left['time'] = dates(left['time'])
        right['valid'] = dates(right['valid'])
        right = right.sort_values('valid', kind='stable')
        matched = pd.merge_asof(left, right, left_on='time', right_on='valid',
                                direction='backward', tolerance=pd.Timedelta(minutes=60),
                                allow_exact_matches=True)
        available = matched.valid.notna()
        if available.any():
            age = matched.loc[available,'time'] - matched.loc[available,'valid']
            if not age.between(pd.Timedelta(0),pd.Timedelta(minutes=60)).all():
                raise RuntimeError('Matched a future weather observation or one older than 60 minutes')
        matched['wx_metar_available_flag'] = available.astype(float)
        f.loc[matched.position.to_numpy(), WX] = matched[WX].to_numpy(dtype=float)
    f.loc[:,REF] = reference_values(d, reference)
    key = pd.MultiIndex.from_arrays([d.ADEP,d.ADES], names=['ADEP','ADES'])
    f['distance_km'] = distance.reindex(key).to_numpy(dtype=float)
    if np.isinf(f.to_numpy()).any():
        raise ValueError('Infinite matched feature value')
    unmatched = f.wx_metar_available_flag.eq(0)
    if not f.loc[unmatched, [c for c in WX if c!='wx_metar_available_flag']].isna().all().all():
        raise RuntimeError('Unexpected weather values on unmatched METAR rows')
    return f

