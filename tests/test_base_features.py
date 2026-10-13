"""Small end-to-end contract test. Run with python -m unittest discover -s tests -v."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from merry_llama.data import training_names
from merry_llama.preprocessor import run_preprocessing
from merry_llama.features import core as c, external as e, base as p

def month_frame(year, month):
    rows = []
    for ai, airport in enumerate(c.TIMEZONES):
        for j, offset in enumerate([0, 30, 60, 60, 120]):
            time = pd.Timestamp(year=year, month=month, day=15, hour=12, tz='UTC') + pd.Timedelta(seconds=offset)
            row = {name: None for name in c.REQUIRED}
            row.update({c.ID:f'{year}-{month}-{airport}-{j}', c.TIME:time,
                        'PHASE_mvt':'DEP' if j < 4 else 'ARR',
                        'ADEP_mvt':airport, 'ADES_mvt':'EDDF',
                        'AIRCRAFT_TYPE_mvt':'A320', 'FLIGHT_RULE_flt':'I',
                        'RUNWAY_mvt':'18', 'STAND_mvt':'A1', 'WK_TBL_CAT_flt':'H' if j % 2 == 0 else 'M',
                        'MARKET_SEGMENT_flt':'L', 'FLIGHT_TYPE_flt':'S',
                        'AIRCRAFT_OPERATOR_flt':'OP', 'CALLSIGN_flt':'OP123', 'FLIGHT_mvt':'X',
                        'TAXITIME_SEC_mvt':600.0 if year == 2025 or j == 4 else None,
                        'BLOCK_TIME_UTC_mvt':time-pd.Timedelta(minutes=10) if year==2025 or j==4 else None})
            for idx, col in enumerate(list(c.T.values())[1:]):
                row[col] = time-pd.Timedelta(minutes=idx+5)
            rows.append(row)
    return pd.DataFrame(rows, columns=c.REQUIRED)


def make_inputs(root):
    rawdir = root / 'data/raw'
    extdir = root / 'data/external'
    rawdir.mkdir(parents=True)
    extdir.mkdir(parents=True)
    frames = {}
    for month, name in enumerate(training_names(), 1):
        frames[name] = month_frame(2025, month)
    frames['ranking.parquet'] = pd.concat([month_frame(2026,1), month_frame(2026,7)], ignore_index=True)
    frames['final_ranking.parquet'] = pd.concat([month_frame(2026,m) for m in [1,2,6,7]], ignore_index=True)
    # Add coverage-start events, repeated takeoff times, absent destination/airport,
    # missing timestamp and one genuine timestamp outside the named month.
    final=frames['final_ranking.parquet']
    extras=[]
    for key, stamp in [('edge1','2026-06-01T00:00:20Z'),('edge2','2026-06-01T00:00:40Z'),
                       ('edge3','2026-06-01T01:01:00Z'),('spill','2025-12-31T23:59:59Z')]:
        row=final.iloc[0].copy(); row[c.ID]=key; row[c.TIME]=pd.Timestamp(stamp); extras.append(row)
    for key, column in [('missing_adep','ADEP_mvt'),('missing_ades','ADES_mvt'),('missing_time',c.TIME)]:
        row=final.iloc[0].copy(); row[c.ID]=key;row[column]=None;extras.append(row)
    frames['final_ranking.parquet']=pd.concat([final,pd.DataFrame(extras)],ignore_index=True)
    for name, frame in frames.items():
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), rawdir/name,
                       coerce_timestamps='us', allow_truncated_timestamps=True)
    
    observations=[]
    refrows=[]
    for airport in c.TIMEZONES:
        times=set()
        for frame in frames.values():
            times.update(frame.loc[frame.ADEP_mvt.eq(airport),c.TIME].dropna())
        for t in sorted(times):
            observations.append({'station':airport,'valid':t,'tmpf':60.0,'sknt':10.0,'vsby':10.0,
                'wxcodes':'-RA BR','metar':f'{airport} 151200Z 18010KT 9999 BKN025',
                'wx_gust_factor_kt':0.0,'wx_gust_reported_flag':0.0,'wx_ceiling_ft':2500.0,
                'wx_no_ceiling_flag':0.0})
        for date in sorted({t.strftime('%Y-%m-%d') for t in times}):
            refrows.append({'airport':airport,'date':date,**{name:123.0 for name in e.REF}})
    pd.DataFrame(observations).to_parquet(extdir/'weather.parquet', index=False)
    pd.DataFrame(refrows).to_csv(extdir/'reference.csv', index=False)
    pd.DataFrame([{'ADEP':a,'ADES':'EDDF','distance_km':100.0} for a in c.TIMEZONES]).to_csv(extdir/'distance.csv',index=False)
    return frames

class BaseFeatureContractTest(unittest.TestCase):
    def test_metadata_to_base_and_reuse(self):
        with TemporaryDirectory(prefix='prc project with spaces ') as tmp:
            root = Path(tmp)
            frames = make_inputs(root)
            rawdir = root / 'data/raw'
            raw_hashes = {name: p.digest(rawdir / name) for name in frames}
            with redirect_stdout(StringIO()):
                run_preprocessing(root)
                report = p.run_base_features(root)
            outdir = root / 'data/interim/base_features'
            self.assertEqual(len(report['files']), 14)
            self.assertEqual(report['ranking_final_overlapping_departure_ids'], 80)
            self.assertEqual(report['contexts']['final_ranking']['outside_declared_periods'], 1)
            self.assertEqual(p.validate_base_outputs(root)['contract'], p.CONTRACT)
            for name in frames:
                self.assertEqual(p.digest(rawdir / name), raw_hashes[name])
                source = pq.read_table(root / 'data/interim/metadata_filled' / name)
                output = pq.read_table(outdir / name)
                self.assertTrue(output.select(source.column_names).equals(source), name)
                df = output.to_pandas()
                self.assertTrue(df.loc[~c.dep_mask(df), p.ADDED].isna().all().all(), name)
                self.assertEqual(len(output.column_names), len(source.column_names) + 46)
            final = pd.read_parquet(outdir / 'final_ranking.parquet').set_index(c.ID)
            ranking = pd.read_parquet(outdir / 'ranking.parquet').set_index(c.ID)
            key = '2026-1-EDDF-2'
            self.assertEqual(ranking.loc[key, 'dep_count_prev_10min'], 2)
            # Same-time departures excluded; the extra missing-destination DEP is a donor.
            self.assertEqual(final.loc[key, 'dep_count_prev_10min'], 3)
            self.assertEqual(ranking.loc[key, 'hs_dep_same_runway_prev_15min'], 1)
            self.assertEqual(final.loc['2026-1-EGLL-0', 'local_hour'], 12)
            self.assertEqual(final.loc['2026-7-EGLL-0', 'local_hour'], 13)
            self.assertEqual(final.loc[key, 'aobt3_taxi_proxy_sec'], 360)
            self.assertEqual(final.loc[key, 'takeoff_minus_schedule_sec'], 300)
            self.assertEqual(final.loc[key, 'wx_precip_flag'], 1)
            self.assertEqual(final.loc[key, 'wx_fog_mist_flag'], 1)
            self.assertTrue(final.loc['missing_adep', e.REF].isna().all())
            self.assertTrue(pd.isna(final.loc['missing_ades', 'distance_km']))
            self.assertEqual(final.loc['missing_time', 'wx_metar_available_flag'], 0)
            self.assertTrue(final.loc['missing_time', e.WX].drop('wx_metar_available_flag').isna().all())
            self.assertTrue(pd.isna(final.loc['edge1', 'dep_count_prev_10min']))
            self.assertEqual(final.loc['edge3', 'dep_count_prev_60min'], 0)
            self.assertEqual(len(final), 207)
            before = {f.name: (p.digest(f), f.stat().st_mtime_ns) for f in outdir.iterdir()}
            with redirect_stdout(StringIO()):
                p.run_base_features(root)
            self.assertEqual(before, {f.name: (p.digest(f), f.stat().st_mtime_ns) for f in outdir.iterdir()})
            # Modified reference coverage invalidates reuse and fails a rebuild before writes.
            reference = root / 'data/external/reference.csv'
            ref = pd.read_csv(reference)
            ref.loc[~ref.date.eq('2026-06-15')].to_csv(reference, index=False)
            with self.assertRaisesRegex(ValueError, 'source changed'):
                p.validate_base_outputs(root)
            with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, 'final_ranking.parquet'):
                p.run_base_features(root, overwrite=True)
            self.assertEqual(before, {f.name: (p.digest(f), f.stat().st_mtime_ns) for f in outdir.iterdir()})

    def test_weather_asof_boundaries_and_distance_null(self):
        airport = 'EDDF'
        valid = pd.Timestamp('2026-01-15T12:00:00Z')
        weather = pd.DataFrame({'station': [airport], 'valid': [valid],
                                **{name: [1.0] for name in e.WX}})
        d = pd.DataFrame({'ADEP': [airport] * 4, 'ADES': ['ZZZZ'] * 4,
                          'time': [valid - pd.Timedelta(seconds=1), valid,
                                   valid + pd.Timedelta(minutes=60),
                                   valid + pd.Timedelta(minutes=60, seconds=1)]}, index=[3, 5, 7, 9])
        reference = pd.DataFrame({'airport': [airport], 'date': ['2026-01-15'],
                                  **{name: [100.0] for name in e.REF}}).set_index(['airport', 'date'])
        distance = pd.Series([], index=pd.MultiIndex.from_tuples([], names=['ADEP', 'ADES']), dtype=float)
        matched = e.match_external(d, weather, reference, distance)
        self.assertEqual(matched.wx_metar_available_flag.tolist(), [0, 1, 1, 0])
        self.assertTrue(matched.distance_km.isna().all())


if __name__ == '__main__':
    unittest.main()
