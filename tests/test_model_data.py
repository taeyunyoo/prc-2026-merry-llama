"""STEP 03 integration and final-period boundary tests, using synthetic inputs."""
from contextlib import redirect_stdout
from io import StringIO
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from merry_llama.preprocessor import run_preprocessing
from merry_llama.features.base import run_base_features, digest
from merry_llama.features import arrival, state
from merry_llama.features.definitions import ADDED, DEFERRED, MOVEMENT_FEATURES, ID
from merry_llama.make_dataset import run_model_data, validate_model_data, read_departures, read_arrivals
from pathlib import Path
import pandas as pd
from test_base_features import make_inputs

def rich_inputs(root):
    make_inputs(root)
    for path in (root/'data/raw').glob('*.parquet'):
        original=pd.read_parquet(path)
        original=original[~original.MVT_ID_mvt.str.startswith(('edge','spill','missing_'))]
        chunks=[]
        for dayoff in (0,1,2,3):
            day=original.copy()
            for name in ['MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt','EOBT_1_flt','IOBT_flt','LOBT_flt','BLOCK_TIME_UTC_mvt']:
                day[name]=pd.to_datetime(day[name],utc=True)+pd.Timedelta(days=dayoff)
            day.MVT_ID_mvt=day.MVT_ID_mvt+f'-day{dayoff}'
            # Keep ties, missing AOBT, missing runway/stand, and invalid training labels.
            d=day[day.PHASE_mvt.eq('DEP')].copy()
            d.loc[d.MVT_ID_mvt.str.contains('-1-day'), 'AOBT_3_flt']=pd.NaT
            d.loc[d.MVT_ID_mvt.str.contains('-3-day3'), 'STAND_mvt']=None
            d.loc[d.MVT_ID_mvt.str.contains('-3-day2'), 'RUNWAY_mvt']=None
            if path.name.startswith('training_'):
                d.loc[d.MVT_ID_mvt.str.contains('-3-day1'),'TAXITIME_SEC_mvt']=-1.0
            chunks.append(d)
            for offset in [5000,4000,3600,2200,1800,600,0,-300]:
                a=day[day.PHASE_mvt.eq('ARR')].copy()
                a['ADES_mvt']=a.ADEP_mvt
                a['MVT_TIME_UTC_mvt']=a.MVT_TIME_UTC_mvt-pd.Timedelta(seconds=120+offset)
                a['TAXITIME_SEC_mvt']=a.MVT_TIME_UTC_mvt.dt.month.astype(float)*100+200
                a['BLOCK_TIME_UTC_mvt']=a.MVT_TIME_UTC_mvt+pd.to_timedelta(a.TAXITIME_SEC_mvt,unit='s')
                a['SCHED_TIME_UTC_mvt']=a.BLOCK_TIME_UTC_mvt-pd.Timedelta(seconds=120)
                a['MVT_ID_mvt']=a.MVT_ID_mvt+f'-arr{offset}'
                a.loc[a.MVT_TIME_UTC_mvt.dt.month.eq(7),'RUNWAY_mvt']='36'
                chunks.append(a)
        frame=pd.concat(chunks,ignore_index=True)
        frame.to_parquet(path,index=False)
    # Resources cover shifted fixture dates; retain sufficient reference coverage.
    ext=root/'data/external'
    ref=pd.read_csv(ext/'reference.csv')
    originals=ref.copy()
    for days in (1,2,3):
        shifted=originals.copy();shifted['date']=(pd.to_datetime(shifted.date)+pd.Timedelta(days=days)).dt.strftime('%Y-%m-%d');ref=pd.concat([ref,shifted])
    ref.drop_duplicates(['airport','date']).to_csv(ext/'reference.csv',index=False)


class ModelDatasetContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory(prefix='prc step03 spaced root ')
        cls.root = Path(cls.temp.name)
        with redirect_stdout(StringIO()):
            rich_inputs(cls.root)
            run_preprocessing(cls.root)
            run_base_features(cls.root)
        cls.base = cls.root / 'data/interim/base_features'
        cls.before = {p.name: digest(p) for p in cls.base.iterdir() if p.is_file()}
        with redirect_stdout(StringIO()):
            cls.report = run_model_data(cls.root)
        cls.output = cls.root / 'data/model_dataset'

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_output_contract_and_reuse(self):
        self.assertEqual(len(self.report['files']), 26)
        self.assertEqual(len(ADDED), 69)
        self.assertEqual(self.before, {p.name: digest(p) for p in self.base.iterdir() if p.is_file()})
        for record in self.report['files']:
            name = Path(record['path']).name
            source = pq.read_table(self.base / name)
            phase = source['PHASE_mvt'].to_pandas().eq('DEP').to_numpy()
            source = source.filter(pa.array(phase))
            output = pq.read_table(self.output / record['path'])
            self.assertTrue(output.select(source.column_names).equals(source), record['path'])
            self.assertEqual(output.num_rows, record['rows'])
            self.assertFalse(set(DEFERRED).intersection(output.column_names))
            d = output.to_pandas()
            if name.startswith('training_'):
                invalid = d.TAXITIME_SEC_mvt.lt(0)
                self.assertTrue(d.loc[invalid, ADDED].isna().all().all())
                self.assertEqual(invalid.sum(), record['invalid_training_targets'])
        # The v50 baseline deliberately stays identical across historical scopes.
        name = 'training_2025-11-01_2025-12-01.parquet'
        full = pd.read_parquet(self.output / 'full' / name)
        validation = pd.read_parquet(self.output / 'validation' / name)
        pd.testing.assert_frame_equal(full[MOVEMENT_FEATURES], validation[MOVEMENT_FEATURES], check_exact=True)
        self.assertFalse(full['arr_taxiin_excess_prev30'].equals(validation['arr_taxiin_excess_prev30']))
        before = {p.relative_to(self.output).as_posix(): (digest(p), p.stat().st_mtime_ns)
                  for p in self.output.rglob('*') if p.is_file()}
        with redirect_stdout(StringIO()):
            run_model_data(self.root)
        self.assertEqual(before, {p.relative_to(self.output).as_posix(): (digest(p), p.stat().st_mtime_ns)
                                 for p in self.output.rglob('*') if p.is_file()})
        # Corrupted state must never be reused. Restore fixture bytes for other tests.
        ref = self.output / '_references/full/apt_med.parquet'
        original = ref.read_bytes()
        try:
            ref.write_bytes(original + b'corrupt')
            with self.assertRaisesRegex(ValueError, 'output changed'):
                validate_model_data(self.root)
        finally:
            ref.write_bytes(original)

    def test_final_coverage_and_frozen_references(self):
        paths = sorted(self.base.glob('training_*.parquet')) + [self.base / 'final_ranking.parquet']
        d, a = read_departures(paths), read_arrivals(paths)
        refs = state.load_references(self.output / '_references/full')
        saved = {name: value.copy(deep=True) for name, value in refs.items()}
        # A June departure ten minutes into new coverage cannot have a complete
        # 30/60-minute arrival history, despite earlier January observations.
        selected = d['_ranking'] & d['_month'].eq(6)
        first = d.index[selected][0]
        d.loc[first, 'MVT_TIME_UTC_mvt'] = pd.Timestamp('2026-06-01T00:10:00Z')
        with redirect_stdout(StringIO()):
            new, applied = arrival.add_arrival_features(d, a, scope='full', context='final_ranking', references=refs)
        self.assertTrue(pd.isna(new.loc[first, 'arr_same_rwy_prev30']))
        self.assertTrue(pd.isna(new.loc[first, 'arr_taxiin_n_prev30']))
        for name, value in saved.items():
            if isinstance(value, pd.Series):
                pd.testing.assert_series_equal(applied[name], value, check_exact=True)
            else:
                pd.testing.assert_frame_equal(applied[name], value, check_exact=True)
        final = pd.read_parquet(self.output / 'full/final_ranking.parquet')
        self.assertFalse(final[ID].duplicated().any())
        # Three strictly earlier departures? Source has one at t=0, one at t=30,
        # and two tied at t=60: ties are excluded and ranking is not double-counted.
        key = '2026-1-EDDF-2-day0'
        self.assertEqual(final.set_index(ID).loc[key, 'tf_dep_rwy_prev_5min'], 2)


if __name__ == '__main__':
    unittest.main()
