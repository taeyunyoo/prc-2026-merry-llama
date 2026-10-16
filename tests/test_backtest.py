"""Synthetic backtest safeguards; no competition data or estimators required."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from merry_llama import backtest as b
from merry_llama import learned_features as learned
from merry_llama.features.base import digest, run_base_features
from merry_llama.features import state
from merry_llama.preprocessor import run_preprocessing
from merry_llama.features.definitions import ID, TIME, TARGET, MEDIAN
from merry_llama.learned_features import RouteMedian
from test_model_data import rich_inputs




class PreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        with redirect_stdout(StringIO()):
            rich_inputs(cls.root)
            run_preprocessing(cls.root)
            run_base_features(cls.root)
        cls.base = cls.root/'data/interim/base_features'

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def audit(self):
        return {'contract': b.CONTRACT, 'reviewer': 'synthetic fixture; not scientific evidence',
                'model_selection': 'published_frozen_exploratory',
                'base_sha256': b.inventory([self.base/n for n in b.training_names()], self.base),
                'checks': {k: {'approved': True, 'evidence': 'synthetic only'} for k in b.CHECKS}}

    def test_gate_and_input_binding(self):
        path = self.root/'audit.json'
        a = self.audit(); path.write_text(json.dumps(a))
        b.validate_audit(self.base, path)
        for key in b.CHECKS:
            rejected = self.audit(); rejected['checks'][key]['approved'] = False
            path.write_text(json.dumps(rejected))
            with self.assertRaisesRegex(ValueError, key): b.validate_audit(self.base, path)
        a['base_sha256'][b.training_names()[0]] = 'wrong'
        path.write_text(json.dumps(a))
        with self.assertRaisesRegex(ValueError, 'SHA-256'): b.validate_audit(self.base, path)

    def test_temporal_and_id_rejection(self):
        paths = [self.base/n for n in b.training_names()]
        b.check_months(paths)
        original = paths[10].read_bytes()
        try:
            d = pd.read_parquet(paths[10]); d.loc[0, TIME] = pd.Timestamp('2025-10-31', tz='UTC')
            d.to_parquet(paths[10], index=False)
            with self.assertRaisesRegex(ValueError, 'outside'): b.check_months(paths)
        finally: paths[10].write_bytes(original)

    def test_holdout_perturbation_cannot_change_fitted_references_or_fit_features(self):
        first, second = self.root/'prepared1', self.root/'prepared2'
        before = {p.name: digest(p) for p in self.base.glob('*') if p.is_file()}
        with redirect_stdout(StringIO()): b.prepare(self.base, first)
        self.assertEqual(before, {p.name: digest(p) for p in self.base.glob('*') if p.is_file()})
        saved = {}
        try:
            for name in b.training_names()[10:]:
                p = self.base/name; saved[p] = p.read_bytes()
                d = pd.read_parquet(p)
                dep = d.PHASE_mvt.eq('DEP'); arr = d.PHASE_mvt.eq('ARR')
                d.loc[dep, TARGET] = -999.  # valid population must not select features
                d.to_parquet(p, index=False)
            label_only = self.root/'prepared-label-only'
            with redirect_stdout(StringIO()): b.prepare(self.base, label_only)
            for name in b.training_names()[10:]:
                pd.testing.assert_frame_equal(pd.read_parquet(first/'evaluation'/name),
                                              pd.read_parquet(label_only/'evaluation'/name))
            for p in saved:
                d = pd.read_parquet(p); arr = d.PHASE_mvt.eq('ARR')
                d.loc[arr, TARGET] = 6500.
                d.loc[arr, 'BLOCK_TIME_UTC_mvt'] = d.loc[arr, TIME] + pd.Timedelta(seconds=6500)
                d.to_parquet(p, index=False)
            with redirect_stdout(StringIO()): b.prepare(self.base, second)
            for p in (first/'fit').glob('*.parquet'):
                pd.testing.assert_frame_equal(pd.read_parquet(p), pd.read_parquet(second/'fit'/p.name))
            for group in ['arrival', 'movement']:
                a = state.load_references(first/'_references'/group)
                c = state.load_references(second/'_references'/group)
                for key in a:
                    if isinstance(a[key], pd.Series): pd.testing.assert_series_equal(a[key], c[key])
                    else: pd.testing.assert_frame_equal(a[key], c[key])
            for name in b.training_names()[10:]:
                d = pd.read_parquet(second/'_masked'/name)
                self.assertTrue(d.loc[d.PHASE_mvt.eq('DEP'), TARGET].eq(0).all())
            tr = pd.concat([pd.read_parquet(p) for p in (first/'fit').glob('*.parquet')])
            tr = tr.loc[tr[TARGET].ge(0) & np.isfinite(tr[TARGET])]
            self.assertTrue(pd.to_datetime(tr[TIME], utc=True).lt(b.CUTOFF).all())
            q = pd.read_parquet(first/'evaluation'/b.training_names()[10])
            # Saved target statistics have training-only donors, even for unseen routes.
            q.loc[:, 'STAND_mvt'] = 'UNSEEN'
            values = RouteMedian().fit(tr).transform(q)
            self.assertTrue(np.isfinite(values).all())
        finally:
            for p, content in saved.items(): p.write_bytes(content)

    def test_learned_fit_restricts_all_donors(self):
        output = self.root/'learned-preparation'
        with redirect_stdout(StringIO()):
            paths, _ = b.prepare(self.base, output)
            train = learned.read_inputs(paths)
        seen = []
        def fit_airside(donors, folder):
            self.assertTrue(donors[TIME].lt(b.CUTOFF).all())
            self.assertTrue((~donors._missing).all())
            seen.append((folder.name, set(donors._month)))
            if folder.name.startswith('airside_oof_'):
                self.assertNotIn(int(folder.name[-2:]), set(donors._month))
        with patch.object(learned, 'fit_airside', side_effect=fit_airside), \
             patch.object(learned, 'predict_regressor', side_effect=lambda folder, q: np.full(len(q), 600.)), \
             redirect_stdout(StringIO()):
            fitted = b.fit_learned(train, output, self.root/'learned-models')
        self.assertEqual(len(seen), 11)
        self.assertEqual(seen[-1], ('airside_full', set(range(1, 11))))
        self.assertTrue(np.isfinite(fitted[[MEDIAN, 'airside_est']]).all().all())
        bad = train.copy(); bad.loc[bad.index[0], TIME] = b.CUTOFF
        with self.assertRaisesRegex(ValueError, 'outside fitting'):
            b.fit_learned(bad, output, self.root/'rejected-learned')

    def test_lirf_fit_receives_only_training_paths(self):
        # Intercept costly estimators; use real preparation and input routing.
        audit = self.root/'orchestration-audit.json'; audit.write_text(json.dumps(self.audit()))
        seen = []
        def lirf(paths, folder):
            seen.extend(paths)
            for p in paths:
                self.assertTrue(pd.to_datetime(pd.read_parquet(p)[TIME], utc=True).lt(b.CUTOFF).all())
            raise RuntimeError('stop before expensive estimators')
        with patch('merry_llama.training.environment', return_value={}), \
             patch('merry_llama.lirf_model.fit_lirf', side_effect=lirf), redirect_stdout(StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'stop before'):
                b.run(self.base, audit, self.root/'orchestration')
        self.assertEqual(len(seen), 10)


if __name__ == '__main__': unittest.main()
