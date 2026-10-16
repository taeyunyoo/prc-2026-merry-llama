"""Metric safeguards run without Parquet or model-library dependencies."""
import unittest
import numpy as np
import pandas as pd
from merry_llama import backtest_metrics as b
from merry_llama.model_config import ID, TIME, TARGET


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.truth = pd.DataFrame({ID: ['a', 'b', 'c', 'd'], 'ADEP': ['EDDF', 'EGLL', 'EDDF', 'EGLL'],
            TIME: pd.to_datetime(['2025-11-01', '2025-12-31', '2025-11-03', '2025-12-02'], utc=True),
            TARGET: [0., 4., -1., np.nan]})
        self.pred = pd.DataFrame({ID: ['d', 'c', 'b', 'a'], 'prediction_sec': [8., 9., 0., 3.]})

    def test_pair_weighted_rmse_and_exclusions(self):
        m = b.score(self.truth, self.pred)
        self.assertAlmostEqual(m['overall']['rmse_sec'], np.sqrt(12.5))
        self.assertEqual(m['overall']['evaluated_pairs'], 2)
        self.assertEqual(m['overall']['excluded_invalid_targets'], 2)
        self.assertEqual(m['airports']['EDDF']['rmse_sec'], 3.)
        self.assertEqual(m['airports']['EGLL']['rmse_sec'], 4.)
        self.assertIsNone(m['airports']['LIRF']['rmse_sec'])

    def test_no_silent_join_or_nonfinite_drops(self):
        for pred in [self.pred.iloc[:3], pd.concat([self.pred, self.pred.iloc[:1]]),
                     self.pred.assign(prediction_sec=np.nan), self.pred.assign(**{ID: ['x','c','b','a']})]:
            with self.assertRaises(ValueError): b.score(self.truth, pred)
        for instant in ['2025-10-31T23:59:59Z', '2026-01-01T00:00:00Z']:
            truth = self.truth.copy(); truth.loc[0, TIME] = pd.Timestamp(instant)
            with self.assertRaises(ValueError): b.score(truth, self.pred)


if __name__ == '__main__': unittest.main()
