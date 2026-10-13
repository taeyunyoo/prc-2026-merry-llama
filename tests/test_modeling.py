"""Fast contracts for STEP 04; python -m unittest discover -s tests -p test_modeling.py -v."""
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from merry_llama.learned_features import RouteMedian, save_median, load_median
from merry_llama.model_config import ID, TARGET
from merry_llama.postprocess import rounded
from merry_llama.submission import write_submission, read_template


class ModelContracts(unittest.TestCase):
    def test_median_fallback_and_portable_state(self):
        train=pd.DataFrame({'ADEP':['A','A','A','B'],'STAND_mvt':['S1','S1','S2','S3'],
                            'RUNWAY_mvt':['R1','R1','R2','R3'],TARGET:[100.,300.,800.,1000.]})
        query=pd.DataFrame({'ADEP':['A','A','A','A','X'],'STAND_mvt':['S1','OTHER','S2',None,None],
                            'RUNWAY_mvt':['R1','R1','OTHER',None,None]})
        model=RouteMedian().fit(train)
        expected=np.array([200.,200.,800.,300.,550.])
        np.testing.assert_array_equal(model.transform(query),expected)
        with TemporaryDirectory() as tmp:
            save_median(model,Path(tmp))
            np.testing.assert_array_equal(load_median(Path(tmp)).transform(query),expected)

    def test_submission_alignment_and_schema(self):
        d=pd.DataFrame({ID:[1.,2.,3.]})
        template=pa.table({ID:[3.,1.,2.],TARGET:pa.nulls(3,type=pa.int32()),'keep':['c','a','b']})
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'prediction.parquet'
            write_submission(template,d,np.array([-.5,1.5,10.49]),p)
            out=pq.read_table(p)
            self.assertTrue(out.schema.equals(template.schema,check_metadata=True))
            self.assertTrue(out['keep'].equals(template['keep']))
            self.assertEqual(out[TARGET].to_pylist(),[10,0,2])
            with self.assertRaisesRegex(ValueError,'contains predictions'):
                read_template(p,d)
            with self.assertRaises(ValueError):rounded([np.inf])


if __name__=='__main__':unittest.main()
