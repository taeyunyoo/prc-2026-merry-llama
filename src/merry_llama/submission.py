"""Align predictions to the supplied blank template and preserve its schema."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .model_config import ID, TARGET
from .postprocess import rounded as postprocess

def read_template(path, ranking):
    table = pq.read_table(path)
    if ID not in table.column_names or TARGET not in table.column_names:
        raise ValueError(f'{path}: expected {ID} and {TARGET}')
    ids = table[ID].to_pandas()
    if ids.isna().any() or ids.duplicated().any() or len(ids) != len(ranking) or not ids.isin(ranking[ID]).all():
        raise ValueError('Submission template IDs do not exactly match ranking DEP IDs')
    if table[TARGET].to_pandas().notna().any():
        raise ValueError('Submission template contains predictions; supply the original blank template.')
    return table

def write_submission(template, ranking, prediction, destination):
    ids = template[ID].to_pandas()
    mapped = ids.map(pd.Series(postprocess(prediction), index=ranking[ID])).to_numpy(float)
    if not np.isfinite(mapped).all():
        raise ValueError('Missing predictions after ID alignment')
    field = template.schema.field(TARGET)
    dtype = pa.float64() if pa.types.is_null(field.type) else field.type
    if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)):
        raise ValueError(f'Unsupported target type: {dtype}')
    values = pa.array(mapped).cast(dtype, safe=True)
    if not np.array_equal(values.to_numpy(), mapped):
        raise ValueError('Target precision loss during casting')
    result = template.set_column(template.schema.get_field_index(TARGET),
        pa.field(TARGET, dtype, nullable=field.nullable, metadata=field.metadata), values)
    for c in template.column_names:
        if c != TARGET and not result[c].equals(template[c]):
            raise AssertionError(f'Template column changed: {c}')
    temporary = destination.with_suffix('.parquet.tmp')
    pq.write_table(result, temporary, compression='snappy')
    if not pq.read_table(temporary).equals(result):
        raise AssertionError('Submission Parquet round trip failed')
    temporary.replace(destination)

