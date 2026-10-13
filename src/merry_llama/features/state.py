"""Portable fitted reference tables: parquet data with a JSON inventory."""
from pathlib import Path
import pandas as pd
from .base import write_json


def save_references(state, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for name, value in state.items():
        if not name.replace('_', '').isalnum():
            raise ValueError(f"Invalid reference name: {name}")
        series = isinstance(value, pd.Series)
        frame = value.to_frame('__value__') if series else value
        index_columns = [f'__index_{i}' for i in range(frame.index.nlevels)]
        index_names = list(frame.index.names)
        stored = frame.reset_index(names=index_columns)
        temporary = folder / f'{name}.parquet.tmp'
        stored.to_parquet(temporary, index=False, compression='snappy')
        restored = pd.read_parquet(temporary).set_index(index_columns)
        restored.index.names = index_names
        pd.testing.assert_frame_equal(restored, frame, check_exact=True)
        temporary.replace(folder / f'{name}.parquet')
        records.append({'name': name, 'series': series,
                        'series_name': value.name if series else None,
                        'index_columns': index_columns, 'index_names': index_names})
    write_json(folder / 'references.json', {'tables': records})


def load_references(folder):
    import json
    folder = Path(folder)
    records = json.loads((folder / 'references.json').read_text(encoding='utf-8'))['tables']
    state = {}
    for record in records:
        name = record['name']
        frame = pd.read_parquet(folder / f'{name}.parquet').set_index(record['index_columns'])
        frame.index.names = record['index_names']
        if record['series']:
            frame = frame['__value__'].rename(record['series_name'])
        state[name] = frame
    return state
