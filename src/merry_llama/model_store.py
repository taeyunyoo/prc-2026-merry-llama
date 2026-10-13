"""Atomic model checkpoints: each completion record hashes its committed files."""
from pathlib import Path
import json
import os
from uuid import uuid4
import numpy as np
import pandas as pd
from .features.base import digest, write_json
from .model_utils import model_input, cb_frame, vocab_of


def load_record(folder):
    folder = Path(folder)
    p = folder / 'complete.json'
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding='utf-8'))
    for name, expected in d['files'].items():
        file = folder / name
        if not file.is_file() or digest(file) != expected:
            raise ValueError(f'Changed/incomplete model artifact: {file}')
    return d


def commit(folder, metadata, names):
    folder = Path(folder)
    metadata = {**metadata, 'files': {n: digest(folder/n) for n in names}}
    write_json(folder/'complete.json', metadata)
    return metadata


def save_model(model, path, kind='lgb'):
    path = Path(path)
    temp = path.with_name('.'+path.name+'.'+uuid4().hex+'.tmp')
    try:
        if kind == 'cb': model.save_model(str(temp))
        else: model.save_model(str(temp))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_frame(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.'+path.name+'.'+uuid4().hex+'.tmp')
    try:
        frame.to_parquet(temp, index=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def fit_regressor(folder, train, y, features, cats, params, kind='lgb', residual=False):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    record = load_record(folder)
    expected = dict(kind=kind, features=features, cats=cats, params=params,
                    residual=residual, training_rows=len(train))
    if record:
        if any(record[k] != v for k,v in expected.items()):
            raise ValueError(f'Model configuration mismatch: {folder}')
        print('REUSE model:', folder.name, flush=True)
        return
    if not len(train) or not np.isfinite(y).all():
        raise ValueError(f'No valid training labels: {folder}')
    print(f'FIT {folder.name}: train={len(train):,}, trees={params.get("n_estimators", params.get("iterations"))}', flush=True)
    if kind == 'cb':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(**params)
        model.fit(cb_frame(train,features,cats), y, cat_features=cats)
        names = ['model.cbm']
        save_model(model, folder/names[0], kind)
    else:
        import lightgbm as lgb
        vocab = vocab_of(train,cats)
        model = lgb.LGBMRegressor(**params)
        model.fit(model_input(train,vocab,features,cats), y, categorical_feature=cats)
        names = ['model.txt','categories.json']
        save_model(model.booster_,folder/names[0])
        write_json(folder/names[1],vocab)
    commit(folder,expected,names)


def predict_regressor(folder, query):
    folder=Path(folder); info=load_record(folder)
    if info is None: raise FileNotFoundError(f'Unfinished model: {folder}')
    if not len(query): return np.empty(0)
    if info['kind']=='cb':
        from catboost import CatBoostRegressor
        m=CatBoostRegressor();m.load_model(str(folder/'model.cbm'))
        pred=m.predict(cb_frame(query,info['features'],info['cats']))
    else:
        import lightgbm as lgb
        vocab=json.loads((folder/'categories.json').read_text())
        m=lgb.Booster(model_file=str(folder/'model.txt'))
        pred=m.predict(model_input(query,vocab,info['features'],info['cats']))
    if info['residual']:
        from .model_config import PROXY
        pred=np.asarray(pred,float)+query[PROXY].to_numpy(float)
    pred=np.asarray(pred,float)
    if not np.isfinite(pred).all(): raise ValueError(f'Nonfinite prediction: {folder}')
    return pred
