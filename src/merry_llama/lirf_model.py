"""V29 mixture plus the v52 forward-OOF residual correction (weight 0.5)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .model_config import *
from .model_utils import clean, utc, assert_ids, lirf_frame_x, lirf_labels, lirf_residual_x
from .model_store import load_record, commit, save_model, atomic_frame
from .features.base import write_json


def read_lirf(paths, training):
    cols=list(dict.fromkeys([ID,TIME,AOBT,'PHASE_mvt','ADEP']+lirf_CORE+lirf_OPS+
                           ([TARGET,'BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt'] if training else [])))
    frames=[]
    for path in paths:
        d=pq.read_table(path,columns=cols).to_pandas()
        keep=clean(d.PHASE_mvt).eq('DEP').fillna(False)&clean(d.ADEP).eq('LIRF').fillna(False)&clean(d[AOBT]).isna()
        d=d.loc[keep].copy()
        d['_month']=utc(d[TIME],errors='coerce').dt.month
        if training:
            y=pd.to_numeric(d[TARGET],errors='coerce')
            valid=np.isfinite(y)&y.ge(0);d=d.loc[valid].copy();d[TARGET]=y.loc[valid].astype(float)
        frames.append(d)
    d=pd.concat(frames,ignore_index=True);assert_ids(d,'LIRF branch')
    if len(d) and not np.isfinite(pd.to_numeric(d[SCHEDULE],errors='coerce')).all():
        raise ValueError('LIRF schedule difference must be finite')
    return d


def encode_frame(x, vocab):
    x=x.copy()
    for c in lirf_CATS:x[c]=pd.Categorical(x[c],categories=vocab[c])
    return x


def fit_mixture(train,folder,threads):
    import lightgbm as lgb
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if load_record(folder):return
    typ=lirf_labels(train);y=train[TARGET].to_numpy(float);classes=np.unique(typ)
    if not np.any(typ==1):raise ValueError('No LIRF type B in fitting period')
    x=lirf_frame_x(train);vocab={c:sorted(x[c].dropna().unique().tolist()) for c in lirf_CATS}
    names=['categories.json'];write_json(folder/names[0],vocab)
    if len(classes)>1:
        model=lgb.LGBMClassifier(**dict(LIRF_PARAMS,n_jobs=threads))
        model.fit(encode_frame(x,vocab),typ,categorical_feature=lirf_CATS)
        classes=model.classes_;save_model(model.booster_,folder/'model.txt');names.append('model.txt')
    b=float(np.median(y[typ==1]));c=float(np.median(y[typ==2])) if np.any(typ==2) else 86400+b
    commit(folder,{'classes':classes.tolist(),'median_B':b,'median_C':c,'training_rows':len(train),'threads':threads},names)


def predict_mixture(query,folder):
    import lightgbm as lgb
    folder=Path(folder);record=load_record(folder)
    if record is None:raise FileNotFoundError(folder)
    p=np.zeros((len(query),3));classes=np.asarray(record['classes'],int)
    if len(query):
        if len(classes)==1:p[:,classes[0]]=1
        else:
            vocab=json.loads((folder/'categories.json').read_text())
            model=lgb.Booster(model_file=str(folder/'model.txt'))
            probability=model.predict(encode_frame(lirf_frame_x(query),vocab))
            if probability.ndim==1:probability=np.column_stack([1-probability,probability])
            p[:,classes]=probability
    base=p[:,0]*query[SCHEDULE].to_numpy(float)+p[:,1]*record['median_B']+p[:,2]*record['median_C']
    return pd.DataFrame({ID:query[ID].to_numpy(),'base':base,'pA':p[:,0],'pB':p[:,1],'pC':p[:,2]})


def fit_lirf(paths, run_dir):
    import lightgbm as lgb
    folder=Path(run_dir)/'lirf';folder.mkdir(parents=True,exist_ok=True)
    raw=read_lirf(paths,training=True)
    if raw.empty:raise ValueError('No LIRF missing-AOBT training flights')
    print(f'LIRF record classifiers: {len(raw):,} rows',flush=True)
    # V29 stored baseline used raw file order, raw predictor precision and default threads.
    fit_mixture(raw,folder/'stored_baseline',None)
    # v52 recomputes probabilities at 4 threads; baseline itself is V29's unrounded value.
    fit_mixture(raw,folder/'probabilities',4)
    raw=raw.sort_values([TIME,ID]).reset_index(drop=True)
    chunks=[]
    for month in range(5,13):
        tr=raw.loc[raw._month.lt(month)];q=raw.loc[raw._month.eq(month)]
        if q.empty:continue
        location=folder/f'oof_{month:02d}'
        fit_mixture(tr,location,4)
        chunks.append(predict_mixture(q,location))
        print(f'LIRF OOF {month:02d}: train={len(tr):,}, query={len(q):,}',flush=True)
    fit=raw.loc[raw._month.ge(5)].reset_index(drop=True)
    if not chunks:raise ValueError('No LIRF residual OOF population')
    bf=pd.concat(chunks,ignore_index=True).set_index(ID).loc[fit[ID]].reset_index()
    destination=folder/'residual';destination.mkdir(exist_ok=True)
    if load_record(destination) is None:
        x=lirf_residual_x(fit,bf);vocab={c:sorted(x[c].dropna().unique().tolist()) for c in lirf_CATS}
        model=lgb.LGBMRegressor(**lirf_REG_PARAMS,n_estimators=200)
        model.fit(encode_frame(x,vocab),fit[TARGET].to_numpy(float)-bf.base.to_numpy(float),categorical_feature=lirf_CATS)
        save_model(model.booster_,destination/'model.txt');write_json(destination/'categories.json',vocab)
        commit(destination,{'features':list(x),'weight':0.5,'training_rows':len(fit)},['model.txt','categories.json'])


def predict_lirf(query,run_dir):
    import lightgbm as lgb
    if query.empty:return np.empty(0),pd.DataFrame()
    folder=Path(run_dir)/'lirf'
    bq=predict_mixture(query,folder/'probabilities')
    stored=predict_mixture(query,folder/'stored_baseline').base.to_numpy(float)
    if not np.array_equal(np.floor(np.maximum(stored,0)+.5),np.floor(np.maximum(bq.base.to_numpy(),0)+.5)):
        raise ValueError('V29 baseline and four-thread rebuild differ after rounding')
    bq['base']=stored
    destination=folder/'residual';record=load_record(destination)
    if record is None:raise FileNotFoundError(destination)
    vocab=json.loads((destination/'categories.json').read_text())
    x=encode_frame(lirf_residual_x(query,bq),vocab)
    if list(x)!=record['features']:raise ValueError('LIRF residual feature order mismatch')
    residual=lgb.Booster(model_file=str(destination/'model.txt')).predict(x)
    result=stored+record['weight']*residual
    if not np.isfinite(result).all():raise ValueError('Nonfinite LIRF prediction')
    audit=pd.DataFrame({ID:query[ID].to_numpy(),'dylan_baseline':stored,'residual':residual,'corrected':result})
    return result,audit
