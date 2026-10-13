"""Source v21 date-OOF medians and v50 month-OOF airside estimators.

STEP 03 remains read-only. Full-year learned state is saved under the model run.
The full inference airside model learns from the same OOF median features as v50.
"""
from pathlib import Path
import json
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .model_config import *
from .model_utils import clean, utc, assert_ids
from .model_store import atomic_frame, fit_regressor, predict_regressor, load_record, commit
from .features.base import write_json

FOLDS, SEED = 5, 42
class RouteMedian:
    KEYS=[('ADEP','STAND_mvt','RUNWAY_mvt'),('ADEP','RUNWAY_mvt'),('ADEP','STAND_mvt'),('ADEP',)]

    def fit(self,train):
        y=train[TARGET].to_numpy(dtype=float)
        if not len(y) or not np.isfinite(y).all() or (y<0).any():
            raise ValueError('중앙값 donor에는 유한한 0 이상 정답이 필요합니다.')
        self.global_median=float(np.median(y))
        self.maps=[train.groupby(list(k),observed=True,dropna=True)[TARGET].median() for k in self.KEYS]
        return self

    def transform(self,frame):
        result=np.full(len(frame),np.nan)
        for keys,table in zip(self.KEYS,self.maps):
            ix=np.flatnonzero(np.isnan(result)&frame[list(keys)].notna().all(axis=1).to_numpy())
            if not len(ix):
                continue
            part=frame.iloc[ix]
            lookup=(pd.Index(part[keys[0]].to_numpy(),name=keys[0]) if len(keys)==1
                    else pd.MultiIndex.from_frame(part[list(keys)]))
            result[ix]=table.reindex(lookup).to_numpy(dtype=float)
        result[np.isnan(result)]=self.global_median
        return result

def oof_median(train):
    # v20와 동일: UTC 날짜별 5-fold, seed 42. 과거 데이터만 쓰는 rolling 통계는 아님.
    day=train[TIME].dt.floor('D')
    days=day.drop_duplicates().sort_values().to_numpy()
    if FOLDS<2 or len(days)<FOLDS:
        raise ValueError('OOF 계산에 충분한 날짜가 없습니다.')
    mapping=pd.Series(np.random.default_rng(SEED).permutation(len(days))%FOLDS,index=pd.Index(days))
    fold=day.map(mapping).to_numpy(dtype=int)
    result=np.full(len(train),np.nan)
    for k in range(FOLDS):
        held=fold==k
        result[held]=RouteMedian().fit(train.loc[~held]).transform(train.loc[held])
        print(f'  OOF {k+1}/{FOLDS}: {held.sum():,}행',flush=True)
    if not np.isfinite(result).all():
        raise RuntimeError('OOF 중앙값 생성 실패')
    return result

def save_median(model, folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    names=[]
    for i,table in enumerate(model.maps):
        name=f'map_{i}.parquet';atomic_frame(table.rename(TARGET).reset_index(),folder/name);names.append(name)
    commit(folder,{'global_median':model.global_median,'keys':model.KEYS},names)


def load_median(folder):
    folder=Path(folder);r=load_record(folder)
    if r is None:raise FileNotFoundError(folder/'complete.json')
    model=RouteMedian();model.global_median=r['global_median']
    model.maps=[pd.read_parquet(folder/f'map_{i}.parquet').set_index(list(keys))[TARGET]
                for i,keys in enumerate(model.KEYS)]
    return model


def read_inputs(paths):
    """v50 load_departures numeric precision before joining movement features."""
    frames=[]
    extra=list(dict.fromkeys(DEP_FEATURES+SIDE_FEATURES+['cf_config']))
    extra.remove('cf_recent_vs_usual_rwy')
    feats=[f for f in PRESENT_FEATURES if f!=MEDIAN]
    for path in paths:
        evaluation=path.name in {'ranking.parquet','final_ranking.parquet'}
        cols=list(dict.fromkeys(feats+extra+[ID,TIME,AOBT,'PHASE_mvt']+([] if evaluation else [TARGET])))
        d=pq.read_table(path,columns=cols).to_pandas()
        d=d.loc[clean(d.PHASE_mvt).eq('DEP').fillna(False)].copy().reset_index(drop=True)
        assert_ids(d,path.name);d[TIME]=utc(d[TIME])
        if d[TIME].isna().any():raise ValueError('Missing movement time')
        for c in MAIN_CATS:d[c]=clean(d[c])
        if not d.ADEP.isin(AIRPORTS).all():raise ValueError('Unexpected departure airport')
        for c in feats+extra:
            if c not in MAIN_CATS:
                d[c]=pd.to_numeric(d[c],errors='raise').astype(np.float64 if c in (PROXY,SCHEDULE) else np.float32)
                if np.isinf(d[c].to_numpy()).any():raise ValueError(f'Infinite {c}')
        d=d.copy()  # consolidate numeric blocks before adding routing fields
        d['_missing']=clean(d[AOBT]).isna()
        if not d._missing.eq(d[PROXY].isna()).all():raise ValueError('AOBT/proxy missingness mismatch')
        d['_special']=d._missing & d.ADEP.eq('LIRF')
        d['_ranking']=evaluation;d['_month']=d[TIME].dt.month
        excluded=0
        if evaluation:d[TARGET]=np.nan
        else:
            d[TARGET]=pd.to_numeric(d[TARGET],errors='raise').astype(float)
            valid=np.isfinite(d[TARGET]) & d[TARGET].ge(0)
            excluded=int((~valid).sum());d=d.loc[valid].copy()
        print(f'LOAD {path.name}: eligible DEP={len(d):,}, invalid targets excluded={excluded}',flush=True)
        frames.append(d)
    out=pd.concat(frames,ignore_index=True);assert_ids(out,'Combined model input')
    return out


def with_median(frame, values, movement_cache):
    d=frame.copy();d[MEDIAN]=np.asarray(values,dtype=np.float32)
    mv=pd.read_parquet(movement_cache,columns=[ID,'cf_recent_taxi_rwy_60min'])
    assert_ids(mv,'Movement precision cache')
    recent=mv.set_index(ID).reindex(d[ID]).cf_recent_taxi_rwy_60min.to_numpy(float)
    # v50 add_features subtracts original float64 movement statistics BEFORE export.
    d['cf_proxy_minus_recent']=(d[PROXY].to_numpy(float)-recent).astype(np.float32)
    d['cf_recent_vs_usual_rwy']=(recent-d[MEDIAN].to_numpy()).astype(np.float32)
    return d


def set_estimate(df, est):
    df['airside_est']=np.asarray(est,dtype=np.float32)
    df['airside_minus_proxy']=(df['airside_est']-df[PROXY]).astype(np.float32)
    return df


def main_precision(frame):
    """v52 prepared_load recomputes these differences after v50 float32 export."""
    d=frame.copy()
    d['cf_proxy_minus_recent']=(d[PROXY]-d.cf_recent_taxi_rwy_60min).astype(np.float32)
    d['cf_recent_vs_usual_rwy']=(d.cf_recent_taxi_rwy_60min-d[MEDIAN]).astype(np.float32)
    return d


def fit_airside(donors, folder):
    sampled=donors.sample(frac=AIR_SAMPLE,random_state=42)
    feats=AIR_CAT+AIR_NUM+SIDE_FEATURES+['cf_config'];cats=AIR_CAT+['cf_config']
    fit_regressor(folder,sampled,sampled[TARGET].clip(upper=AIR_CAP),feats,cats,AIR_PARAMS)


def fit_learned(train, dataset_dir, run_dir):
    directory=Path(run_dir)/'learned';directory.mkdir(parents=True,exist_ok=True)
    med_path=directory/'median_oof'
    med_record=load_record(med_path)
    if med_record:
        tab=pd.read_parquet(med_path/'values.parquet')
        if not np.array_equal(tab[ID],train[ID]):raise ValueError('Median OOF IDs changed')
        values=tab[MEDIAN].to_numpy()
    else:
        values=oof_median(train)
        med_path.mkdir(parents=True,exist_ok=True)
        atomic_frame(pd.DataFrame({ID:train[ID],MEDIAN:values}),med_path/'values.parquet')
        commit(med_path,{'scope':'full-2025 UTC-day five-fold, seed 42'},['values.parquet'])
    if load_record(directory/'median_full') is None:
        save_median(RouteMedian().fit(train),directory/'median_full')
    train=with_median(train,values,Path(dataset_dir)/'_movement/train.parquet')
    present=train.loc[~train._missing]
    estimates=np.full(len(train),np.nan)
    for month in range(1,13):
        mask=train._month.eq(month).to_numpy();query=train.loc[mask]
        if query.empty:continue
        folder=directory/f'airside_oof_{month:02d}'
        pred_folder=directory/f'airside_predictions_{month:02d}'
        old=load_record(pred_folder)
        if old:
            tab=pd.read_parquet(pred_folder/'values.parquet')
            if not np.array_equal(tab[ID],query[ID]):raise ValueError('Airside OOF IDs changed')
            values=tab.airside_est.to_numpy()
        else:
            donors=present.loc[present._month.ne(month)]
            if len(donors)<3:raise ValueError(f'Insufficient airside donors for month {month}')
            fit_airside(donors,folder)
            values=predict_regressor(folder,query)
            pred_folder.mkdir(parents=True,exist_ok=True)
            atomic_frame(pd.DataFrame({ID:query[ID].to_numpy(),'airside_est':values}),pred_folder/'values.parquet')
            commit(pred_folder,{'month':month},['values.parquet'])
        estimates[mask]=values
        print(f'AIRSIDE OOF {month:02d}: {len(query):,} rows complete',flush=True)
    if not np.isfinite(estimates).all():raise ValueError('Airside OOF missing values')
    fit_airside(present,directory/'airside_full')
    return main_precision(set_estimate(train,estimates))


def apply_learned(query, movement_cache, run_dir):
    directory=Path(run_dir)/'learned'
    values=load_median(directory/'median_full').transform(query)
    q=with_median(query,values,movement_cache)
    return main_precision(set_estimate(q,predict_regressor(directory/'airside_full',q)))
