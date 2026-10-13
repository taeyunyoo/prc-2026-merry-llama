"""Full-year refit and saved-model routing for the frozen v52 ensemble."""
from pathlib import Path
import gc
import json
import numpy as np
from .model_config import *
from .model_utils import target
from .model_store import fit_regressor, predict_regressor
from .features.base import write_json


def fit_models(train, run_dir, threads=-1):
    folder=Path(run_dir)/'estimators';folder.mkdir(parents=True,exist_ok=True)
    for airport in AIRPORTS:
        tr=train.loc[~train._missing & train.ADEP.eq(airport)]
        if tr.empty:
            print(f'NO PRESENT MODEL {airport}: no training rows',flush=True)
            continue
        for key,(kind,params) in MAIN_MODELS.items():
            trees=FIXED_TREES[key][airport]
            if kind=='cb':
                cp={k:v for k,v in params.items() if k not in ['iterations','od_type','od_wait','use_best_model','verbose']}
                params=dict(cp,task_type='CPU',thread_count=threads,iterations=trees,verbose=100,allow_writing_files=False)
            else:params=dict(params,n_estimators=trees,n_jobs=threads)
            fit_regressor(folder/(airport+'_'+key),tr,target(tr,True),MAIN_FEATURES,MAIN_CATS,params,kind,True)
            gc.collect()
    tr=train.loc[train._missing & ~train._special]
    if tr.empty:raise ValueError('No pooled missing9 training population')
    fit_regressor(folder/'missing9_lgb',tr,target(tr,False),MISSING_FEATURES,MODEL_CATS,
                  dict(PARAMS,n_estimators=MISSING9_LGB_TREES,n_jobs=threads))
    cp={k:v for k,v in CB_PARAMS.items() if k not in ['iterations','od_type','od_wait','use_best_model','verbose']}
    fit_regressor(folder/'missing9_cb',tr,target(tr,False),M9_CB_FEATURES,MAIN_CATS,
                  dict(cp,task_type='CPU',thread_count=threads,iterations=FIXED_TREES['m9_cb'],verbose=100,allow_writing_files=False),'cb')
    eligible=[];counts={}
    for airport in AIRPORTS:
        if airport=='LIRF':continue
        local=tr.loc[tr.ADEP.eq(airport)];counts[airport]=len(local)
        if len(local)<MIN_LOCAL_ROWS:continue
        eligible.append(airport)
        fit_regressor(folder/('local_'+airport),local,target(local,False),M9_CB_FEATURES,MAIN_CATS,
                      dict(LOCAL_PARAMS,n_estimators=LOCAL_TREES))
    write_json(folder/'routing.json',{'local_airports':eligible,'local_counts':counts,'local_weight':WEIGHT,
                                    'present_airports':AIRPORTS,'present_models':list(MAIN_MODELS)})


def predict_models(query,run_dir,lirf_prediction):
    folder=Path(run_dir)/'estimators';routing=json.loads((folder/'routing.json').read_text())
    prediction=np.full(len(query),np.nan)
    for airport in AIRPORTS:
        mask=(~query._missing & query.ADEP.eq(airport)).to_numpy();q=query.loc[mask]
        if q.empty:continue
        components=[predict_regressor(folder/(airport+'_'+key),q) for key in routing['present_models']]
        prediction[mask]=np.mean(components,axis=0)
        print(f'PREDICT {airport}: AOBT present={len(q):,}',flush=True)
    mask=(query._missing & ~query._special).to_numpy();q=query.loc[mask]
    if len(q):
        a=predict_regressor(folder/'missing9_lgb',q);c=predict_regressor(folder/'missing9_cb',q)
        prediction[mask]=(a+q.airside_est.to_numpy(float)+c)/3
    prediction[query._special.to_numpy()]=lirf_prediction
    for airport in routing['local_airports']:
        mask=(query._missing & ~query._special & query.ADEP.eq(airport)).to_numpy();q=query.loc[mask]
        if q.empty:continue
        p=predict_regressor(folder/('local_'+airport),q)
        prediction[mask]=(1-routing['local_weight'])*prediction[mask]+routing['local_weight']*p
    if not np.isfinite(prediction).all():raise ValueError('Uncovered model branch/nonfinite prediction')
    return prediction
