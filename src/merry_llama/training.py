"""STEP 04 orchestration; no training/ranking/final target leakage at inference."""
from pathlib import Path
from contextlib import contextmanager
from importlib.metadata import version, PackageNotFoundError
import gc
import json
import os
import platform
import shutil
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .data import ProjectPaths, training_names
from .features.base import digest, write_json
from .make_dataset import validate_model_data
from .model_config import REQUIRED_VERSIONS, ID, TARGET
from . import model_config
from .model_utils import assert_ids
from .model_store import load_record
from .learned_features import read_inputs, fit_learned, apply_learned
from .lirf_model import fit_lirf, read_lirf, predict_lirf
from .modeling import fit_models, predict_models
from .postprocess import apply_rules, load_rules
from .submission import read_template, write_submission

CONTRACT='prc2026.models.v1'
MODULES=['model_config.py','model_utils.py','model_store.py','learned_features.py','lirf_model.py',
         'modeling.py','postprocess.py','submission.py','training.py']
# These two released modules differed only in orchestration/reporting and the
# now-removed comparison helper. Model fitting and prediction arithmetic are
# unchanged. Accept that exact release without rewriting saved model manifests.
LEGACY_WORKFLOW_HASHES={
    'training.py':'36bed6708a7d3e9a1aa51121cdb6192f1509806a890c360592b8fe8aaad55e4d',
    'submission.py':'edb9ff128bb153a58d457f0de8fec44bfa46cbc808fbc224a03639ed9e35772c',
}


def implementation():
    folder=Path(__file__).parent
    return {name:digest(folder/name) for name in MODULES}


def compatible_implementation(saved):
    current=implementation()
    return saved==current or saved=={**current,**LEGACY_WORKFLOW_HASHES}


def environment(allow_version_mismatch=False):
    actual={};missing=[]
    for name,expected in REQUIRED_VERSIONS.items():
        try:actual[name]=version(name)
        except PackageNotFoundError:missing.append(name)
    mismatch={n:{'expected':v,'installed':actual.get(n)} for n,v in REQUIRED_VERSIONS.items() if actual.get(n)!=v}
    if missing or (mismatch and not allow_version_mismatch):
        raise RuntimeError('Model environment differs from the source run:\n'+json.dumps(mismatch,indent=2)+
            '\nUse the original traffic environment, or install requirements-model.txt in a dedicated environment. '
            'No packages or completed data were changed. --allow-version-mismatch permits an explicitly non-exact experiment only.')
    print('Model versions:',json.dumps(actual),flush=True)
    if mismatch:print('NON-EXACT ENVIRONMENT EXPERIMENT:',json.dumps(mismatch),flush=True)
    return {'python':platform.python_version(),'packages':actual,'mismatch':mismatch}


def run_paths(project,run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('run-id may contain letters, numbers, underscores and hyphens only')
    return project.model_dir/run_id, project.report_dir/run_id


@contextmanager
def locked(folder):
    folder.mkdir(parents=True,exist_ok=True);lock=folder/'RUNNING.lock'
    try:fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'Run lock exists: {lock}. Wait for the active run. After a crash, confirm the old process has stopped before deleting only this lock.')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:yield
    finally:lock.unlink(missing_ok=True)


def snapshot_files(folder):
    return {p.relative_to(folder).as_posix():digest(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p != folder/'complete.json' and p.name != 'RUNNING.lock'}


def validate_run(folder):
    marker=folder/'complete.json'
    d=json.loads(marker.read_text())
    if d.get('contract')!=CONTRACT:raise ValueError('Unknown saved-model contract')
    cfg=json.loads((folder/'run_config.json').read_text())
    if not compatible_implementation(cfg['implementation']):raise ValueError('Saved run uses different model code. Use its matching checkout.')
    for relative,h in d['files'].items():
        p=folder/relative
        if not p.is_file() or digest(p)!=h:raise ValueError(f'Saved model changed/missing: {p}')
    # Also validate nested completion records (excluded from the flat snapshot).
    for record in folder.rglob('complete.json'):
        if record!=marker:load_record(record.parent)
    return cfg


def preflight(project,allow_version_mismatch=False):
    env=environment(allow_version_mismatch)
    data=project.model_data_dir/'full'
    paths=[data/n for n in training_names()]
    rank=data/'ranking.parquet';template=project.raw_dir/'submitting.parquet'
    for p in paths+[rank,template]:
        if not p.is_file():raise FileNotFoundError(p)
    required=set(model_config.PRESENT_FEATURES+model_config.DEP_FEATURES+model_config.SIDE_FEATURES+
                 ['cf_config',ID,model_config.TIME,model_config.AOBT,'PHASE_mvt'])
    required-={model_config.MEDIAN,'cf_recent_vs_usual_rwy'}
    for p in paths+[rank]:
        missing=required-set(pq.read_schema(p).names)
        if p in paths:missing|={TARGET,'BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt'}-set(pq.read_schema(p).names)
        if missing:raise ValueError(f'{p.name}: missing {sorted(missing)}')
    r=pq.read_table(rank,columns=[ID]).to_pandas();assert_ids(r,'ranking')
    read_template(template,r)
    load_rules(project.config_dir)
    provenance=json.loads((project.config_dir/'model_provenance.json').read_text())
    for name,h in provenance['rules_sha256'].items():
        if digest(project.config_dir/name)!=h:raise ValueError('Frozen rule asset changed: '+name)
    print('Verifying completed STEP 03 and upstream inputs (SHA-256)...',flush=True)
    validate_model_data(project.root)
    print('CHECK COMPLETE: environment, STEP 03, model columns, rules and ranking/template IDs.',flush=True)
    return env,paths


def _predict(project,folder,dataset):
    if dataset not in {'ranking','final'}:raise ValueError('dataset must be ranking or final')
    name='ranking.parquet' if dataset=='ranking' else 'final_ranking.parquet'
    template_path=project.raw_dir/('submitting.parquet' if dataset=='ranking' else 'final_submitting.parquet')
    path=project.model_data_dir/'full'/name
    query=read_inputs([path])
    template=read_template(template_path,query)
    query=apply_learned(query,project.model_data_dir/'_movement'/name,folder)
    raw=read_lirf([path],training=False)
    raw=raw.set_index(ID).loc[query.loc[query._special,ID]].reset_index()
    lirf,_=predict_lirf(raw,folder)
    prediction=predict_models(query,folder,lirf)
    prediction,_,rule_summary=apply_rules(query,prediction,folder/'configs')
    project.submission_dir.mkdir(parents=True,exist_ok=True)
    output=project.submission_dir/('merry-llama.parquet' if dataset=='ranking' else 'merry-llama_final.parquet')
    # Distinct run IDs must not silently overwrite a previous run's deliverable.
    if output.exists():
        previous=pq.read_table(output,columns=[ID,TARGET]).to_pandas()
        current=pd.Series(prediction,index=query[ID]).reindex(previous[ID]).to_numpy(float)
        if len(previous)!=len(query) or not np.array_equal(previous[TARGET].to_numpy(float),current):
            output=project.submission_dir/(output.stem+'_'+folder.name+'.parquet')
    write_submission(template,query,prediction,output)
    report={'status':'complete','dataset':dataset,'rows':len(query),'output':str(output),
            'sha256':digest(output),'rules':rule_summary,'saved_model_inference':True}
    print(json.dumps(report,indent=2),flush=True)
    return report


def train(root=None,run_id='v55_reproduction',threads=-1,check_only=False,allow_version_mismatch=False):
    if threads==0 or threads < -1:raise ValueError('threads must be -1 or positive')
    project=ProjectPaths.from_root(root);folder,_=run_paths(project,run_id)
    env,paths=preflight(project,allow_version_mismatch)
    if check_only:return
    started=time.time()
    config={'contract':CONTRACT,'implementation':implementation(),'environment':env,'threads':threads,
            'step03_sha256':digest(project.model_data_dir/'step03_complete.json'),
            'config_files':{name:digest(project.config_dir/name) for name in ['model_provenance.json','v54_category95_rules.csv','v55_category95_rules.csv']},
            'method':'v52 + frozen v54 + frozen v55; full refit, fixed trees; no holdout RMSE claimed'}
    with locked(folder):
        cfg=folder/'run_config.json'
        if cfg.exists():
            saved=json.loads(cfg.read_text())
            if not compatible_implementation(saved['implementation']):
                raise ValueError('Saved run uses different model code. Use its matching checkout.')
            if {**saved,'implementation':config['implementation']}!=config:
                raise ValueError('Inputs/code/environment/settings changed. Keep this run intact and choose a new --run-id.')
        else:
            write_json(cfg,config)
            shutil.copytree(project.config_dir,folder/'configs',dirs_exist_ok=True)
            shutil.copytree(project.model_data_dir/'_references',folder/'feature_references',dirs_exist_ok=True)
            settings={k:v for k,v in vars(model_config).items() if k.isupper() or k in ['lirf_CORE','lirf_OPS','lirf_CATS','lirf_REG_PARAMS']}
            write_json(folder/'resolved_model_settings.json',settings)
        if (folder/'complete.json').exists():
            validate_run(folder);print('REUSE completed model bundle.',flush=True)
        else:
            fit_lirf(paths,folder)
            data=read_inputs(paths)
            data=fit_learned(data,project.model_data_dir,folder)
            fit_models(data,folder,threads)
            del data;gc.collect()
            # Recheck data after a long fit before committing its model bundle.
            validate_model_data(project.root)
            write_json(folder/'complete.json',{'contract':CONTRACT,'status':'complete','files':snapshot_files(folder)})
            validate_run(folder)
        report=_predict(project,folder,'ranking')
    print(f'COMPLETE: model training and ranking submission ({report["rows"]:,} rows).',flush=True)
    print(f'Elapsed minutes: {(time.time()-started)/60:.1f}',flush=True)
    return report


def predict(root=None,run_id='v55_reproduction',dataset='final'):
    if dataset not in {'ranking','final'}:raise ValueError('dataset must be ranking or final')
    project=ProjectPaths.from_root(root);folder,_=run_paths(project,run_id)
    cfg=validate_run(folder)
    env=environment(bool(cfg['environment']['mismatch']))
    if env!=cfg['environment']:raise ValueError('Inference environment differs from saved training environment')
    validate_model_data(project.root)
    if digest(project.model_data_dir/'step03_complete.json')!=cfg['step03_sha256']:raise ValueError('STEP 03 differs from saved run')
    with locked(folder):return _predict(project,folder,dataset)
