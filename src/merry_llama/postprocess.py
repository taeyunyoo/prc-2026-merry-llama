"""Apply the submitted, frozen v54 then v55 Category95 rules."""
from pathlib import Path
import numpy as np
import pandas as pd
from .model_config import ID, AOBT, SCHEDULE
from .model_utils import clean

FIELDS=['ADES','AIRCRAFT_OPERATOR_flt','FLIGHT_mvt','CALLSIGN_flt','AIRCRAFT_TYPE',
        'STAND_mvt','RUNWAY_mvt','callsign_prefix3','flight_prefix2']


def rounded(p):
    p=np.asarray(p,float)
    if not np.isfinite(p).all():raise ValueError('Nonfinite prediction')
    return np.floor(np.maximum(p,0)+.5)


def load_rules(folder):
    rules=[]
    for version in ['v54','v55']:
        r=pd.read_csv(Path(folder)/f'{version}_category95_rules.csv',keep_default_na=False,dtype={'value':'string'})
        if version=='v54':
            if not r.selected.astype(str).str.lower().eq('true').all():raise ValueError('Unselected v54 rule')
            r['airport']='LIRF';r['aobt_state']='missing'
        if not r.feature.isin(FIELDS).all() or not r.aobt_state.isin(['present','missing']).all():raise ValueError('Invalid rule fields')
        n=pd.to_numeric(r.n);k=pd.to_numeric(r.matches);rate=pd.to_numeric(r.rate)
        if not ((n>=10)&(n%1==0)&(k%1==0)&(k>=0)&(k<=n)&(rate>=.95)).all() or not np.allclose(rate,k/n):raise ValueError('Invalid frozen rule statistics')
        if r[['feature','airport','aobt_state','value']].duplicated().any() or clean(r.value).isna().any():raise ValueError('Invalid rule keys')
        r['version']=version;rules.append(r)
    return pd.concat(rules,ignore_index=True)


def apply_rules(frame,prediction,folder):
    rules=load_rules(folder);p=rounded(prediction)
    airport=clean(frame.ADEP)
    state=pd.Series(np.where(clean(frame[AOBT]).isna(),'missing','present'),index=frame.index)
    x={}
    for f in rules.feature.unique():
        if f=='callsign_prefix3':x[f]=clean(frame.CALLSIGN_flt).str[:3]
        elif f=='flight_prefix2':x[f]=clean(frame.FLIGHT_mvt).str[:2]
        else:x[f]=clean(frame[f])
    schedule=pd.to_numeric(frame[SCHEDULE],errors='coerce').to_numpy(float)
    finite=np.isfinite(schedule);audits=[];summary=[]
    for version in ['v54','v55']:
        old=p.copy();flag=np.zeros(len(p),bool);reasons=[[] for _ in range(len(p))]
        for r in rules.loc[rules.version.eq(version)].itertuples():
            m=(airport.eq(r.airport)&state.eq(r.aobt_state)&x[r.feature].eq(r.value)).fillna(False).to_numpy(bool)
            if version=='v54' and (m&~finite).any():raise ValueError('v54 flagged nonfinite schedule')
            m &= finite;flag |= m
            for i in np.flatnonzero(m):reasons[i].append(f'{r.feature}={r.value}')
        p[flag]=rounded(schedule[flag])
        ix=np.flatnonzero(flag)
        audits.append(pd.DataFrame({ID:frame[ID].to_numpy()[ix],'stage':version,'airport':airport.to_numpy()[ix],
            'aobt_state':state.to_numpy()[ix],'before':old[ix],'after':p[ix],
            'changed':p[ix]!=old[ix],'matched_rules':[' | '.join(reasons[i]) for i in ix]}))
        summary.append({'stage':version,'flagged':int(flag.sum()),'changed':int((old!=p).sum())})
    return p,pd.concat(audits,ignore_index=True),summary
