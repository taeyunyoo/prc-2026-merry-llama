"""Frozen v42 arrival feature engine (40 columns).

Derived from the supplied cell6_arrival_features_v42.py, with inactive
experiments removed. Definitions and calculation order are preserved, including
historical window/fallback choices. An empty direction-change list yields the
source's intended 1440-minute sentinel instead of raising IndexError.

This is batch competition feature engineering, not an online prediction API.
The next-stand feature intentionally accesses a later arrival record. The
baseline 55 features, including heavy/super counts, are never overwritten here.
"""
import numpy as np
import pandas as pd
from .definitions import TIME, AOBT, ARRIVAL_FEATURES
from .base import periods
EPOCH = pd.Timestamp('1970-01-01', tz='UTC')
RWY_WIN = 60

def to_sec(s):
    t = pd.to_datetime(s, utc=True, errors='coerce', format='mixed')
    return ((t - EPOCH) / pd.Timedelta('1s')).to_numpy(float)

def cnt(sorted_times, lo, hi):
    return (np.searchsorted(sorted_times, hi, 'left') - np.searchsorted(sorted_times, lo, 'left')).astype(float)

def add_arrival_features(dep, arr, *, scope="full", context="ranking", references=None):
    """Return (enriched departures, fitted reference tables).

    Passing references applies saved training state without fitting new tables.
    scope controls the original reference cutoff; context controls coverage only.
    """
    if scope not in {"full", "validation"} or context not in {"ranking", "final_ranking"}:
        raise ValueError("Invalid arrival scope/context")
    final = scope == "full"
    fitting = references is None
    state = {} if fitting else dict(references)

    def reference(name, fit):
        if fitting:
            state[name] = fit()
        if name not in state:
            raise ValueError(f"Missing saved arrival reference: {name}")
        return state[name]

    cutoff = pd.Timestamp('2026-01-01' if final else '2025-11-01', tz='UTC')
    T = to_sec(dep[TIME])
    A = to_sec(dep[AOBT])
    S = to_sec(dep['SCHED_TIME_UTC_mvt'])
    R = np.where(np.isnan(A), S, A)
    coverage = periods("training") + periods(context)
    starts = np.array([start.timestamp() for start, _ in coverage])
    ends = np.array([end.timestamp() for _, end in coverage])
    k = np.clip(np.searchsorted(starts, T, 'right') - 1, 0, len(starts) - 1)
    p0, p1 = starts[k], ends[k]
    a = arr
    a['L'] = to_sec(a.LAND)
    a['B'] = to_sec(a.INBLOCK)
    a['S'] = to_sec(a.SCHED)
    ok_lb = (a.B >= a.L) & (a.B - a.L < 7200)
    past = (a.LAND < cutoff).to_numpy()
    ga = a.groupby('APT').indices
    new = {}

    def put(name, idx, v):
        new.setdefault(name, np.full(len(dep), np.nan))
        new[name][idx] = v
    apt_med = reference('apt_med', lambda: a[past & a.TAXI_IN.between(0, 7200)].groupby('APT').TAXI_IN.median())
    st_med = reference('st_med', lambda: a[past & a.TAXI_IN.between(0, 7200)].groupby(['APT', 'STAND']).TAXI_IN.median())
    rw_num = pd.to_numeric(a.RUNWAY.astype('string').str[:2], errors='coerce').to_numpy(float)
    ref_dir = reference('ref_dir', lambda: pd.Series(rw_num[past]).groupby(a.APT.to_numpy()[past]).agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan))
    same_dir = np.abs((rw_num * 10 - a.APT.map(ref_dir).to_numpy(float) * 10 + 180) % 360 - 180) < 90
    a['DIR'] = np.where(np.isnan(rw_num), np.nan, same_dir.astype(float))
    for apt, di in dep.groupby('ADEP').indices.items():
        print(f'  Arrival statistics: {apt}', flush=True)
        ai = ga.get(apt)
        if ai is None:
            continue
        g = a.iloc[ai]
        t, r, aa = (T[di], R[di], A[di])
        pastok = lambda lo: lo >= p0[di]
        futok = lambda hi: hi <= p1[di]
        L = np.sort(g.L.dropna().to_numpy())
        gv = g[ok_lb.to_numpy()[ai]]
        Lv = np.sort(gv.L.to_numpy())
        Bv = np.sort(gv.B.to_numpy())
        on_ground = lambda q: (np.searchsorted(Lv, q, 'right') - np.searchsorted(Bv, q, 'right')).astype(float)
        h = g[g.B.notna() & g.TAXI_IN.between(0, 7200)].sort_values('B')
        base = pd.Series(list(zip(h.APT, h.STAND))).map(st_med).fillna(apt_med.get(apt, np.nan)).to_numpy(float)
        bs = h.B.to_numpy()
        tx = h.TAXI_IN.to_numpy(float)
        ex = tx - base
        c1 = np.concatenate([[0], np.cumsum(tx)])
        c2 = np.concatenate([[0], np.cumsum(np.nan_to_num(ex))])

        def wmean(c, lo_t, hi_t):
            hi = np.searchsorted(bs, hi_t, 'left')
            lo = np.searchsorted(bs, lo_t, 'left')
            n = hi - lo
            return (np.where(n > 0, (c[hi] - c[lo]) / np.maximum(n, 1), np.nan), n.astype(float))
        m30, n30 = wmean(c1, t - 1800, t)
        m60, _ = wmean(c1, t - 3600, t - 1800)
        e30, _ = wmean(c2, t - 1800, t)
        bad = ~pastok(t - 3600)
        for nm, v in [('arr_taxiin_mean_prev30', m30), ('arr_taxiin_n_prev30', n30), ('arr_taxiin_excess_prev30', e30), ('arr_taxiin_trend', m30 - m60)]:
            v = v.copy()
            v[bad] = np.nan
            put(nm, di, v)
        h = g[g.L.notna()].sort_values('L')
        ls = h.L.to_numpy()
        rw = h.RUNWAY.astype('string').fillna('NA').to_numpy()
        hi = np.searchsorted(ls, t, 'left')
        lo = np.searchsorted(ls, t - RWY_WIN * 60, 'left')
        n = hi - lo
        rws = np.unique(rw)
        if not len(rws):
            raise ValueError(f'{apt}: no valid arrival landing times for v42 runway features')
        M = np.stack([np.concatenate([[0], np.cumsum(rw == x)])[hi] - np.concatenate([[0], np.cumsum(rw == x)])[lo] for x in rws], 1)
        P = M / np.maximum(n, 1)[:, None]
        dom = np.where(n > 0, np.array([f'{apt}_{x}' for x in rws], dtype=object)[P.argmax(1)], None)
        new.setdefault('arr_rwy_dom', np.full(len(dep), None, dtype=object))
        new['arr_rwy_dom'][di] = dom
        put('arr_rwy_dom_share60', di, np.where(n > 0, P.max(1), np.nan))
        with np.errstate(divide='ignore', invalid='ignore'):
            put('arr_rwy_entropy60', di, np.where(n > 0, -np.where(P > 0, P * np.log2(P), 0).sum(1), np.nan))
        d = np.nan_to_num(h.DIR.to_numpy())
        cd = np.concatenate([[0], np.cumsum(d)])
        lo6 = np.searchsorted(ls, t - 3600, 'left')
        n6 = hi - lo6
        main = np.where(n6 > 0, ((cd[hi] - cd[lo6]) / np.maximum(n6, 1) >= 0.5).astype(float), np.nan)
        put('arr_dir_main60', di, main)
        hL = np.arange(1, len(ls) + 1)
        lL = np.searchsorted(ls, ls - 3600, 'left')
        st = ((cd[hL] - cd[lL]) / np.maximum(hL - lL, 1) >= 0.5).astype(int)
        chg = ls[np.where(np.diff(st) != 0)[0] + 1]
        kk = np.searchsorted(chg, t, 'right') - 1
        put('arr_min_since_dir_change', di, np.minimum(np.where(kk >= 0, (t - chg[np.maximum(kk, 0)]) / 60, 1440), 1440) if len(chg) else np.full(len(t), 1440.0))
        myrw = dep.RUNWAY_mvt.iloc[di].astype('string').fillna('NA').to_numpy()
        for w in (30, 60):
            lo2 = np.searchsorted(ls, t - w * 60, 'left')
            same = np.zeros(len(di))
            for j, x in enumerate(rws):
                c = np.concatenate([[0], np.cumsum(rw == x)])
                s_ = myrw == x
                same[s_] = (c[hi] - c[lo2])[s_]
            n2 = (hi - lo2).astype(float)
            v = same.astype(float)
            v[~pastok(t - w * 60)] = np.nan
            put(f'arr_same_rwy_prev{w}', di, v)
            put(f'arr_same_rwy_share{w}', di, np.where(n2 > 0, v / np.maximum(n2, 1), np.nan))
    print('  Matching prior arrivals (4B/N1)...', flush=True)
    q = pd.DataFrame({'i': np.arange(len(dep)), 'APT': dep.ADEP.astype(str).to_numpy(), 'STAND': dep.STAND_mvt.astype(str).to_numpy(), 'TYPE': dep.AIRCRAFT_TYPE.astype(str).to_numpy(), 'AL': dep.FLIGHT_mvt.astype('string').str[:2].astype(str).to_numpy(), 'R': R, 'T': T, 'S': S})
    q = q[~np.isnan(q.R) & ~dep.STAND_mvt.isna().to_numpy()].sort_values('R')
    h = a[a.B.notna() & a.STAND.notna()].copy()
    h = h.assign(APT=h.APT.astype(str), STAND=h.STAND.astype(str), TYPE=h.TYPE.astype(str), AL=h.FLIGHT.astype('string').str[:2].astype(str), R=h.B, IN_S=h.S, IN_L=h.L, IN_TX=h.TAXI_IN).sort_values('R')
    keys = ['APT', 'STAND', 'TYPE', 'AL']
    m = pd.merge_asof(q, h[keys + ['R', 'B', 'IN_S', 'IN_L', 'IN_TX']], on='R', by=keys, direction='backward', tolerance=86400.0)
    last_any = pd.merge_asof(q[['APT', 'STAND', 'R']], h[['APT', 'STAND', 'R', 'B']].rename(columns={'B': 'B_any'}), on='R', by=['APT', 'STAND'], direction='backward')['B_any'].to_numpy()
    dq = q[['APT', 'STAND', 'R']].assign(R_prev=q.R)
    prev_dep = pd.merge_asof(q[['APT', 'STAND', 'R']], dq, on='R', by=['APT', 'STAND'], direction='backward', allow_exact_matches=False)['R_prev'].to_numpy()
    B = m.B.to_numpy()
    turn = q.R.to_numpy() - B
    strict = ~np.isnan(B) & (last_any == B) & (np.isnan(prev_dep) | (prev_dep < B)) & (turn >= 900) & (turn <= 43200)
    use = strict
    idx = q.i.to_numpy()
    put('inb_turnaround_sec', idx, np.where(use, turn, np.nan))
    put('inb_T_minus_inblock_sec', idx, np.where(use, q['T'].to_numpy() - B, np.nan))
    put('inb_sched_turn_sec', idx, np.where(use, q.S.to_numpy() - m.IN_S.to_numpy(), np.nan))
    put('inb_match_strict', idx, np.where(np.isnan(B), np.nan, strict.astype(float)))
    g_ = B - m.IN_S.to_numpy()
    rec = (np.abs(g_) <= 60) | (np.abs(g_ + 3600) <= 60) | (np.abs(g_ + 7200) <= 60)
    put('inb_sched_record', idx, np.where(use & ~np.isnan(g_), rec.astype(float), np.nan))
    trainrows = (~dep._ranking & (dep._month.le(10) | final)).to_numpy()
    print('  Matching supplementary arrivals and typical stay (B2)...', flush=True)
    d_al = dep.FLIGHT_mvt.astype('string').str[:2].astype(str).to_numpy()
    a_al = a.FLIGHT.astype('string').str[:2].astype(str).to_numpy()
    hv_ok = (a.B.notna() & a.STAND.notna()).to_numpy()

    def codes(cols_dep, cols_arr):
        n = len(dep)
        mi = pd.MultiIndex.from_arrays([np.concatenate([x, y]) for x, y in zip(cols_dep, cols_arr)])
        c = pd.factorize(mi)[0]
        return (c[:n], c[n:])
    k4d, k4a = codes([dep.ADEP.astype(str).to_numpy(), dep.STAND_mvt.astype(str).to_numpy(), dep.AIRCRAFT_TYPE.astype(str).to_numpy(), d_al], [a.APT.astype(str).to_numpy()[hv_ok], a.STAND.astype(str).to_numpy()[hv_ok], a.TYPE.astype(str).to_numpy()[hv_ok], a_al[hv_ok]])
    k2d, k2a = codes([dep.ADEP.astype(str).to_numpy(), dep.STAND_mvt.astype(str).to_numpy()], [a.APT.astype(str).to_numpy()[hv_ok], a.STAND.astype(str).to_numpy()[hv_ok]])
    hB = a.B.to_numpy()[hv_ok]
    H4 = pd.DataFrame({'k': k4a, 'R': hB, 'B': hB}).sort_values('R')
    H2 = pd.DataFrame({'k': k2a, 'R': hB, 'B_any': hB}).sort_values('R')
    hv = np.sort(k4a.astype(float) * 10000000000.0 + hB)

    def match(Rv):
        ok = ~np.isnan(Rv) & ~dep.STAND_mvt.isna().to_numpy()
        q = pd.DataFrame({'i': np.arange(len(dep))[ok], 'k4': k4d[ok], 'k2': k2d[ok], 'R': Rv[ok]}).sort_values('R')
        m = pd.merge_asof(q, H4.rename(columns={'k': 'k4'}), on='R', by='k4', direction='backward', tolerance=86400.0)
        last_any = pd.merge_asof(q[['k2', 'R']], H2.rename(columns={'k': 'k2'}), on='R', by='k2', direction='backward')['B_any'].to_numpy()
        prev = pd.merge_asof(q[['k2', 'R']], q[['k2', 'R']].assign(Rp=q.R), on='R', by='k2', direction='backward', allow_exact_matches=False)['Rp'].to_numpy()
        Bm = m.B.to_numpy()
        rr = q.R.to_numpy()
        turn = rr - Bm
        strict = ~np.isnan(Bm) & (last_any == Bm) & (np.isnan(prev) | (prev < Bm)) & (turn >= 900) & (turn <= 43200)
        base = q.k4.to_numpy().astype(float) * 10000000000.0
        cn = np.searchsorted(hv, base + rr, 'left') - np.searchsorted(hv, base + rr - 86400, 'left')
        out = pd.DataFrame(index=q.i.to_numpy(), data={'B': Bm, 'strict': strict.astype(float), 'cand': cn.astype(float)})
        return out.reindex(np.arange(len(dep)))
    m1 = match(R)
    m2 = match(np.where(np.isnan(A), T, A))
    use1 = m1.strict.fillna(0).to_numpy() == 1
    use2 = m2.strict.fillna(0).to_numpy() == 1
    turn = np.where(use1 & ~np.isnan(A), A - m1.B.to_numpy(), np.nan)
    al = dep.FLIGHT_mvt.astype('string').str[:2].astype(str).to_numpy()
    ty = dep.AIRCRAFT_TYPE.astype(str).to_numpy()
    ap = dep.ADEP.astype(str).to_numpy()
    ok = trainrows & ~np.isnan(turn)
    ref = pd.DataFrame({'ap': ap[ok], 'al': al[ok], 'ty': ty[ok], 't': turn[ok]})
    g3 = reference('g3', lambda: ref.groupby(['ap', 'al', 'ty']).t.agg(['median', 'size']))
    g2 = reference('g2', lambda: ref.groupby(['ap', 'ty']).t.agg(['median', 'size']))
    g1 = reference('g1', lambda: ref.groupby('ap').t.median())
    k3 = pd.MultiIndex.from_arrays([ap, al, ty])
    k2 = pd.MultiIndex.from_arrays([ap, ty])
    med3 = g3['median'].where(g3['size'] >= 30).reindex(k3).to_numpy()
    med2 = g2['median'].where(g2['size'] >= 30).reindex(k2).to_numpy()
    typical = np.where(~np.isnan(med3), med3, np.where(~np.isnan(med2), med2, pd.Series(ap).map(g1).to_numpy(float)))
    put('inb_turn_dev', np.arange(len(dep)), turn - typical)
    put('inb_turn_ref_level', np.arange(len(dep)), np.where(~np.isnan(med3), 3.0, np.where(~np.isnan(med2), 2.0, 1.0)))
    put('inb_slack_sec', np.arange(len(dep)), np.where(use2, S - m2.B.to_numpy(), np.nan))
    put('inb2_T_minus_inblock', np.arange(len(dep)), np.where(use2, T - m2.B.to_numpy(), np.nan))
    put('inb_cand_n', np.arange(len(dep)), m2.cand.to_numpy())
    put('inb_conf', np.arange(len(dep)), np.where(m2.B.isna(), 0.0, np.where(m2.strict.fillna(0).to_numpy() == 1, 2.0, 1.0)))
    print('  Route-adjusted taxi-in (C2)...', flush=True)
    ok = a.B.notna() & a.L.notna() & a.TAXI_IN.between(0, 7200)
    pa = a[ok & past]
    g_rs = reference('g_rs', lambda: pa.groupby(['APT', 'RUNWAY', 'STAND']).TAXI_IN.agg(med='median', n='size', p80=lambda s: s.quantile(0.8)))
    g_r = reference('g_r', lambda: pa.groupby(['APT', 'RUNWAY']).TAXI_IN.agg(med='median', p80=lambda s: s.quantile(0.8)))
    h = a[ok]
    k_rs = pd.MultiIndex.from_arrays([h.APT, h.RUNWAY, h.STAND])
    k_r = pd.MultiIndex.from_arrays([h.APT, h.RUNWAY])
    n = g_rs.n.reindex(k_rs).fillna(0).to_numpy()
    w = n / (n + 20.0)
    up_med = g_r.med.reindex(k_r).to_numpy()
    up_med = np.where(np.isnan(up_med), h.APT.map(apt_med).to_numpy(float), up_med)
    up_p80 = g_r.p80.reindex(k_r).to_numpy()
    base = np.where(n > 0, w * np.nan_to_num(g_rs.med.reindex(k_rs).to_numpy()) + (1 - w) * up_med, up_med)
    p80 = np.where(n > 0, w * np.nan_to_num(g_rs.p80.reindex(k_rs).to_numpy()) + (1 - w) * np.nan_to_num(up_p80, nan=np.nan), up_p80)
    hh = pd.DataFrame({'APT': h.APT.to_numpy(), 'B': h.B.to_numpy(), 'dev': h.TAXI_IN.to_numpy(float) - base, 'exc': (h.TAXI_IN.to_numpy(float) > p80).astype(float)}).dropna(subset=['dev'])
    for apt, di in dep.groupby('ADEP').indices.items():
        g = hh[hh.APT == apt].sort_values('B')
        if g.empty:
            continue
        bs = g.B.to_numpy()
        t = T[di]
        c1 = np.concatenate([[0], np.cumsum(g.dev.to_numpy())])
        c2 = np.concatenate([[0], np.cumsum(g.exc.to_numpy())])
        hi = np.searchsorted(bs, t, 'left')
        lo = np.searchsorted(bs, t - 1800, 'left')
        nn = hi - lo
        bad = t - 3600 < p0[di]
        v = np.where(nn > 0, (c1[hi] - c1[lo]) / np.maximum(nn, 1), np.nan)
        v[bad] = np.nan
        put('arr_taxiin_routedev_mean30', di, v)
        v = np.where(nn > 0, (c2[hi] - c2[lo]) / np.maximum(nn, 1), np.nan)
        v[bad] = np.nan
        put('arr_taxiin_p80_share30', di, v)
        rm = pd.Series(g.dev.to_numpy(), index=pd.to_datetime(bs, unit='s')).rolling('30min').median().to_numpy()
        last = hi - 1
        v = np.where((nn > 0) & (last >= 0), rm[np.maximum(last, 0)], np.nan)
        v[bad] = np.nan
        put('arr_taxiin_routedev_med30', di, v)
        put('arr_taxiin_since_last', di, np.where(hi > 0, (t - bs[np.maximum(hi - 1, 0)]) / 60, np.nan))
    print('  Next stand arrival (NXa)...', flush=True)
    okb = (a.B.notna() & a.STAND.notna()).to_numpy()
    hh = pd.DataFrame({'APT': a.APT.astype(str).to_numpy()[okb], 'ST': a.STAND.astype(str).to_numpy()[okb], 'B': a.B.to_numpy()[okb], 'L': a.L.to_numpy()[okb]}).sort_values('B')
    gs = {k: (g.B.to_numpy(), g.L.to_numpy()) for k, g in hh.groupby(['APT', 'ST'])}
    nx = np.full(len(dep), np.nan)
    lx = np.full(len(dep), np.nan)
    key_apt = dep.ADEP.astype(str).to_numpy()
    key_st = dep.STAND_mvt.astype(str).to_numpy()
    for (ap_, st_), di in pd.Series(np.arange(len(dep))).groupby([key_apt, key_st]).indices.items():
        v = gs.get((ap_, st_))
        if v is None:
            continue
        b, l = v
        r = R[di]
        ok = ~np.isnan(r)
        j = np.searchsorted(b, r[ok] - 600, 'left')
        has = j < len(b)
        t1 = np.full(ok.sum(), np.nan)
        t2 = np.full(ok.sum(), np.nan)
        t1[has] = b[j[has]]
        t2[has] = l[j[has]]
        nx[di[ok]] = t1
        lx[di[ok]] = t2
    miss_st = dep.STAND_mvt.isna().to_numpy()
    nx[miss_st] = np.nan
    lx[miss_st] = np.nan
    put('stand_next_inblock_minus_aobt', np.arange(len(dep)), np.where(np.isnan(A), np.nan, nx - A))
    for apt, di in dep.groupby('ADEP').indices.items():
        ai = ga.get(apt)
        if ai is None:
            continue
        g = a.iloc[ai]
        t, aa = (T[di], A[di])
        bad = lambda lo: lo < p0[di]
        Lg = g.L.to_numpy()
        Bg = g.B.to_numpy()
        Sg = g.S.to_numpy()
        okk = ~np.isnan(Bg) & ~np.isnan(Sg)
        o = np.argsort(Bg[okk])
        bs = Bg[okk][o]
        gg = bs - Sg[okk][o]
        rec = ((np.abs(gg) <= 60) | (np.abs(gg + 3600) <= 60) | (np.abs(gg + 7200) <= 60)).astype(float)
        cr = np.concatenate([[0], np.cumsum(rec)])
        for nm, qq in [('T', t), ('A', aa)]:
            q_ = np.nan_to_num(qq)
            hi = np.searchsorted(bs, q_, 'left')
            lo = np.searchsorted(bs, q_ - 7200, 'left')
            n = (hi - lo).astype(float)
            sh = np.where(n > 0, (cr[hi] - cr[lo]) / np.maximum(n, 1), np.nan)
            inval = bad(q_ - 7200) | np.isnan(qq)
            sh[inval] = np.nan
            put(f'arr_schedrec_share120_{nm}', di, sh)
            if nm == 'T':
                n[inval] = np.nan
                put('arr_schedrec_n120_T', di, n)
        okL = ~np.isnan(Lg)
        oL = np.argsort(Lg[okL])
        ls = Lg[okL][oL]
        rw_l = g.RUNWAY.astype('string').fillna('NA').to_numpy(dtype=object).astype(str)[okL][oL]
        myrw = dep.RUNWAY_mvt.iloc[di].astype('string').fillna('NA').to_numpy(dtype=object).astype(str)
        same30 = np.zeros(len(di))
        during = np.full(len(di), np.nan)
        prev20 = np.zeros(len(di))
        since = np.full(len(di), np.nan)
        gmed = np.full(len(di), np.nan)
        gstd = np.full(len(di), np.nan)
        gmax = np.full(len(di), np.nan)
        n30r = np.zeros(len(di))
        for x in np.unique(myrw):
            s_ = myrw == x
            lx = ls[rw_l == x]
            tt = t[s_]
            if len(lx) == 0:
                continue
            same30[s_] = cnt(lx, tt - 1800, tt)
            prev20[s_] = cnt(lx, tt - 1200, tt)
            n30r[s_] = same30[s_]
            a_ = aa[s_]
            ok_ = ~np.isnan(a_)
            dv = np.full(s_.sum(), np.nan)
            ok_ &= ~(a_ > tt)
            dv[ok_] = cnt(lx, a_[ok_], tt[ok_])
            during[s_] = dv
            gap = np.diff(lx, prepend=np.nan)
            gap[gap > 1800] = np.nan
            ser = pd.Series(gap, index=pd.to_datetime(lx, unit='s'))
            rmed = ser.rolling('30min').median().to_numpy()
            rstd = ser.rolling('30min').std().to_numpy()
            rmax = ser.rolling('30min').max().to_numpy()
            j = np.searchsorted(lx, tt, 'left') - 1
            okj = j >= 0
            jj = np.maximum(j, 0)
            sl = np.where(okj, tt - lx[jj], np.nan)
            fresh = okj & (sl <= 1800)
            since[s_] = np.where(okj & (sl <= 3600), sl, np.nan)
            gmed[s_] = np.where(fresh, rmed[jj], np.nan)
            gstd[s_] = np.where(fresh, rstd[jj], np.nan)
            gmax[s_] = np.where(fresh, rmax[jj], np.nan)
        put('rwy_land_during_taxi', di, during)
        put('rwy_land_during_taxi_per10', di, during / np.maximum(t - aa, 60) * 600)
        v = prev20.copy()
        v[bad(t - 1200)] = np.nan
        put('rwy_land_prev20', di, v)
        b30 = bad(t - 3600)
        for nm, v in [('rwy_since_last_land', since), ('rwy_gap_med30', gmed), ('rwy_gap_std30', gstd), ('rwy_gap_max30', gmax), ('rwy_land_n30', n30r.copy())]:
            v = v.copy()
            v[b30] = np.nan
            put(nm, di, v)
    for k_ in list(new):
        v = new.pop(k_)
        dep[k_] = v if v.dtype == object else v.astype(np.float32)
    for name in ARRIVAL_FEATURES:
        if name not in dep:
            dep[name] = None if name == 'arr_rwy_dom' else np.nan
    return dep, state
