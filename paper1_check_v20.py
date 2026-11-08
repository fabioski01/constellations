"""Recompute manuscript checks with the pinned Paper 1 numerical routines."""
from __future__ import annotations
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import platform
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import paper1_dense_solver as dense
from paper1_dense_checks import CASES
from paper1_validation_report import summarize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/paper1/v20-checks'
EPOCH = datetime(2025, 5, 5, 12, tzinfo=timezone.utc)
OMEGA = 7.2921159e-5

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

@contextmanager
def rotation(kind):
    original = dense.earth_coefficients
    if kind == 'uniform':
        theta = original(EPOCH)[0]
        dense.earth_coefficients = lambda epoch: (theta, OMEGA, 0., 0.)
    try:
        yield
    finally:
        dense.earth_coefficients = original

def nominal(case, offset, latitudes, longitudes, tolerances=(.25,), kind='gmst'):
    started = time.perf_counter()
    with rotation(kind), redirect_stdout(io.StringIO()):
        result = dense.corrected_grid(case, offset, EPOCH, latitudes, longitudes, tolerances)
    print(json.dumps(dict(case=case['case_id'], offset=offset, rotation=kind,
        grid=[len(latitudes), len(longitudes)], seconds=round(time.perf_counter()-started,3))), flush=True)
    return result

def pilot():
    source = ROOT / 'results/paper1/validation-36x18/per_maneuver.csv'
    frame = pd.read_csv(source)
    result = summarize(frame)
    result['within_1e_6_degree_of_boundary'] = int((frame.nominal_longitude_margin_deg < 1e-6).sum())
    result['csv_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    result['recount_scope'] = 'Recalculated from the saved per-maneuver CSV; trajectories not rerun.'
    save_json(OUT / 'pilot_recount.json', result)

def visit_counts(case, latitudes, longitudes, result=None, offset=None, kind='gmst'):
    """Enumerate latitude crossings, retaining the requested longitude criterion."""
    with redirect_stdout(io.StringIO()):
        p = dense.parameters(case, 100. if offset is None else offset, EPOCH)
    theta0, c1, c2, c3 = dense.earth_coefficients(EPOCH)
    if kind == 'uniform':
        c1,c2,c3 = OMEGA,0.,0.
    theta = lambda t: theta0+c1*t+c2*t*t+c3*t*t*t
    phi = np.arctan((1-dense.reference.WGS84_e2)*np.tan(np.deg2rad(latitudes)))
    first = np.arcsin(np.sin(phi)/np.sin(p.i)) % (2*np.pi)
    branches = [np.broadcast_to(u[:,None],(len(latitudes),len(longitudes)))
                for u in (first,(np.pi-first)%(2*np.pi))]
    windows = np.array([24.,48.,72.,168.])*3600
    counts = np.zeros((4,len(latitudes),len(longitudes)),dtype=np.int16)
    if result is None:
        anchor_time = np.zeros(counts.shape[1:])
        anchor_u = np.full(counts.shape[1:],p.u0)
        anchor_node = np.full(counts.shape[1:],p.Omega0)
        insertion = anchor_time
        gamma,node_rate = p.du_G,p.dOm_G
    else:
        tw,td = result['wait_s'][0],result['drift_s'][0]
        anchor_time = result['response_s'][0]
        anchor_u = np.where(result['branch'][0]==0,branches[0],branches[1])
        anchor_node = p.Omega0+p.dOm_G*tw+p.dOm_T1*p.T1+p.dOm_D*td+p.dOm_T2*p.T2
        insertion = tw+p.T1+(td+p.T2 if p.T2 else 0.)
        _,wr,mr = dense.reference.j2_rates(case['a_r_km'],0.,p.i)
        node_rate = dense.reference.j2_rates(case['a_r_km'],0.,p.i)[0]
        gamma = wr+mr
    lon = np.deg2rad(longitudes)[None,:]
    for branch,u in enumerate(branches):
        q = np.arctan2(np.cos(p.i)*np.sin(u),np.cos(u))
        k0 = int(np.floor(np.nanmin((insertion-anchor_time)*gamma/(2*np.pi))))-2
        k1 = int(np.ceil(np.nanmax((windows[-1]-anchor_time)*gamma/(2*np.pi))))+2
        for k in range(k0,k1+1):
            dt = (u-anchor_u+2*np.pi*k)/gamma
            t = anchor_time+dt
            geographic = np.abs(np.rad2deg(dense.wrap(anchor_node+node_rate*dt+q-theta(t)-lon))) <= .25
            # The numerical guard only reconciles algebraically identical insertion events.
            eligible = geographic & (t >= insertion-1e-7) & (t >= -1e-7)
            for j,w in enumerate(windows):
                counts[j] += (eligible & (t <= w+1e-7)).astype(np.int16)
    return counts

def observations():
    latitudes=np.linspace(-49.,49.,180)
    longitudes=np.linspace(-180.,180.,360,endpoint=False)
    summary={}
    for kind in ['gmst','uniform']:
        case=dict(CASES[1])
        inc=np.deg2rad(case['inclination_deg'])
        def residual(a):
            node,w,m=dense.reference.j2_rates(a,0.,inc)
            return (w+m)/(OMEGA-node)-15
        # Use the same precise repeat root in both rotation comparisons.
        case['a_r_km']=brentq(residual,6471.,9371.,xtol=1e-11)
        counts={'initial':visit_counts(case,latitudes,longitudes,kind=kind)}
        timings={}
        for label,offset in [('direct',case['a_r_km']-case['a_g_km']),('excursion',100.)]:
            result=nominal(case,offset,latitudes,longitudes,kind=kind)
            counts[label]=visit_counts(case,latitudes,longitudes,result,offset,kind)
            timings[label]=result
            np.savez_compressed(OUT / (label+'_'+kind+'_observations.npz'),
                latitudes=latitudes,longitudes=longitudes,counts=counts[label],**result)
        rows=[]
        for j,w in enumerate([24,48,72,168]):
            row={'window_h':w}
            for label,array in counts.items():
                row[label+'_mean']=float(array[j].mean())
                row[label+'_at_least_two_pct']=float(100*(array[j]>=2).mean())
                row[label+'_at_least_one_pct']=float(100*(array[j]>=1).mean())
            rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / ('observations_'+kind+'.csv'),index=False)
        summary[kind]=dict(regional_radius_km=case['a_r_km'],rows=rows,
            excursion_earlier_pct=float(100*(timings['excursion']['response_s']<timings['direct']['response_s']).mean()))
        print(json.dumps({kind:summary[kind]},indent=2),flush=True)
    save_json(OUT/'observation_summary.json',summary)

def scalar_count(case, latitude, longitude, window_s, result, index, offset, kind, details=False):
    """Locate latitude roots numerically from insertion, independently of branch enumeration."""
    from reconfig_verification import earth_angle_elapsed, gmst_rad
    with redirect_stdout(io.StringIO()):
        p=dense.parameters(case,offset,EPOCH)
    if result is None:
        start=0.; phase=p.u0; node=p.Omega0; gamma=p.du_G; node_rate=p.dOm_G
    else:
        tw=float(result['wait_s'][index]); td=float(result['drift_s'][index])
        if p.T2:
            start=tw+p.T1+td+p.T2
            phase=p.u0+p.du_G*tw+p.du_T1*p.T1+p.du_D*td+p.du_T2*p.T2
            node=p.Omega0+p.dOm_G*tw+p.dOm_T1*p.T1+p.dOm_D*td+p.dOm_T2*p.T2
        else:
            start=tw+p.T1; phase=p.u0+p.du_G*tw+p.du_T1*p.T1
            node=p.Omega0+p.dOm_G*tw+p.dOm_T1*p.T1
        node_rate,w,m=dense.reference.j2_rates(case['a_r_km'],0.,p.i); gamma=w+m
    if start>window_s:return 0
    phi=np.arctan((1-dense.reference.WGS84_e2)*np.tan(np.deg2rad(latitude)))
    f=lambda t: np.sin(p.i)*np.sin(phase+gamma*(t-start))-np.sin(phi)
    # Bracket at latitude extrema: a uniform time mesh can miss two nearby crossings.
    k0=int(np.floor((phase-np.pi/2)/np.pi))-1
    k1=int(np.ceil((phase+gamma*(window_s-start)-np.pi/2)/np.pi))+1
    extrema=start+(np.pi/2+np.pi*np.arange(k0,k1+1)-phase)/gamma
    times=np.concatenate(([start],extrema[(extrema>start)&(extrema<window_s)],[window_s]))
    roots=[]
    for a,b in zip(times[:-1],times[1:]):
        fa,fb=f(a),f(b)
        if abs(fa)<1e-12:roots.append(float(a))
        if fa*fb<0:roots.append(float(brentq(f,a,b,xtol=1e-9)))
    if abs(f(window_s))<1e-12:roots.append(window_s)
    unique=[]
    for t in sorted(roots):
        if not unique or t-unique[-1]>1e-6:unique.append(t)
    total=0; diagnostic=[]
    for t in unique:
        u=phase+gamma*(t-start)
        earth=(gmst_rad(EPOCH)+OMEGA*t if kind=='uniform' else earth_angle_elapsed(EPOCH,t))
        lon=node+node_rate*(t-start)+np.arctan2(np.cos(p.i)*np.sin(u),np.cos(u))-earth
        error=float(np.rad2deg(dense.wrap(lon-np.deg2rad(longitude))))
        total+=int(abs(error)<=.25)
        if abs(error)<1:diagnostic.append((t/3600,error))
    return diagnostic if details else total

def crosschecks():
    records=[]
    summary=json.loads((OUT/'observation_summary.json').read_text())
    latitudes=np.linspace(-49.,49.,180);longitudes=np.arange(-180.,180.)
    for kind in ['gmst','uniform']:
        case=dict(CASES[1],a_r_km=summary[kind]['regional_radius_km'])
        initial=visit_counts(case,latitudes,longitudes,kind=kind)
        for label in ['initial','direct','excursion']:
            offset=(case['a_r_km']-case['a_g_km'] if label=='direct' else 100.)
            if label=='initial':result=None;counts=initial
            else:
                with np.load(OUT/(label+'_'+kind+'_observations.npz')) as archive:
                    result={k:archive[k] for k in archive.files};counts=result['counts']
            for i,j in [(0,0),(0,120),(0,240),(89,0),(89,120),(89,240),(179,0),(179,120),(179,240)]:
                for wi,w in enumerate([24,48,72,168]):
                    n=scalar_count(case,latitudes[i],longitudes[j],w*3600,result,(0,i,j),offset,kind)
                    expected=int(counts[wi,i,j])
                    records.append(dict(rotation=kind,strategy=label,latitude=latitudes[i],
                        longitude=longitudes[j],window_h=w,enumerated=expected,root_located=n))
                    assert n==expected,records[-1]
    pd.DataFrame(records).to_csv(OUT/'independent_visit_checks.csv',index=False)
    # Recompute the 324-target nominal comparison at the propagation campaign's precise regional root.
    case=dict(CASES[1],a_r_km=summary['uniform']['regional_radius_km'])
    coarse_lat=np.linspace(-49.,49.,18);coarse_lon=np.linspace(-180.,180.,18,endpoint=False)
    direct=nominal(case,case['a_r_km']-case['a_g_km'],coarse_lat,coarse_lon)
    excursion=nominal(case,100.,coarse_lat,coarse_lon)
    from paper1_analysis import outbound_cost_m_s
    cost=outbound_cost_m_s(case['a_g_km'],case['a_g_km']+100.,case['a_r_km'])
    a_g,a_d,a_r=case['a_g_km'],case['a_g_km']+100.,case['a_r_km']
    vc=lambda a:np.sqrt(398600.4418/a)
    vt=lambda a,b:np.sqrt(398600.4418*(2/a-2/(a+b)))
    burns=np.array([vt(a_g,a_d)-vc(a_g),vc(a_d)-vt(a_d,a_g),vt(a_d,a_r)-vc(a_d),vc(a_r)-vt(a_r,a_d)])*1000
    combined=abs(burns[0])+abs(burns[1]+burns[2])+abs(burns[3])
    zero=np.abs(excursion['drift_s'][0])<=1e-7
    diff=(direct['response_s'][0]-excursion['response_s'][0])/3600
    paired=dict(requested=324,direct_mean_h=float(direct['response_s'].mean()/3600),
        excursion_mean_h=float(excursion['response_s'].mean()/3600),
        excursion_earlier=int((diff>.01/3600).sum()),mean_difference_h=float(diff.mean()),
        median_difference_h=float(np.median(diff)),zero_drift_count=int(zero.sum()),
        separate_cost_m_s=float(cost),combined_zero_drift_cost_m_s=float(combined),
        physical_cost_mean_m_s=float(np.where(zero,combined,cost).mean()),
        independent_visit_comparisons=len(records))
    save_json(OUT/'crosscheck_summary.json',paired)
    print(json.dumps(paired,indent=2),flush=True)

def freeze_manifest():
    sources=['paper1_check_v20.py','paper1_v20_figures.py','paper1_dense_solver.py',
        'paper1_dense_checks.py','reconfig_driftANDwaitTime_multicore.py',
        'reconfig_verification.py','paper1_validation_report.py','paper1_analysis.py']
    save_json(OUT/'verification_manifest.json',dict(
        base_commit='0443aa0e98666cb6531ddb0c97d2840b32a6e47f',
        epoch_utc=EPOCH.isoformat(),initial_mean_node_deg=0.,initial_mean_phase_deg=0.,
        grid=dict(n_latitude=180,n_longitude=360,latitude_band_deg=49.),
        coast_caps_h=[200.,300.],longitude_tolerance_deg=.25,
        observation_windows_h=[24,48,72,168],python=platform.python_version(),
        numpy=np.__version__,pandas=pd.__version__,
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        scope='Nominal recomputations, saved pilot CSV recount, and independent event checks. The controlled 21-setting Cartesian campaign is not rerun.'))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pilot()
    latitudes = np.linspace(-49., 49., 180)
    longitudes = np.linspace(-180., 180., 360, endpoint=False)
    case = CASES[1]
    results = {}
    for label, offset, tolerances in [('tolerance',50.,(.001,.005,.01,.25,1.,5.)),
                                     ('direct',case['a_r_km']-case['a_g_km'],(.25,)),
                                     ('excursion',100.,(.25,))]:
        result = nominal(case, offset, latitudes, longitudes, tolerances)
        np.savez_compressed(OUT / (label+'_gmst.npz'), latitudes=latitudes,
            longitudes=longitudes, tolerances=tolerances, **result)
        results[label] = [dict(tolerance_deg=tol, accepted=int(result['success'][j].sum()),
            mean_response_h=float(np.nanmean(result['response_s'][j])/3600),
            median_response_h=float(np.nanmedian(result['response_s'][j])/3600))
            for j,tol in enumerate(tolerances)]
        save_json(OUT / 'nominal_summary.json', results)
    print(json.dumps(results, indent=2), flush=True)
    observations()
    crosschecks()
    freeze_manifest()

if __name__ == '__main__':
    main()
