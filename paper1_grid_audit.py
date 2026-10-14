"""Recompute nominal geographic errors and aggregates from saved grid times."""
import argparse
from datetime import datetime
import hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from reconfig_driftANDwaitTime_multicore import AnalyticRaanPhasingPlanner,Orbit,WGS84_e2
from reconfig_verification import earth_angle_elapsed


def wrap_pm_pi(angle):
    return (angle+math.pi)%(2*math.pi)-math.pi


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads((args.data_dir/'manifest.json').read_text())
    r=pd.read_csv(args.data_dir/'per_target.csv'); s=pd.read_csv(args.data_dir/'summary.csv')
    errors=[]
    for name,digest in manifest['code_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            errors.append('source_hash:'+name)
    unique=r.drop_duplicates('unique_request_id')
    if len(unique)!=manifest['unique_solver_requests']: errors.append('unique_count')
    cache={}; longitude_errors=[]; latitude_errors=[]; cost_errors=[]; transfer_errors=[]
    for row in unique.itertuples():
        key=(row.case_id,row.offset_km)
        if key not in cache:
            cache[key]=AnalyticRaanPhasingPlanner(
                Orbit(row.a_g_km,math.radians(row.inclination_deg),math.radians(row.node_deg),math.radians(row.phase_deg)),
                row.a_r_km,row.offset_km,datetime.fromisoformat(row.epoch_utc),
                row.target_latitude_deg,row.target_longitude_deg,
                drift_time_cap_days=row.drift_cap_hours/24,n_jobs=1,
                wait_time_cap_hours=row.wait_cap_hours,response_horizon_hours=None)
        planner=cache[key]
        mu=manifest['constants']['mu_km3_s2']
        def dv(a,b):
            return (abs(math.sqrt(mu/a)*(math.sqrt(2*b/(a+b))-1))
                    +abs(math.sqrt(mu/b)*(1-math.sqrt(2*a/(a+b)))))
        expected_cost=1000*(dv(planner.a_G,planner.a_D)+dv(planner.a_D,planner.rom_a))
        cost_errors.append(abs(expected_cost-row.outbound_cost_m_s))
        if not row.success: continue
        transfer_errors.extend([abs(row.transfer1_s-planner.T1),abs(row.transfer2_s-planner.T2)])
        u=(planner.u0+planner.du_G*row.wait_s+planner.du_T1*row.transfer1_s
            +planner.du_D*row.drift_s+planner.du_T2*row.transfer2_s)
        node=(planner.Omega0+planner.dOm_G*row.wait_s+planner.dOm_T1*row.transfer1_s
              +planner.dOm_D*row.drift_s+planner.dOm_T2*row.transfer2_s)
        psi=math.atan2(math.cos(planner.i)*math.sin(u),math.cos(u))
        longitude=wrap_pm_pi(node+psi-earth_angle_elapsed(planner.epoch,row.response_s))
        longitude_error=math.degrees(wrap_pm_pi(longitude-math.radians(row.target_longitude_deg)))
        geocentric=math.asin(max(-1,min(1,math.sin(planner.i)*math.sin(u))))
        latitude=math.degrees(math.atan(math.tan(geocentric)/(1-WGS84_e2)))
        latitude_error=latitude-row.target_latitude_deg
        longitude_errors.append(abs(longitude_error-row.longitude_error_deg))
        latitude_errors.append(abs(latitude_error-row.latitude_error_deg))
        if abs(longitude_error)>row.tolerance_deg: errors.append(f'forward_longitude:{row.unique_request_id}')
        if abs(latitude_error)>1e-7: errors.append(f'forward_latitude:{row.unique_request_id}')
    mean_errors=[]
    for row in s.itertuples():
        g=r[(r.study==row.study)&(r.case_id==row.case_id)&(r.policy==row.policy)
            &np.isclose(r.offset_km,row.offset_km,rtol=0,atol=1e-8)&(r.tolerance_deg==row.tolerance_deg)]
        a=g[g.success]
        if len(g)!=60 or g.target_id.nunique()!=60 or len(a)!=row.accepted_count:
            errors.append('aggregate_count')
        for field in ['response_s','wait_s','drift_s']:
            if len(a):
                expected=math.fsum(a[field])/len(a)
                mean_errors.append(abs(expected-getattr(row,'mean_'+field)))
        if len(a):
            weighted=math.fsum(a.response_s*a.area_weight)/math.fsum(a.area_weight)
            if abs(weighted-row.area_weighted_mean_response_s)>1e-6: errors.append('area_weighted_mean')
    if max(mean_errors,default=0)>1e-6: errors.append('conditional_mean')
    if max(cost_errors,default=0)>1e-7: errors.append('outbound_cost')
    if max(transfer_errors,default=0)>1e-6: errors.append('transfer_duration')
    if max(longitude_errors,default=0)>1e-7 or max(latitude_errors,default=0)>1e-7:
        errors.append('saved_geographic_error')
    accepted=unique[unique.success]
    component_error=(accepted.response_s-accepted[['wait_s','transfer1_s','drift_s','transfer2_s']].sum(axis=1)).abs()
    if component_error.max()>1e-6: errors.append('saved_time_components')
    result=dict(unique_requests=len(unique),accepted_unique_requests=int(unique.success.sum()),
        source_hashes_agree=not any(e.startswith('source_hash:') for e in errors),failures=errors,
        maximum_longitude_error_reconstruction_difference_deg=max(longitude_errors,default=0),
        maximum_latitude_error_reconstruction_difference_deg=max(latitude_errors,default=0),
        maximum_conditional_mean_difference_s=max(mean_errors,default=0),
        maximum_outbound_cost_difference_m_s=max(cost_errors,default=0),
        maximum_transfer_duration_difference_s=max(transfer_errors,default=0),
        maximum_time_component_error_s=float(component_error.max()),
        scope='Reconstructed nominal geometry and saved aggregates; not Cartesian execution validation.')
    (args.data_dir/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if errors: raise SystemExit('Independent saved-record audit failed.')


if __name__=='__main__': main()
