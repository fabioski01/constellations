"""Location, altitude and longitude-tolerance studies on a declared target grid.

Preserves the wait/two-transfer/drift maneuver and nominal model. This runner
does not establish continuous-altitude or spatial convergence or validate an
executed Cartesian maneuver. Individual failures remain in the output.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from paper1_analysis import run_request, summarize, check_tolerance_nesting
from reconfig_driftANDwaitTime_multicore import R_EARTH, MU_EARTH, J2
from reconfig_phasing_RGT import PhasingRGT


def build_grid(grid):
    n_lat, n_lon = int(grid['n_lat']), int(grid['n_lon'])
    lo, hi = float(grid['latitude_min_deg']), float(grid['latitude_max_deg'])
    if n_lat < 2 or n_lon < 2 or not -90 < lo < hi < 90:
        raise ValueError('Need at least two samples per axis and a valid latitude band.')
    if grid.get('longitude_endpoint', False):
        raise ValueError('The periodic longitude endpoint must be excluded.')
    latitudes = np.linspace(lo, hi, n_lat)
    longitudes = float(grid['longitude_start_deg']) + np.arange(n_lon) * 360.0 / n_lon
    if np.any(longitudes < -180) or np.any(longitudes >= 180):
        raise ValueError('Grid longitudes must lie in [-180,180).')
    # Representative latitude cells end at the declared band boundary.
    edges = np.r_[lo, (latitudes[:-1] + latitudes[1:])/2, hi]
    latitude_weights = np.diff(np.sin(np.deg2rad(edges)))
    latitude_weights /= latitude_weights.sum()
    rows = []
    for j, lat in enumerate(latitudes):
        for k, lon in enumerate(longitudes):
            rows.append(dict(target_id=len(rows)+1, latitude_index=j,
                longitude_index=k, target_latitude_deg=float(lat),
                target_longitude_deg=float(lon), area_weight=float(latitude_weights[j]/n_lon)))
    return pd.DataFrame(rows)


def physical_key(task):
    return (task['case_id'], task['target_id'], task['offset_km'], task['tolerance_deg'])


def build_tasks(config, targets):
    tasks, cases = [], []
    for definition in config['cases']:
        case = dict(definition)
        a_r, ok, _ = PhasingRGT().solve_for_semi_major_axis(i=case['inclination_deg'], e=0.0,
            No=config['repeat_revolutions'], Nd=config['repeat_days'])
        if not ok:
            raise RuntimeError(f'Regional repeat root failed for {case["case_id"]}.')
        case.update(a_g_km=R_EARTH+case['global_altitude_km'], a_r_km=float(a_r))
        cases.append(case)
        offsets = sorted(set(float(a_r)-case['a_g_km'] if x == 'regional' else float(x)
                             for x in config['offsets_km']))
        knobs = {k:config[k] for k in ['epoch_utc','node_deg','phase_deg','wait_cap_hours',
            'drift_cap_hours','horizon_hours','scan_discovery','scan_bracket','longitude_neighbor_pad']}
        knobs['prune_phase_wraps'] = config.get('prune_phase_wraps', False)
        for target in targets.to_dict('records'):
            for offset in offsets:
                tasks.append(dict(case, **target, **knobs, study='altitude', policy='joint',
                    offset_km=offset, tolerance_deg=config['tolerance_deg']))
            if case['case_id'] in config.get('tolerance_cases', []):
                tol_offsets = config['tolerance_offsets_by_case'].get(case['case_id'],
                    config['tolerance_offsets_by_case']['default'])
                for offset in tol_offsets:
                    for tol in config['tolerances_deg']:
                        tasks.append(dict(case, **target, **knobs, study='tolerance', policy='joint',
                            offset_km=float(offset), tolerance_deg=float(tol)))
    return tasks, cases


def summarize_grid(records):
    summary = summarize(records)
    data = records[records.study == 'tolerance']
    smallest = data.tolerance_deg.min()
    narrow_ids = {}
    for (case, offset), group in data[data.tolerance_deg == smallest].groupby(['case_id','offset_km']):
        narrow_ids[case,offset] = set(group.loc[group.success, 'target_id'])
    extra = []
    for (study, case, policy, offset, tol), group in records.groupby(
            ['study','case_id','policy','offset_km','tolerance_deg'], sort=True):
        accepted = group[group.success]
        by72 = accepted[accepted.response_s <= 72*3600 + 1e-6]
        row = dict(study=study,case_id=case,policy=policy,offset_km=offset,tolerance_deg=tol,
                   accepted_by_72h_count=len(by72), accepted_by_72h_fraction=len(by72)/len(group),
                   minimum_response_s=float(accepted.response_s.min()) if len(accepted) else np.nan,
                   accepted_area_fraction=float(accepted.area_weight.sum()))
        for field in ['response_s','wait_s','drift_s']:
            row['area_weighted_mean_'+field] = (float(np.average(accepted[field],weights=accepted.area_weight))
                                                if len(accepted) else np.nan)
        common = accepted[accepted.target_id.isin(narrow_ids.get((case,offset), set()))] if study == 'tolerance' else accepted.iloc[:0]
        row['narrow_common_count'] = len(common)
        row['narrow_common_mean_response_s'] = float(common.response_s.mean()) if len(common) else np.nan
        extra.append(row)
    return summary.merge(pd.DataFrame(extra),on=['study','case_id','policy','offset_km','tolerance_deg'],validate='one_to_one')


def paired_tolerances(records):
    rows = []
    data = records[records.study == 'tolerance']
    for (case, target, offset), group in data.groupby(['case_id','target_id','offset_km']):
        group = group.sort_values('tolerance_deg')
        base = group.iloc[0]
        for row in group.itertuples():
            both = bool(base.success and row.success)
            rows.append(dict(case_id=case,target_id=target,offset_km=offset,
                target_latitude_deg=row.target_latitude_deg,target_longitude_deg=row.target_longitude_deg,
                baseline_tolerance_deg=base.tolerance_deg,tolerance_deg=row.tolerance_deg,
                baseline_success=bool(base.success),success=bool(row.success),both_feasible=both,
                newly_accepted=bool(row.success and not base.success),
                baseline_response_s=base.response_s,response_s=row.response_s,
                paired_time_saved_s=base.response_s-row.response_s if both else np.nan))
    return pd.DataFrame(rows)


def paired_locations(records):
    e2 = records[(records.study=='altitude') & (records.case_id=='E2')]
    cols = ['target_id','target_latitude_deg','target_longitude_deg','success','response_s','wait_s','drift_s']
    joined = e2[e2.offset_km==50][cols].merge(e2[e2.offset_km==-200][cols],
        on=['target_id','target_latitude_deg','target_longitude_deg'],suffixes=('_plus50','_minus200'),validate='one_to_one')
    joined['both_feasible'] = joined.success_plus50 & joined.success_minus200
    joined['paired_time_saved_s'] = np.where(joined.both_feasible,
        joined.response_s_plus50-joined.response_s_minus200,np.nan)
    return joined


def audit_records(records, summary, targets, requested, unique_count):
    accepted = records[records.success]
    time_error = (accepted.response_s-accepted[['wait_s','transfer1_s','drift_s','transfer2_s']].sum(axis=1)).abs()
    errors = []
    if len(records)!=requested: errors.append('request_count')
    if records.unique_request_id.nunique()!=unique_count: errors.append('unique_request_count')
    if abs(targets.area_weight.sum()-1)>1e-12: errors.append('area_weight_sum')
    if (records.status=='numerical_failure').any(): errors.append('numerical_failures')
    if (accepted.longitude_error_deg.abs()>accepted.tolerance_deg).any(): errors.append('longitude_acceptance')
    if (accepted.latitude_error_deg.abs()>1e-7).any(): errors.append('latitude_acceptance')
    if time_error.max()>1e-6: errors.append('time_components')
    if (accepted[['wait_s','drift_s','transfer1_s','transfer2_s']] < -1e-8).any().any(): errors.append('negative_duration')
    if (accepted.wait_s>accepted.wait_cap_hours*3600+1e-6).any(): errors.append('wait_cap')
    if (accepted.drift_s>accepted.drift_cap_hours*3600+1e-6).any(): errors.append('drift_cap')
    capped = accepted[accepted.horizon_hours.notna()]
    if (capped.response_s>capped.horizon_hours*3600+1e-6).any(): errors.append('response_horizon')
    if (summary.requested_count!=len(targets)).any(): errors.append('summary_denominator')
    nesting = check_tolerance_nesting(records)
    if nesting: errors.append('tolerance_nesting')
    return dict(requested=len(records),unique_solver_requests=unique_count,targets=len(targets),
        accepted=int(records.success.sum()),numerical_failures=int((records.status=='numerical_failure').sum()),
        failures=errors,tolerance_nesting_violations=nesting,
        maximum_time_component_error_s=float(time_error.max()),
        maximum_accepted_latitude_error_deg=float(accepted.latitude_error_deg.abs().max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/paper1_grid_10x6.json'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args()
    if args.workers<1: parser.error('Workers must be positive.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()): parser.error('Use an empty output directory.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    config=json.loads(args.config.read_text())
    targets=build_grid(config['grid']); targets.to_csv(args.output_dir/'targets.csv',index=False)
    tasks,cases=build_tasks(config,targets)
    unique={}
    for task in tasks: unique.setdefault(physical_key(task),task)
    sources=['constants.py','reconfig_driftANDwaitTime_multicore.py','reconfig_phasing_RGT.py',
             'reconfig_verification.py','reconfig_linear_reference.py','paper1_analysis.py','paper1_grid_analysis.py']
    manifest=dict(config=config,actual_cases=cases,generated_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),workers=args.workers,requested_records=len(tasks),
        unique_solver_requests=len(unique),package_versions={n:version(n) for n in ('numpy','pandas','matplotlib')},
        constants=dict(mu_km3_s2=MU_EARTH,J2=J2,spherical_and_gravity_radius_km=R_EARTH),
        code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        primary_weighting='Equal coordinate-point weights; conditional means retain explicit accepted counts.',
        secondary_weighting='Spherical representative-cell areas within the common latitude band; conditional on acceptance.',
        scope=config['scope'])
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    solved={}
    checkpoint=None
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True,exist_ok=True)
        checkpoint=args.checkpoint.open('w')
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i,row in enumerate(pool.map(run_request,unique.values(),chunksize=1),1):
                row['unique_request_id']=i
                solved[physical_key(row)]=row
                if checkpoint:
                    checkpoint.write(json.dumps(row)+'\n')
                    if i%50==0: checkpoint.flush()
                if i%100==0 or i==len(unique):
                    print(f'Completed {i}/{len(unique)} unique solver requests',flush=True)
    finally:
        if checkpoint: checkpoint.close()
    records=pd.DataFrame([dict(solved[physical_key(task)],study=task['study']) for task in tasks])
    records.to_csv(args.output_dir/'per_target.csv',index=False)
    summary=summarize_grid(records); summary.to_csv(args.output_dir/'summary.csv',index=False)
    paired_tolerances(records).to_csv(args.output_dir/'tolerance_paired.csv',index=False)
    paired_locations(records).to_csv(args.output_dir/'location_paired.csv',index=False)
    checks=audit_records(records,summary,targets,len(tasks),len(unique))
    (args.output_dir/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks),flush=True)
    if checks['failures']: raise SystemExit('Grid study failed its audit; inspect checks.json.')


if __name__=='__main__':
    main()
