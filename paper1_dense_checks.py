"""Compare dense batched calculations against unchanged scalar sources."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from math import radians
from concurrent.futures import ProcessPoolExecutor
import argparse, json, time
import numpy as np
import pandas as pd
import reconfig_driftANDwaitTime_multicore as scalar_corrected
from paper1_dense_solver import corrected_grid, original_latitude, load_original, legacy_clock

EPOCH=datetime(2025,5,5,12,tzinfo=timezone.utc)
CASES=[dict(case_id='E1',a_g_km=6871.,a_r_km=6874.316367790103,inclination_deg=50.),
       dict(case_id='E2',a_g_km=7071.,a_r_km=6874.316367790103,inclination_deg=50.),
       dict(case_id='E3',a_g_km=7371.,a_r_km=6874.316367790103,inclination_deg=50.),
       dict(case_id='E4',a_g_km=7071.,a_r_km=6884.090533949435,inclination_deg=60.),
       dict(case_id='E5',a_g_km=7071.,a_r_km=6939.75319480896,inclination_deg=98.)]


def clock_check():
    """Check the discontinuities that control legacy band-edge decisions."""
    original=load_original()
    evaluate=legacy_clock(EPOCH,1820000)
    centres=np.array([1.,86400.,92097.,148767.,720000.,1080000.,1800000.])
    times=np.concatenate([centres+delta for delta in
                          [-1.51e-6,-.51e-6,-.5e-6,-.49e-6,0.,.49e-6,.5e-6,.51e-6]])
    expected=np.array([original.gmst_rad(EPOCH+timedelta(seconds=float(t))) for t in times])
    actual=evaluate(times)
    error=float(np.max(np.abs(expected-actual)))
    if error!=0.:raise AssertionError(f'Original clock mismatch: {error}')
    return dict(compared_times=len(times),maximum_radian_difference=error)


def scalar_check(task):
    case,offset,latitude,longitudes,tolerances=task
    start=time.perf_counter()
    original=load_original()
    batch_old=original_latitude(case,offset,EPOCH,latitude,longitudes,tolerances)
    batch_new=corrected_grid(case,offset,EPOCH,[latitude],longitudes,tolerances)
    rows=[]
    for version,module,batch in [('original',original,batch_old),('corrected',scalar_corrected,batch_new)]:
        for ti,tolerance in enumerate(tolerances):
            for j,longitude in enumerate(longitudes):
                options=dict(drift_time_cap_days=300/24.,n_jobs=1)
                if version=='corrected':options['prune_phase_wraps']=True
                p=module.AnalyticRaanPhasingPlanner(module.Orbit(case['a_g_km'],radians(case['inclination_deg']),0.,0.),
                    case['a_r_km'],offset,EPOCH,latitude,longitude,**options)
                s=p.solve_min_time([tolerance])
                index=(ti,j) if version=='original' else (ti,0,j)
                success=bool(s.get('success'))
                other=bool(batch['success'][index])
                time_error=(float(batch['response_s'][index])-s['total_time_s']) if success and other else None
                row=dict(version=version,case_id=case['case_id'],offset_km=offset,
                         latitude_deg=latitude,longitude_deg=longitude,tolerance_deg=tolerance,
                         scalar_success=success,batch_success=other,response_difference_s=time_error,
                         scalar_response_s=s.get('total_time_s'),batch_response_s=float(batch['response_s'][index]),
                         scalar_longitude_error_deg=s.get('lon_err_deg'),
                         batch_longitude_error_deg=float(batch['longitude_error_deg'][index]))
                rows.append(row)
    return rows,time.perf_counter()-start


def archive_check(data_dir):
    records=pd.read_csv(data_dir/'per_target.csv').drop_duplicates('unique_request_id')
    issues=[];max_time=0.;compared=0
    for (case_id,offset),group in records.groupby(['case_id','offset_km']):
        case=next(c for c in CASES if c['case_id']==case_id)
        lats=np.sort(group.target_latitude_deg.unique());lons=np.sort(group.target_longitude_deg.unique())
        tolerances=np.sort(group.tolerance_deg.unique())
        out=corrected_grid(case,offset,EPOCH,lats,lons,tolerances)
        for row in group.itertuples():
            index=(int(np.flatnonzero(tolerances==row.tolerance_deg)[0]),
                   int(np.flatnonzero(lats==row.target_latitude_deg)[0]),
                   int(np.flatnonzero(lons==row.target_longitude_deg)[0]))
            success=bool(out['success'][index]);compared+=1
            error=float(out['response_s'][index]-row.response_s) if success and row.success else 0.
            max_time=max(max_time,abs(error))
            if success!=row.success or abs(error)>.01:
                issues.append(dict(case_id=case_id,offset_km=offset,latitude=row.target_latitude_deg,
                                   longitude=row.target_longitude_deg,tolerance=row.tolerance_deg,
                                   scalar_success=bool(row.success),batch_success=success,response_difference_s=error))
    return dict(compared_unique_requests=compared,maximum_response_difference_s=max_time,disagreements=issues)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--archive-dir',type=Path,default=Path('results/paper1/grid-10x6-20261014'))
    p.add_argument('--archive-only',action='store_true')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    clock=clock_check()
    archive=archive_check(args.archive_dir)
    (args.output_dir/'batch_archive_check.json').write_text(json.dumps(archive,indent=2)+'\n')
    print(json.dumps({'archive_requests':archive['compared_unique_requests'],'archive_disagreements':len(archive['disagreements']),
                      'archive_max_difference_s':archive['maximum_response_difference_s']}),flush=True)
    if args.archive_only:return
    tasks=[]
    for ci,case in enumerate(CASES):
        for offset,latitude in [(50.,27.2),(0.,-38.1),(case['a_r_km']-case['a_g_km'],5.4)]:
            tasks.append((case,offset,latitude,[-120.,37.],[.001,.25]))
    tasks.extend([(CASES[1],-200.,-49.,[-180.,0.],[.25,5.]),
                  (CASES[1],200.,49.,[-60.,120.],[.001,.25]),
                  (CASES[0],5.,16.3,[-120.,60.],[.25]),
                  (CASES[0],-5.,-16.3,[-120.,60.],[.25]),
                  (CASES[4],50.,85.,[-60.,120.],[.25]),
                  (CASES[4],50.,97.,[-60.,120.],[.25]),
                  (CASES[0],180.,-49.,[-60.],[.25])])
    all_rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(rows,seconds) in enumerate(pool.map(scalar_check,tasks),1):
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(args.output_dir/'batch_scalar_checks.csv',index=False)
            print(json.dumps({'completed_groups':i,'groups':len(tasks),'group_seconds':seconds}),flush=True)
    mismatches=[r for r in all_rows if r['scalar_success']!=r['batch_success'] or
                (r['response_difference_s'] is not None and abs(r['response_difference_s'])>.01)]
    report=dict(scalar_comparisons=len(all_rows),mismatches=mismatches,
                maximum_response_difference_s=max(abs(r['response_difference_s'] or 0.) for r in all_rows),
                archive=archive,legacy_clock=clock,
                scope='Nominal solver equivalence; not executed Cartesian dynamics.')
    (args.output_dir/'batch_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='archive'}),flush=True)


if __name__=='__main__':main()
