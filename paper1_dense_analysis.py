"""Checkpointed original/corrected dense spatial studies for Paper 1.

Raw per-target arrays live in a declared external output directory. Compact
summaries and manifests can be committed separately from the full data bundle.
Each latitude chunk is atomic and checksummed; resume requires unchanged inputs
and numerical source hashes. All configured targets, including invalid ones,
remain in the output arrays and denominators.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, hashlib, json, os, platform, time
import numpy as np
import pandas as pd
from paper1_dense_checks import CASES
from paper1_dense_solver import corrected_grid, original_latitude
from paper1_analysis import outbound_cost_m_s

ROOT=Path(__file__).resolve().parent
SOURCES=['paper1_dense_solver.py','paper1_dense_analysis.py','paper1_dense_checks.py','paper1_analysis.py','reconfig_driftANDwaitTime_multicore.py',
         'reconfig_verification.py','constants.py','reference/paper1_original/reconfig_driftANDwaitTime_multicore.py']
FIELDS=['response_s','wait_s','drift_s','longitude_error_deg','latitude_error_deg','branch','phase_wrap','longitude_wrap','success']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    temporary=Path(str(path)+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=True)+'\n')
    temporary.replace(path)


def experiments(config):
    cases=[dict(c,node_deg=config['node_deg'],phase_deg=config['phase_deg']) for c in CASES]
    out=[]
    def add(profile,case,offset,lat_count,lon_count,band,tolerances,original=True,shift=0.):
        slug=f"{offset:+.8f}".replace('+','p').replace('-','m').replace('.','d')
        item=dict(experiment_id=f'{profile}_{case["case_id"]}_{slug}',profile=profile,
                  case=case,offset_km=float(offset),n_lat=int(lat_count),n_lon=int(lon_count),
                  latitude_band_deg=float(band),longitude_shift_deg=float(shift),
                  tolerances_deg=list(map(float,tolerances)),original=bool(original))
        out.append(item)
    regular=[x for x in config['matched_offsets_km'] if isinstance(x,(int,float))]
    for case in cases:
        regional=case['a_r_km']-case['a_g_km']
        for offset in regular+[regional]:
            tols=config['tolerances_deg'] if offset==50 or (case['case_id']=='E2' and offset in [-200,200]) else [.25]
            add('common',case,offset,config['n_latitude'],config['n_longitude'],49.,tols)
        lower=-200 if case['case_id']=='E1' else -600 if case['case_id']=='E3' else -300
        for offset in np.arange(lower,200.001,config['refined_altitude_step_km']):
            if offset not in regular:
                add('refined_altitude',case,offset,config['n_latitude'],config['n_longitude'],49.,[.25],original=False)
        # Resolution comparisons keep one declared coordinate measure. The
        # latitude grids are not claimed to be nested; the longitude shift is
        # a separate check for repeat-track sampling interactions.
        offsets=[50.,-200.,200.] if case['case_id']=='E2' else [50.]
        for offset in offsets:
            for label,nlat,nlon,shift in [('coarse',45,90,0.),('medium',90,180,0.),
                                          ('shifted',180,360,.5)]:
                add(label,case,offset,nlat,nlon,49.,config['tolerances_deg'],original=False,shift=shift)
    for case in cases[3:]:
        band=case['inclination_deg']-1.
        for offset in [0.,50.,case['a_r_km']-case['a_g_km']]:
            add('native_requested',case,offset,180,360,band,
                config['tolerances_deg'] if offset==50 else [.25])
    for case,nlat,band in zip(cases,[72,80,120,80,120],[49.,49.,49.,59.,97.]):
        offsets=[100.,200.] if case['case_id']=='E1' else [50.,case['a_r_km']-case['a_g_km']]
        for offset in offsets:add('thesis_grid_bridge',case,offset,nlat,240,band,[.25])
        add('thesis_tolerance_grid',case,50.,22 if case['case_id'] not in ['E4','E5'] else 24 if case['case_id']=='E4' else 36,
            72,band,config['tolerances_deg'])
    # Exact source-artifact reconstruction uses the six requested locations
    # explicitly encoded in the only historical CSV supplied in the repo.
    for offset in np.arange(-200.,200.001,10.):
        add('historical_csv',cases[0],offset,2,3,49.,[.25])
    return out


def chunk_worker(task):
    experiment,first,last,config,path_string=task
    start=time.perf_counter();path=Path(path_string)
    latitude=np.linspace(-experiment['latitude_band_deg'],experiment['latitude_band_deg'],experiment['n_lat'])[first:last]
    longitude=config['longitude_start_deg']+np.arange(experiment['n_lon'])*360/experiment['n_lon']+experiment['longitude_shift_deg']
    epoch=datetime.fromisoformat(config['epoch_utc'])
    common=(experiment['case'],experiment['offset_km'],epoch)
    corrected=corrected_grid(*common,latitude,longitude,experiment['tolerances_deg'],
        config['wait_cap_hours'],config['drift_cap_hours'])
    arrays=dict(latitude_deg=latitude,longitude_deg=longitude,
                tolerances_deg=np.array(experiment['tolerances_deg']),
                latitude_status=corrected['latitude_status'])
    for key in FIELDS:arrays['corrected_'+key]=corrected[key]
    arrays['corrected_transfer1_s']=np.array(corrected['transfer1_s'])
    arrays['corrected_transfer2_s']=np.array(corrected['transfer2_s'])
    arrays['longitude_derivative_bounds_rad_s']=corrected['derivative_bounds_rad_s']
    if experiment['original']:
        originals=[original_latitude(*common,float(lat),longitude,experiment['tolerances_deg'],
                                    drift_hours=config['drift_cap_hours']) for lat in latitude]
        for key in FIELDS+['strict_acceptance','source_within_longitude_tolerance']:
            arrays['original_'+key]=np.stack([row[key] for row in originals],axis=1)
        arrays['original_transfer1_s']=np.array(originals[0]['transfer1_s'])
        arrays['original_transfer2_s']=np.array(originals[0]['transfer2_s'])
    temporary=Path(str(path)+'.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temporary.replace(path)
    return dict(experiment_id=experiment['experiment_id'],latitude_start=first,latitude_stop=last,
                relative_path=str(path.name),sha256=digest(path),runtime_s=time.perf_counter()-start,
                configured_targets=int(len(latitude)*len(longitude)*len(experiment['tolerances_deg'])),
                corrected_accepted=int(corrected['success'].sum()))


def area_weights(latitudes,nlon,band):
    if band>90.:
        return np.full((len(latitudes),nlon),np.nan)
    edges=np.r_[-band,(latitudes[1:]+latitudes[:-1])/2,band]
    row=np.sin(np.deg2rad(edges[1:]))-np.sin(np.deg2rad(edges[:-1]))
    return np.broadcast_to(row[:,None]/nlon,(len(latitudes),nlon)).copy()


def summarize_experiment(experiment,paths,config):
    loaded=[np.load(path) for path in paths]
    latitudes=np.concatenate([d['latitude_deg'] for d in loaded])
    n=experiment['n_lat']*experiment['n_lon']
    weights=area_weights(latitudes,experiment['n_lon'],experiment['latitude_band_deg'])
    result=[]
    for ti,tolerance in enumerate(experiment['tolerances_deg']):
        data={}
        for version in ['corrected','original']:
            if version=='original' and not experiment['original']:continue
            arrays={key:np.concatenate([d[version+'_'+key][ti] for d in loaded],axis=0) for key in FIELDS}
            if version=='original':arrays['strict_acceptance']=np.concatenate([d['original_strict_acceptance'][ti] for d in loaded],axis=0)
            data[version]=arrays
            accepted=arrays['success'].astype(bool)
            strict=arrays.get('strict_acceptance',accepted)
            response=arrays['response_s'][accepted]
            row=dict(experiment_id=experiment['experiment_id'],profile=experiment['profile'],
                case_id=experiment['case']['case_id'],offset_km=experiment['offset_km'],
                tolerance_deg=tolerance,version=version,n_lat=experiment['n_lat'],n_lon=experiment['n_lon'],
                latitude_band_deg=experiment['latitude_band_deg'],longitude_shift_deg=experiment['longitude_shift_deg'],
                requested_count=n,reported_accepted_count=int(accepted.sum()),strict_accepted_count=int(strict.sum()),
                false_acceptance_count=int((accepted&~strict).sum()),
                reported_mean_response_h=float(np.mean(response)/3600) if len(response) else np.nan,
                strict_mean_response_h=float(np.mean(arrays['response_s'][strict])/3600) if strict.any() else np.nan,
                median_response_h=float(np.median(response)/3600) if len(response) else np.nan,
                p90_response_h=float(np.quantile(response,.9)/3600) if len(response) else np.nan,
                maximum_response_h=float(np.max(response)/3600) if len(response) else np.nan,
                mean_wait_h=float(np.mean(arrays['wait_s'][accepted])/3600) if accepted.any() else np.nan,
                mean_drift_h=float(np.mean(arrays['drift_s'][accepted])/3600) if accepted.any() else np.nan,
                area_weighted_mean_response_h=float(np.average(arrays['response_s'][accepted],weights=weights[accepted])/3600) if accepted.any() and np.all(np.isfinite(weights)) else np.nan,
                outbound_cost_m_s=outbound_cost_m_s(experiment['case']['a_g_km'],experiment['case']['a_g_km']+experiment['offset_km'],experiment['case']['a_r_km']),
                fraction_within_72h=float(np.sum(accepted&(arrays['response_s']<=72*3600)))/n,
                maximum_accepted_longitude_error_deg=float(np.max(np.abs(arrays['longitude_error_deg'][accepted]))) if accepted.any() else np.nan,
                maximum_accepted_latitude_error_deg=float(np.max(np.abs(arrays['latitude_error_deg'][accepted]))) if accepted.any() else np.nan,
                invalid_coordinate_count=int(np.sum(np.abs(latitudes)>90)*experiment['n_lon']),
                unreachable_coordinate_count=int(sum(np.sum(d['latitude_status']==3)*experiment['n_lon'] for d in loaded)))
            result.append(row)
        if 'original' in data:
            old,new=data['original'],data['corrected']
            reported=old['success'].astype(bool)&new['success'].astype(bool)
            audited=old['strict_acceptance']&new['success'].astype(bool)
            difference=new['response_s']-old['response_s']
            result[-1].update(common_reported_pairs=int(reported.sum()),common_strict_pairs=int(audited.sum()),
                reported_paired_mean_change_h=float(np.mean(difference[reported])/3600) if reported.any() else np.nan,
                strict_paired_mean_change_h=float(np.mean(difference[audited])/3600) if audited.any() else np.nan,
                strict_paired_median_change_s=float(np.median(difference[audited])) if audited.any() else np.nan,
                strict_paired_max_absolute_change_h=float(np.max(np.abs(difference[audited]))/3600) if audited.any() else np.nan,
                strict_pairs_changed_by_over_1s=int(np.sum(audited&(np.abs(difference)>1))),
                strict_pairs_changed_by_over_1h=int(np.sum(audited&(np.abs(difference)>3600))))
    for d in loaded:d.close()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/paper1_dense_360x180.json'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--profiles',nargs='*')
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args(); config=json.loads(args.config.read_text())
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    shards=output/'per_target';shards.mkdir(exist_ok=True)
    all_experiments=experiments(config)
    selected=[e for e in all_experiments if not args.profiles or e['profile'] in args.profiles]
    sources={name:digest(ROOT/name) for name in SOURCES}
    manifest_path=output/'manifest.json'
    identity=dict(config=config,source_sha256=sources,experiments=all_experiments)
    if manifest_path.exists():
        if not (args.resume or args.summarize_only):raise ValueError('Output exists; use --resume or a new directory.')
        old=json.loads(manifest_path.read_text())
        if any(old[key]!=value for key,value in identity.items()):raise ValueError('Configuration or numerical source changed; use a fresh output directory.')
    else:
        atomic_json(manifest_path,dict(identity,generated_utc=datetime.now(timezone.utc).isoformat(),
            python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
            grid_order='n_longitude by n_latitude; array axes tolerance, latitude, longitude',
            scope=config['scope']))
    progress_path=output/'checkpoint.jsonl'
    completed={}
    if progress_path.exists():
        for line in progress_path.read_text().splitlines():
            row=json.loads(line);key=(row['experiment_id'],row['latitude_start'])
            path=shards/row['relative_path']
            if not path.is_file() or digest(path)!=row['sha256']:raise ValueError('A saved chunk failed its checksum.')
            completed[key]=row
    tasks=[];all_tasks=[]
    for experiment in selected:
        for first in range(0,experiment['n_lat'],config['latitude_chunk_size']):
            last=min(first+config['latitude_chunk_size'],experiment['n_lat'])
            path=shards/f'{experiment["experiment_id"]}_lat{first:03d}.npz'
            task=(experiment,first,last,config,str(path));all_tasks.append(task)
            if (experiment['experiment_id'],first) not in completed:tasks.append(task)
    print(json.dumps({'experiments':len(selected),'latitude_chunks':len(all_tasks),'pending_chunks':len(tasks),
        'configured_targets':sum(e['n_lat']*e['n_lon']*len(e['tolerances_deg']) for e in selected),
        'source_hashes_verified':True}),flush=True)
    if not args.summarize_only:
        with progress_path.open('a',buffering=1) as stream,ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(chunk_worker,task):task for task in tasks}
            for index,future in enumerate(as_completed(futures),1):
                row=future.result();stream.write(json.dumps(row)+'\n');stream.flush()
                completed[(row['experiment_id'],row['latitude_start'])]=row
                if index%6==0 or index==len(tasks):
                    print(json.dumps({'finished_new_chunks':index,'pending_initial':len(tasks),
                        'experiment_id':row['experiment_id'],'chunk_seconds':row['runtime_s']}),flush=True)
    summary=[];finished=[]
    for experiment in all_experiments:
        entries=[completed.get((experiment['experiment_id'],first)) for first in range(0,experiment['n_lat'],config['latitude_chunk_size'])]
        if all(entries):
            summary.extend(summarize_experiment(experiment,[shards/e['relative_path'] for e in entries],config))
            finished.append(experiment['experiment_id'])
    pd.DataFrame(summary).to_csv(output/'summary.csv',index=False)
    atomic_json(output/'completion.json',dict(finished_experiments=finished,total_experiments=len(all_experiments),
        complete=len(finished)==len(all_experiments),finished_chunks=len(completed),
        total_solver_target_pairs=sum(e['n_lat']*e['n_lon']*len(e['tolerances_deg'])*(2 if e['original'] else 1) for e in all_experiments),
        bytes_in_raw_shards=sum(p.stat().st_size for p in shards.glob('*.npz'))))
    print(json.dumps({'finished_experiments':len(finished),'total_experiments':len(all_experiments)}),flush=True)


if __name__=='__main__':main()
