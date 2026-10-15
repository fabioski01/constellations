"""Audit saved dense runs by independently reconstructing nominal geometry.

This checks elapsed coast times, direct secular-rate formulas, transfer
durations, continuous calendar GMST, latitude/longitude residuals, counts and
tolerance nesting. It does not integrate executed Cartesian maneuvers.
The original solver's own-clock acceptance and common-clock acceptance are
kept separate, because integer-second GMST is one of the audited defects.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, math
import numpy as np
import pandas as pd

MU=398600.4418
J2=.0010826266
RADIUS=6371.
E2=(1/298.257223563)*(2-1/298.257223563)
TAU=2*math.pi


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rates(a,e,i):
    n=math.sqrt(MU/a**3); ci=math.cos(i)
    node=-1.5*J2*n*(RADIUS/(a*(1-e*e)))**2*ci
    peri=.75*J2*n*(RADIUS/(a*(1-e*e)))**2*(5*ci*ci-1)
    anomaly=n+.75*J2*n*(RADIUS/a)**2*(1-e*e)**(-1.5)*(3*ci*ci-1)
    return node, peri+anomaly


def transfer(a,b,corrected):
    centre=(a+b)/2; eccentricity=abs(b-a)/(a+b)
    duration=0. if corrected and a==b else math.pi*math.sqrt(centre**3/MU)
    return centre,eccentricity,duration


def reconstruct(case,offset,epoch,wait,drift,corrected):
    inclination=math.radians(case['inclination_deg'])
    global_a=case['a_g_km'];drift_a=global_a+offset;regional_a=case['a_r_km']
    a1,e1,t1=transfer(global_a,drift_a,corrected)
    a2,e2,t2=transfer(drift_a,regional_a,corrected)
    ng,ug=rates(global_a,0.,inclination);nd,ud=rates(drift_a,0.,inclination)
    n1,u1=rates(a1,e1,inclination);n2,u2=rates(a2,e2,inclination)
    total=wait+t1+drift+t2
    phase=np.remainder(math.radians(case['phase_deg'])+ug*wait+u1*t1+ud*drift+u2*t2,TAU)
    node=np.remainder(math.radians(case['node_deg'])+ng*wait+n1*t1+nd*drift+n2*t2,TAU)
    days=(epoch-datetime(2000,1,1,12,tzinfo=timezone.utc)).total_seconds()/86400.
    centuries=days/36525.;elapsed=total/86400./36525.
    base=(280.46061837+360.98564736629*days+.000387933*centuries**2-centuries**3/38710000.)%360.
    rotation=base+(360.98564736629*total/86400.+.000387933*elapsed*(2*centuries+elapsed)
                  -elapsed*(3*centuries**2+3*centuries*elapsed+elapsed**2)/38710000.)
    longitude=(node+np.arctan2(math.cos(inclination)*np.sin(phase),np.cos(phase))
               -np.deg2rad(rotation)+math.pi)%TAU-math.pi
    geocentric=np.arcsin(np.clip(math.sin(inclination)*np.sin(phase),-1.,1.))
    latitude=np.rad2deg(np.arctan(np.tan(geocentric)/(1-E2)))
    return total,np.rad2deg(longitude),latitude,t1,t2


def quantiles(values,prefix):
    a=np.asarray(values)
    return {prefix+name:float(function(a)) if len(a) else None for name,function in
            [('mean',np.mean),('median',np.median),('p10',lambda x:np.quantile(x,.1)),
             ('p90',lambda x:np.quantile(x,.9)),('min',np.min),('max',np.max)]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true',help='Audit completed experiments only and label the report partial.')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.data_dir/'manifest.json').read_text())
    completion=json.loads((args.data_dir/'completion.json').read_text())
    if not completion['complete'] and not args.allow_partial:raise ValueError('A complete study is required for the final audit.')
    experiments=[e for e in manifest['experiments'] if e['experiment_id'] in completion['finished_experiments']]
    root=Path(__file__).resolve().parent
    for name,expected in manifest['source_sha256'].items():
        if sha256(root/name)!=expected:raise ValueError('Numerical source changed: '+name)
    checkpoints=[json.loads(line) for line in (args.data_dir/'checkpoint.jsonl').read_text().splitlines()]
    indexed={(r['experiment_id'],r['latitude_start']):r for r in checkpoints}
    epoch=datetime.fromisoformat(manifest['config']['epoch_utc'])
    maximum={k:0. for k in ['corrected_longitude_residual_deg','corrected_latitude_residual_deg',
        'saved_longitude_residual_difference_deg','saved_latitude_residual_difference_deg',
        'time_component_difference_s','transfer_duration_difference_s']}
    failures=[];rows=[];tolerance_rows=[];total_corrected=0;total_original=0;minimum_coast=0.
    for number,experiment in enumerate(experiments,1):
        chunks=[]
        for first in range(0,experiment['n_lat'],manifest['config']['latitude_chunk_size']):
            entry=indexed[(experiment['experiment_id'],first)]
            path=args.data_dir/'per_target'/entry['relative_path']
            if sha256(path)!=entry['sha256']:raise ValueError('Saved chunk changed: '+str(path))
            with np.load(path) as item:chunks.append({k:item[k] for k in item.files})
        latitude=np.concatenate([x['latitude_deg'] for x in chunks])
        longitude=chunks[0]['longitude_deg']
        tolerances=chunks[0]['tolerances_deg'];lat=latitude[None,:,None];lon=longitude[None,None,:]
        data={}
        for version in ['corrected','original']:
            if version=='original' and not experiment['original']:continue
            arrays={field:np.concatenate([x[version+'_'+field] for x in chunks],axis=1) for field in
                    ['response_s','wait_s','drift_s','longitude_error_deg','latitude_error_deg','success','branch','phase_wrap','longitude_wrap']}
            accepted=arrays['success'].astype(bool)
            if version=='corrected':total_corrected+=accepted.size
            else:total_original+=accepted.size
            with np.errstate(invalid='ignore'):
                total,final_lon,final_lat,t1,t2=reconstruct(experiment['case'],experiment['offset_km'],epoch,
                    arrays['wait_s'],arrays['drift_s'],version=='corrected')
                longitude_error=(final_lon-lon+180.)%360.-180.
                latitude_error=final_lat-lat
            coordinate=np.isfinite(lat)&(np.abs(lat)<=90.)
            # Independent arithmetic is allowed 1e-9 degrees in this audit.
            # Saved planner acceptance retains the exact requested band.
            longitude_ok=np.abs(longitude_error)<=tolerances[:,None,None]+1e-9
            latitude_ok=np.abs(latitude_error)<=1e-7
            feasible=(arrays['wait_s']>=-1e-8)&(arrays['wait_s']<=manifest['config']['wait_cap_hours']*3600+1e-8)
            feasible&=(arrays['drift_s']>=-1e-8)&(arrays['drift_s']<=manifest['config']['drift_cap_hours']*3600+1e-8)
            continuous=accepted&longitude_ok&latitude_ok&coordinate&feasible
            arrays['continuous']=continuous;data[version]=arrays
            if accepted.any():
                minimum_coast=min(minimum_coast,float(np.min(arrays['wait_s'][accepted])),float(np.min(arrays['drift_s'][accepted])))
                maximum['time_component_difference_s']=max(maximum['time_component_difference_s'],float(np.max(np.abs(total[accepted]-arrays['response_s'][accepted]))))
                for field,expected in [('transfer1_s',t1),('transfer2_s',t2)]:
                    maximum['transfer_duration_difference_s']=max(maximum['transfer_duration_difference_s'],max(float(abs(x[version+'_'+field]-expected)) for x in chunks))
            if version=='corrected' and accepted.any():
                for name,values in [('corrected_longitude_residual_deg',longitude_error),('corrected_latitude_residual_deg',latitude_error),
                                    ('saved_longitude_residual_difference_deg',longitude_error-arrays['longitude_error_deg']),
                                    ('saved_latitude_residual_difference_deg',latitude_error-arrays['latitude_error_deg'])]:
                    maximum[name]=max(maximum[name],float(np.max(np.abs(values[accepted]))))
                if np.any(accepted&~continuous):failures.append(dict(experiment_id=experiment['experiment_id'],kind='corrected_forward_acceptance',count=int(np.sum(accepted&~continuous))))
            for ti,tolerance in enumerate(tolerances):
                mask=accepted[ti];strict=continuous[ti]
                row={k:experiment[k] for k in ['experiment_id','profile','offset_km','n_lat','n_lon','latitude_band_deg','longitude_shift_deg']}
                row.update(case_id=experiment['case']['case_id'],version=version,tolerance_deg=float(tolerance),
                    requested_count=int(mask.size),reported_count=int(mask.sum()),continuous_accepted_count=int(strict.sum()),
                    continuous_rejected_reported_count=int(np.sum(mask&~strict)),
                    longitude_rejected_count=int(np.sum(mask&~longitude_ok[ti])),
                    latitude_rejected_count=int(np.sum(mask&~latitude_ok[ti])),
                    continuous_mean_response_h=float(np.mean(arrays['response_s'][ti][strict])/3600) if strict.any() else None)
                rows.append(row)
            if version=='corrected' and len(tolerances)>1:
                for ti in range(1,len(tolerances)):
                    missing=accepted[ti-1]&~accepted[ti]
                    worsening=accepted[ti-1]&accepted[ti]&(arrays['response_s'][ti]>arrays['response_s'][ti-1]+.01)
                    if missing.any() or worsening.any():failures.append(dict(experiment_id=experiment['experiment_id'],kind='tolerance_nesting',lost=int(missing.sum()),worsened=int(worsening.sum())))
                    common=accepted[0]&accepted[ti]
                    saved=(arrays['response_s'][0]-arrays['response_s'][ti])[common]/3600.
                    changes=(arrays['branch'][0]!=arrays['branch'][ti])|(arrays['phase_wrap'][0]!=arrays['phase_wrap'][ti])|(arrays['longitude_wrap'][0]!=arrays['longitude_wrap'][ti])
                    item=dict(experiment_id=experiment['experiment_id'],profile=experiment['profile'],case_id=experiment['case']['case_id'],
                        offset_km=experiment['offset_km'],tolerance_deg=float(tolerances[ti]),paired_count=int(common.sum()),
                        discrete_alternative_changes=int(np.sum(common&changes)))
                    item.update(quantiles(saved,'reduction_h_'));tolerance_rows.append(item)
        if 'original' in data:
            old,new=data['original'],data['corrected']
            for ti,tolerance in enumerate(tolerances):
                both=old['continuous'][ti]&new['success'][ti]
                change=(new['response_s'][ti]-old['response_s'][ti])[both]/3600.
                row=next(r for r in reversed(rows) if r['version']=='original' and r['tolerance_deg']==float(tolerance))
                row.update(continuous_paired_count=int(both.sum()),continuous_pairs_changed_over_1s=int(np.sum(np.abs(change)>1/3600)),
                    continuous_pairs_changed_over_1h=int(np.sum(np.abs(change)>1)),
                    corrected_only_count=int(np.sum(new['success'][ti]&~old['continuous'][ti])))
                row.update(quantiles(change,'continuous_paired_change_h_'))
        if number%30==0:print(json.dumps({'audited_experiments':number}),flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir/'continuous_clock_audit.csv',index=False)
    pd.DataFrame(tolerance_rows).to_csv(args.output_dir/'tolerance_paired_summary.csv',index=False)
    report=dict(complete_experiments=len(experiments),full_study_complete=completion['complete'],corrected_target_requests=total_corrected,
        original_target_requests=total_original,maximum=maximum,minimum_saved_coast_s=minimum_coast,
        failures=failures,source_and_chunk_hashes_verified=True,
        audit_longitude_arithmetic_allowance_deg=1e-9,
        scope='Independent nominal secular geometry and continuous calendar clock; no executed Cartesian integration.')
    (args.output_dir/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if failures:raise SystemExit('Saved-record audit failed.')


if __name__=='__main__':main()
