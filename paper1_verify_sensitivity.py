"""Independently recount sweep outcomes from saved event and burn evidence."""
from __future__ import annotations

import argparse
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper1_validate_grid import sha256

ROOT=Path(__file__).resolve().parent


def require(condition,message):
    if not condition:raise AssertionError(message)


@lru_cache(maxsize=None)
def origin_records(relative_path,digest):
    path=ROOT/relative_path
    require(sha256(path)==digest,'Reused archive checksum mismatch: '+relative_path)
    with gzip.open(path,'rt') as handle:
        return {r['row']['case_index']:r for r in map(json.loads,handle)}


def verify(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    completion=json.loads((folder/'completion.json').read_text())
    require(completion['complete'],'Campaign must be complete before verification.')
    for name,digest in manifest['source_sha256'].items():
        require(sha256(ROOT/name)==digest,'Numerical source/input checksum mismatch: '+name)
    checkpoints={}
    for line in (folder/'checkpoint.jsonl').read_text().splitlines():
        item=json.loads(line);key=(item['experiment_id'],item['case_index'])
        require(key not in checkpoints,'Duplicate checkpoint.')
        require(sha256(folder/item['path'])==item['sha256'],'Checkpoint checksum mismatch.')
        checkpoints[key]=item
    require(len(checkpoints)==manifest['requested_count'],'Missing or excess checkpoints.')
    targets={r['case_index']:r for r in manifest['targets']}
    experiment_ids={e['experiment_id'] for e in manifest['experiments']}
    require(set(k[0] for k in checkpoints)==experiment_ids,'Unexpected experiment membership.')
    results=[];reused=0;maximum_position=0.;maximum_time=0.
    for experiment in manifest['experiments']:
        identifier=experiment['experiment_id'];context=experiment['context'];scenario=context['scenario']
        tolerance=scenario['longitude_tolerance_deg']
        counts=dict(requested=0,nominal_accepted=0,numerically_completed=0,nominal_infeasible=0,
                    numerical_failures=0,matched_crossing_found=0,matched_longitude_within_tolerance=0,
                    matched_access_accepted=0,post_insertion_window_accepted=0,matched_before_insertion=0)
        folder_case=folder/'experiments'/identifier
        raw_bytes=[]
        for index,target in targets.items():
            checkpoint=checkpoints[(identifier,index)]
            content=(folder/checkpoint['path']).read_bytes();raw_bytes.append(content)
            record=json.loads(content);row=record['row'];counts['requested']+=1
            for key,value in target.items():require(row[key]==value,'Target coordinate/grid weight changed.')
            if 'reused_from' in record:
                reused+=1;origin=record['reused_from']
                original=origin_records(origin['archive'],origin['sha256'])[origin['case_index']]
                for key in ['nominal','evidence','matched_event','convergence','burns']:
                    require(record.get(key)==original.get(key),'Reused physical evidence changed: '+key)
            counts['nominal_accepted']+=bool(row['nominal_success'])
            if row['status']=='nominal_infeasible':
                counts['nominal_infeasible']+=1
                require(not record['nominal']['success'] and not row['nominal_success'],'Infeasible target misclassified.')
                require(not any(row[k] for k in ['matched_access_accepted','post_insertion_window_accepted',
                                                'matched_longitude_within_tolerance']),'Infeasible target reported successful.')
                continue
            require(row['status']=='completed','Numerically unresolved request: '+identifier)
            counts['numerically_completed']+=1
            convergence=record['convergence']
            maximum_position=max(maximum_position,convergence['maximum_position_difference_m'])
            maximum_time=max(maximum_time,convergence['maximum_latitude_crossing_time_difference_s'])
            require(convergence['initial_state_identical'],'Convergence compared different initial states.')
            require(convergence['maximum_position_difference_m']<=.1,'Position convergence failure.')
            require(convergence['maximum_latitude_crossing_time_difference_s']<=.001,'Event-time convergence failure.')
            require(convergence['latitude_crossing_count_primary']==convergence['latitude_crossing_count_tighter'],
                    'Event count convergence failure.')
            require(all(convergence[k] for k in ['passed','post_window_status_agrees','matched_status_agrees']),
                    'Classification convergence failure.')
            access=record['evidence']['access']
            half_period=context['repeat_s']/scenario['repeat_revolutions']*manifest['config']['matched_event_half_window_nodal_periods']
            candidates=[e for e in access['crossings'] if e['branch']==row['nominal_branch'] and
                        abs(e['time_s']-row['nominal_response_s'])<=half_period]
            matched=min(candidates,key=lambda e:abs(e['time_s']-row['nominal_response_s'])) if candidates else None
            require(matched==record['matched_event'],'Matched crossing selection changed or favored success.')
            require(bool(matched)==bool(row['matched_crossing_found']),'Missing-event flag disagrees.')
            if matched:
                counts['matched_crossing_found']+=1
                longitude=abs(matched['longitude_residual_deg'])<=tolerance
                eligible=matched['time_s']>=row['insertion_s']
                full=longitude and eligible
                counts['matched_longitude_within_tolerance']+=longitude
                counts['matched_access_accepted']+=full
                counts['matched_before_insertion']+=not eligible
                require(longitude==bool(row['matched_longitude_within_tolerance']),'Longitude acceptance recount failed.')
                require(eligible==bool(row['matched_after_insertion']),'Insertion eligibility recount failed.')
                require(full==bool(row['matched_access_accepted']),'Full matched acceptance recount failed.')
                model_error=(matched['longitude_residual_deg']-record['nominal']['lon_err_deg']+180)%360-180
                require(abs(model_error-row['matched_longitude_model_error_deg'])<1e-10,'Longitude error conflates prediction and target offset.')
                require(abs(matched['time_s']-row['nominal_response_s']-row['matched_time_error_s'])<1e-10,
                        'Event-time error recount failed.')
            accepted=[e for e in access['crossings'] if e['time_s']>=row['insertion_s'] and
                      abs(e['longitude_residual_deg'])<=tolerance]
            counts['post_insertion_window_accepted']+=bool(accepted)
            require(bool(accepted)==bool(row['post_insertion_window_accepted']),'Extended-window recount failed.')
            require(access['first_accepted_crossing']==(accepted[0] if accepted else None),'First delayed access recount failed.')
            require(record['evidence']['repeat']['status']=='measured','Repeat closure not measured.')
            require(record['evidence']['maximum_burn_position_jump_m']==0,'A commanded impulse reset position.')
            require(record['evidence']['maximum_impulse_residual_m_s']<1e-9,'Impulse realization residual too large.')
        with gzip.open(folder_case/'case_evidence.jsonl.gz','rb') as handle:
            require(handle.read()==b''.join(raw_bytes),'Compressed archive differs from exact checkpoints.')
        saved=json.loads((folder_case/'summary.json').read_text())
        for key,value in counts.items():require(saved['counts'][key]==value,'Aggregate count mismatch: '+key)
        frame=pd.read_csv(folder_case/'per_maneuver.csv')
        require(len(frame)==len(targets),'CSV omitted requests.')
        for column,key in [('matched_longitude_within_tolerance','matched_longitude_within_tolerance'),
                           ('matched_access_accepted','matched_access_accepted'),
                           ('post_insertion_window_accepted','post_insertion_window_accepted')]:
            require(int(frame[column].sum())==counts[key],'CSV flag count mismatch.')
        for field in ['matched_longitude_model_error_deg','matched_time_error_s']:
            values=pd.to_numeric(frame.get(field,pd.Series(dtype=float)),errors='coerce').dropna().to_numpy()
            stats=saved['statistics'][field]
            require(len(values)==stats['count'],'Error statistic denominator mismatch.')
            if len(values):require(np.isclose(np.abs(values).mean(),stats['mean_absolute'],rtol=1e-11,atol=1e-12),
                                   'MAE did not reproduce from the CSV.')
        results.append(dict(experiment_id=identifier,counts=counts))
    combined=pd.read_csv(folder/'all_maneuvers.csv')
    require(len(combined)==manifest['requested_count'],'Combined CSV omitted target-setting pairs.')
    require(not combined.duplicated(['experiment_id','case_index']).any(),'Duplicate combined CSV requests.')
    result=dict(verified=True,requested=manifest['requested_count'],settings=len(results),reused_records=reused,
                maximum_sampled_position_difference_m=maximum_position,maximum_crossing_time_difference_s=maximum_time,
                all_input_and_source_hashes_match=True,all_checkpoint_and_archive_bytes_match=True,
                all_event_acceptance_counts_independently_reproduced=True,all_reused_physical_evidence_unchanged=True,
                all_repeat_closures_measured=True,no_impulse_position_resets=True,
                experiments=results,verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='experiments'},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    args=parser.parse_args();verify(args.data_dir.resolve())
