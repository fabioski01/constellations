"""Compare phase-wrap pruning and search refinement on declared grid requests."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from paper1_analysis import run_request
from paper1_grid_analysis import build_grid, build_tasks


def evaluate(task):
    pruned = run_request(dict(task,prune_phase_wraps=True))
    full = run_request(dict(task,prune_phase_wraps=False))
    refined = run_request(dict(task,prune_phase_wraps=True,scan_discovery=320,
                              scan_bracket=320,longitude_neighbor_pad=12))
    row={k:task[k] for k in ['case_id','target_id','target_latitude_deg',
                             'target_longitude_deg','offset_km','tolerance_deg']}
    for name,result in [('pruned',pruned),('full',full),('refined',refined)]:
        row[name+'_success']=result['success']
        row[name+'_response_s']=result['response_s']
        row[name+'_runtime_s']=result['runtime_s']
        row[name+'_status']=result['status']
    row['pruning_status_agrees']=pruned['success']==full['success']
    row['refinement_status_agrees']=pruned['success']==refined['success']
    for name,result in [('pruning',full),('refinement',refined)]:
        difference=abs(pruned['response_s']-result['response_s']) if pruned['success'] and result['success'] else np.nan
        row[name+'_time_difference_s']=difference
        row[name+'_agrees']=row[name+'_status_agrees'] and (not pruned['success'] or difference<=.01)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/paper1_grid_10x6.json'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    c=json.loads(args.config.read_text())
    tasks,_=build_tasks(c,build_grid(c['grid']))
    selected=[t for t in tasks if t['study']=='altitude' and t['target_id'] in (1,34)
              and t['offset_km'] in (-200.0,50.0)]
    selected += [t for t in tasks if t['study']=='tolerance' and t['case_id']=='E2'
                 and (t['target_id'],t['offset_km'],t['tolerance_deg']) in
                 [(20,50.0,.001),(59,50.0,5.0),(20,200.0,.001),(59,200.0,5.0)]]
    assert len(selected)==24
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(evaluate,selected))
    d=pd.DataFrame(rows)
    d.to_csv(args.output_dir/'search_checks.csv',index=False)
    result=dict(requests=len(d),pruning_disagreements=int((~d.pruning_agrees).sum()),
        refinement_disagreements=int((~d.refinement_agrees).sum()),
        maximum_pruning_time_difference_s=float(d.pruning_time_difference_s.max()),
        maximum_refinement_time_difference_s=float(d.refinement_time_difference_s.max()),
        baseline_scans=[160,160],baseline_neighbor_pad=8,
        refined_scans=[320,320],refined_neighbor_pad=12,time_comparison_allowance_s=.01,
        scope='A declared 24-request check, not proof of band-search, altitude or target-grid convergence.')
    (args.output_dir/'search_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if result['pruning_disagreements'] or result['refinement_disagreements']:
        raise SystemExit('Search comparison failed; inspect search_checks.csv.')


if __name__=='__main__':
    main()
