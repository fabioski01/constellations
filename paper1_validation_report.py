"""Statistics and scientific figures for saved connected-propagation results."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def distribution(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(count=0)
    return dict(count=len(a), mean=float(a.mean()), mean_absolute=float(np.abs(a).mean()),
                rms=float(np.sqrt(np.mean(a*a))), median=float(np.median(a)),
                absolute_p50=float(np.quantile(np.abs(a), .5)),
                absolute_p90=float(np.quantile(np.abs(a), .9)),
                absolute_p95=float(np.quantile(np.abs(a), .95)),
                absolute_p99=float(np.quantile(np.abs(a), .99)),
                minimum=float(a.min()), maximum=float(a.max()), maximum_absolute=float(np.abs(a).max()))


def summarize(frame):
    n = len(frame)
    completed = frame.status.eq('completed')
    nominal = frame.nominal_success.fillna(False).astype(bool)
    counts = dict(requested=n, nominal_accepted=int(nominal.sum()),
                  numerically_completed=int(completed.sum()),
                  nominal_infeasible=int(frame.status.eq('nominal_infeasible').sum()),
                  numerical_failures=int(frame.status.isin(['numerical_failure', 'convergence_failure']).sum()))
    for field in ['matched_crossing_found', 'matched_longitude_within_tolerance',
                  'matched_access_accepted', 'post_insertion_window_accepted']:
        mask = completed & frame[field].fillna(False).astype(bool)
        counts[field] = int(mask.sum())
    counts['matched_before_insertion'] = int((completed & frame.matched_crossing_found.fillna(False).astype(bool) &
        frame.get('matched_after_insertion', pd.Series(False,index=frame.index)).eq(False)).sum())
    counts['later_success_without_matched_acceptance'] = int((completed &
        frame.post_insertion_window_accepted.fillna(False).astype(bool) &
        ~frame.matched_access_accepted.fillna(False).astype(bool)).sum())
    rates = {}
    for field in ['matched_longitude_within_tolerance', 'matched_access_accepted', 'post_insertion_window_accepted']:
        mask = completed & frame[field].fillna(False).astype(bool)
        rates[field+'_percent_of_requested'] = 100*counts[field]/n if n else None
        rates[field+'_percent_of_nominal_accepted'] = 100*counts[field]/counts['nominal_accepted'] if counts['nominal_accepted'] else None
        total_weight = float(frame.area_weight.sum())
        rates[field+'_spherical_area_percent_of_requested'] = 100*float(frame.loc[mask,'area_weight'].sum())/total_weight if total_weight else None
    categories = frame.loc[completed, 'failure_category'].value_counts().to_dict()
    statistics = {}
    for field in ['nominal_response_s','wait_s','drift_s','nominal_longitude_margin_deg',
                  'matched_time_error_s','matched_longitude_model_error_deg',
                  'matched_longitude_residual_deg','matched_longitude_excess_deg',
                  'matched_target_distance_km','at_prediction_latitude_residual_deg',
                  'at_prediction_longitude_model_error_deg','at_prediction_surface_model_error_km',
                  'first_accepted_post_insertion_delay_s',
                  'terminal_mean_a_error_m','terminal_mean_eccentricity',
                  'terminal_mean_inclination_error_deg','repeat_node_closure_km',
                  'repeat_interval_error_s','repeat_nominal_time_closure_km',
                  'convergence_maximum_position_difference_m',
                  'convergence_maximum_crossing_time_difference_s']:
        statistics[field] = distribution(frame.loc[completed, field]) if field in frame else dict(count=0)
    return dict(counts=counts, rates=rates, failure_categories=categories, statistics=statistics)


def write_figures(output, manifest, frame, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from paper1_validate_maneuver import _land

    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.titlesize':8.5,
                         'svg.hashsalt':'paper1-grid-validation-36x18'})
    good = frame[frame.status.eq('completed')].copy()
    matched = good[good.matched_crossing_found.astype(bool)]
    tolerance = manifest['context']['scenario']['longitude_tolerance_deg']
    figure = plt.figure(figsize=(6.8,8.8))
    grid = figure.add_gridspec(3,2,height_ratios=[1,1,.85])
    error_map = figure.add_subplot(grid[0,:])
    outcome_map = figure.add_subplot(grid[1,:])
    cdf = figure.add_subplot(grid[2,0])
    duration = figure.add_subplot(grid[2,1])
    for axis in [error_map,outcome_map]:
        _land(axis)
        axis.grid(alpha=.2)
    values = matched.matched_longitude_model_error_deg.to_numpy()
    bound = max(float(np.max(np.abs(values))),1e-8)
    points = error_map.scatter(matched.target_longitude_deg,matched.target_latitude_deg,
                               c=values,s=11,marker='s',cmap='RdBu_r',
                               norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),zorder=3)
    figure.colorbar(points,ax=error_map,pad=.02,label='Propagated minus nominal longitude [deg]')
    error_map.set_title('(a) Longitude prediction error at matched crossings')
    colors = {'accepted':'#009E73','longitude_only':'#E69F00',
              'before_insertion_only':'#56B4E9','longitude_and_before_insertion':'#CC79A7',
              'no_matched_crossing':'#777777','nominal_infeasible':'#222222',
              'numerical_failure':'#D55E00','convergence_failure':'#D55E00'}
    names = {'accepted':'Accepted at matched event','longitude_only':'Longitude only',
             'before_insertion_only':'Before insertion only',
             'longitude_and_before_insertion':'Longitude + before insertion',
             'no_matched_crossing':'No matched event','nominal_infeasible':'Nominally infeasible',
             'numerical_failure':'Numerical failure','convergence_failure':'Convergence failure'}
    handles = []
    for category in colors:
        mask = (frame.failure_category.eq(category) & frame.status.eq('completed')) if category not in ['nominal_infeasible','numerical_failure','convergence_failure'] else frame.status.eq(category)
        subset=frame[mask]
        if len(subset):
            outcome_map.scatter(subset.target_longitude_deg,subset.target_latitude_deg,
                                color=colors[category],s=11,marker='s',zorder=3)
            handles.append(Line2D([],[],linestyle='',marker='s',markersize=5,color=colors[category],label=f'{names[category]} ({len(subset)})'))
    outcome_map.legend(handles=handles,loc='upper center',fontsize=7,ncol=2,framealpha=.95)
    outcome_map.set_title('(b) Original longitude and insertion requirements')
    for column,color,label in [('matched_longitude_model_error_deg','#0072B2','Prediction error'),
                               ('matched_longitude_residual_deg','#D55E00','Target offset')]:
        ordered=np.sort(np.abs(matched[column].to_numpy()))
        cdf.step(ordered,np.arange(1,len(ordered)+1)/len(ordered),where='post',color=color,label=label)
    cdf.axvline(tolerance,color='#009E73',linestyle='--',label=f'Offset limit ({tolerance:g} deg)')
    cdf.set(xlabel='Absolute longitude quantity [deg]',ylabel='Coordinate fraction',
            ylim=(0,1.02),title='(c) Longitude error distributions')
    cdf.legend(fontsize=8,loc='lower right');cdf.grid(alpha=.2)
    scatter=duration.scatter(matched.nominal_response_s/3600,matched.matched_longitude_model_error_deg,
                              c=matched.target_latitude_deg,cmap='viridis',s=5,alpha=.8)
    figure.colorbar(scatter,ax=duration,pad=.02,label='Target latitude [deg]')
    duration.axhline(0,color='#777777',linewidth=.8)
    duration.set(xlabel='Nominal response [h]',ylabel='Longitude prediction error [deg]',
                 title='(d) Error versus duration')
    duration.grid(alpha=.2)
    counts=summary['counts'];rates=summary['rates'];stats=summary['statistics']
    figure.suptitle('Connected J2 execution on 36 × 18 targets\nE2, +50 km, 0.25 deg longitude tolerance',fontsize=10)
    footer=(f'{counts["numerically_completed"]}/{counts["requested"]} requests numerically completed. '
            f'Longitude within band: {rates["matched_longitude_within_tolerance_percent_of_requested"]:.2f}%.\n'
            f'Full matched-event acceptance: {rates["matched_access_accepted_percent_of_requested"]:.2f}%. '
            f'Longitude prediction MAE: {stats["matched_longitude_model_error_deg"]["mean_absolute"]:.5f} deg.\n'
            'One request state; numerical mean initialization; fixed impulses; no correction.')
    figure.text(.08,.025,footer,fontsize=7,va='bottom',linespacing=1.5)
    figure.subplots_adjust(left=.10,right=.96,bottom=.15,top=.90,wspace=.65,hspace=.55)
    for extension in ['png','svg']:
        figure.savefig(output/f'validation_grid.{extension}',dpi=180,
                       metadata={'Date':None} if extension=='svg' else None)
    plt.close(figure)

    figure,axes=plt.subplots(1,2,figsize=(6.8,3.3))
    timing=axes[0]
    span=float(matched.matched_time_error_s.max()-matched.matched_time_error_s.min())
    xmin=min(-.05,float(matched.matched_time_error_s.min())-.08*span)
    xmax=max(.05,float(matched.matched_time_error_s.max())+.08*span)
    timing.fill_between([0,xmax],-tolerance,tolerance,color='#009E73',alpha=.12)
    timing.axvline(0,color='#333333',linewidth=.9)
    timing.plot([0,xmax],[tolerance,tolerance],color='#009E73',linewidth=.8)
    timing.plot([0,xmax],[-tolerance,-tolerance],color='#009E73',linewidth=.8)
    sc=timing.scatter(matched.matched_time_error_s,matched.matched_longitude_residual_deg,
                      c=matched.nominal_response_s/3600,cmap='viridis',s=12,alpha=.8)
    figure.colorbar(sc,ax=timing,label='Nominal response [h]',pad=.02)
    timing.set(xlabel='Crossing minus nominal insertion [s]',
               ylabel='Longitude offset from target [deg]',xlim=(xmin,xmax),title='(a) Matched-event feasibility')
    timing.text(.02,.98,'Required: time ≥ 0\nand longitude inside band',
                 transform=timing.transAxes,va='top',fontsize=6.5)
    repeat=axes[1]
    repeat.scatter(good.nominal_response_s/3600,good.repeat_node_closure_km,
                    c='#0072B2',s=12,alpha=.7)
    repeat.set(xlabel='Nominal response [h]',ylabel='15-node surface closure [km]',
               title='(b) Achieved repeat closure')
    for ax in axes:ax.grid(alpha=.2)
    figure.tight_layout(pad=1.7)
    for extension in ['png','svg']:
        figure.savefig(output/f'validation_diagnostics.{extension}',dpi=180,
                       metadata={'Date':None} if extension=='svg' else None)
    plt.close(figure)


def write_report(output, manifest, rows):
    output=Path(output)
    frame=pd.DataFrame(rows).sort_values('case_index')
    frame.to_csv(output/'per_maneuver.csv',index=False,float_format='%.15g')
    summary=summarize(frame)
    summary.update(schema='paper1_connected_grid_summary_v1',study_id=manifest['context']['grid_config']['study_id'],
                   complete=len(frame)==manifest['requested_count'],
                   report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   statistical_scope='Descriptive statistics of this fixed coordinate grid; no random-sample confidence intervals or extrapolation to other configurations.',
                   acceptance_definition=manifest['acceptance'],limitations=manifest['limitations'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    groups=[]
    for name in ['target_latitude_deg','nominal_branch']:
        if name in frame:
            for key,subset in frame.groupby(name,dropna=False):
                s=summarize(subset)
                groups.append(dict(grouping=name,group=str(key),**s['counts'],**s['rates'],
                    longitude_prediction_mae_deg=s['statistics']['matched_longitude_model_error_deg'].get('mean_absolute'),
                    longitude_prediction_p95_deg=s['statistics']['matched_longitude_model_error_deg'].get('absolute_p95'),
                    timing_mae_s=s['statistics']['matched_time_error_s'].get('mean_absolute')))
    for field in ['nominal_response_s','wait_s','drift_s']:
        if field not in frame:continue
        bins=pd.qcut(frame[field],4,duplicates='drop')
        for key,subset in frame.groupby(bins,observed=True):
            s=summarize(subset)
            groups.append(dict(grouping=field+'_quartile',group=str(key),**s['counts'],**s['rates'],
                longitude_prediction_mae_deg=s['statistics']['matched_longitude_model_error_deg'].get('mean_absolute'),
                longitude_prediction_p95_deg=s['statistics']['matched_longitude_model_error_deg'].get('absolute_p95'),
                timing_mae_s=s['statistics']['matched_time_error_s'].get('mean_absolute')))
    pd.DataFrame(groups).to_csv(output/'grouped_summary.csv',index=False,float_format='%.15g')
    write_figures(output,manifest,frame,summary)
    c=summary['counts'];r=summary['rates'];s=summary['statistics']
    text=(f'# Connected maneuver validation on a 36 by 18 grid\n\n'
          f'This unchanged E2 +50 km experiment requests {c["requested"]} targets, of which '
          f'{c["nominal_accepted"]} have nominal solutions and {c["numerically_completed"]} '
          f'complete the numerical convergence checks. There are {c["numerical_failures"]} numerical failures.\n\n'
          f'The corresponding latitude crossing is inside the original longitude band for '
          f'{c["matched_longitude_within_tolerance"]} requests '
          f'({r["matched_longitude_within_tolerance_percent_of_requested"]:.4f}%). '
          f'Including the requirement that insertion is complete, {c["matched_access_accepted"]} '
          f'({r["matched_access_accepted_percent_of_requested"]:.4f}%) are accepted at the matched event. '
          f'The longer post-insertion window contains acceptable access for '
          f'{c["post_insertion_window_accepted"]} requests '
          f'({r["post_insertion_window_accepted_percent_of_requested"]:.4f}%); '
          'that separate quantity does not validate the planned response time.\n\n'
          f'Longitude prediction error is propagated minus nominal longitude at corresponding crossings. '
          f'Its mean is {s["matched_longitude_model_error_deg"]["mean"]:.8f} deg, '
          f'MAE {s["matched_longitude_model_error_deg"]["mean_absolute"]:.8f} deg, '
          f'RMS {s["matched_longitude_model_error_deg"]["rms"]:.8f} deg, '
          f'95th absolute percentile {s["matched_longitude_model_error_deg"]["absolute_p95"]:.8f} deg, '
          f'and maximum absolute error {s["matched_longitude_model_error_deg"]["maximum_absolute"]:.8f} deg. '
          'This differs from the propagated longitude offset from the target.\n\n'
          f'The matched latitude-crossing timing MAE is {s["matched_time_error_s"]["mean_absolute"]:.6f} s. '
          f'Among extended-window successes, the median delay beyond prediction is '
          f'{s["first_accepted_post_insertion_delay_s"]["median"]/3600:.6f} h. '
          f'The mean 15-node closure is {s["repeat_node_closure_km"]["mean"]:.6f} km.\n\n'
          'Every target retains a record, including rejected or unresolved cases. '
          'Primary rates count target coordinates equally; spherical cell-weighted rates are separate. '
          'No empirical margin, correction or additional force has been introduced. '
          'This single configuration and request state do not establish an all-configuration success rate.\n')
    (output/'results.md').write_text(text)
    files=['manifest.json','completion.json','checkpoint.jsonl','per_maneuver.csv','grouped_summary.csv',
           'summary.json','case_evidence.jsonl.gz','validation_grid.png','validation_grid.svg',
           'validation_diagnostics.png','validation_diagnostics.svg','results.md',
           'los_angeles_validation.png','los_angeles_validation.svg','verification.json']
    checksums={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in files if (output/name).exists()}
    (output/'checksums.json').write_text(json.dumps(checksums,indent=2)+'\n')
    print(json.dumps(dict(counts=c,rates=r),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    args=parser.parse_args()
    output=args.data_dir.resolve()
    manifest=json.loads((output/'manifest.json').read_text())
    if not (output/'case_evidence.jsonl.gz').exists():
        raise FileNotFoundError('The full raw evidence is distributed separately. Place case_evidence.jsonl.gz in the data directory, or run paper1_validate_grid.py to reproduce it.')
    with gzip.open(output/'case_evidence.jsonl.gz','rt') as handle:
        rows=[json.loads(line)['row'] for line in handle]
    write_report(output,manifest,rows)


if __name__=='__main__':main()
