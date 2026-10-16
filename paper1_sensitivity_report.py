"""Generate comparison tables and publication figures from saved sweep evidence.

This script never plans or propagates a maneuver. All curves are descriptive
summaries of the same geographic coordinates at the sampled settings.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper1_validation_report import distribution, summarize

ROOT = Path(__file__).resolve().parent
COLORS = dict(error='#0072B2', tail='#D55E00', longitude='#0072B2', full='#009E73',
              window='#CC79A7', nominal='#666666', timing='#6F4C9B')
RATE_FIELDS = [
    ('longitude', 'matched_longitude_within_tolerance_percent_of_requested', 'Longitude only'),
    ('full', 'matched_access_accepted_percent_of_requested', 'Full matched access'),
    ('window', 'post_insertion_window_accepted_percent_of_requested', 'Extended-window access'),
    ('nominal', 'nominal_acceptance_percent', 'Nominally feasible')]


def load_campaign(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    records, frames = [], {}
    for entry in manifest['experiments']:
        identifier = entry['experiment_id']
        path = folder/'experiments'/identifier/'per_maneuver.csv'
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if len(frame) != manifest['target_count_per_experiment']:
            raise ValueError('Incomplete per-experiment CSV: '+identifier)
        frames[identifier] = frame
        summary = summarize(frame)
        record = dict(experiment_id=identifier, sweeps=';'.join(entry['sweeps']),
                      labels=';'.join(entry['labels']),
                      **{key:entry['context']['scenario'][key] for key in
                         ['global_altitude_km','inclination_deg','drift_offset_km','longitude_tolerance_deg']},
                      regional_altitude_km=entry['context']['a_r_km']-6371.,
                      **summary['counts'], **summary['rates'])
        record['nominal_acceptance_percent'] = 100*record['nominal_accepted']/record['requested']
        margins=pd.to_numeric(frame.get('nominal_longitude_margin_deg',pd.Series(dtype=float)),errors='coerce').dropna()
        record['near_boundary_nominal_count']=int((margins<1e-6).sum())
        record['near_boundary_percent_of_nominal']=100*record['near_boundary_nominal_count']/record['nominal_accepted'] if record['nominal_accepted'] else np.nan
        for field, values in summary['statistics'].items():
            for stat, value in values.items():
                record[field+'__'+stat] = value
        timing = pd.to_numeric(frame.get('matched_time_error_s',pd.Series(dtype=float)),errors='coerce').dropna()
        for q in [.05,.25,.5,.75,.95]:
            record[f'timing_q{int(100*q):02}_s'] = float(timing.quantile(q)) if len(timing) else np.nan
        records.append(record)
    return manifest, pd.DataFrame(records), frames


def save_figure(figure, output, name):
    for extension in ['png','svg','pdf']:
        options = dict(dpi=200, bbox_inches='tight', facecolor='white')
        if extension == 'svg': options['metadata'] = {'Date':None}
        if extension == 'pdf': options['metadata'] = {'CreationDate':None,'ModDate':None}
        figure.savefig(output/(name+'.'+extension),**options)


def plot_column(axes, data, xfield, xlabel, *, log=False, special=False):
    from matplotlib.lines import Line2D
    data = data.sort_values(xfield)
    ordinary = data[~data.labels.str.contains('exact_regional_coincidence',na=False)]
    if special:
        ordinary = ordinary[ordinary.drift_offset_km.ne(0)]
    error, acceptance, timing = axes
    for field, color, label, style in [
        ('matched_longitude_model_error_deg__mean_absolute',COLORS['error'],'Mean absolute error','-'),
        ('matched_longitude_model_error_deg__absolute_p95',COLORS['tail'],'95th absolute percentile','--')]:
        error.plot(ordinary[xfield],ordinary[field],marker='o',markersize=4,
                   color=color,linestyle=style,label=label,linewidth=1.3)
    for color_key,field,label in RATE_FIELDS:
        acceptance.plot(ordinary[xfield],ordinary[field],marker='o',markersize=3.5,
                        color=COLORS[color_key],linestyle='--' if color_key=='nominal' else '-',
                        linewidth=1.0 if color_key=='nominal' else 1.3,label=label)
    x = ordinary[xfield].to_numpy(float)
    timing.fill_between(x,ordinary.timing_q05_s.to_numpy(float),ordinary.timing_q95_s.to_numpy(float),
                        color=COLORS['timing'],alpha=.14,label='5th–95th percentile')
    timing.plot(x,ordinary.timing_q50_s,color=COLORS['timing'],marker='o',markersize=4,
                label='Median',linewidth=1.3)
    timing.axhline(0,color='#444444',linewidth=.8,linestyle=':')
    if special:
        for mask,marker,label in [
            (data.drift_offset_km.eq(0),'D','Global/drift coincidence'),
            (data.labels.str.contains('exact_regional_coincidence',na=False),'^','Drift/regional coincidence')]:
            subset=data[mask]
            for field,color in [('matched_longitude_model_error_deg__mean_absolute',COLORS['error']),
                                ('matched_longitude_model_error_deg__absolute_p95',COLORS['tail'])]:
                error.scatter(subset[xfield],subset[field],marker=marker,s=38,color=color,
                              edgecolors='#222222',linewidths=.6,zorder=5)
            for color_key,field,_ in RATE_FIELDS:
                acceptance.scatter(subset[xfield],subset[field],marker=marker,s=30,color=COLORS[color_key],
                                   edgecolors='#222222',linewidths=.5,zorder=5)
            timing.scatter(subset[xfield],subset.timing_q50_s,marker=marker,s=38,color=COLORS['timing'],
                           edgecolors='#222222',linewidths=.6,zorder=5)
            timing.vlines(subset[xfield],subset.timing_q05_s,subset.timing_q95_s,color=COLORS['timing'],alpha=.55)
        extra=[Line2D([],[],marker=marker,color='#444444',linestyle='',markerfacecolor='white',label=label)
               for marker,label in [('D','Global/drift coincidence'),('^','Drift/regional coincidence')]]
        handles,labels=error.get_legend_handles_labels()
        error.legend(handles=handles+extra,fontsize=7,loc='center right',framealpha=.95)
    else:
        error.legend(fontsize=7,loc='center right',framealpha=.95)
    acceptance.set_ylim(-3,105)
    acceptance.legend(fontsize=7,loc='upper left',bbox_to_anchor=(.01,.92),
                      borderaxespad=.2,ncol=2,framealpha=.95)
    timing.legend(fontsize=7,loc='best')
    error.set_ylabel('Longitude prediction error [deg]')
    error.set_ylim(bottom=0)
    acceptance.set_ylabel('Percent of requested targets')
    timing.set_ylabel('Crossing-time error [s]')
    timing.set_xlabel(xlabel)
    for ax in axes:
        ax.grid(alpha=.22)
        if log: ax.set_xscale('log')
        if not log: ax.set_xticks(sorted(data[xfield].unique()) if not special else [-200,-100,0,100,200])


def make_figures(output, manifest, table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.titlesize':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'svg.hashsalt':'paper1-sensitivity-plus100'})
    n=manifest['target_count_per_experiment']
    grid=('geographic probes only' if manifest.get('probe') else
          f'{manifest["config"]["n_longitude"]} × {manifest["config"]["n_latitude"]}')
    subsets={name:table[table.sweeps.str.split(';').apply(lambda values:name in values)]
             for name in ['tolerance','drift_offset','orbit']}
    specifications=[
        ('tolerance', 'longitude_tolerance_deg','Longitude tolerance [deg]',
         'Tolerance sensitivity: 700 km, 50°, Δa = +100 km','sensitivity_tolerance',True),
        ('drift_offset','drift_offset_km','Drift offset Δa [km]',
         'Drift-offset sensitivity: 700 km, 50°, tolerance = 0.25°','sensitivity_drift_offset',False)]
    for group,xfield,xlabel,title,name,log in specifications:
        data=subsets[group]
        if len(data)<2: continue
        figure,axes=plt.subplots(3,1,figsize=(7.2,8.2),sharex=True)
        plot_column(axes,data,xfield,xlabel,log=log,special=group=='drift_offset')
        for i,ax in enumerate(axes):ax.set_title(['(a) Prediction accuracy','(b) Access acceptance','(c) Event timing'][i],loc='left')
        figure.suptitle(title+'\n'+f'{n} common targets ({grid}); re-planned at every setting',fontsize=11)
        figure.tight_layout(rect=(0,.055,1,.94),h_pad=1.6)
        footer=(
                    'Fixed request state; connected Cartesian gravity + J2; unchanged numerical mean convention and impulses.\n'
                    'Extended-window access may be delayed. Errors use completed requests; acceptance retains all requests.\n'
                    'Connections join sampled settings and do not establish continuous trends.')
        if group=='drift_offset':footer+='\nAt regional coincidence, access can follow insertion; a negative prediction-time error can remain eligible.'
        figure.text(.10,.015,footer,fontsize=7,va='bottom')
        save_figure(figure,output,name);plt.close(figure)
    orbit=subsets['orbit']
    if len(orbit)>=3:
        figure,axes=plt.subplots(3,2,figsize=(12,8.2),sharex='col')
        altitude=orbit[orbit.inclination_deg.eq(50)].sort_values('global_altitude_km')
        inclination=orbit[orbit.global_altitude_km.eq(700)].sort_values('inclination_deg')
        plot_column(axes[:,0],altitude,'global_altitude_km','Initial mean altitude [km]')
        plot_column(axes[:,1],inclination,'inclination_deg','Inclination [deg]')
        for column,label in enumerate(['Altitude slice: i = 50°','Inclination slice: h = 700 km']):
            axes[0,column].set_title(label)
        figure.suptitle('Orbit sensitivity: Δa = +100 km, longitude tolerance = 0.25°\n'
                        f'{n} common targets ({grid}), latitude band ±49°',fontsize=12)
        figure.tight_layout(rect=(0,.055,1,.94),h_pad=1.8,w_pad=2.4)
        figure.text(.07,.015,'Five orbit configurations; the shared 700 km / 50° case appears in both slices.\n'
                    'The 15/1 regional orbit is recomputed at each inclination. These slices do not isolate altitude–inclination interactions.',
                    fontsize=8,va='bottom')
        save_figure(figure,output,'sensitivity_orbits');plt.close(figure)
        for data,xfield,xlabel,title,name in [
            (altitude,'global_altitude_km','Initial mean altitude [km]',
             'Altitude sensitivity: i = 50°, Δa = +100 km, tolerance = 0.25°','sensitivity_altitude'),
            (inclination,'inclination_deg','Inclination [deg]',
             'Inclination sensitivity: h = 700 km, Δa = +100 km, tolerance = 0.25°','sensitivity_inclination')]:
            figure,axes=plt.subplots(3,1,figsize=(7.2,8.2),sharex=True)
            plot_column(axes,data,xfield,xlabel)
            for i,ax in enumerate(axes):
                ax.set_title(['(a) Prediction accuracy','(b) Access acceptance','(c) Event timing'][i],loc='left')
            figure.suptitle(title+'\n'+f'{n} common targets ({grid}), latitude band ±49°',fontsize=11)
            figure.tight_layout(rect=(0,.055,1,.94),h_pad=1.6)
            figure.text(.10,.015,'The regional 15/1 repeat orbit is recomputed at each inclination.\n'
                        'One request state; unchanged mean convention and impulses; no correction.\n'
                        'Connections join sampled settings and do not establish a continuous trend.',fontsize=7,va='bottom')
            save_figure(figure,output,name);plt.close(figure)
    # Terminal orbit and node-indexed closure remain diagnostics, with no pass limit invented.
    figure,axes=plt.subplots(3,2,figsize=(10,9))
    for row,(group,xfield,xlabel) in enumerate([
        ('tolerance','longitude_tolerance_deg','Longitude tolerance [deg]'),
        ('drift_offset','drift_offset_km','Drift offset Δa [km]'),
        ('orbit','labels','Orbit configuration')]):
        data=subsets[group].sort_values('drift_offset_km' if group=='drift_offset' else
                                        'longitude_tolerance_deg' if group=='tolerance' else 'global_altitude_km')
        if group=='orbit':
            data=data.copy()
            data['_case_label']=[next((p for p in str(value).split(';') if p in ['E1','E2','E3','E4','E5']),str(value)) for value in data.labels]
            data=data.sort_values('_case_label')
        x=np.arange(len(data)) if group=='orbit' else data[xfield].to_numpy(float)
        for column,field,ylabel in [
            (0,'terminal_mean_a_error_m__mean_absolute','Terminal semimajor-axis MAE [m]'),
            (1,'repeat_node_closure_km__mean','Mean 15-node surface closure [km]')]:
            ax=axes[row,column]
            if group=='drift_offset':
                ordinary=data[~data.labels.str.contains('exact_regional_coincidence',na=False) & data.drift_offset_km.ne(0)]
                ax.plot(ordinary[xfield],ordinary[field],marker='o',markersize=4,color=COLORS['error'],linewidth=1.1)
                for mask,marker in [(data.drift_offset_km.eq(0),'D'),
                                    (data.labels.str.contains('exact_regional_coincidence',na=False),'^')]:
                    ax.scatter(data.loc[mask,xfield],data.loc[mask,field],marker=marker,s=34,
                               color=COLORS['error'],edgecolors='#222222',linewidths=.6,zorder=5)
            elif group=='orbit':
                ax.bar(x,data[field],color=COLORS['error'],width=.55)
            else:
                ax.plot(x,data[field],marker='o',markersize=4,color=COLORS['error'],linewidth=1.1)
            ax.set(xlabel=xlabel,ylabel=ylabel)
            ax.set_ylim(bottom=0)
            if group=='tolerance':ax.set_xscale('log')
            if group=='orbit':
                ax.set_xticks(x,data['_case_label'])
            ax.grid(alpha=.22)
    figure.suptitle(f'Terminal-orbit and repeat diagnostics: {n} common targets per setting',fontsize=12)
    figure.tight_layout(rect=(0,.035,1,.96),h_pad=1.8)
    figure.text(.08,.01,'Numerical-mean convention unchanged. No terminal-element or repeat-closure acceptance limit is prescribed.',fontsize=8)
    save_figure(figure,output,'sensitivity_terminal_repeat');plt.close(figure)


def refinement_comparison(folder, refinements, output):
    import matplotlib.pyplot as plt
    _,coarse,_=load_campaign(folder)
    fine_tables=[];fine_manifests=[]
    for refinement in refinements:
        fine_manifest,fine,_=load_campaign(refinement)
        fine_tables.append(fine);fine_manifests.append(fine_manifest)
    fine=pd.concat(fine_tables,ignore_index=True)
    if fine.experiment_id.duplicated().any():raise ValueError('An experiment has multiple refinement inputs.')
    common=coarse.merge(fine,on='experiment_id',suffixes=('_coarse','_fine'))
    fields=['matched_longitude_model_error_deg__mean_absolute',
            'matched_longitude_within_tolerance_percent_of_requested',
            'matched_access_accepted_percent_of_requested',
            'post_insertion_window_accepted_percent_of_requested',
            'matched_time_error_s__mean_absolute']
    rows=[]
    for _,record in common.iterrows():
        result=dict(experiment_id=record.experiment_id)
        for field in fields:
            result[field+'_coarse']=record[field+'_coarse']
            result[field+'_fine']=record[field+'_fine']
            result[field+'_change']=record[field+'_fine']-record[field+'_coarse']
        rows.append(result)
    pd.DataFrame(rows).to_csv(output/'grid_refinement.csv',index=False,float_format='%.15g')
    figure,axes=plt.subplots(1,2,figsize=(10,3.8))
    x=np.arange(len(common))
    for offset,suffix,label,color in [(-.06,'coarse','18 × 18','#0072B2'),(.06,'fine','36 × 18','#D55E00')]:
        axes[0].scatter(x+offset,common['matched_longitude_model_error_deg__mean_absolute_'+suffix],color=color,label=label,s=38)
        axes[1].scatter(x+offset,common['matched_access_accepted_percent_of_requested_'+suffix],color=color,label=label,s=38)
    labels=[f'Δa={row.drift_offset_km_coarse:g} km\n{row.global_altitude_km_coarse:g} km, {row.inclination_deg_coarse:g}°\nδ={row.longitude_tolerance_deg_coarse:g}°'
            for row in common.itertuples()]
    for ax in axes:
        ax.set_xticks(x,labels,fontsize=8);ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].set_ylabel('Longitude prediction MAE [deg]')
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel('Full matched access [% of requested]')
    figure.suptitle('Selected longitude-grid refinement checks',fontsize=11)
    figure.tight_layout(rect=(0,0,1,.93))
    save_figure(figure,output,'sensitivity_grid_refinement');plt.close(figure)
    return dict(fine_grids=[m['config']['n_longitude'] for m in fine_manifests],comparisons=rows)


def common_target_comparison(output,manifest,table,frames):
    """Keep the error comparison on the same feasible coordinate cohort."""
    import matplotlib.pyplot as plt
    rows=[]
    for group in ['tolerance','drift_offset','orbit']:
        entries=[e for e in manifest['experiments'] if group in e['sweeps']]
        common=set.intersection(*[set(frames[e['experiment_id']].loc[
            frames[e['experiment_id']].status.eq('completed') &
            frames[e['experiment_id']].matched_crossing_found,'case_index']) for e in entries])
        for entry in entries:
            frame=frames[entry['experiment_id']]
            subset=frame[frame.case_index.isin(common)]
            values=distribution(subset.matched_longitude_model_error_deg)
            rows.append(dict(sweep=group,experiment_id=entry['experiment_id'],common_target_count=len(common),
                longitude_prediction_mae_deg=values.get('mean_absolute'),
                longitude_prediction_p95_deg=values.get('absolute_p95'),
                full_acceptance_percent_of_common_targets=100*subset.matched_access_accepted.sum()/len(common) if common else np.nan))
    paired=pd.DataFrame(rows)
    paired.to_csv(output/'common_target_comparison.csv',index=False,float_format='%.15g')
    data=table.merge(paired[paired.sweep.eq('drift_offset')],on='experiment_id').sort_values('drift_offset_km')
    count=int(data.common_target_count.iloc[0])
    figure,axes=plt.subplots(1,2,figsize=(10,4.3))
    special=data.labels.str.contains('exact_regional_coincidence',na=False) | data.drift_offset_km.eq(0)
    for ax,primary,shared,ylabel in [
        (axes[0],'matched_longitude_model_error_deg__mean_absolute','longitude_prediction_mae_deg','Mean absolute prediction error [deg]'),
        (axes[1],'matched_longitude_model_error_deg__absolute_p95','longitude_prediction_p95_deg','95th absolute prediction error [deg]')]:
        for field,color,label,style in [(primary,COLORS['error'],'Completed targets at each setting','-'),
                                       (shared,COLORS['tail'],f'Same {count} targets at every setting','--')]:
            ax.plot(data.loc[~special,'drift_offset_km'],data.loc[~special,field],color=color,
                    linestyle=style,marker='o',markersize=3.5,label=label,linewidth=1.2)
            for mask,marker in [(data.drift_offset_km.eq(0),'D'),
                                (data.labels.str.contains('exact_regional_coincidence',na=False),'^')]:
                ax.scatter(data.loc[mask,'drift_offset_km'],data.loc[mask,field],marker=marker,
                    s=52 if field==shared else 30,facecolors='none' if field==shared else color,
                    edgecolors=color,linewidths=1.1,zorder=5)
        ax.set(xlabel='Drift offset Δa [km]',ylabel=ylabel,ylim=(0,None))
        ax.set_xticks([-200,-100,0,100,200]);ax.grid(alpha=.22)
        ax.legend(fontsize=7,loc='best',framealpha=.95)
    figure.suptitle('Drift-offset accuracy with a fixed feasible target cohort\n700 km, 50°, longitude tolerance = 0.25°',fontsize=11)
    figure.tight_layout(rect=(0,.09,1,.91),w_pad=2)
    figure.text(.08,.015,'Diamonds: global/drift coincidence; triangles: drift/regional coincidence.\n'
                'This supplemental error comparison holds coordinates fixed; primary acceptance fractions retain all 324 requests.',fontsize=8)
    save_figure(figure,output,'sensitivity_common_targets');plt.close(figure)
    return paired


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--refinement-dir',type=Path,action='append')
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    output=args.data_dir.resolve()
    manifest,table,frames=load_campaign(output)
    complete=len(table)==len(manifest['experiments'])
    if not complete and not args.partial:raise ValueError('The campaign is incomplete; use --partial for provisional plots.')
    table.to_csv(output/'sensitivity_summary.csv',index=False,float_format='%.15g')
    make_figures(output,manifest,table)
    paired=common_target_comparison(output,manifest,table,frames) if complete else None
    refinement=refinement_comparison(output,[p.resolve() for p in args.refinement_dir],output) if args.refinement_dir else None
    readable=table[['experiment_id','global_altitude_km','inclination_deg','drift_offset_km','longitude_tolerance_deg',
                    'requested','nominal_accepted','numerically_completed','numerical_failures',
                    'matched_longitude_model_error_deg__mean_absolute',
                    'matched_longitude_model_error_deg__absolute_p95',
                    'matched_longitude_within_tolerance_percent_of_requested',
                    'matched_access_accepted_percent_of_requested',
                    'post_insertion_window_accepted_percent_of_requested',
                    'matched_time_error_s__mean_absolute','terminal_mean_a_error_m__mean_absolute','repeat_node_closure_km__mean',
                    'near_boundary_percent_of_nominal']]
    readable.to_csv(output/'comparison_table.csv',index=False,float_format='%.15g')
    text=['# Controlled propagation sensitivity study',
          f'\nStatus: {"complete" if complete else "PROVISIONAL; incomplete sweep"}. {len(table)} unique completed settings; '
          f'{manifest["target_count_per_experiment"]} identical geographic targets per setting. '
          'The reference is 700 km, 50° inclination, +100 km drift offset and 0.25° longitude tolerance.\n',
          'Every target is re-planned at every setting and the resulting fixed impulse sequence is executed in connected Cartesian central gravity plus J2. '
          'Every new trajectory is repeated with tighter integration settings. No margin, retiming, correction or added force is introduced.\n',
          'The orbit study uses five configurations: altitude at 50° inclination, and inclination at 700 km. The common geographic band is ±49°; '
          'the regional 15/1 repeat orbit is recomputed for each inclination. The experiment does not measure an altitude–inclination interaction surface.\n',
          'Longitude prediction error compares propagated and nominal longitude at corresponding latitude crossings. Longitude-only acceptance tests the original '
          'target band; full matched acceptance additionally requires insertion eligibility. Extended-window acceptance may be delayed and does not validate '
          'the planned response time. Acceptance uses every requested coordinate; conditional rates, error sample counts and nominal infeasibility remain explicit '
          'in sensitivity_summary.csv. Separate spherical cell-area fractions do not represent a target-demand distribution.\n',
          '| Altitude km | Inclination ° | Δa km | Tolerance ° | Nominal / requested | Longitude MAE ° | Longitude pass % | Full matched % | Extended-window % |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in table.itertuples(index=False,name=None):
        record=dict(zip(table.columns,row))
        text.append('| '+ ' | '.join([
            f'{record["global_altitude_km"]:g}',f'{record["inclination_deg"]:g}',f'{record["drift_offset_km"]:.6g}',
            f'{record["longitude_tolerance_deg"]:g}',f'{record["nominal_accepted"]}/{record["requested"]}',
            f'{record["matched_longitude_model_error_deg__mean_absolute"]:.6f}',
            f'{record["matched_longitude_within_tolerance_percent_of_requested"]:.2f}',
            f'{record["matched_access_accepted_percent_of_requested"]:.2f}',
            f'{record["post_insertion_window_accepted_percent_of_requested"]:.2f}'])+' |')
    text.extend(['\n## Interpretation limits\n',
                 'These are descriptive results for a fixed request epoch, initial mean node and phase, deterministic target grid and stated numerical mean convention. '
                 'Small numerical convergence differences establish integration stability, while mean-state realization, secular approximation and fixed-impulse execution '
                 'remain combined sources of model discrepancy. Their causal contributions have not been isolated. A wider tolerance changes both the acceptance band and '
                 'the maneuver selected by the planner; neither propagated acceptance nor prediction error is assumed to be monotonic.\n',
                 'Coincident global/drift and drift/regional cases remove a transfer and have different insertion/access timing relationships. They are marked separately '
                 'in the drift-offset figure and must not be interpreted as ordinary points on a smooth drift-altitude trend.\n'])
    if paired is not None:
        cohort=int(paired.loc[paired.sweep.eq('drift_offset'),'common_target_count'].iloc[0])
        text.append(f'The supplemental common-target comparison uses the same {cohort} nominally feasible, numerically completed coordinates at every drift offset. '
                    'It checks the influence of changing target composition on the error distributions. Primary access fractions continue to retain every requested coordinate. '
                    'Tolerance and orbit cohorts, together with the conditional statistics, are retained in common_target_comparison.csv.\n')
    if refinement:
        text.append('Selected 36×18 longitude-grid refinements are reported in grid_refinement.csv and sensitivity_grid_refinement. '
                    'They retain all 18 latitude rows and reuse only physically identical existing requests with verified numerical source hashes. '
                    'They do not establish latitude-grid or request-state convergence.\n')
    (output/'results.md').write_text('\n'.join(text)+'\n')
    products=[p for p in output.iterdir() if p.is_file() and p.name not in ['checksums.json','progress.json']]
    checksums={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(products)}
    (output/'checksums.json').write_text(json.dumps(checksums,indent=2)+'\n')
    print(json.dumps(dict(complete=complete,settings=len(table),targets_per_setting=manifest['target_count_per_experiment'],
                         numerical_failures=int(table.numerical_failures.sum()),report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
