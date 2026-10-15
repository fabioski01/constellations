"""Export dense-study figures through the project's existing plotting modules.

Numerical arrays are read from checkpoints; plotting never reruns the solver.
"""
from pathlib import Path
from datetime import datetime
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reconfig_plot_from_csv import plot_timesAvgDeltaA_from_csv
from reconfig_plotter_multicore import (Orbit, plot_saved_altitude_summary,
    plot_saved_tolerance_summary, plot_saved_heatmaps, plot_saved_geographic_gain)


def load_grid(data_dir,experiment,tolerance=.25,version='corrected'):
    parts=[]
    for file in sorted((data_dir/'per_target').glob(experiment['experiment_id']+'_lat*.npz')):
        with np.load(file) as d:parts.append({k:d[k] for k in d.files})
    index=int(np.flatnonzero(np.isclose(parts[0]['tolerances_deg'],tolerance,rtol=0,atol=1e-12))[0])
    out=dict(latitude_deg=np.concatenate([d['latitude_deg'] for d in parts]),
             longitude_deg=parts[0]['longitude_deg'],latitude_band_deg=experiment['latitude_band_deg'])
    for field in ['response_s','wait_s','drift_s','branch','phase_wrap','longitude_wrap','success']:
        out[field]=np.concatenate([d[version+'_'+field][index] for d in parts],axis=0)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    figures=output/'figures';figures.mkdir(exist_ok=True)
    tables=output/'tables';tables.mkdir(exist_ok=True)
    manifest=json.loads((args.data_dir/'manifest.json').read_text())
    completion=json.loads((args.data_dir/'completion.json').read_text())
    if not completion['complete']:raise ValueError('Final figures require the complete configured study.')
    summary=pd.read_csv(args.data_dir/'summary.csv')
    physical=pd.read_csv(output/'continuous_clock_audit.csv')
    experiments=manifest['experiments']
    cases=list({e['case']['case_id']:e['case'] for e in experiments}.values())
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.titlesize':10,
        'legend.fontsize':8,'xtick.labelsize':9,'ytick.labelsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    def save(fig,name):
        for extension in ['png','pdf']:fig.savefig(figures/(name+'.'+extension),dpi=250)
        plt.close(fig)
    common=summary[summary.profile.isin(['common','refined_altitude'])].copy()
    # Retain the original four-column CSV interface, with explicit version and
    # full-precision sidecar configuration. Never merge different target grids.
    for case in cases:
        exported={}
        for version in ['corrected','original']:
            g=common[(common.case_id==case['case_id'])&(common.version==version)&
                     np.isclose(common.tolerance_deg,.25)].sort_values('offset_km')
            data=pd.DataFrame(dict(delta_a_km=g.offset_km,avg_total_s=g.reported_mean_response_h*3600,
                avg_wait_s=g.mean_wait_h*3600,avg_drift_s=g.mean_drift_h*3600,
                accepted_count=g.reported_accepted_count,requested_count=g.requested_count))
            name=tables/f'TimesAvgDeltaA_180x360_GOM{case["a_g_km"]:.0f}_ROM{case["a_r_km"]:.0f}_{case["case_id"]}_{version}.csv'
            data.to_csv(name,index=False);exported[version]=name
        fig=plot_timesAvgDeltaA_from_csv(exported['corrected'],outdir=str(figures),
            gom_a_km=case['a_g_km'],rom_a_km=case['a_r_km'],comparison_csv_path=exported['original'],
            figure_title=f'{case["case_id"]} | 360 × 180 targets | 0.25° longitude tolerance')
        plt.close(fig)
        (tables/f'{case["case_id"]}_exact_configuration.json').write_text(json.dumps(case,indent=2)+'\n')
    save(plot_saved_altitude_summary(common,cases[:3],title='Altitude trade | 64,800 common target coordinates'),
         'paper_figure_2_altitude')
    save(plot_saved_altitude_summary(common,[cases[i] for i in [1,3,4]],
         title='Inclination configurations | identical ±49° geographic domain'),'paper_figure_4_inclination')
    def select(case,offset,profile='common'):
        return next(e for e in experiments if e['profile']==profile and e['case']['case_id']==case and
                    np.isclose(e['offset_km'],offset,atol=1e-8,rtol=0))
    e2=next(c for c in cases if c['case_id']=='E2')
    epoch=datetime.fromisoformat(manifest['config']['epoch_utc'])
    gom=Orbit(e2['a_g_km'],np.deg2rad(e2['inclination_deg']),0.,0.)
    land=Path(__file__).resolve().parent/'reference/maps/ne_110m_land.geojson'
    grids=[]
    for offset in [50.,-200.]:
        grid=load_grid(args.data_dir,select('E2',offset));grid['label']=f'Δa={offset:+g} km'
        grids.append(grid)
    save(plot_saved_heatmaps(grids,gom=gom,epoch=epoch,land_geojson=land,
         title='E2 response geography | 360 × 180 targets | 0.25° band'),'paper_figure_3_targets')
    tol=summary[(summary.profile=='common')&summary.case_id.eq('E2')&summary.offset_km.isin([-200.,50.,200.])]
    save(plot_saved_tolerance_summary(tol,case_id='E2',offsets_km=[-200.,50.,200.],
         title='E2 tolerance sensitivity | 64,800 targets per curve'),'paper_figure_5_tolerance')
    base=load_grid(args.data_dir,select('E2',50.),.001)
    wider=load_grid(args.data_dir,select('E2',50.),5.)
    save(plot_saved_geographic_gain(base,narrow_s=base['response_s'],wide_s=wider['response_s'],
         land_geojson=land,title='E2 at +50 km | paired geographic effect of tolerance'),'paper_figure_7_tolerance_location')
    paired=np.isfinite(grids[0]['response_s'])&np.isfinite(grids[1]['response_s'])
    change=(grids[0]['response_s']-grids[1]['response_s'])[paired]/3600.
    location=dict(paired_count=int(paired.sum()),minus200_earlier_count=int(np.sum(change>1/3600)),
        plus50_earlier_count=int(np.sum(change<-1/3600)),mean_reduction_h=float(change.mean()),
        median_reduction_h=float(np.median(change)),p10_reduction_h=float(np.quantile(change,.1)),
        p90_reduction_h=float(np.quantile(change,.9)))
    (output/'location_paired_summary.json').write_text(json.dumps(location,indent=2)+'\n')
    # Compare resolutions at identical cases/offsets/tolerances, and record a
    # longitude-shift sensitivity separately from the refinement sequence.
    convergence=[]
    for row in summary[(summary.version=='corrected')&summary.profile.isin(['coarse','medium','shifted'])].itertuples():
        dense=summary[(summary.version=='corrected')&(summary.profile=='common')&(summary.case_id==row.case_id)&
            np.isclose(summary.offset_km,row.offset_km,rtol=0,atol=1e-8)&np.isclose(summary.tolerance_deg,row.tolerance_deg,rtol=0,atol=1e-12)].iloc[0]
        convergence.append(dict(case_id=row.case_id,offset_km=row.offset_km,tolerance_deg=row.tolerance_deg,
            profile=row.profile,n_lon=row.n_lon,n_lat=row.n_lat,requested_count=row.requested_count,
            accepted_fraction=row.reported_accepted_count/row.requested_count,
            mean_response_h=row.reported_mean_response_h,dense_mean_response_h=dense.reported_mean_response_h,
            difference_from_dense_h=row.reported_mean_response_h-dense.reported_mean_response_h,
            relative_difference_percent=100*(row.reported_mean_response_h/dense.reported_mean_response_h-1),
            accepted_fraction_difference=row.reported_accepted_count/row.requested_count-dense.reported_accepted_count/dense.requested_count))
    pd.DataFrame(convergence).to_csv(output/'grid_convergence.csv',index=False)
    pilot_path=Path(__file__).resolve().parent/'results/paper1/grid-10x6-20261014/summary.csv'
    if pilot_path.exists():
        pilot=pd.read_csv(pilot_path).drop_duplicates(['case_id','offset_km','tolerance_deg'])
        changes=[]
        for row in summary[(summary.profile=='common')&(summary.version=='corrected')].itertuples():
            previous=pilot[pilot.case_id.eq(row.case_id)&np.isclose(pilot.offset_km,row.offset_km,rtol=0,atol=1e-8)&
                           np.isclose(pilot.tolerance_deg,row.tolerance_deg,rtol=0,atol=1e-12)]
            if previous.empty:continue
            previous=previous.iloc[0]
            changes.append(dict(case_id=row.case_id,offset_km=row.offset_km,tolerance_deg=row.tolerance_deg,
                pilot_mean_response_h=previous.mean_response_s/3600.,dense_mean_response_h=row.reported_mean_response_h,
                dense_minus_pilot_mean_h=row.reported_mean_response_h-previous.mean_response_s/3600.,
                scope='Corrected solver in both; sampling change, not an original/corrected code comparison.'))
        pd.DataFrame(changes).to_csv(output/'pilot_to_dense_sampling.csv',index=False)
    # Audit plot uses a common continuous clock; source-clock counts remain
    # available in summary.csv under their separate definitions.
    fig,axes=plt.subplots(2,1,figsize=(8,5.8),sharex=True)
    for ci,case in enumerate(cases):
        g=physical[(physical.profile=='common')&(physical.version=='original')&
            (physical.case_id==case['case_id'])&np.isclose(physical.offset_km,50.)].sort_values('tolerance_deg')
        axes[0].semilogx(g.tolerance_deg,g.continuous_rejected_reported_count/g.reported_count,'o-',label=case['case_id'])
        g=summary[(summary.profile=='common')&(summary.version=='corrected')&
            (summary.case_id==case['case_id'])&np.isclose(summary.offset_km,50.)].sort_values('tolerance_deg')
        axes[1].semilogx(g.tolerance_deg,g.reported_mean_response_h,'o-',label=case['case_id'])
    axes[0].set_ylabel('Original reports failing\ncommon-clock acceptance')
    axes[1].set_ylabel('Corrected mean [h]');axes[1].set_xlabel('Longitude tolerance [deg]')
    axes[0].legend(ncol=5);axes[0].set_ylim(-.03,1.03)
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout();save(fig,'paper_dense_correction_audit')
    metadata=dict(data_manifest_sha256=hashlib.sha256((args.data_dir/'manifest.json').read_bytes()).hexdigest(),
        plotting_source_sha256={name:hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest() for name in
            ['reconfig_plot_from_csv.py','reconfig_plotter_multicore.py','paper1_dense_figures.py']},
        outputs=[p.name for p in sorted(figures.iterdir()) if p.suffix in ['.png','.pdf']],
        scope='Saved per-target cells, existing plot framework; no resimulation or smoothing.')
    (figures/'figure_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'figures':len(metadata['outputs']),'location':location}),flush=True)


if __name__=='__main__':main()
