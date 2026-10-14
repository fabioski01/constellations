"""Exact plots of the grid study; no interpolation between target locations."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS=dict(E1='#0072B2',E2='#D55E00',E3='#009E73',E4='#CC79A7',E5='#8B4513')
LABELS=dict(E1='E1 500 km 50°',E2='E2 700 km 50°',E3='E3 1000 km 50°',E4='E4 700 km 60°',E5='E5 700 km 98°')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    args=p.parse_args()
    output=args.data_dir/'figures'; output.mkdir(exist_ok=True)
    records=pd.read_csv(args.data_dir/'per_target.csv')
    summary=pd.read_csv(args.data_dir/'summary.csv')
    targets=pd.read_csv(args.data_dir/'targets.csv')
    paired=pd.read_csv(args.data_dir/'tolerance_paired.csv')
    lats=np.sort(targets.target_latitude_deg.unique())
    lons=np.sort(targets.target_longitude_deg.unique())
    n=len(targets)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.labelsize':10,
        'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    def save(fig,name):
        for extension in ('png','pdf'):
            fig.savefig(output/(name+'.'+extension),dpi=250)
        plt.close(fig)
    def altitude(cases,name):
        fig,axes=plt.subplots(3,1,figsize=(7,6.1),sharex=True,
            gridspec_kw={'height_ratios':[1.5,.8,1]})
        for case in cases:
            data=records[(records.study=='altitude')&(records.case_id==case)]
            g=summary[(summary.study=='altitude')&(summary.case_id==case)].sort_values('offset_km')
            regional=data.a_r_km.iloc[0]-data.a_g_km.iloc[0]
            regular=g[(g.offset_km-regional).abs()>1e-8]
            costs=data.groupby('offset_km').outbound_cost_m_s.first()
            axes[0].plot(regular.offset_km,regular.mean_response_s/3600,'o-',color=COLORS[case],label=LABELS[case],lw=1.2,ms=3)
            axes[1].plot(regular.offset_km,regular.feasible_fraction,'o-',color=COLORS[case],lw=1.2,ms=3)
            axes[2].plot(regular.offset_km,costs.reindex(regular.offset_km),'o-',color=COLORS[case],lw=1.2,ms=3)
            exact=g[(g.offset_km-regional).abs()<1e-8].iloc[0]
            for ax,value in zip(axes,[exact.mean_response_s/3600,exact.feasible_fraction,costs.loc[exact.offset_km]]):
                if np.isfinite(value): ax.plot(regional,value,'*',ms=11,color=COLORS[case],zorder=5)
        axes[0].set_ylabel('Conditional mean response [h]')
        peak=summary[(summary.study=='altitude')&summary.case_id.isin(cases)].mean_response_s.max()/3600
        axes[0].set_ylim(0,peak*1.22)
        axes[0].legend(frameon=False,ncols=3,loc='upper center')
        axes[0].set_title(f'{n} common targets | Longitude band 0.25° | Wait/drift caps 200/300 h')
        axes[1].set_ylabel('Accepted fraction'); axes[1].set_ylim(-.04,1.08); axes[1].set_yticks([0,.5,1])
        axes[2].set_ylabel('Outbound cost [m/s]'); axes[2].set_xlabel('Drift offset relative to global orbit [km]')
        for ax in axes:
            ax.axvline(0,color='.45',linestyle='--',lw=.8); ax.grid(alpha=.22)
        fig.tight_layout(h_pad=1.1); save(fig,name)
    altitude(['E1','E2','E3'],'paper_figure_2_altitude')
    altitude(['E2','E4','E5'],'paper_figure_4_inclination')

    def matrix(group,field):
        return group.pivot(index='target_latitude_deg',columns='target_longitude_deg',values=field).reindex(index=lats,columns=lons).to_numpy(dtype=float)
    def coordinate_axes(ax,show_y=True,show_x=True):
        ax.set_xticks(range(len(lons)),[f'{v:g}' for v in lons])
        ax.set_yticks(range(len(lats)),[f'{v:.1f}' for v in lats])
        if show_y: ax.set_ylabel('Geodetic latitude [deg]')
        else: ax.tick_params(labelleft=False)
        if show_x: ax.set_xlabel('Longitude [deg]')
        else: ax.tick_params(labelbottom=False)
    fig=plt.figure(figsize=(7.5,7.1))
    grid=fig.add_gridspec(3,3,width_ratios=[1,1,.04],left=.1,right=.88,bottom=.09,top=.91,hspace=.24,wspace=.17)
    e2=records[(records.study=='altitude')&(records.case_id=='E2')]
    cmap=plt.colormaps['viridis'].copy(); cmap.set_bad('#dddddd')
    for i,(field,label) in enumerate([('response_s','Response [h]'),('wait_s','Waiting [h]'),('drift_s','Drift [h]')]):
        valid=e2[e2.offset_km.isin([50,-200])&e2.success]
        vmax=valid[field].max()/3600
        for j,offset in enumerate([50,-200]):
            ax=fig.add_subplot(grid[i,j]); g=e2[e2.offset_km==offset].copy()
            g.loc[~g.success,field]=np.nan
            values=matrix(g,field)/3600
            im=ax.imshow(np.ma.masked_invalid(values),origin='lower',aspect='auto',interpolation='nearest',cmap=cmap,vmin=0,vmax=vmax)
            failed=np.argwhere(~np.isfinite(values))
            if len(failed): ax.scatter(failed[:,1],failed[:,0],marker='x',color='.3',s=24,lw=1)
            coordinate_axes(ax,j==0,i==2)
            if i==0: ax.set_title(f'Offset {offset:+g} km | {int(g.success.sum())}/{n} accepted')
        fig.colorbar(im,cax=fig.add_subplot(grid[i,2])).set_label(label)
    fig.suptitle('E2 target-coordinate arrays | No spatial interpolation',fontsize=11,y=.975)
    save(fig,'paper_figure_3_targets')

    tol=summary[(summary.study=='tolerance')&(summary.offset_km==50)]
    fig,axes=plt.subplots(3,1,figsize=(7,6.7),sharex=True,
        gridspec_kw={'height_ratios':[1.3,1.3,.7]})
    for case,g in tol.groupby('case_id'):
        g=g.sort_values('tolerance_deg')
        axes[0].semilogx(g.tolerance_deg,g.mean_response_s/3600,'o-',color=COLORS[case],label=LABELS[case],ms=3,lw=1.2)
        axes[2].semilogx(g.tolerance_deg,g.accepted_by_72h_fraction,'o-',color=COLORS[case],ms=3,lw=1.2)
    axes[0].set_title(f'Offset +50 km | {n} fixed targets | Wait/drift caps 200/300 h')
    axes[0].set_ylim(0,tol.mean_response_s.max()/3600*1.28)
    axes[0].set_ylabel('Conditional mean response [h]'); axes[0].legend(frameon=False,ncols=3,loc='upper right')
    e2tol=records[(records.study=='tolerance')&(records.case_id=='E2')&(records.offset_km==50)]
    narrow=e2tol.tolerance_deg.min()
    common=set(e2tol.loc[(e2tol.tolerance_deg==narrow)&e2tol.success,'target_id'])
    for field,label,color,style in [('response_s','Total','black','-'),('wait_s','Waiting','#0072B2','--'),('drift_s','Drift','#D55E00','-.')]:
        g=e2tol[e2tol.target_id.isin(common)&e2tol.success].groupby('tolerance_deg')[field].mean()/3600
        axes[1].semilogx(g.index,g.values,marker='o',ms=3,lw=1.2,label=label,color=color,linestyle=style)
    axes[1].set_ylabel('E2 common-target mean [h]')
    axes[1].legend(frameon=False,ncols=3,loc='upper right')
    axes[1].text(.02,.05,f'Fixed narrow-band subset: n = {len(common)}',transform=axes[1].transAxes,fontsize=8)
    axes[2].set_ylabel('Fraction within 72 h'); axes[2].set_ylim(-.03,1.08); axes[2].set_yticks([0,.5,1])
    axes[2].set_xlabel('Requested longitude tolerance [deg]')
    for ax in axes: ax.grid(alpha=.22); ax.set_ylim(bottom=0)
    fig.tight_layout(h_pad=1.0); save(fig,'paper_figure_5_tolerance')

    fig=plt.figure(figsize=(7.8,5.8))
    grid=fig.add_gridspec(2,4,width_ratios=[1,1,1,.045],left=.1,right=.9,bottom=.11,top=.88,hspace=.25,wspace=.13)
    selections=paired[(paired.case_id=='E2')&paired.offset_km.isin([50,200])&paired.tolerance_deg.isin([.25,1,5])]
    vmax=selections.paired_time_saved_s.max()/3600
    for i,offset in enumerate([50,200]):
        for j,band in enumerate([.25,1,5]):
            ax=fig.add_subplot(grid[i,j]); g=selections[(selections.offset_km==offset)&(selections.tolerance_deg==band)]
            values=matrix(g,'paired_time_saved_s')/3600
            im=ax.imshow(np.ma.masked_invalid(values),origin='lower',aspect='auto',interpolation='nearest',cmap=cmap,vmin=0,vmax=vmax)
            failed=np.argwhere(~np.isfinite(values))
            if len(failed): ax.scatter(failed[:,1],failed[:,0],marker='x',color='.3',s=24,lw=1)
            coordinate_axes(ax,j==0,i==1)
            if i==0: ax.set_title(f'Longitude band {band:g}°')
            if j==0: ax.set_ylabel(f'Offset {offset:+g} km\nGeodetic latitude [deg]')
    fig.colorbar(im,cax=fig.add_subplot(grid[:,3])).set_label('Response reduction from 0.001° [h]')
    fig.suptitle('E2 tolerance benefit by target location | Paired accepted responses',fontsize=11,y=.97)
    save(fig,'paper_figure_7_tolerance_location')
    metadata=dict(data_sha256={name:hashlib.sha256((args.data_dir/name).read_bytes()).hexdigest()
        for name in ['per_target.csv','summary.csv','targets.csv','tolerance_paired.csv']},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs=[p.name for p in sorted(output.iterdir()) if p.suffix in ('.png','.pdf')])
    (output/'figure_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata['outputs']))


if __name__=='__main__':
    main()
