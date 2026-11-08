"""Additional nominal checks and the Earth-rotation sensitivity figure."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd
from paper1_check_v20 import CASES, OUT, nominal

def tolerance_case(task):
    case,offset=task
    tolerances=[.001,.005,.01,.25,1.,5.]
    latitudes=np.linspace(-49.,49.,180);longitudes=np.arange(-180.,180.)
    r=nominal(case,offset,latitudes,longitudes,tolerances)
    path=OUT/f"tolerance_{case['case_id']}_{offset:+g}.npz"
    np.savez_compressed(path,**r)
    return [dict(case_id=case['case_id'],offset_km=offset,tolerance_deg=tol,
        accepted=int(r['success'][j].sum()),mean_h=float(np.nanmean(r['response_s'][j])/3600),
        median_h=float(np.nanmedian(r['response_s'][j])/3600),
        p90_h=float(np.nanquantile(r['response_s'][j],.9)/3600),
        within72_pct=float(100*np.mean(r['success'][j]&(r['response_s'][j]<=72*3600))))
        for j,tol in enumerate(tolerances)]

def more_checks():
    tasks=[(case,50.) for case in CASES]+[(CASES[1],-200.),(CASES[1],200.)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=[row for group in pool.map(tolerance_case,tasks) for row in group]
    pd.DataFrame(rows).to_csv(OUT/'seven_tolerance_sweeps.csv',index=False)
    r=dict(np.load(OUT/'tolerance_E2_+50.npz'))
    paired=[]
    for j,tol in [(3,.25),(5,5.)]:
        change=(r['response_s'][0]-r['response_s'][j])/3600
        switch=(r['branch'][0]!=r['branch'][j])|(r['phase_wrap'][0]!=r['phase_wrap'][j])|(r['longitude_wrap'][0]!=r['longitude_wrap'][j])
        paired.append(dict(tolerance=tol,mean_reduction_h=float(change.mean()),
            median_reduction_s=float(np.median(change)*3600),switch_count=int(switch.sum())))
    (OUT/'paired_tolerance_checks.json').write_text(json.dumps(paired,indent=2)+'\n')
    print(json.dumps(paired,indent=2),flush=True)

def figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d=json.loads((OUT/'observation_summary.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(10.7,4.4),layout='constrained')
    colors={'initial':'#66717c','direct':'#176c86','excursion':'#a34a1c'}
    markers={'initial':'s','direct':'o','excursion':'^'}
    for kind,style in [('uniform','-'),('gmst','--')]:
        for label in ['initial','direct','excursion']:
            if kind=='gmst' and label=='initial':continue
            rows=d[kind]['rows'];x=[r['window_h'] for r in rows]
            for ax,key in zip(axes,['_mean','_at_least_two_pct']):
                ax.plot(x,[r[label+key] for r in rows],style,color=colors[label],
                    marker=markers[label],markersize=5,linewidth=1.7,
                    markerfacecolor=colors[label] if kind=='uniform' else 'white')
    for ax in axes:
        ax.grid(axis='y',alpha=.2);ax.set_xticks([24,48,72,168]);ax.set_xlabel('Window after request [h]')
        ax.set_ylim(bottom=0)
    axes[0].set(title='(a) Nominal visit count',ylabel='Mean accepted visits per target')
    axes[1].set(title='(b) Repeated nominal access',ylabel='Requests with at least two visits [%]',ylim=(0,103))
    handles=[Line2D([],[],color=colors[k],marker=markers[k],lw=1.7,label=label)
        for k,label in [('initial','Initial orbit'),('direct','Direct reconfiguration'),('excursion','+100 km excursion')]]
    handles += [Line2D([],[],color='.25',lw=1.7,label='Uniform rotation'),
                Line2D([],[],color='.25',lw=1.7,ls='--',label='Calendar GMST')]
    axes[0].legend(handles=handles,loc='upper left',frameon=False,fontsize=8.5)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/('opportunities_rotation.'+ext),dpi=250)
    plt.close(fig)

if __name__=='__main__':
    more_checks()
    figure()
