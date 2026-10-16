"""Publication-layout version of the unchanged Los Angeles validation example."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from math import cos,sin
from pathlib import Path

import numpy as np

from paper1_validate_maneuver import _land,_track,solve_scenario
from reconfig_cartesian_validation import (
    Gravity,IntegrationSettings,Phase,circular_secular_rates,execute_maneuver,
    ground_point,maneuver_schedule,wrap,
)

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=ROOT/'results/paper1/validation-los-angeles/summary.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/paper1/validation-36x18')
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.titlesize':9,
                         'svg.hashsalt':'paper1-los-angeles-publication'})
    summary=json.loads(args.summary.read_text());config=summary['config']
    gravity,epoch,planner,solution,schedule,repeat_s,end_s,initial,initialization=solve_scenario(config)
    np.testing.assert_array_equal(initial,np.asarray(summary['initial_cartesian_state']))
    trajectory=execute_maneuver(initial,schedule,end_s,gravity,
                                config['target_latitude_deg'],IntegrationSettings(**config['integration']))
    figure,(world,zoom)=plt.subplots(2,1,figsize=(6.8,6.4),gridspec_kw={'height_ratios':[1.15,1]})
    _land(world)
    colors=['#0072B2','#56B4E9','#009E73','#E69F00','#8e5ea2']
    phases=[*schedule.phases,Phase(schedule.predicted_access_s,schedule.predicted_access_s+6500,'Achieved regional orbit')]
    for index,phase in enumerate(phases):
        windows=[(phase.start_s,phase.end_s)]
        if phase.end_s-phase.start_s>14400:
            windows=[(phase.start_s,phase.start_s+3600),(phase.end_s-3600,phase.end_s)]
            world.text(.5,.98,f'{(windows[1][0]-windows[0][1])/3600:.2f} h of waiting omitted between faded ends',
                       transform=world.transAxes,ha='center',va='top',fontsize=7,
                       bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
        for j,(lo,hi) in enumerate(windows):
            times=np.linspace(lo,hi,max(2,int((hi-lo)/config['plot_sample_step_s'])+1))
            points=[ground_point(trajectory.state(t),epoch,t) for t in times]
            _track(world,points,color=colors[index],linewidth=1.2,
                   fade_start=len(windows)>1 and j>0,fade_end=len(windows)>1 and j==0,
                   label=f'{phase.label} ({(phase.end_s-phase.start_s)/3600:.2f} h)' if j==0 else None)
            if len(windows)>1:
                t,(lat,lon)=(hi,points[-1]) if j==0 else (lo,points[0])
                world.annotate(f'{t/3600:.2f} h',(lon,lat),xytext=(-5,10) if j==0 else (5,10),
                               ha='right' if j==0 else 'left',textcoords='offset points',fontsize=7,
                               bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
    for index,burn in enumerate(trajectory.burn_records,1):
        lat,lon=ground_point(np.array(burn['after_state']),epoch,burn['time_s'])
        world.scatter(lon,lat,s=20,color='black',marker='^',zorder=5)
        world.annotate(str(index),(lon,lat),xytext=(4,4),textcoords='offset points',fontsize=7)
    lat,lon=ground_point(trajectory.state(0),epoch,0)
    world.scatter(lon,lat,s=26,facecolors='white',edgecolors=colors[0],zorder=6)
    world.annotate('Start, t = 0 h',(lon,lat),xytext=(-7,-12),ha='right',textcoords='offset points',
                   fontsize=7,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
    world.scatter(config['target_longitude_deg'],config['target_latitude_deg'],marker='*',s=65,
                   color='#D55E00',edgecolors='black',linewidths=.4,zorder=6)
    world.set_title('(a) Connected maneuver and numbered impulses')
    world.legend(loc='lower left',ncol=2,fontsize=7,framealpha=.95,columnspacing=1)
    world.grid(alpha=.2)

    times=schedule.predicted_access_s+np.linspace(-2,2,1201)
    actual=np.array([ground_point(trajectory.state(t),epoch,t) for t in times])
    actual_lon=np.rad2deg(wrap(np.deg2rad(actual[:,1]-config['target_longitude_deg'])))
    actual_lat=actual[:,0]-config['target_latitude_deg']
    node_rate,phase_rate=circular_secular_rates(planner.rom_a,planner.i,gravity)
    nominal=[]
    for t in times:
        dt=t-schedule.predicted_access_s
        u=solution['u_f_rad']+phase_rate*dt;node=solution['Omega_f_rad']+node_rate*dt
        direction=np.array([cos(node)*cos(u)-sin(node)*sin(u)*cos(planner.i),
                            sin(node)*cos(u)+cos(node)*sin(u)*cos(planner.i),sin(u)*sin(planner.i)])
        nominal.append(ground_point(np.r_[planner.rom_a*direction,np.zeros(3)],epoch,t))
    nominal=np.array(nominal)
    zoom.plot(actual_lon,actual_lat,color=colors[0],label='Connected propagation')
    zoom.plot(np.rad2deg(wrap(np.deg2rad(nominal[:,1]-config['target_longitude_deg']))),
               nominal[:,0]-config['target_latitude_deg'],'--',color='#777777',label='Nominal regional reference')
    tolerance=config['longitude_tolerance_deg']
    zoom.plot([-tolerance,tolerance],[0,0],color='#009E73',linewidth=4,alpha=.65,label='Accepted longitude band at latitude crossing')
    zoom.scatter(0,0,marker='*',s=65,color='#D55E00',edgecolors='black',linewidths=.4,zorder=6)
    nearest=summary['evidence']['access']['nearest_crossing_to_prediction']
    zoom.scatter(nearest['longitude_residual_deg'],nearest['latitude_residual_deg'],
                   color='#D55E00',s=30,zorder=7,label='Actual latitude crossing (before insertion)')
    zoom.scatter(solution['lon_err_deg'],0,marker='D',s=27,facecolors='white',edgecolors='#444444',label='Nominal access',zorder=6)
    at=summary['evidence']['access']['at_predicted_time']
    zoom.scatter(at['longitude_residual_deg'],at['latitude_residual_deg'],marker='x',s=35,
                   color=colors[0],label='Propagated at predicted time',zorder=7)
    zoom.set(xlim=(-.29,.32),ylim=(-.03,.03),xlabel='Longitude offset from target [deg]',
              ylabel='Latitude offset from target [deg]',title='(b) Target neighborhood and acceptance at the latitude crossing')
    zoom.legend(loc='upper left',fontsize=7,framealpha=.95)
    zoom.grid(alpha=.2)
    figure.text(.12,.025,f'Nominal response: {schedule.predicted_access_s/3600:.3f} h. '
                  f'Actual crossing: {nearest["time_from_prediction_s"]:+.3f} s, '
                  f'{nearest["longitude_residual_deg"]:+.4f} deg longitude offset.\n'
                  'Numerical mean initialization; fixed RTN transverse impulses; no targeting correction.',fontsize=7)
    figure.subplots_adjust(left=.12,right=.97,bottom=.12,top=.95,hspace=.43)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for extension in ['png','svg']:
        figure.savefig(args.output_dir/f'los_angeles_validation.{extension}',dpi=220,
                       metadata={'Date':None} if extension=='svg' else None)
    plt.close(figure)


if __name__=='__main__':main()
