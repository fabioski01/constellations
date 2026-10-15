"""Batched nominal latitude-crossing solves for reproducible dense-grid studies.

The corrected calculation eliminates drift in favour of elapsed response time.
For the declared fixed-inclination Earth cases the continuous longitude is
monotone on each capped phase-wrap interval. This permits direct band-entry
inversion rather than repeating longitude discovery for each target.
This module is an analysis accelerator; the scalar planner remains the reference.
"""
from __future__ import annotations

from datetime import datetime
from math import pi, sin, cos, atan2, radians
import numpy as np

import reconfig_driftANDwaitTime_multicore as reference
from reconfig_verification import days_from_j2000, gmst_rad

TAU = 2.0 * pi


def load_original():
    """Load the byte-preserved source without its import-time CLI side effect."""
    import importlib.util
    from pathlib import Path
    import sys
    name = 'paper1_original_scalar'
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parent/'reference/paper1_original/reconfig_driftANDwaitTime_multicore.py'
    spec = importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    previous=sys.argv
    sys.argv=[str(path)]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv=previous
    return module


_LEGACY_CLOCK={}


def legacy_clock(epoch, maximum_seconds):
    """Exactly reproduce the original integer-second Julian-date clock.

    Calendar fields, operation order, the large Julian-date subtraction and
    radian wrapping are retained. Fractional microseconds round as timedelta.
    The table is shared by every target, phase wrap, altitude and tolerance.
    """
    from datetime import timedelta
    key=(epoch.isoformat(),int(maximum_seconds))
    if key not in _LEGACY_CLOCK:
        seconds=np.arange(-2,int(maximum_seconds)+3,dtype=np.int64)
        midnight=epoch.replace(hour=0,minute=0,second=0,microsecond=0)
        offset=epoch.hour*3600+epoch.minute*60+epoch.second
        absolute=seconds+offset
        days=np.floor_divide(absolute,86400)
        sod=absolute-days*86400
        hour=sod//3600
        minute=(sod%3600)//60
        second=sod%60
        jd=np.empty(len(seconds))
        for day in np.unique(days):
            date=midnight+timedelta(days=int(day))
            y,m=date.year,date.month
            if m<=2:
                y-=1; m+=12
            aa=y//100; bb=2-aa+aa//4
            take=days==day
            decimal_day=date.day+(hour[take]+(minute[take]+second[take]/60.)/60.)/24.
            jd[take]=np.floor(365.25*(y+4716))+np.floor(30.6001*(m+1))+decimal_day+bb-1524.5
        dd=jd-2451545.
        centuries=dd/36525.
        degrees=280.46061837+360.98564736629*dd+.000387933*centuries*centuries-(centuries**3)/38710000.
        angles=np.mod(degrees*(pi/180.),TAU)
        _LEGACY_CLOCK[key]=angles
    table=_LEGACY_CLOCK[key]
    def evaluate(elapsed):
        elapsed=np.asarray(elapsed)
        whole=np.floor(elapsed)
        # timedelta rounds the fractional part, not the large product t*1e6.
        # The latter loses bits at half-microsecond clock discontinuities and
        # can change which side a legacy band-edge bisection returns.
        micros=np.rint((elapsed-whole)*1e6).astype(np.int64)+epoch.microsecond
        ticks=whole.astype(np.int64)+np.floor_divide(micros,1000000)
        if np.any(ticks < -2) or np.any(ticks > len(table)-3):
            raise ValueError('Elapsed time exceeds the declared legacy clock table.')
        return table[ticks+2]
    return evaluate


def earth_coefficients(epoch):
    d0 = days_from_j2000(epoch)
    t0 = d0 / 36525.0
    seconds_per_century = 86400.0 * 36525.0
    c1 = (360.98564736629 / 86400.0
          + (2.0 * .000387933 * t0 - 3.0 * t0*t0 / 38710000.0) / seconds_per_century) * pi/180
    c2 = (.000387933 - 3.0 * t0 / 38710000.0) / seconds_per_century**2 * pi/180
    c3 = -1.0 / (38710000.0 * seconds_per_century**3) * pi/180
    return gmst_rad(epoch), c1, c2, c3


def wrap(value):
    return (value + pi) % TAU - pi


def parameters(case, offset_km, epoch, wait_hours=200., drift_hours=300.):
    p = reference.AnalyticRaanPhasingPlanner(
        reference.Orbit(case['a_g_km'], radians(case['inclination_deg']),
                        radians(case.get('node_deg', 0.)), radians(case.get('phase_deg', 0.))),
        case['a_r_km'], float(offset_km), epoch, 0., 0.,
        drift_time_cap_days=drift_hours/24., wait_time_cap_hours=wait_hours,
        n_jobs=1)
    return p


def corrected_grid(case, offset_km, epoch, latitudes, longitudes,
                   tolerances=(.25,), wait_hours=200., drift_hours=300.):
    """Solve a full coordinate grid, retaining all requested coordinates.

    Output arrays have shape (tolerance, latitude, longitude). Invalid and
    unreachable latitudes retain NaN times and an explicit geometric status.
    There is no extra response deadline; the separate coast caps bound time.
    """
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    tolerances = np.asarray(tolerances, dtype=float)
    if np.any(~np.isfinite(tolerances)) or np.any((tolerances < 0) | (tolerances >= 180)):
        raise ValueError('Tolerances must be finite and in [0, 180) degrees.')
    p = parameters(case, offset_km, epoch, wait_hours, drift_hours)
    shape = (len(tolerances), len(latitudes), len(longitudes))
    best = np.full(shape, np.inf)
    output = {key: np.full(shape, np.nan) for key in
              ['wait_s', 'drift_s', 'longitude_error_deg', 'latitude_error_deg']}
    output.update(branch=np.full(shape, -1, dtype=np.int8),
                  phase_wrap=np.full(shape, -9999, dtype=np.int16),
                  longitude_wrap=np.full(shape, -9999, dtype=np.int16))
    valid_lat = np.isfinite(latitudes) & (np.abs(latitudes) <= 90.)
    phi = np.arctan((1-reference.WGS84_e2) * np.tan(np.deg2rad(latitudes)))
    ratio = np.sin(phi) / sin(p.i)
    reachable = valid_lat & (np.abs(ratio) <= 1 + 1e-12)
    status = np.where(~valid_lat, 2, np.where(~reachable, 3, 1)).astype(np.int8)
    if not np.any(reachable):
        output.update(success=np.zeros(shape,dtype=bool),response_s=np.full(shape,np.nan),
                      latitude_status=status,transfer1_s=p.T1,transfer2_s=p.T2,
                      derivative_bounds_rad_s=np.empty((0,2)))
        return output
    u_first = np.arcsin(np.clip(ratio, -1., 1.)) % TAU
    branches = [u_first, (pi-u_first) % TAU]
    longitude = np.deg2rad(longitudes)[None, :]
    theta0, c1, c2, c3 = earth_coefficients(epoch)
    gamma_g, gamma_d = p.du_G, p.du_D
    if min(gamma_g, gamma_d) <= 0:
        raise ValueError('The batch inversion requires positive phase rates.')
    r = gamma_d / gamma_g
    beta = 1.0-r
    node_slope = p.dOm_D-r*p.dOm_G
    transfer_s = p.T1+p.T2
    transfer_phase = p.du_T1*p.T1+p.du_T2*p.T2
    transfer_node = p.dOm_T1*p.T1+p.dOm_T2*p.T2
    wait_cap, drift_cap = wait_hours*3600., drift_hours*3600.
    max_phase = gamma_g*wait_cap + gamma_d*drift_cap
    inward = np.minimum(1e-10, np.deg2rad(tolerances)*1e-4)
    bands = np.maximum(0., np.deg2rad(tolerances)-inward)
    derivative_bounds=[]

    def theta(t):
        return theta0+c1*t+c2*t*t+c3*t*t*t

    for branch, u in enumerate(branches):
        const = -transfer_phase + u-p.u0
        psi = np.arctan2(cos(p.i)*np.sin(u), np.cos(u))
        m_start = int(np.floor((-const[reachable]/TAU).min()))-2
        m_end = int(np.floor(((max_phase-const[reachable])/TAU).max()))+2
        for m in range(m_start, m_end+1):
            phase = const+TAU*m
            lo = np.maximum(0., np.minimum(phase/gamma_d, (phase-gamma_g*wait_cap)/gamma_d))
            hi = np.minimum(drift_cap, np.maximum(phase/gamma_d, (phase-gamma_g*wait_cap)/gamma_d))
            geometrically_valid = reachable & (hi >= lo-1e-8)
            hi = np.maximum(lo, hi)
            if not np.any(geometrically_valid):
                continue
            lower_bound = phase/max(gamma_g,gamma_d)+transfer_s
            active = (best > lower_bound[None,:,None]-1e-6) & geometrically_valid[None,:,None]
            if not np.any(active):
                continue
            c = phase/gamma_g+transfer_s
            early_td = hi if beta < 0 else lo
            late_td = lo if beta < 0 else hi
            early_tw = (phase-gamma_d*early_td)/gamma_g
            late_tw = (phase-gamma_d*late_td)/gamma_g
            early_t = early_tw+p.T1+early_td+p.T2
            late_t = late_tw+p.T1+late_td+p.T2
            early_phi = p.Omega0+p.dOm_G*early_tw+transfer_node+p.dOm_D*early_td+psi-theta(early_t)
            early_error = wrap(early_phi[:,None]-longitude)
            for ti, (tol, band) in enumerate(zip(tolerances, bands)):
                if not np.any(active[ti]):
                    continue
                t = np.broadcast_to(early_t[:,None], shape[1:]).copy()
                td = np.broadcast_to(early_td[:,None], shape[1:]).copy()
                endpoint = np.abs(early_error) <= np.deg2rad(tol)
                if abs(beta) > 1e-15:
                    effective_node = node_slope/beta
                    derivative = effective_node-(c1+2*c2*early_t+3*c3*early_t**2)
                    derivative_late = effective_node-(c1+2*c2*late_t+3*c3*late_t**2)
                    if np.any(np.maximum(derivative,derivative_late)[geometrically_valid] >= 0):
                        raise ValueError('Longitude is not decreasing with response time in this case.')
                    derivative_bounds.append((float(np.min(derivative[geometrically_valid])),
                                              float(np.max(derivative[geometrically_valid]))))
                    k = np.floor((early_phi[:,None]-longitude+band)/TAU)
                    boundary = longitude+TAU*k+band
                    difference = boundary-early_phi[:,None]
                    t += np.where(endpoint, 0., difference/derivative[:,None])
                    for _ in range(3):
                        phi_t = early_phi[:,None] + effective_node*(t-early_t[:,None]) - (theta(t)-theta(early_t[:,None]))
                        slope = effective_node-(c1+2*c2*t+3*c3*t*t)
                        t -= np.where(endpoint, 0., (phi_t-boundary)/slope)
                    td = early_td[:,None]+(t-early_t[:,None])/beta
                else:
                    active[ti] &= endpoint
                tw = (phase[:,None]-gamma_d*td)/gamma_g
                total = tw+p.T1+td+p.T2
                uf = (p.u0+gamma_g*tw+p.du_T1*p.T1+gamma_d*td+p.du_T2*p.T2) % TAU
                node = (p.Omega0+p.dOm_G*tw+p.dOm_T1*p.T1+p.dOm_D*td+p.dOm_T2*p.T2) % TAU
                final_psi = np.arctan2(cos(p.i)*np.sin(uf),np.cos(uf))
                lon_error = np.rad2deg(wrap(node+final_psi-theta(total)-longitude))
                phi_f = np.arcsin(np.clip(sin(p.i)*np.sin(uf),-1.,1.))
                latitude_f = np.rad2deg(np.arctan(np.tan(phi_f)/(1-reference.WGS84_e2)))
                lat_error = latitude_f-latitudes[:,None]
                okay = active[ti] & (td >= -1e-8) & (td <= drift_cap+1e-8) & (tw >= -1e-8) & (tw <= wait_cap+1e-8)
                okay &= (np.abs(lon_error) <= tol) & (np.abs(lat_error) <= 1e-7)
                improve = okay & (total < best[ti])
                if np.any(improve):
                    best[ti][improve]=total[improve]
                    for key, values in [('wait_s',tw),('drift_s',td),('longitude_error_deg',lon_error),('latitude_error_deg',lat_error)]:
                        output[key][ti][improve]=values[improve]
                    output['branch'][ti][improve]=branch
                    output['phase_wrap'][ti][improve]=m
                    unwrapped=p.Omega0+p.dOm_G*tw+transfer_node+p.dOm_D*td+psi[:,None]-theta(total)
                    output['longitude_wrap'][ti][improve]=np.rint((unwrapped-longitude)/TAU)[improve].astype(np.int16)
    output['success']=np.isfinite(best)
    output['response_s']=np.where(output['success'],best,np.nan)
    output['latitude_status']=status
    output['transfer1_s']=p.T1
    output['transfer2_s']=p.T2
    output['derivative_bounds_rad_s']=np.array(derivative_bounds) if derivative_bounds else np.empty((0,2))
    return output


def original_latitude(case, offset_km, epoch, latitude, longitudes,
                      tolerances=(.25,), drift_hours=300., prune=True):
    """Batch the ORIGINAL worker across longitude, preserving its defects.

    Retains its 160/160 scans, 200 h wait cap, wrapped integer-second clock,
    bisection stopping rule, omitted-leg timing, near-miss fallback and final
    branch choice. It shares target-independent scan geometry. Far-neighbour
    near misses duplicate the scan minimum/maximum; their first representative
    is retained. Source agreement is checked separately against scalar runs.
    """
    old=load_original()
    p=old.AnalyticRaanPhasingPlanner(
        old.Orbit(case['a_g_km'],radians(case['inclination_deg']),
                  radians(case.get('node_deg',0.)),radians(case.get('phase_deg',0.))),
        case['a_r_km'],float(offset_km),epoch,float(latitude),0.,
        drift_time_cap_days=drift_hours/24.,n_jobs=1)
    longitudes=np.asarray(longitudes,dtype=float)
    lon=longitudes*(pi/180.)
    tolerances=np.asarray(tolerances,dtype=float)
    nl,nt=len(lon),len(tolerances)
    theta=legacy_clock(epoch,int((200+drift_hours)*3600+20000))
    gamma_g,gamma_d=p.du_G,p.du_D
    wait_cap,drift_cap=200.*3600.,drift_hours*3600.
    transfer_s=p.T1+p.T2
    transfer_phase=p.du_T1*p.T1+p.du_T2*p.T2
    fields=['response_s','wait_s','drift_s','longitude_error_deg','latitude_error_deg','phase_wrap','longitude_wrap','hit_type']
    selected=[]
    for branch,u in enumerate(old.u_required_branches(float(latitude),p.i)):
        strict={key:np.full((nt,nl),np.nan) for key in fields}
        fallback={key:np.full((nt,nl),np.nan) for key in fields}
        strict_time=np.full((nt,nl),np.inf)
        fallback_error=np.full((nt,nl),np.inf)
        fallback_time=np.full((nt,nl),np.inf)
        const=-transfer_phase+u-p.u0
        psi=atan2(cos(p.i)*sin(u),cos(u))
        uf=old.wrap_2pi(u)
        psi_final=atan2(cos(p.i)*sin(uf),cos(uf))
        phi_gc=np.arcsin(np.clip(sin(p.i)*sin(uf),-1.,1.))
        lat_final=np.arctan(np.tan(phi_gc)/(1-reference.WGS84_e2))/(pi/180.)
        amin=0.
        amax=gamma_g*wait_cap+gamma_d*drift_cap
        m_start=int((amin-const)//TAU)-2
        m_end=int((amax-const)//TAU)+2
        for m in range(m_start,m_end+1):
            phase=const+TAU*m
            t0=phase/gamma_d
            tc=(phase-gamma_g*wait_cap)/gamma_d
            lo=max(0.,min(t0,tc));hi=min(drift_cap,max(t0,tc))
            if hi <= lo+1e-6:
                continue
            lower_bound=phase/max(gamma_g,gamma_d)+transfer_s
            active=(strict_time >= lower_bound-1e-6) if prune else np.ones_like(strict_time,dtype=bool)
            if not np.any(active):
                break

            def geometry(td):
                tw=(phase-gamma_d*td)/gamma_g
                total=tw+p.T1+td+p.T2
                om=p.Omega0+p.dOm_G*tw+p.dOm_T1*p.T1+p.dOm_D*td+p.dOm_T2*p.T2
                ph=om+psi-theta(total)
                return tw,total,om,ph

            grid=lo+(hi-lo)*(np.arange(161)/160.)
            tw_grid,_,_,phi_grid=geometry(grid)
            discover=(tw_grid >= -1e-9)&(tw_grid-wait_cap <= 1e-9)
            bracket=(tw_grid >= -1e-12)&(tw_grid-wait_cap <= 1e-12)
            if not np.any(discover) or not np.any(bracket):
                continue
            kmin=np.rint((phi_grid[discover].min()-lon)/TAU).astype(int)
            kmax=np.rint((phi_grid[discover].max()-lon)/TAU).astype(int)
            if np.max(kmax-kmin)>2:
                raise ValueError('Legacy discovery-set assumption needs the general path.')
            kbase=int(kmin.min())-8
            nk=int(kmax.max())+8-kbase+1
            had=np.zeros((nt,nl,nk),dtype=bool)

            def package(indices,td,k,kind,tol_indices):
                if len(indices)==0:
                    return
                tw,total,om,_=geometry(td)
                keep=(tw>=0.)&(tw<=wait_cap)
                indices=indices[keep];td=td[keep];k=k[keep]
                tw=tw[keep];total=total[keep];om=om[keep]
                if len(indices)==0:
                    return
                err=wrap(wrap(np.mod(om,TAU)+psi_final-theta(total))-lon[indices])/(pi/180.)
                values=dict(response_s=total,wait_s=tw,drift_s=td,longitude_error_deg=err,
                            latitude_error_deg=np.full(len(indices),lat_final-latitude),
                            phase_wrap=np.full(len(indices),m),longitude_wrap=k,
                            hit_type=np.full(len(indices),kind))
                for ti in tol_indices:
                    if kind!=3:
                        had[ti,indices,k-kbase]=True
                    accepted=(np.abs(err)<=tolerances[ti])&active[ti,indices]
                    # Stable target/time sorting replicates minimum-time selection.
                    candidates=np.flatnonzero(accepted)
                    order=candidates[np.lexsort((total[candidates],indices[candidates]))]
                    first=np.r_[True,indices[order][1:]!=indices[order][:-1]] if len(order) else np.array([],dtype=bool)
                    take=order[first]
                    take=take[total[take]<strict_time[ti,indices[take]]]
                    strict_time[ti,indices[take]]=total[take]
                    for key in fields:strict[key][ti,indices[take]]=values[key][take]
                    missing=np.isinf(strict_time[ti,indices])&active[ti,indices]
                    candidates=np.flatnonzero(missing)
                    order=candidates[np.lexsort((total[candidates],np.abs(err[candidates]),indices[candidates]))]
                    first=np.r_[True,indices[order][1:]!=indices[order][:-1]] if len(order) else np.array([],dtype=bool)
                    take=order[first]
                    index=indices[take];error=np.abs(err[take])
                    improve=(error<fallback_error[ti,index])|((error==fallback_error[ti,index])&(total[take]<fallback_time[ti,index]))
                    take=take[improve];index=indices[take]
                    fallback_error[ti,index]=np.abs(err[take]);fallback_time[ti,index]=total[take]
                    for key in fields:fallback[key][ti,index]=values[key][take]

            def roots(level,which_tolerances,kind):
                lower=np.minimum(phi_grid[:-1],phi_grid[1:])
                upper=np.maximum(phi_grid[:-1],phi_grid[1:])
                k_lo=np.ceil((lower[None,:]-lon[:,None]-level-1e-12)/TAU).astype(int)
                k_hi=np.floor((upper[None,:]-lon[:,None]-level+1e-12)/TAU).astype(int)
                allowed=bracket[:-1]&bracket[1:]
                active_lon=np.any(active[list(which_tolerances)],axis=0)
                for extra in range(2):
                    kk=k_lo+extra
                    mask=(kk<=k_hi)&allowed[None,:]&active_lon[:,None]
                    mask&=(kk>=kmin[:,None]-8)&(kk<=kmax[:,None]+8)
                    ix,jx=np.nonzero(mask)
                    if len(ix)==0:
                        continue
                    k=kk[ix,jx]
                    target=lon[ix]+TAU*k
                    fa=(phi_grid[jx]-target)-level
                    fb=(phi_grid[jx+1]-target)-level
                    crossing=fa*fb<=0.
                    ix=ix[crossing];jx=jx[crossing];k=k[crossing]
                    target=target[crossing];fa=fa[crossing]
                    if len(ix)==0:continue
                    a=grid[jx].copy();b=grid[jx+1].copy()
                    done=np.zeros(len(ix),dtype=bool)
                    result=np.zeros(len(ix))
                    for _ in range(80):
                        ix_live=np.flatnonzero(~done)
                        if len(ix_live)==0:break
                        mid=.5*(a[ix_live]+b[ix_live])
                        tw,_,_,ph=geometry(mid)
                        fm=(ph-target[ix_live])-level
                        invalid=(tw < -1e-12)|(tw-wait_cap > 1e-12)
                        fm=np.where(invalid,1e9,fm)
                        finish=(~invalid)&((np.abs(fm)<1e-12)|((b[ix_live]-a[ix_live])<1e-6))
                        ii=ix_live[finish];result[ii]=mid[finish];done[ii]=True
                        left=(fa[ix_live]*fm<=0.)&(~invalid)&(~finish)
                        jj=ix_live[left];b[jj]=mid[left]
                        right=(~left)&(~finish)
                        jj=ix_live[right];a[jj]=mid[right];fa[jj]=fm[right]
                    result[~done]=.5*(a[~done]+b[~done])
                    package(ix,result,k,kind,which_tolerances)

            roots(0.,range(nt),0)
            for ti,tolerance in enumerate(tolerances):
                if not np.any(active[ti]):continue
                roots(tolerance*(pi/180.),[ti],1)
                roots(-tolerance*(pi/180.),[ti],2)

            # Only central nearest-wrap samples and the first far neighbours
            # can produce distinct nearest coarse samples. Extra far wraps
            # duplicate the same sample at the min/max of phi_grid.
            for slot in range(int(np.max(kmax-kmin))+3):
                if slot==0:
                    ks=kmin-8
                elif slot==int(np.max(kmax-kmin))+2:
                    ks=kmax+8
                else:
                    ks=kmin+slot-1
                residual=np.abs(phi_grid[None,:]-(lon+TAU*ks)[:,None])
                residual[:,~bracket]=np.inf
                nearest=np.argmin(residual,axis=1)
                for ti in range(nt):
                    eligible=active[ti]&(~had[ti,np.arange(nl),ks-kbase])
                    if slot not in [0,int(np.max(kmax-kmin))+2]:eligible&=(ks<=kmax)
                    ix=np.flatnonzero(eligible)
                    package(ix,grid[nearest[ix]],ks[ix],3,[ti])
        choose_strict=np.isfinite(strict_time)
        answer={key:np.where(choose_strict,strict[key],fallback[key]) for key in fields}
        answer['branch']=np.full((nt,nl),branch,dtype=np.int8)
        answer['source_within_longitude_tolerance']=choose_strict
        selected.append(answer)
    first,second=selected
    # Original code chooses an earlier fallback branch even if the other
    # branch satisfies the tolerance. Preserve that behaviour for the audit.
    take_second=np.isfinite(second['response_s']) & ((~np.isfinite(first['response_s'])) |
                                                  (second['response_s']<first['response_s']))
    result={key:np.where(take_second,second[key],first[key]) for key in first}
    result['success']=np.isfinite(result['response_s'])
    result['strict_acceptance']=result['success'] & (np.abs(result['longitude_error_deg'])<=tolerances[:,None])
    result['strict_acceptance'] &= (np.abs(result['latitude_error_deg'])<=1e-7)&(abs(latitude)<=90.)
    result['transfer1_s']=p.T1;result['transfer2_s']=p.T2
    return result
