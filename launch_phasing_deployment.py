"""RAAN Phasing Deployment Planner

Compute time + Δv to phase satellites in RAAN using J2 drift.
Reuses core utilities from reconfig_driftANDwaitTime_multicore.

Use case: Deploy N satellites together; phase them to target RAANs by
temporarily changing altitude (J2 RAAN drift), then return to nominal.

Key Logic:
- Max Δa is derived from max Δv budget (Hohmann transfer cost)
- Each satellite uses optimal Δa to minimize time while staying under Δv budget
- Smaller RAAN offsets may use smaller Δa (less propellant)

Inputs:
    initial_orbit (Orbit): nominal deployment orbit (circular)
    target_raan_offsets (list[float]): desired RAAN separations (deg), e.g. [90, 180, 270]
    max_time_days (float): deployment time budget (e.g., 180 days)
    max_dv_budget_ms (float): max Δv per satellite (m/s) - constrains max Δa
    detailed_diagnostics (bool): compute alternative scenarios for infeasible cases

Outputs:
    deployment_plan (dict): per-satellite {delta_a_km, time_s, dv_ms, feasible} + total stats

Author: Fabio Meloni - TU Delft MSc Student - NASA JPL Visiting Researcher
"""
import os


from reconfig_driftANDwaitTime_multicore import (
    j2_rates, hohmann_half, Orbit, deg2rad, rad2deg, MU_EARTH, R_EARTH
)
import numpy as np
from scipy.optimize import minimize_scalar, brentq

def compute_hohmann_dv(a_nom: float, delta_a_km: float) -> float:
    """
    Compute total Δv (m/s) for a round-trip Hohmann transfer.
    
    Parameters:
    -----------
    a_nom : float
        Nominal semi-major axis (km)
    delta_a_km : float
        Altitude change (km)
    
    Returns:
    --------
    dv_total_ms : float
        Total Δv in m/s (4 burns: up, circularize, down, circularize)
    """
    a_drift = a_nom + delta_a_km
    
    # Velocities
    v_nom = np.sqrt(MU_EARTH / a_nom)
    v_drift = np.sqrt(MU_EARTH / a_drift)
    
    # Transfer orbit (Hohmann)
    a_transfer = (a_nom + a_drift) / 2
    v_transfer_at_nom = np.sqrt(MU_EARTH * (2/a_nom - 1/a_transfer))
    v_transfer_at_drift = np.sqrt(MU_EARTH * (2/a_drift - 1/a_transfer))
    
    # Δv for each burn
    dv1 = abs(v_transfer_at_nom - v_nom)      # raise apogee
    dv2 = abs(v_drift - v_transfer_at_drift)  # circularize at drift
    dv3 = abs(v_transfer_at_drift - v_drift)  # lower perigee
    dv4 = abs(v_nom - v_transfer_at_nom)      # circularize back
    
    return (dv1 + dv2 + dv3 + dv4) * 1000  # km/s → m/s


def max_delta_a_from_dv_budget(a_nom: float, max_dv_ms: float) -> float:
    """
    Find maximum |Δa| allowed by Δv budget via binary search.
    
    Returns:
    --------
    max_delta_a : float (km)
    """
    # Binary search for max |Δa| where Hohmann Δv <= budget
    def dv_cost(abs_delta_a):
        return compute_hohmann_dv(a_nom, abs_delta_a) - max_dv_ms
    
    # Search in [0, 500] km range (reasonable for LEO/MEO)
    try:
        max_abs_da = brentq(dv_cost, 0.1, 500.0)
    except ValueError:
        # Budget too small or too large
        if compute_hohmann_dv(a_nom, 500.0) < max_dv_ms:
            max_abs_da = 500.0
        else:
            max_abs_da = 0.1
    
    return max_abs_da


def compute_raan_phasing_time_and_dv(
    initial_orbit: Orbit,
    target_raan_offset_deg: float,
    delta_a_km: float
) -> dict:
    """
    Compute time and Δv to phase one satellite by target RAAN offset.
    
    Strategy:
    1. Hohmann to drift orbit (a_nominal + Δa)
    2. Coast in drift orbit until RAAN accumulates target offset (via J2)
    3. Hohmann back to nominal
    
    Returns:
    --------
    result : dict
        {
            'success': bool,
            'time_drift_s': float,
            'time_total_s': float,  # includes 2× Hohmann
            'dv_total_ms': float,   # sum of 4 burns (2 Hohmanns)
            'delta_a_km': float,
            'delta_raan_accumulated_deg': float
        }
    """
    a_nom = initial_orbit.a_km
    i = initial_orbit.i_rad
    a_drift = a_nom + delta_a_km
    
    if a_drift <= R_EARTH + 200:
        return {'success': False, 'note': 'Drift altitude too low'}
    
    # Hohmann time
    _, _, t_hohmann = hohmann_half(a_nom, a_drift)
    
    # Δv (use dedicated function)
    dv_total_ms = compute_hohmann_dv(a_nom, delta_a_km)
    
    # J2 RAAN drift rates
    dOm_nom, _, _ = j2_rates(a_nom, 0.0, i)
    dOm_drift, _, _ = j2_rates(a_drift, 0.0, i)
    
    # Relative RAAN drift rate (rad/s)
    dOm_rel = dOm_drift - dOm_nom
    
    if abs(dOm_rel) < 1e-16:
        return {'success': False, 'note': 'No differential RAAN drift'}
    
    # Drift time to accumulate target offset
    target_raan_rad = deg2rad(target_raan_offset_deg)
    t_drift = target_raan_rad / dOm_rel
    
    if t_drift < 0:
        return {'success': False, 'note': 'Wrong drift direction (need opposite Δa sign)'}
    
    # Total time
    t_total = 2*t_hohmann + t_drift
    
    return {
        'success': True,
        'time_drift_s': t_drift,
        'time_hohmann_s': t_hohmann,
        'time_total_s': t_total,
        'dv_total_ms': dv_total_ms,
        'delta_a_km': delta_a_km,
        'delta_raan_accumulated_deg': target_raan_offset_deg
    }


def find_optimal_delta_a_for_raan_offset(
    initial_orbit: Orbit,
    target_raan_offset_deg: float,
    max_time_days: float,
    max_dv_ms: float
) -> dict:
    """
    Find optimal Δa that minimizes deployment time while respecting constraints.
    
    Strategy:
    - Compute max |Δa| from Δv budget
    - Search over [0, max_Δa] (both +/- directions) to minimize time
    - Return best feasible solution
    """
    max_time_s = max_time_days * 86400
    a_nom = initial_orbit.a_km
    
    # Compute max |Δa| from Δv budget
    max_abs_delta_a = max_delta_a_from_dv_budget(a_nom, max_dv_ms)
    
    # Objective: minimize time
    def objective(delta_a):
        result = compute_raan_phasing_time_and_dv(initial_orbit, target_raan_offset_deg, delta_a)
        if not result['success']:
            return 1e15  # penalty
        if result['time_total_s'] > max_time_s:
            return 1e15  # exceeds time budget
        return result['time_total_s']
    
    # Search both positive and negative Δa
    best_result = None
    best_time = 1e15
    
    for sign in [+1, -1]:
        try:
            res = minimize_scalar(
                objective, 
                bounds=(0.1, max_abs_delta_a), 
                method='bounded'
            )
            if res.fun < best_time:
                best_time = res.fun
                best_delta_a = sign * res.x
                best_result = compute_raan_phasing_time_and_dv(
                    initial_orbit, target_raan_offset_deg, best_delta_a
                )
        except:
            continue
    
    if best_result is None or not best_result['success']:
        return {
            'success': False, 
            'note': f'No feasible Δa in ±{max_abs_delta_a:.1f} km for {target_raan_offset_deg:.0f}° offset'
        }
    
    return best_result


def compute_alternative_scenarios(
    initial_orbit: Orbit,
    target_raan_offset_deg: float,
    max_time_days: float,
    max_dv_ms: float
) -> dict:
    """
    For infeasible cases, compute two alternative scenarios:
    1. Time needed if using full Δv budget
    2. Δv needed to meet time constraint
    
    Returns:
    --------
    scenarios : dict
        {
            'use_full_dv': {'delta_a_km', 'time_days', 'dv_ms'},
            'meet_time_budget': {'delta_a_km', 'time_days', 'dv_ms'}
        }
    """
    a_nom = initial_orbit.a_km
    max_time_s = max_time_days * 86400
    
    scenarios = {}
    
    # Scenario 1: Use full Δv budget, see resulting time
    max_abs_delta_a = max_delta_a_from_dv_budget(a_nom, max_dv_ms)
    
    best_time_with_full_dv = 1e15
    best_result_full_dv = None
    
    for sign in [+1, -1]:
        result = compute_raan_phasing_time_and_dv(
            initial_orbit, target_raan_offset_deg, sign * max_abs_delta_a
        )
        if result['success'] and result['time_total_s'] < best_time_with_full_dv:
            best_time_with_full_dv = result['time_total_s']
            best_result_full_dv = result
    
    if best_result_full_dv:
        scenarios['use_full_dv'] = {
            'delta_a_km': best_result_full_dv['delta_a_km'],
            'time_days': best_result_full_dv['time_total_s'] / 86400,
            'dv_ms': best_result_full_dv['dv_total_ms']
        }
    else:
        scenarios['use_full_dv'] = None
    
    # Scenario 2: Meet time budget, find required Δv
    def time_objective(abs_delta_a):
        """Find Δa that gives exactly max_time_s"""
        results = []
        for sign in [+1, -1]:
            result = compute_raan_phasing_time_and_dv(
                initial_orbit, target_raan_offset_deg, sign * abs_delta_a
            )
            if result['success']:
                results.append((result['time_total_s'], result))
        
        if not results:
            return 1e15
        
        # Return time difference from target (minimize absolute difference)
        best_time, _ = min(results, key=lambda x: abs(x[0] - max_time_s))
        return abs(best_time - max_time_s)
    
    try:
        # Search for Δa that gives time closest to max_time_s
        res = minimize_scalar(time_objective, bounds=(0.1, 500.0), method='bounded')
        
        # Evaluate both signs at optimal Δa
        results_at_optimal = []
        for sign in [+1, -1]:
            result = compute_raan_phasing_time_and_dv(
                initial_orbit, target_raan_offset_deg, sign * res.x
            )
            if result['success']:
                results_at_optimal.append(result)
        
        if results_at_optimal:
            # Pick the one closest to max_time_s
            best_for_time = min(results_at_optimal, 
                               key=lambda r: abs(r['time_total_s'] - max_time_s))
            
            scenarios['meet_time_budget'] = {
                'delta_a_km': best_for_time['delta_a_km'],
                'time_days': best_for_time['time_total_s'] / 86400,
                'dv_ms': best_for_time['dv_total_ms']
            }
        else:
            scenarios['meet_time_budget'] = None
    except:
        scenarios['meet_time_budget'] = None
    
    return scenarios


def plan_multi_satellite_raan_deployment(
    initial_orbit: Orbit,
    target_raan_offsets_deg: list[float],
    max_time_days: float = 180,
    max_dv_budget_ms: float = 50.0,
    detailed_diagnostics: bool = False
) -> dict:
    """
    Plan RAAN phasing for multiple satellites with optimized Δa per satellite.
    
    Parameters:
    -----------
    detailed_diagnostics : bool
        If True, compute alternative scenarios for infeasible cases:
        - Time needed if using full Δv budget
        - Δv needed to meet time constraint
    
    Returns:
    --------
    plan : dict
        {
            'satellites': [
                {'id': 0, 'target_raan_offset_deg': 0, 'delta_a_km': 0, 'time_days': 0, 'dv_ms': 0, 'feasible': True},
                {'id': 1, 'target_raan_offset_deg': 90, 'delta_a_km': X, 'time_days': Y, 'dv_ms': Z, 'feasible': bool,
                 'alternatives': {...}},  # only if infeasible and detailed_diagnostics=True
                ...
            ],
            'max_time_days': float,
            'max_dv_ms': float,
            'all_feasible': bool,
            'recommendation': str
        }
    """
    satellites = []
    
    # Reference satellite (id=0): no phasing needed
    satellites.append({
        'id': 0,
        'target_raan_offset_deg': 0.0,
        'delta_a_km': 0.0,
        'time_s': 0.0,
        'time_days': 0.0,
        'dv_ms': 0.0,
        'feasible': True
    })
    
    # Optimize Δa for each offset
    for idx, offset_deg in enumerate(target_raan_offsets_deg, start=1):
        result = find_optimal_delta_a_for_raan_offset(
            initial_orbit, offset_deg, max_time_days, max_dv_budget_ms
        )
        
        if not result['success']:
            sat_entry = {
                'id': idx,
                'target_raan_offset_deg': offset_deg,
                'delta_a_km': None,
                'time_s': None,
                'time_days': None,
                'dv_ms': None,
                'feasible': False,
                'note': result.get('note', 'Failed')
            }
            
            # Compute alternative scenarios if requested
            if detailed_diagnostics:
                alternatives = compute_alternative_scenarios(
                    initial_orbit, offset_deg, max_time_days, max_dv_budget_ms
                )
                sat_entry['alternatives'] = alternatives
            
            satellites.append(sat_entry)
            continue
        
        satellites.append({
            'id': idx,
            'target_raan_offset_deg': offset_deg,
            'delta_a_km': result['delta_a_km'],
            'time_s': result['time_total_s'],
            'time_days': result['time_total_s'] / 86400,
            'dv_ms': result['dv_total_ms'],
            'feasible': True
        })
    
    all_feasible = all(sat['feasible'] for sat in satellites)
    max_time = max((sat['time_days'] for sat in satellites if sat['time_days'] is not None), default=0)
    max_dv = max((sat['dv_ms'] for sat in satellites if sat['dv_ms'] is not None), default=0)
    
    recommendation = "Single launch sufficient" if all_feasible else "Multiple launches required"
    
    return {
        'satellites': satellites,
        'max_time_days': max_time,
        'max_dv_ms': max_dv,
        'all_feasible': all_feasible,
        'recommendation': recommendation
    }


def plot_deployment_time_vs_delta_a(
    initial_orbit: Orbit,
    target_raan_offsets_deg: list[float],
    delta_a_grid_km: np.ndarray,
    nominal_alts_km: list[float] = (500.0, 700.0, 1000.0),
    save_path: str = "results/launch/raan_deploy_time_vs_delta_a.png"
) -> None:
    """Plot deployment time (days, log scale) vs Δa.
    - Color encodes target RAAN offset (e.g., 90/180/270 deg)
    - Line style encodes nominal circular altitude h0 (500/700/1000 km)
    - Bottom axes show round-trip Δv [m/s] for h0 = 500, 700, 1000 km
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(11, 8))
    ax = plt.gca()

    # Color per RAAN target
    color_map = {
        30: "tab:red",       # ADDED
        60: "tab:purple",    # ADDED
        90: "tab:blue",
        180: "tab:orange",
        270: "tab:green",
    }
    # Line style per nominal altitude
    style_map = {
        500.0: "-",
        700.0: "--",
        1000.0: ":",
    }

    # Collect all times to set sensible log limits
    all_times = []

    for offset in target_raan_offsets_deg:
        color = color_map.get(float(offset), None)
        for h0 in nominal_alts_km:
            # Build orbit at requested nominal altitude, reusing i, Omega, u, e
            orb = Orbit(
                a_km=R_EARTH + float(h0),
                i_rad=initial_orbit.i_rad,
                Omega_rad=initial_orbit.Omega_rad,
                u_rad=initial_orbit.u_rad,
                e=initial_orbit.e,
            )

            times_days = []
            for da in delta_a_grid_km:
                res = compute_raan_phasing_time_and_dv(orb, offset, da)
                if res.get('success', False) and res['time_total_s'] > 0:
                    val = res['time_total_s'] / 86400.0
                    times_days.append(val)
                    all_times.append(val)
                else:
                    times_days.append(np.nan)  # hide infeasible regions

            # Do not create per-curve legend entries; we'll add a compact custom legend below
            ax.plot(
                delta_a_grid_km, times_days, label="_nolegend_", linewidth=2,
                color=color, linestyle=style_map.get(float(h0), "-.")
            )

    # Y axis: exponential/log scale
    ax.set_yscale('log')
    if all_times:
        y_min = max(1e-1, np.nanmin(all_times) * 0.8)
        y_max = np.nanmax(all_times) * 1.2
        if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
            ax.set_ylim(y_min, y_max)

    # Reference horizontal lines (with labels)
    # Keep: 120, 180, 360, 720 days; add 3y/4y/5y with years in label
    ref_days = [120, 180, 360, 720, 1095, 1460, 1825]  # 3,4,5 years ≈ 365×{3,4,5}
    for d in ref_days:
        ax.axhline(d, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
        if d in (1095, 1460, 1825):
            years = {1095: "3y", 1460: "4y", 1825: "5y"}[d]
            label = f"{d} d ({years})"
        else:
            label = f"{d} d"
        ax.text(0.985, d, label, transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=9, color="gray")

    ax.set_xlabel("Δa (km)")
    ax.set_ylabel("Deployment time (days, log scale)")
    # ax.set_title("Deployment time vs Δa (color: ΔRAAN, line style: h0)")
    ax.grid(True, which='both', alpha=0.3)

    # Compact legend: colors for RAAN targets + line styles for nominal altitudes
    # Color legend
    color_handles = []
    for off in sorted(set(float(o) for o in target_raan_offsets_deg)):
        col = color_map.get(off, "tab:gray")
        color_handles.append(Line2D([0], [0], color=col, lw=2, label=f"ΔRAAN {off:.0f}°"))
    # Style legend (use black lines with different linestyles)
    style_handles = []
    for h0 in nominal_alts_km:
        style_handles.append(Line2D([0], [0], color="k", lw=2,
                                    linestyle=style_map.get(float(h0), "-."),
                                    label=f"h0 = {h0:.0f} km"))
    ax.legend(handles=color_handles + style_handles, ncol=2, frameon=True)

    # ---------- Bottom Δv axes aligned to Δa (0 km ↔ 0 m/s) ----------
    # Use true functional mapping via secondary_xaxis so ticks coincide exactly.
    xlim = ax.get_xlim()  # keep after plotting
    ax.set_xlim(xlim)

    def make_dv_axes(alt_nom_km: float, offset_px: int, label: str):
        a_nom = R_EARTH + alt_nom_km

        # forward: Δa -> signed Δv (so mapping is 1-1 and 0 aligns)
        def fwd(da_km):
            da = np.asarray(da_km, dtype=float)
            return np.sign(da) * np.vectorize(lambda x: compute_hohmann_dv(a_nom, abs(x)))(da)

        # inverse: signed Δv -> Δa (solve with brentq on |Δa|)
        def inv(dv_signed):
            dv = np.asarray(dv_signed, dtype=float)
            out = np.empty_like(dv, dtype=float)
            hi0 = max(abs(xlim[0]), abs(xlim[1])) if max(abs(xlim[0]), abs(xlim[1])) > 0 else 500.0

            for idx, v in np.ndenumerate(dv):
                if abs(v) < 1e-9:
                    out[idx] = 0.0
                    continue
                target = abs(v)
                # grow upper bound until dv(hi) >= target (monotonic)
                hi = hi0
                for _ in range(8):
                    if compute_hohmann_dv(a_nom, hi) >= target:
                        break
                    hi *= 1.8
                try:
                    root = brentq(lambda x: compute_hohmann_dv(a_nom, x) - target, 0.0, hi, maxiter=100)
                    out[idx] = np.sign(v) * root
                except Exception:
                    out[idx] = np.nan
            return out

        sec = ax.secondary_xaxis('bottom', functions=(fwd, inv))
        sec.spines['bottom'].set_position(('outward', offset_px))
        sec.set_xlabel(label)
        sec.tick_params(axis="x", labelsize=9, pad=2)
        # Mirror main x limits so ticks align exactly
        sec.set_xlim(fwd(np.array(xlim)))

        return sec

    # More vertical spacing so axes don't overlap and sit close to main axis
    make_dv_axes(500.0, 35,  "Round-trip Δv [m/s]  @ h0 = 500 km")
    make_dv_axes(700.0, 70,  "Round-trip Δv [m/s]  @ h0 = 700 km")
    make_dv_axes(1000.0, 105, "Round-trip Δv [m/s]  @ h0 = 1000 km")

    # Extra bottom margin so stacked axes are not clipped
    plt.gcf().subplots_adjust(bottom=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    # plt.show()

def plot_raan_over_time_by_delta_a(
    initial_orbit: Orbit,
    delta_as_km: list[float] | None = None,
    duration_days: float = 800.0,  # slightly over 2 years
    save_path: str = "results/launch/raan_offset_over_time_by_delta_a.png"
) -> None:
    """
    Plot ΔRAAN(t) for a set of Δa values provided (can be negative or positive).
    Current model: RAAN drift starts after the first Hohmann leg and then
    grows linearly at (Ω̇_drift − Ω̇_nom). Return leg not included.

    Annotations:
    - Inline labels sit along each line, with earlier placement for large |Δa|
      to keep them inside plot bounds.
    - Labels for |Δa| ≥ 300 km use larger tilt and offset for readability.
    """
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Default Δa sampling: up to +500 km, not below -200 km
    if delta_as_km is None:
        delta_as_km = [-200, -150, -100, -50, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    a_nom = initial_orbit.a_km
    i = initial_orbit.i_rad
    OmG, _, _ = j2_rates(a_nom, 0.0, i)

    t = np.linspace(0.0, duration_days * 86400.0, 1000)  # seconds

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    y_min_plot, y_max_plot = -180.0, 360.0

    for da in delta_as_km:
        da = float(da)
        a_drift = a_nom + da

        # First transfer duration
        _, _, T1 = hohmann_half(a_nom, a_drift)

        # Relative RAAN rate during drift
        OmD, _, _ = j2_rates(a_drift, 0.0, i)
        dOm_rel = OmD - OmG  # rad/s

        # Piecewise: no RAAN change before T1, then linear drift
        dOm_t = np.where(t <= T1, 0.0, dOm_rel * (t - T1))  # rad
        dOm_deg = np.degrees(dOm_t)

        linestyle = "--" if da < 0 else "-"
        ax.plot(t / 86400.0, dOm_deg, linestyle, linewidth=2, color="black")

        # ---------- Inline label placement ----------
        abs_da = abs(da)

        # Place labels earlier for large |Δa|, and much earlier for -150 km
        if da <= -150:
            label_frac = 0.25  # much closer to start
        elif abs_da >= 500:
            label_frac = 0.35
        elif abs_da >= 450:
            label_frac = 0.40
        elif abs_da >= 400:
            label_frac = 0.45
        elif abs_da >= 350:
            label_frac = 0.50
        elif abs_da >= 300 or (da < 0 and abs_da >= 150):
            label_frac = 0.55
        else:
            label_frac = 0.70

        t_label_s = label_frac * duration_days * 86400.0
        t_label_days = label_frac * duration_days

        # RAAN at label time (deg)
        raan_label_deg = np.degrees(dOm_rel * max(0.0, t_label_s - T1))

        # Larger vertical offset for |Δa| >= 300
        base_offset = 10 if da > 0 else -10
        if abs_da >= 300:
            base_offset = 12 if da > 0 else -12
        y_lbl = raan_label_deg + base_offset
        y_lbl = float(np.clip(y_lbl, y_min_plot + 6.0, y_max_plot - 6.0))

        # Rotation based on slope, add ±(2–3)° more for large |Δa|
        slope_deg_per_day = np.degrees(dOm_rel) * 86400.0 if abs(dOm_rel) > 1e-16 else 0.0
        rotation_base = float(np.degrees(np.arctan2(slope_deg_per_day, duration_days / (y_max_plot - y_min_plot))))
        extra = 11.0 if abs_da >= 300 else 4.0
        rotation = rotation_base + (extra if da > 0 else -extra)

        ax.text(
            t_label_days, y_lbl,
            f"Δa={da:+.0f}",
            color="black", fontsize=8, ha="center", va="center",
            rotation=rotation, clip_on=True,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0)
        )

    # Vertical guide lines (time)
    ref_days = [30, 60, 180, 360, 720]
    for d in ref_days:
        ax.axvline(d, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(d, y_min_plot + 5, f"{d} d", color="gray", fontsize=8,
                va="bottom", ha="center", rotation=90, clip_on=True)

    # Horizontal guide lines (RAAN), emphasize -180, 180, 360 and ensure ticks exist
    for r in [-180, -90, -60, 60, 90, 180]:
        ax.axhline(r, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(360, color="gray", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.axhline(180, color="gray", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.axhline(-180, color="gray", linestyle="--", linewidth=1.1, alpha=0.8)
    # Ensure y-axis ticks include -180, 180, 360
    yticks = [-180, -90, -60, 0, 60, 90, 180, 360]
    ax.set_yticks(yticks)

    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("ΔRAAN relative to nominal (deg)", fontsize=11)
    # ax.set_title("RAAN offset growth vs time for different Δa (current drift model)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration_days)
    ax.set_ylim(y_min_plot, y_max_plot)

    fig.savefig(save_path, dpi=220)
    plt.close(fig)

# Example usage
if __name__ == "__main__":
    from datetime import datetime, timezone
    
    # Nominal orbit (e.g., 500 km circular, 51.6° inclination)
    nominal = Orbit(
        a_km=R_EARTH + 500,
        i_rad=deg2rad(51.6),
        Omega_rad=deg2rad(0),
        u_rad=deg2rad(0),
        e=0.0
    )
    
    # Target RAAN offsets
    target_offsets = [30, 60, 90, 180, 270]  # degrees
    
    # Constraints
    max_time_days = 360  # days
    max_dv_ms = 400  # m/s per satellite (this now constrains max Δa)
    
    plan = plan_multi_satellite_raan_deployment(
        nominal, target_offsets, max_time_days, max_dv_ms,
        detailed_diagnostics=True  # Enable alternative scenarios
    )
    
    print("=== RAAN Phasing Deployment Plan ===")
    print(f"Δv budget per satellite: {max_dv_ms} m/s")
    print(f"Time budget: {max_time_days} days")
    print(f"Max deployment time: {plan['max_time_days']:.1f} days")
    print(f"Max Δv used: {plan['max_dv_ms']:.1f} m/s")
    print(f"All satellites feasible: {plan['all_feasible']}")
    print(f"Recommendation: {plan['recommendation']}\n")
    
    for sat in plan['satellites']:
        if sat['feasible']:
            print(f"Satellite {sat['id']}: RAAN offset {sat['target_raan_offset_deg']:>3.0f}° → "
                  f"Δa={sat['delta_a_km']:>6.1f} km, "
                  f"t={sat['time_days']:>5.1f} days, "
                  f"Δv={sat['dv_ms']:>5.1f} m/s ✓")
        else:
            note = sat.get('note', 'Failed')
            print(f"Satellite {sat['id']}: RAAN offset {sat['target_raan_offset_deg']:>3.0f}° → ✗ {note}")
            
            # Show alternatives if available
            if 'alternatives' in sat:
                alts = sat['alternatives']
                print(f"  Alternative scenarios:")
                
                if alts['use_full_dv']:
                    full_dv = alts['use_full_dv']
                    print(f"    • Using full Δv budget ({max_dv_ms} m/s): "
                          f"time = {full_dv['time_days']:.1f} days "
                          f"(Δa = {full_dv['delta_a_km']:.1f} km)")
                else:
                    print(f"    • Using full Δv budget: Not achievable")
                
                if alts['meet_time_budget']:
                    meet_time = alts['meet_time_budget']
                    print(f"    • Meeting time budget ({max_time_days} days): "
                          f"Δv = {meet_time['dv_ms']:.1f} m/s "
                          f"(Δa = {meet_time['delta_a_km']:.1f} km)")
                else:
                    print(f"    • Meeting time budget: Not achievable")

    # 1) Deployment time vs Δa for each satellite (target offset)
    da_grid = np.linspace(-500, 500, 201)  # extend Δa to ±500 km
    da_grid[np.abs(da_grid) < 1e-6] = np.nan  # avoid Δa=0 singularity in plot
    plot_deployment_time_vs_delta_a(nominal, target_offsets, da_grid, nominal_alts_km=[500.0, 700.0, 1000.0])

    # 2) RAAN over time for different Δa values up to +500 km, not below -200 km
    plot_raan_over_time_by_delta_a(
        nominal,
        delta_as_km=[-200, -150, -100, -50, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        duration_days=800.0
    )