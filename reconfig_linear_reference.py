"""Bounded exact-point reference for the Paper 1 constant-rotation model.

Implements the timing matrix and horizon row bounds (manuscript Eqs. 12–15).
This is separate from the calendar-GMST longitude-band planner. It does not
validate Cartesian execution, nor prove a continuous-altitude optimum.
"""
from math import atan2, ceil, cos, floor, isfinite, pi, sin
import numpy as np
from reconfig_verification import gmst_rad, required_latitude_branches, TargetGeometryError

EARTH_OMEGA_RAD_S = 7.2921159e-5


def _wrap(x):
    return (x + pi) % (2.0 * pi) - pi


def row_wrap_bounds(a, c, budget_s):
    """Complete row bounds over Tw,Td >= 0 and Tw+Td <= budget."""
    if budget_s < 0:
        return None
    lower = budget_s * np.minimum(0.0, np.minimum(a[:, 0], a[:, 1]))
    upper = budget_s * np.maximum(0.0, np.maximum(a[:, 0], a[:, 1]))
    return [(ceil((lo - rhs) / (2*pi) - 1e-12), floor((hi - rhs) / (2*pi) + 1e-12))
            for lo, hi, rhs in zip(lower, upper, c)]


def solve_point(planner, horizon_s, *, wrap_padding=0, earth_omega=EARTH_OMEGA_RAD_S):
    """Earliest center-point access within finite caps and a total horizon.

    wrap_padding enlarges the proven row bounds for an independent enumeration
    check; it is not a tuning parameter required for the default solve.
    Coincident initial and drift orbits reduce to one physical initial coast.
    """
    if not isfinite(horizon_s) or horizon_s < 0 or wrap_padding < 0 or int(wrap_padding) != wrap_padding:
        raise ValueError('A finite nonnegative horizon and integer padding are required.')
    budget = horizon_s - planner.T1 - planner.T2
    if budget < 0:
        return {'success': False, 'status': 'infeasible', 'model': 'linear_constant_rotation_point'}
    try:
        branches = required_latitude_branches(planner.target_lat_deg, planner.i)
    except TargetGeometryError as exc:
        return {'success': False, 'status': exc.status, 'note': str(exc), 'model': 'linear_constant_rotation_point'}
    a = np.array([[planner.du_G, planner.du_D],
                  [planner.dOm_G - earth_omega, planner.dOm_D - earth_omega]], dtype=float)
    row_scale = np.max(np.abs(a), axis=1)
    balanced = a / row_scale[:, None]
    condition = float(np.linalg.cond(balanced))
    transfer_phase = planner.du_T1 * planner.T1 + planner.du_T2 * planner.T2
    transfer_node = planner.dOm_T1 * planner.T1 + planner.dOm_T2 * planner.T2
    theta0 = gmst_rad(planner.epoch)
    transfer_s = planner.T1 + planner.T2
    candidates = []
    enumerated = 0
    recorded_bounds = []
    for b, u in enumerate(branches):
        if b == 1 and abs(_wrap(u - branches[0])) <= 1e-12:
            continue
        q = atan2(cos(planner.i)*sin(u), cos(u))
        c = np.array([u - planner.u0 - transfer_phase,
                      planner.target_lon_rad - q - (planner.Omega0 - theta0)
                      - transfer_node + earth_omega * transfer_s])
        bounds = row_wrap_bounds(a, c, budget)
        recorded_bounds.append(bounds)
        if planner.a_D == planner.a_G:
            # No pseudoinverse. Phase fixes the combined initial coast.
            for m in range(bounds[0][0] - wrap_padding, bounds[0][1] + wrap_padding + 1):
                coast = (c[0] + 2*pi*m) / planner.du_G
                if coast < -1e-8 or coast > budget + 1e-8:
                    continue
                coast = max(0.0, coast)
                wait = min(coast, planner.wait_time_cap) if planner.wait_time_cap is not None else coast
                drift = coast - wait
                if drift > planner.drift_time_cap + 1e-8:
                    continue
                k = round((a[1,0] * coast - c[1]) / (2*pi))
                enumerated += 1
                residual = a @ [wait, drift] - c - 2*pi*np.array([m, k])
                if np.max(np.abs(residual)) <= 1e-9:
                    candidates.append((wait + drift + transfer_s, wait, drift, b, m, k, residual))
        else:
            if not isfinite(condition) or condition > 1e12:
                return {'success': False, 'status': 'ill_conditioned', 'condition_number': condition,
                        'model': 'linear_constant_rotation_point'}
            for m in range(bounds[0][0] - wrap_padding, bounds[0][1] + wrap_padding + 1):
                for k in range(bounds[1][0] - wrap_padding, bounds[1][1] + wrap_padding + 1):
                    enumerated += 1
                    rhs = c + 2*pi*np.array([m, k])
                    wait, drift = np.linalg.solve(balanced, rhs / row_scale)
                    total = wait + drift + transfer_s
                    if min(wait, drift) < -1e-8 or total > horizon_s + 1e-8:
                        continue
                    wait, drift = max(0.0, float(wait)), max(0.0, float(drift))
                    if (planner.wait_time_cap is not None and wait > planner.wait_time_cap + 1e-8) or drift > planner.drift_time_cap + 1e-8:
                        continue
                    residual = a @ [wait, drift] - rhs
                    if np.max(np.abs(residual)) <= 1e-9:
                        candidates.append((wait + drift + transfer_s, wait, drift, b, m, k, residual))
    metadata = {'model': 'linear_constant_rotation_point', 'condition_number': condition,
                'wrap_bounds': recorded_bounds, 'enumerated': enumerated,
                'horizon_s': horizon_s, 'earth_omega_rad_s': earth_omega}
    if not candidates:
        return dict(metadata, success=False, status='infeasible')
    total, wait, drift, b, m, k, residual = min(candidates, key=lambda x: x[0])
    return dict(metadata, success=True, status='accepted', total_time_s=total, T_wait=wait,
                T_d=drift, T1=planner.T1, T2=planner.T2,
                branch='ascending' if b == 0 else 'descending', m_u_wrap=m, k_lon_wrap=k,
                max_equation_residual_rad=float(np.max(np.abs(residual))))
