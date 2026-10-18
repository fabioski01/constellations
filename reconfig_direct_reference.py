"""Independent nominal reference: wait, one Hohmann transfer, regional coast.

The search is written separately from the wait/drift candidate search. It uses
the same declared secular rates, radial surface geometry and continuous GMST.
This is a physical reference strategy, not a reproduction of another paper.
"""
from math import atan2, ceil, cos, floor, isfinite, pi, sin, sqrt

from scipy.optimize import brentq

from reconfig_driftANDwaitTime_multicore import j2_rates, MU_EARTH
from reconfig_verification import (
    days_from_j2000, earth_angle_elapsed, required_latitude_branches,
)

TAU = 2 * pi


def _earth_rate(epoch, time_s):
    centuries = (days_from_j2000(epoch) + time_s / 86400) / 36525
    degrees_per_day = (360.98564736629 +
                       (0.000775866 * centuries - 3 * centuries**2 / 38710000) / 36525)
    return degrees_per_day * pi / 180 / 86400


def solve_direct(planner, tolerance_deg):
    """Minimize nominal response under matched initial-wait/regional-coast caps.

    Finite phase-wrap bounds follow from the rectangular coast-time bounds.
    At each wrap, phase eliminates the regional coast, leaving a scalar wait
    interval. Longitude-band boundaries are bracketed without a discovery grid.
    The derivative guard rejects a nonmonotone interval instead of assuming it.
    """
    wait_cap, coast_cap = planner.wait_time_cap, planner.drift_time_cap
    if (wait_cap is None or not all(isfinite(x) and x >= 0 for x in (wait_cap, coast_cap))
            or not isfinite(tolerance_deg) or not 0 < tolerance_deg < 180):
        raise ValueError('Finite coast caps and 0 < tolerance < 180 degrees are required.')
    a_g, a_r, inc = planner.a_G, planner.rom_a, planner.i
    transfer_a = (a_g + a_r) / 2
    transfer_e = abs(a_g - a_r) / (a_g + a_r)
    transfer_s = 0.0 if a_g == a_r else pi * sqrt(transfer_a**3 / MU_EARTH)
    ng, wg, mg = j2_rates(a_g, 0, inc)
    nr, wr, mr = j2_rates(a_r, 0, inc)
    nt, wt, mt = j2_rates(transfer_a, transfer_e, inc)
    gamma_g, gamma_r, gamma_t = wg + mg, wr + mr, wt + mt
    if min(gamma_g, gamma_r) <= 0:
        raise ValueError('Positive secular phase rates are required.')
    tol = tolerance_deg * pi / 180
    candidates = []
    intervals = 0
    for branch_index, required_u in enumerate(required_latitude_branches(planner.target_lat_deg, inc)):
        constant = required_u - planner.u0 - gamma_t * transfer_s
        lower = ceil(-constant / TAU - 1e-12)
        upper = floor((gamma_g * wait_cap + gamma_r * coast_cap - constant) / TAU + 1e-12)
        q = atan2(cos(inc) * sin(required_u), cos(required_u))
        for m in range(lower, upper + 1):
            phase_increment = constant + TAU * m
            lo = max(0.0, (phase_increment - gamma_r * coast_cap) / gamma_g)
            hi = min(wait_cap, phase_increment / gamma_g)
            response_slope = 1 - gamma_g / gamma_r
            response_constant = transfer_s + phase_increment / gamma_r
            if planner.response_horizon is not None:
                edge = planner.response_horizon - response_constant
                if abs(response_slope) < 1e-15:
                    if edge < 0:
                        continue
                elif response_slope > 0:
                    hi = min(hi, edge / response_slope)
                else:
                    lo = max(lo, edge / response_slope)
            if hi < lo or hi < 0:
                continue
            intervals += 1

            def coast(wait):
                return (phase_increment - gamma_g * wait) / gamma_r

            def longitude(wait):
                regional_coast = coast(wait)
                elapsed = wait + transfer_s + regional_coast
                return (planner.Omega0 + ng * wait + nt * transfer_s + nr * regional_coast
                        + q - earth_angle_elapsed(planner.epoch, elapsed))

            def derivative(wait):
                elapsed = wait + transfer_s + coast(wait)
                return ng - nr * gamma_g / gamma_r - _earth_rate(planner.epoch, elapsed) * response_slope

            derivatives = [derivative(t) for t in (lo, (lo + hi) / 2, hi)]
            if min(derivatives) <= 0 <= max(derivatives) and max(abs(x) for x in derivatives) > 1e-16:
                raise RuntimeError('Direct reference encountered nonmonotone longitude.')
            left, right = longitude(lo), longitude(hi)
            low_lon, high_lon = min(left, right), max(left, right)
            k_lo = ceil((low_lon - planner.target_lon_rad - tol) / TAU - 1e-12)
            k_hi = floor((high_lon - planner.target_lon_rad + tol) / TAU + 1e-12)
            for k in range(k_lo, k_hi + 1):
                center = planner.target_lon_rad + TAU * k
                trial = [lo, hi]
                for boundary in (center - tol, center + tol):
                    f_lo, f_hi = left - boundary, right - boundary
                    if f_lo * f_hi < 0:
                        root = brentq(lambda w: longitude(w) - boundary, lo, hi, xtol=1e-8)
                        trial.extend((root, max(lo, root - 1e-6), min(hi, root + 1e-6)))
                for wait in trial:
                    regional_coast = coast(wait)
                    if regional_coast < -1e-8 or regional_coast > coast_cap + 1e-8:
                        continue
                    error = (longitude(wait) - planner.target_lon_rad + pi) % TAU - pi
                    if abs(error) > tol:
                        continue
                    candidates.append(dict(
                        success=True, T_wait=float(wait), T_d=float(max(0, regional_coast)),
                        T1=transfer_s, T2=0.0,
                        total_time_s=float(wait + transfer_s + regional_coast),
                        insertion_time_s=float(wait + transfer_s),
                        branch='ascending' if branch_index == 0 else 'descending',
                        m_u_wrap=m, k_lon_wrap=k, lon_err_deg=error * 180 / pi))
    if not candidates:
        return dict(success=False, searched_phase_intervals=intervals)
    return dict(min(candidates, key=lambda row: row['total_time_s']),
                searched_phase_intervals=intervals, reference='independent_direct_transfer')
