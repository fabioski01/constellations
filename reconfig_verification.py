"""Shared nominal-model checks for the Earth wait-and-drift planners.

These helpers validate analytical candidates, not executed Cartesian maneuvers.
The original spherical/J2 constants and UT1 approximately equal to UTC convention
are retained. Do not use this module as a high-fidelity propagation claim.
"""
from datetime import datetime, timezone
from math import asin, atan, degrees, isfinite, pi, sin, tan

J2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
TAU = 2.0 * pi


class TargetGeometryError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def utc_datetime(dt):
    """Interpret a naive input as UTC; normalize aware inputs to UTC."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def days_from_j2000(dt):
    return (utc_datetime(dt) - J2000).total_seconds() / 86400.0


def datetime_to_jd(dt):
    return 2451545.0 + days_from_j2000(dt)


def _gmst_degrees(d):
    t = d / 36525.0
    return 280.46061837 + 360.98564736629 * d + 0.000387933 * t * t - t**3 / 38710000.0


def gmst_rad(dt):
    # Compute the small day offset directly, without subtracting two large JDs.
    return (_gmst_degrees(days_from_j2000(dt)) % 360.0) * pi / 180.0


def earth_angle_elapsed(epoch, elapsed_s):
    """Continuous GMST angle anchored to the wrapped angle at the request.

    Root searches use floating elapsed seconds, avoiding datetime microsecond
    rounding and wrapped-angle jumps. Only final longitude residuals are wrapped.
    """
    d0 = days_from_j2000(epoch)
    delta = float(elapsed_s) / 86400.0
    t0, dt = d0 / 36525.0, delta / 36525.0
    correction = 0.000387933 * dt * (2.0 * t0 + dt)
    correction -= dt * (3.0 * t0 * t0 + 3.0 * t0 * dt + dt * dt) / 38710000.0
    return gmst_rad(epoch) + (360.98564736629 * delta + correction) * pi / 180.0


def required_latitude_branches(lat_deg, inclination_rad):
    if not isfinite(lat_deg) or not -90.0 <= lat_deg <= 90.0:
        raise TargetGeometryError('invalid_coordinate', 'Geodetic latitude must lie in [-90, 90] degrees.')
    if not isfinite(inclination_rad) or not 0.0 <= inclination_rad <= pi:
        raise TargetGeometryError('invalid_orbit', 'Inclination must lie in [0, pi].')
    phi = atan((1.0 - WGS84_E2) * tan(lat_deg * pi / 180.0))
    reach = sin(inclination_rad)
    if reach <= 1e-14:
        if abs(phi) > 1e-12:
            raise TargetGeometryError('unreachable', 'Latitude is unreachable at this inclination.')
        # Equatorial targets do not define two isolated latitude crossings.
        raise TargetGeometryError('unsupported_geometry', 'Equatorial orbits require a continuous-phase target search.')
    ratio = sin(phi) / reach
    if abs(ratio) > 1.0 + 1e-12:
        raise TargetGeometryError('unreachable', 'Latitude is unreachable at this inclination.')
    u = asin(max(-1.0, min(1.0, ratio)))
    return u % TAU, (pi - u) % TAU


def reachable_geodetic_latitude_deg(inclination_deg):
    if not isfinite(inclination_deg) or not 0.0 <= inclination_deg <= 180.0:
        raise ValueError('Inclination must lie in [0, 180] degrees.')
    phi = min(inclination_deg, 180.0 - inclination_deg) * pi / 180.0
    return degrees(atan(tan(phi) / (1.0 - WGS84_E2)))


def target_grid(inclination_deg, lat_band_deg, n_lat, n_lon):
    import numpy as np
    limit = reachable_geodetic_latitude_deg(float(inclination_deg))
    if not isfinite(lat_band_deg) or not 0.0 <= lat_band_deg <= min(90.0, limit) + 1e-10:
        raise ValueError(f'Latitude band must be between 0 and {limit:.9f} geodetic degrees.')
    if n_lat < 1 or n_lon < 1 or int(n_lat) != n_lat or int(n_lon) != n_lon:
        raise ValueError('Target-grid dimensions must be positive integers.')
    return (np.linspace(-lat_band_deg, lat_band_deg, int(n_lat)),
            np.linspace(-180.0, 180.0, int(n_lon), endpoint=False))


def validate_tolerances(schedule, default):
    values = list(default if schedule is None else schedule)
    if not values or any(not isfinite(v) or not 0.0 <= v <= 180.0 for v in values):
        raise ValueError('Longitude tolerances must be a nonempty sequence in [0, 180] degrees.')
    if any(a > b for a, b in zip(values, values[1:])):
        raise ValueError('The tolerance ladder must be nondecreasing.')
    return values


def clip_horizon(lo, hi, phase_increment, gamma_g, gamma_d, transfer_s, horizon_s):
    """Intersect a phase line with the optional total response deadline."""
    if horizon_s is None:
        return lo, hi
    constant = phase_increment / gamma_g + transfer_s
    slope = 1.0 - gamma_d / gamma_g
    if abs(slope) <= 1e-15:
        return (lo, hi) if constant <= horizon_s + 1e-8 else (1.0, 0.0)
    edge = (horizon_s - constant) / slope
    return (lo, min(hi, edge)) if slope > 0.0 else (max(lo, edge), hi)


def candidate_is_feasible(sol, requested_tolerance_deg, wait_cap_s, drift_cap_s, horizon_s=None):
    if not sol.get('success'):
        return False
    try:
        tw, td, total, error = (float(sol[k]) for k in ('T_wait', 'T_d', 'total_time_s', 'lon_err_deg'))
        return (all(isfinite(v) for v in (tw, td, total, error))
                and tw >= 0.0 and td >= 0.0
                and (wait_cap_s is None or tw <= wait_cap_s + 1e-8)
                and td <= drift_cap_s + 1e-8
                and (horizon_s is None or total <= horizon_s + 1e-8)
                and abs(error) <= requested_tolerance_deg
                and abs(float(sol.get('lat_err_deg', 0.0))) <= 1e-7)
    except (KeyError, TypeError, ValueError):
        return False


def select_branch_candidate(candidates, schedule, branch, wait_cap_s, drift_cap_s, horizon_s=None):
    for tol in schedule:
        valid = [c for c in candidates if candidate_is_feasible(c, tol, wait_cap_s, drift_cap_s, horizon_s)]
        if valid:
            best = dict(min(valid, key=lambda s: s['total_time_s']))
            best['status'] = 'accepted'
            best['selection_meta'] = {
                'policy': 'earliest_within_tol', 'tolerance_used_deg': tol,
                'num_candidates_considered': len(candidates),
                'wait_cap_s': wait_cap_s, 'wait_cap_hr': None if wait_cap_s is None else wait_cap_s / 3600.0,
                'drift_cap_s': drift_cap_s, 'drift_cap_hr': drift_cap_s / 3600.0,
                'response_horizon_s': horizon_s,
                'chosen_m': best['m_u_wrap'], 'chosen_k': best['k_lon_wrap'],
            }
            return best
    diagnostic = min(candidates, key=lambda s: (abs(s['lon_err_deg']), s['total_time_s'])) if candidates else None
    if diagnostic is not None:
        diagnostic = dict(diagnostic, success=False, status='diagnostic_near_miss')
    return {'success': False, 'status': 'infeasible', 'branch': branch,
            'note': 'No candidate satisfies the requested tolerance and time limits.',
            'diagnostic': diagnostic}


def add_event_metadata(sol, planner):
    if sol['T1'] == 0.0 and sol['T2'] == 0.0:
        sol['insertion_time_s'] = 0.0  # Already in the regional orbit at request.
    else:
        sol['insertion_time_s'] = sol['T_wait'] + sol['T1'] if sol['T2'] == 0.0 else sol['total_time_s']
    sol['access_time_s'] = sol['total_time_s']
    sol['departure_time_s'] = sol['T_wait'] if sol['T1'] > 0.0 else sol['T_wait'] + sol['T_d']
    sol['transfer1_omitted'] = sol['T1'] == 0.0
    sol['transfer2_omitted'] = sol['T2'] == 0.0
    sol['initial_coast_s'] = sol['T_wait'] + sol['T_d'] if sol['T1'] == 0.0 else sol['T_wait']
    sol['regional_coast_s'] = sol['T_d'] if sol['T2'] == 0.0 else 0.0
    sol['model'] = 'secular_j2_calendar_gmst_longitude_band'
    return sol
