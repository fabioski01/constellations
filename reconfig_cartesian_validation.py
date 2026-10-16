"""Continuous Cartesian execution of the Paper 1 impulsive maneuver.

Units are km, s, radians and km/s. This is a fixed-axis two-body/J2 model,
with the planner's GMST convention used only for surface coordinates. It is
independent of SGP4 and never resets a trajectory to a desired staging orbit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, pi, sin, sqrt
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp, trapezoid

from reconfig_verification import WGS84_E2, earth_angle_elapsed

TAU = 2.0 * pi


def wrap(angle):
    return (np.asarray(angle) + pi) % TAU - pi


@dataclass(frozen=True)
class Gravity:
    mu: float = 398600.4418
    radius_km: float = 6371.0
    j2: float = 1.0826266e-3

    def __post_init__(self):
        if not all(np.isfinite(x) for x in (self.mu, self.radius_km, self.j2)):
            raise ValueError('Gravity parameters must be finite.')
        if self.mu <= 0 or self.radius_km <= 0:
            raise ValueError('Gravity parameter and reference radius must be positive.')


@dataclass(frozen=True)
class IntegrationSettings:
    rtol: float = 2e-11
    position_atol_km: float = 1e-9
    velocity_atol_km_s: float = 1e-12
    max_step_s: float = 60.0

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in (
                self.rtol, self.position_atol_km, self.velocity_atol_km_s,
                self.max_step_s)):
            raise ValueError('Integrator tolerances and maximum step must be positive.')

    def options(self):
        return dict(method='DOP853', rtol=self.rtol,
                    atol=np.array([self.position_atol_km] * 3 +
                                  [self.velocity_atol_km_s] * 3),
                    max_step=self.max_step_s, dense_output=True)


def rhs(_time, state, gravity: Gravity):
    r, v = state[:3], state[3:]
    radius = np.linalg.norm(r)
    if radius <= 0:
        raise ValueError('A Cartesian state cannot lie at the gravity center.')
    acceleration = -gravity.mu * r / radius**3
    if gravity.j2:
        z2_r2 = (r[2] / radius)**2
        factor = 1.5 * gravity.j2 * gravity.mu * gravity.radius_km**2 / radius**5
        acceleration += factor * np.array([
            r[0] * (5 * z2_r2 - 1), r[1] * (5 * z2_r2 - 1),
            r[2] * (5 * z2_r2 - 3)])
    return np.concatenate((v, acceleration))


def conserved_energy(state, gravity: Gravity):
    r, v = state[:3], state[3:]
    radius = np.linalg.norm(r)
    p2 = (3 * (r[2] / radius)**2 - 1) / 2
    return (np.dot(v, v) / 2 - gravity.mu / radius +
            gravity.mu * gravity.j2 * gravity.radius_km**2 * p2 / radius**3)


def state_from_elements(elements, mu):
    """Convert [a, e cos(omega), e sin(omega), i, node, omega+M] to r,v."""
    a, ex, ey, inc, node, mean_u = np.asarray(elements, dtype=float)
    eccentricity = np.hypot(ex, ey)
    if not np.all(np.isfinite(elements)) or a <= 0 or eccentricity >= 1:
        raise ValueError('Only finite elliptic orbital elements are supported.')
    omega = atan2(ey, ex) if eccentricity > 1e-15 else 0.0
    anomaly = float(wrap(mean_u - omega))
    eccentric_anomaly = anomaly
    for _ in range(30):
        change = ((eccentric_anomaly - eccentricity * sin(eccentric_anomaly) - anomaly) /
                  (1 - eccentricity * cos(eccentric_anomaly)))
        eccentric_anomaly -= change
        if abs(change) < 1e-14:
            break
    else:
        raise RuntimeError('Kepler equation did not converge.')
    p = np.array([cos(node), sin(node), 0.0])
    q = np.array([-sin(node) * cos(inc), cos(node) * cos(inc), sin(inc)])
    peri = cos(omega) * p + sin(omega) * q
    transverse = -sin(omega) * p + cos(omega) * q
    ce, se = cos(eccentric_anomaly), sin(eccentric_anomaly)
    beta = sqrt(1 - eccentricity**2)
    radius = a * (1 - eccentricity * ce)
    r = a * ((ce - eccentricity) * peri + beta * se * transverse)
    v = sqrt(mu * a) / radius * (-se * peri + beta * ce * transverse)
    return np.concatenate((r, v))


def elements_from_state(state, mu):
    """Nonsingular-in-eccentricity elements; equatorial planes are excluded."""
    r, v = np.asarray(state[:3]), np.asarray(state[3:])
    radius = np.linalg.norm(r)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    if hn == 0 or np.hypot(h[0], h[1]) / hn < 1e-9:
        raise ValueError('Radial or equatorial states are not supported by this element chart.')
    inc = atan2(np.hypot(h[0], h[1]), h[2])
    node = atan2(h[0], -h[1])
    p = np.array([cos(node), sin(node), 0.0])
    q = np.array([-sin(node) * cos(inc), cos(node) * cos(inc), sin(inc)])
    ev = np.cross(v, h) / mu - r / radius
    ex, ey = np.dot(ev, p), np.dot(ev, q)
    eccentricity = np.hypot(ex, ey)
    true_u = atan2(np.dot(r, q), np.dot(r, p))
    mean_u = true_u
    if eccentricity > 1e-12:
        omega = atan2(ey, ex)
        true_anomaly = float(wrap(true_u - omega))
        eccentric_anomaly = atan2(sqrt(max(0.0, 1 - eccentricity**2)) * sin(true_anomaly),
                                 eccentricity + cos(true_anomaly))
        mean_anomaly = eccentric_anomaly - eccentricity * sin(eccentric_anomaly)
        mean_u += float(wrap(mean_anomaly - true_anomaly))
    denominator = 2 / radius - np.dot(v, v) / mu
    if denominator <= 0 or eccentricity >= 1:
        raise ValueError('Only bound elliptic states are supported.')
    return np.array([1 / denominator, ex, ey, inc, node, mean_u])


def circular_secular_rates(a, inclination, gravity):
    n = sqrt(gravity.mu / a**3)
    factor = gravity.j2 * n * (gravity.radius_km / a)**2
    ci = cos(inclination)
    node = -1.5 * factor * ci
    phase = n + .75 * factor * (3 * ci**2 - 1) + .75 * factor * (5 * ci**2 - 1)
    return node, phase


def averaged_elements(state, gravity, period_s, settings=None, samples=513):
    """Symmetric one-period time average of nonsingular osculating elements.

    This explicitly defined numerical averaging convention is NOT an exact
    Brouwer-Lyddane, SGP4, or DSST mean-element transformation.
    """
    settings = settings or IntegrationSettings(rtol=2e-12, max_step_s=30)
    times = np.linspace(-period_s / 2, period_s / 2, samples)
    backward = solve_ivp(lambda t, y: rhs(t, y, gravity), (0, times[0]), state,
                        **settings.options())
    forward = solve_ivp(lambda t, y: rhs(t, y, gravity), (0, times[-1]), state,
                       **settings.options())
    if not backward.success or not forward.success:
        raise RuntimeError('Mean-state initialization propagation failed.')
    values = np.array([elements_from_state(
        backward.sol(t) if t < 0 else forward.sol(t), gravity.mu) for t in times])
    values[:, 4:] = np.unwrap(values[:, 4:], axis=0)
    return trapezoid(values, times, axis=0) / period_s


def mean_matched_initial_state(mean_elements, gravity, tolerance=1e-10,
                               max_iterations=12):
    """Match the initial orbit's numerical mean; never use a target or burns."""
    target = np.asarray(mean_elements, dtype=float)
    if np.hypot(target[1], target[2]) > 1e-12:
        raise ValueError('The initial mean orbit must be circular for this study.')
    if abs(sin(target[3])) < 1e-8:
        raise ValueError('Equatorial initial orbits need another element chart.')
    _, phase_rate = circular_secular_rates(target[0], target[3], gravity)
    period = TAU / phase_rate
    candidate = target.copy()
    scale = np.array([target[0], 1, 1, 1, 1, 1])
    for iteration in range(max_iterations):
        state = state_from_elements(candidate, gravity.mu)
        measured = averaged_elements(state, gravity, period)
        residual = measured - target
        residual[4:] = wrap(residual[4:])
        if np.max(np.abs(residual / scale)) < tolerance:
            return state, dict(
                mode='symmetric_one_orbit_numerical_mean', iterations=iteration + 1,
                averaging_period_s=float(period), samples=513,
                normalized_residual=float(np.max(np.abs(residual / scale))),
                target_mean_elements=target.tolist(),
                initial_osculating_elements=candidate.tolist(),
                measured_mean_elements=measured.tolist(),
                residual_elements=residual.tolist(),
                note='Numerical averaging convention; not an exact theory-specific mean conversion.')
        candidate -= residual
    raise RuntimeError(f'Initial mean matching did not converge: {residual.tolist()}')


@dataclass(frozen=True)
class Burn:
    time_s: float
    label: str
    transverse_delta_v_km_s: float


@dataclass(frozen=True)
class Phase:
    start_s: float
    end_s: float
    label: str


@dataclass
class ManeuverSchedule:
    burns: list[Burn]
    phases: list[Phase]
    insertion_s: float
    predicted_access_s: float
    departure_s: float


def maneuver_schedule(a_g, a_d, a_r, wait_s, drift_s, mu):
    """Nominal Hohmann magnitudes and times; RTN directions come from execution."""
    if not all(np.isfinite(v) for v in (a_g, a_d, a_r, wait_s, drift_s, mu)):
        raise ValueError('Maneuver inputs must be finite.')
    if min(a_g, a_d, a_r, mu) <= 0 or min(wait_s, drift_s) < 0:
        raise ValueError('Radii and mu must be positive, coasts nonnegative.')
    burns, phases = [], []
    clock = float(wait_s)
    if wait_s:
        phases.append(Phase(0.0, clock, 'Initial wait'))

    def transfer(a, b, number):
        nonlocal clock
        if a == b:
            return
        semimajor = (a + b) / 2
        duration = pi * sqrt(semimajor**3 / mu)
        dv_depart = sqrt(mu * (2 / a - 1 / semimajor)) - sqrt(mu / a)
        dv_arrive = sqrt(mu / b) - sqrt(mu * (2 / b - 1 / semimajor))
        burns.append(Burn(clock, f'T{number} departure', dv_depart))
        phases.append(Phase(clock, clock + duration, f'Transfer {number}'))
        clock += duration
        burns.append(Burn(clock, f'T{number} arrival', dv_arrive))

    transfer(a_g, a_d, 1)
    first_insertion = clock if a_g != a_d else 0.0
    if drift_s:
        phases.append(Phase(clock, clock + drift_s, 'Intermediate coast'))
    clock += drift_s
    transfer(a_d, a_r, 2)
    insertion = clock if a_d != a_r else first_insertion
    departure = wait_s if a_g != a_d else (wait_s + drift_s if a_d != a_r else None)
    return ManeuverSchedule(burns, phases, float(insertion), float(clock), departure)


@dataclass
class Arc:
    start_s: float
    end_s: float
    solution: Any


@dataclass
class Trajectory:
    initial_state: np.ndarray
    end_s: float
    arcs: list[Arc] = field(default_factory=list)
    burn_records: list[dict] = field(default_factory=list)
    latitude_crossings_s: list[float] = field(default_factory=list)
    ascending_nodes_s: list[float] = field(default_factory=list)
    right_states: dict[float, np.ndarray] = field(default_factory=dict)
    evaluations: int = 0

    def state(self, time_s, side='right'):
        t = float(time_s)
        if t < -1e-7 or t > self.end_s + 1e-7:
            raise ValueError('Requested state is outside the propagated interval.')
        t = min(self.end_s, max(0.0, t))
        if side == 'right' and t in self.right_states:
            return self.right_states[t].copy()
        for arc in self.arcs:
            if arc.start_s - 1e-9 <= t <= arc.end_s + 1e-9:
                return np.asarray(arc.solution.sol(t))
        if t == 0:
            return self.initial_state.copy()
        raise RuntimeError(f'No propagation arc covers t={t}.')


def execute_maneuver(initial_state, schedule, end_s, gravity, target_latitude_deg,
                     settings=None):
    """Propagate r,v continuously and apply the fixed signed RTN-T impulses."""
    settings = settings or IntegrationSettings()
    if end_s < schedule.predicted_access_s or not np.isfinite(end_s):
        raise ValueError('Propagation must cover the complete scheduled maneuver.')
    if not -90 < target_latitude_deg < 90:
        raise ValueError('The latitude event requires a nonpolar target.')
    state = np.asarray(initial_state, dtype=float).copy()
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError('A finite six-component Cartesian initial state is required.')
    gc_lat = atan2((1 - WGS84_E2) * sin(np.deg2rad(target_latitude_deg)),
                   cos(np.deg2rad(target_latitude_deg)))

    def latitude_event(_t, y):
        return y[2] / np.linalg.norm(y[:3]) - sin(gc_lat)

    def ascending_node(_t, y):
        return y[2]

    ascending_node.direction = 1
    trajectory = Trajectory(state.copy(), float(end_s))
    clock = 0.0
    burns_at = {}
    for burn in schedule.burns:
        burns_at.setdefault(burn.time_s, []).append(burn)
    endpoints = sorted(set([*burns_at, float(end_s)]))
    for endpoint in endpoints:
        if endpoint > clock:
            result = solve_ivp(lambda t, y: rhs(t, y, gravity), (clock, endpoint),
                               state, events=(latitude_event, ascending_node),
                               **settings.options())
            if not result.success:
                raise RuntimeError(result.message)
            trajectory.arcs.append(Arc(clock, endpoint, result))
            trajectory.evaluations += result.nfev
            trajectory.latitude_crossings_s.extend(result.t_events[0].tolist())
            trajectory.ascending_nodes_s.extend(result.t_events[1].tolist())
            state = result.y[:, -1].copy()
            clock = endpoint
        for burn in burns_at.get(endpoint, []):
            before = state.copy()
            radial = state[:3] / np.linalg.norm(state[:3])
            normal = np.cross(state[:3], state[3:])
            normal /= np.linalg.norm(normal)
            transverse = np.cross(normal, radial)
            delta_v = burn.transverse_delta_v_km_s * transverse
            state[3:] += delta_v
            trajectory.burn_records.append(dict(
                time_s=float(endpoint), label=burn.label,
                signed_transverse_delta_v_m_s=1000 * burn.transverse_delta_v_km_s,
                delta_v_km_s=delta_v.tolist(), before_state=before.tolist(),
                after_state=state.tolist(),
                position_jump_m=float(1000 * np.linalg.norm(state[:3] - before[:3])),
                impulse_residual_m_s=float(1000 * np.linalg.norm(state[3:] - before[3:] - delta_v))))
        trajectory.right_states[float(endpoint)] = state.copy()

    def deduplicate(values):
        result = []
        for t in sorted(values):
            if not result or t - result[-1] > 1e-6:
                result.append(float(t))
        return result

    trajectory.latitude_crossings_s = deduplicate(trajectory.latitude_crossings_s)
    trajectory.ascending_nodes_s = deduplicate(trajectory.ascending_nodes_s)
    return trajectory


def ground_point(state, epoch, time_s):
    """Radial WGS84 surface subpoint, matching the planner's target geometry.

    This is the ellipsoid intersection of the geocentric radius ray, not the
    normal-foot geodetic coordinates of the spacecraft at its nonzero altitude.
    """
    x, y, z = state[:3]
    latitude = atan2(z, (1 - WGS84_E2) * np.hypot(x, y))
    longitude = float(wrap(atan2(y, x) - earth_angle_elapsed(epoch, time_s)))
    return np.rad2deg([latitude, longitude])


def surface_distance_km(first_lat_lon, second_lat_lon, radius_km=6371.0):
    """Spherical diagnostic distance; geographic acceptance remains the band."""
    a, b = np.deg2rad(first_lat_lon), np.deg2rad(second_lat_lon)
    h = sin((b[0] - a[0]) / 2)**2 + cos(a[0]) * cos(b[0]) * sin((b[1] - a[1]) / 2)**2
    return float(2 * radius_km * np.arctan2(sqrt(max(0, h)), sqrt(max(0, 1 - min(1, h)))))


def evaluate_access(trajectory, schedule, epoch, target_latitude_deg,
                    target_longitude_deg, tolerance_deg):
    if not np.isfinite(tolerance_deg) or not 0 <= tolerance_deg <= 180:
        raise ValueError('Longitude tolerance must be in [0, 180] degrees.')
    crossings = []
    for t in trajectory.latitude_crossings_s:
        state = trajectory.state(t)
        lat, lon = ground_point(state, epoch, t)
        error = float(np.rad2deg(wrap(np.deg2rad(lon - target_longitude_deg))))
        radial_rate = np.dot(state[:3], state[3:]) / np.linalg.norm(state[:3])
        latitude_direction = state[5] - state[2] * radial_rate / np.linalg.norm(state[:3])
        eligible = t >= schedule.insertion_s
        crossings.append(dict(
            time_s=t, time_from_prediction_s=t - schedule.predicted_access_s,
            latitude_deg=float(lat), longitude_deg=float(lon),
            latitude_residual_deg=float(lat - target_latitude_deg),
            longitude_residual_deg=error, normalized_longitude_error=abs(error) / tolerance_deg if tolerance_deg else None,
            branch='ascending' if latitude_direction > 0 else 'descending',
            eligible_after_insertion=bool(eligible),
            accepted=bool(eligible and abs(error) <= tolerance_deg)))
    nearest = min(crossings, key=lambda x: abs(x['time_from_prediction_s'])) if crossings else None
    eligible = [c for c in crossings if c['eligible_after_insertion']]
    accepted = [c for c in eligible if c['accepted']]
    at_prediction = ground_point(trajectory.state(schedule.predicted_access_s), epoch,
                                 schedule.predicted_access_s)
    return dict(
        accepted=bool(accepted),
        first_accepted_crossing=accepted[0] if accepted else None,
        first_eligible_crossing=eligible[0] if eligible else None,
        nearest_crossing_to_prediction=nearest,
        at_predicted_time=dict(
            latitude_deg=float(at_prediction[0]), longitude_deg=float(at_prediction[1]),
            latitude_residual_deg=float(at_prediction[0] - target_latitude_deg),
            longitude_residual_deg=float(np.rad2deg(wrap(np.deg2rad(at_prediction[1] - target_longitude_deg)))),
            spherical_ground_distance_km=surface_distance_km(
                at_prediction, [target_latitude_deg, target_longitude_deg])),
        search_start_s=schedule.insertion_s, search_end_s=trajectory.end_s,
        crossings=crossings)


def evaluate_repeat(trajectory, schedule, epoch, revolutions, nominal_repeat_s):
    nodes = [t for t in trajectory.ascending_nodes_s if t >= schedule.insertion_s + 1e-6]
    if len(nodes) < revolutions + 1:
        return dict(status='insufficient_node_returns', available_nodes=len(nodes))
    first, last = nodes[0], nodes[revolutions]
    p0 = ground_point(trajectory.state(first), epoch, first)
    p1 = ground_point(trajectory.state(last), epoch, last)
    result = dict(status='measured', revolutions=revolutions,
                  reference_node_time_s=first, return_node_time_s=last,
                  achieved_repeat_interval_s=last - first,
                  repeat_interval_error_s=last - first - nominal_repeat_s,
                  node_longitude_closure_deg=float(np.rad2deg(wrap(np.deg2rad(p1[1] - p0[1])))),
                  node_surface_closure_km=surface_distance_km(p0, p1))
    if first + nominal_repeat_s <= trajectory.end_s:
        p_at_interval = ground_point(trajectory.state(first + nominal_repeat_s), epoch,
                                     first + nominal_repeat_s)
        result['scheduled_interval_surface_closure_km'] = surface_distance_km(p0, p_at_interval)
    return result
