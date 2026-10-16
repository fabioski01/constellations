"""Generate one connected-maneuver validation figure and its numerical evidence."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from math import cos, pi, radians, sin
import os
from pathlib import Path
import platform

import numpy as np
from scipy.optimize import brentq

from reconfig_cartesian_validation import (
    Gravity, IntegrationSettings, Phase, circular_secular_rates, elements_from_state,
    averaged_elements, evaluate_access, evaluate_repeat, execute_maneuver,
    ground_point, maneuver_schedule, mean_matched_initial_state,
    state_from_elements, wrap,
)
from reconfig_driftANDwaitTime_multicore import (
    AnalyticRaanPhasingPlanner, Orbit, MU_EARTH, R_EARTH, J2, j2_rates,
)

ROOT = Path(__file__).resolve().parent


def atomic_text(path, text):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


def atomic_csv(path, records):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        if records:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    os.replace(temporary, path)


def solve_scenario(config):
    gravity = Gravity(**config['gravity'])
    for key in ['repeat_revolutions', 'repeat_earth_rotations']:
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer.')
    # The corrected nominal planner uses repository constants. Do not silently
    # compare it with a different gravity model or a different planet.
    if (gravity.mu, gravity.radius_km, gravity.j2) != (MU_EARTH, R_EARTH, J2):
        raise ValueError('This runner requires the same Earth constants as the nominal planner.')
    inclination = radians(config['inclination_deg'])
    ratio = config['repeat_revolutions'] / config['repeat_earth_rotations']
    earth_rate = config['earth_rotation_rate_rad_s']

    def repeat_residual(a):
        node, periapsis, mean = j2_rates(a, 0.0, inclination)
        return (periapsis + mean) / (earth_rate - node) - ratio

    a_r = brentq(repeat_residual, R_EARTH + 100, R_EARTH + 3000, xtol=1e-9)
    a_g = R_EARTH + config['global_altitude_km']
    a_d = a_g + config['drift_offset_km']
    if min(a_g, a_d, a_r) <= R_EARTH:
        raise ValueError('This Earth scenario contains a radius inside the reference surface.')
    initial_elements = np.array([
        a_g, 0, 0, inclination, radians(config['node_deg']), radians(config['phase_deg'])])
    epoch = datetime.fromisoformat(config['epoch_utc'])
    planner = AnalyticRaanPhasingPlanner(
        Orbit(a_g, inclination, initial_elements[4], initial_elements[5]),
        a_r, config['drift_offset_km'], epoch,
        config['target_latitude_deg'], config['target_longitude_deg'],
        n_jobs=1, wait_time_cap_hours=config['wait_cap_hours'],
        drift_time_cap_days=config['drift_cap_hours'] / 24,
        response_horizon_hours=config['response_horizon_hours'],
        prune_phase_wraps=True)
    solution = planner.solve_min_time([config['longitude_tolerance_deg']])
    if not solution['success']:
        raise RuntimeError(f'Nominal planner found no accepted maneuver: {solution.get("note")}')
    schedule = maneuver_schedule(a_g, a_d, a_r, solution['T_wait'], solution['T_d'], gravity.mu)
    if abs(schedule.predicted_access_s - solution['total_time_s']) > 1e-6:
        raise RuntimeError('The Cartesian burn schedule does not match the planner timing.')
    if abs(schedule.insertion_s - solution['insertion_time_s']) > 1e-6:
        raise RuntimeError('The Cartesian schedule disagrees with the insertion definition.')
    _, gamma = circular_secular_rates(a_r, inclination, gravity)
    repeat_s = 2 * pi * config['repeat_revolutions'] / gamma
    intervals = float(config['post_access_repeat_intervals'])
    if intervals < 1.1:
        raise ValueError('At least 1.1 repeat intervals are needed for node-indexed closure.')
    end_s = schedule.predicted_access_s + intervals * repeat_s
    if config['initialization'] == 'numerical_mean':
        initial_state, initialization = mean_matched_initial_state(initial_elements, gravity)
    elif config['initialization'] == 'osculating':
        initial_state = state_from_elements(initial_elements, gravity.mu)
        initialization = dict(mode='direct_osculating',
                              note='Planner mean elements used as osculating elements; initialization mismatch is not removed.')
    else:
        raise ValueError('Initialization must be numerical_mean or osculating.')
    return gravity, epoch, planner, solution, schedule, repeat_s, end_s, initial_state, initialization


def numerical_evidence(config, gravity, epoch, schedule, repeat_s, trajectory, a_r):
    access = evaluate_access(trajectory, schedule, epoch,
                             config['target_latitude_deg'], config['target_longitude_deg'],
                             config['longitude_tolerance_deg'])
    terminal = elements_from_state(trajectory.state(schedule.insertion_s), gravity.mu)
    mean_terminal = averaged_elements(trajectory.state(schedule.insertion_s), gravity,
                                      repeat_s / config['repeat_revolutions'])
    return dict(
        access=access,
        repeat=evaluate_repeat(trajectory, schedule, epoch,
                               config['repeat_revolutions'], repeat_s),
        terminal=dict(
            osculating_elements=terminal.tolist(),
            numerical_mean_elements=mean_terminal.tolist(),
            numerical_mean_a_error_m=float(1000 * (mean_terminal[0] - a_r)),
            numerical_mean_eccentricity=float(np.hypot(mean_terminal[1], mean_terminal[2])),
            numerical_mean_inclination_error_deg=float(np.rad2deg(mean_terminal[3]) - config['inclination_deg']),
            note='Mean is the stated one-orbit numerical average of the postburn free orbit, not a formal theory-specific mean.'),
        burn_count=len(trajectory.burn_records),
        total_commanded_delta_v_m_s=sum(abs(b['signed_transverse_delta_v_m_s']) for b in trajectory.burn_records),
        maximum_burn_position_jump_m=max((b['position_jump_m'] for b in trajectory.burn_records), default=0),
        maximum_impulse_residual_m_s=max((b['impulse_residual_m_s'] for b in trajectory.burn_records), default=0))


def compare_integrations(first, second):
    times = np.unique(np.concatenate((np.linspace(0, first.end_s, 2001),
                                       [b['time_s'] for b in first.burn_records])))
    differences = np.array([first.state(t) - second.state(t) for t in times])
    result = dict(
        initial_state_identical=bool(np.array_equal(first.initial_state, second.initial_state)),
        compared_samples=len(times),
        maximum_position_difference_m=float(1000 * np.max(np.linalg.norm(differences[:, :3], axis=1))),
        maximum_velocity_difference_m_s=float(1000 * np.max(np.linalg.norm(differences[:, 3:], axis=1))),
        latitude_crossing_count_primary=len(first.latitude_crossings_s),
        latitude_crossing_count_tighter=len(second.latitude_crossings_s))
    if len(first.latitude_crossings_s) == len(second.latitude_crossings_s):
        result['maximum_latitude_crossing_time_difference_s'] = max(
            (abs(a - b) for a, b in zip(first.latitude_crossings_s, second.latitude_crossings_s)), default=0)
    return result


def _track(axis, points, *, fade_start=False, fade_end=False, **kwargs):
    values = np.asarray(points).copy()
    jumps = np.where(np.abs(np.diff(values[:, 1])) > 180)[0] + 1
    values[jumps] = np.nan
    if fade_start or fade_end:
        from matplotlib.collections import LineCollection
        from matplotlib.colors import to_rgba
        segments = np.stack((values[:-1, ::-1], values[1:, ::-1]), axis=1)
        fraction = (np.arange(len(segments)) + .5) / len(segments)
        opacity = np.ones(len(segments))
        if fade_start:
            opacity *= np.clip(fraction / .22, 0, 1)
        if fade_end:
            opacity *= np.clip((1 - fraction) / .22, 0, 1)
        colors = np.tile(to_rgba(kwargs.pop('color')), (len(segments), 1))
        colors[:, 3] *= opacity
        collection = LineCollection(segments, colors=colors, **kwargs)
        axis.add_collection(collection)
        return [collection]
    return axis.plot(values[:, 1], values[:, 0], **kwargs)


def _land(axis):
    from matplotlib.collections import PolyCollection
    data = json.loads((ROOT / 'reference/maps/ne_110m_land.geojson').read_text())
    polygons = []
    for feature in data['features']:
        geometry = feature['geometry']
        groups = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']
        polygons.extend(np.asarray(group[0]) for group in groups)
    axis.add_collection(PolyCollection(polygons, facecolors='#e9e9e9',
                                       edgecolors='#999999', linewidths=.35, zorder=0))
    axis.set(xlim=(-180, 180), ylim=(-85, 85), xlabel='Longitude [deg]', ylabel='Latitude [deg]')
    axis.set_xticks(np.arange(-180, 181, 60))
    axis.set_yticks(np.arange(-60, 61, 30))


def make_figure(config, epoch, planner, solution, schedule, trajectory, evidence, repeat_s, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                         'axes.titlesize': 11, 'svg.hashsalt': 'paper1-connected-validation'})
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10),
                                gridspec_kw={'height_ratios': [1.1, 1]})
    world, zoom, latitude_axis, longitude_axis = axes.flat
    _land(world)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    phase_colors = ['#0072B2', '#56B4E9', '#009E73', '#E69F00', '#8e5ea2']
    phases = [*schedule.phases,
              Phase(schedule.predicted_access_s,
                                       min(trajectory.end_s, schedule.predicted_access_s + 6500),
                                       'Achieved regional orbit')]
    step = config['plot_sample_step_s']
    omitted_intervals = []
    for index, phase in enumerate(phases):
        color = phase_colors[index % len(phase_colors)]
        windows = [(phase.start_s, phase.end_s)]
        if phase.end_s - phase.start_s > 14400:
            windows = [(phase.start_s, phase.start_s + 3600), (phase.end_s - 3600, phase.end_s)]
            omitted_start, omitted_end = windows[0][1], windows[1][0]
            omitted_intervals.append(
                f'{phase.label}: {(omitted_end-omitted_start)/3600:.2f} h omitted '
                f'(t = {omitted_start/3600:.2f}\u2013{omitted_end/3600:.2f} h)')
        for j, (lo, hi) in enumerate(windows):
            times = np.linspace(lo, hi, max(2, int((hi - lo) / step) + 1))
            points = [ground_point(trajectory.state(t), epoch, t) for t in times]
            _track(world, points, color=color, linewidth=1.3,
                   fade_start=len(windows) > 1 and j > 0,
                   fade_end=len(windows) > 1 and j < len(windows) - 1,
                   label=f'{phase.label} ({(phase.end_s-phase.start_s)/3600:.2f} h)' if j == 0 else None)
            if len(windows) > 1:
                cut_time, (cut_lat, cut_lon) = (hi, points[-1]) if j == 0 else (lo, points[0])
                world.annotate(f'{cut_time/3600:.2f} h', (cut_lon, cut_lat),
                               xytext=(-8, 12) if j == 0 else (8, 12),
                               ha='right' if j == 0 else 'left',
                               textcoords='offset points', fontsize=7, color=color,
                               arrowprops=dict(arrowstyle='-', color=color, alpha=.6, linewidth=.7),
                               bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1.5))
    start_lat, start_lon = ground_point(trajectory.state(0), epoch, 0)
    world.scatter(start_lon, start_lat, s=42, facecolors='white', edgecolors=phase_colors[0],
                  linewidths=1.5, zorder=7)
    world.annotate('Simulation start\n$t = 0$ h', (start_lon, start_lat),
                   xytext=(-12, -24), ha='right', textcoords='offset points', fontsize=8,
                   arrowprops=dict(arrowstyle='-', color=phase_colors[0], linewidth=.8),
                   bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2), zorder=8)
    if omitted_intervals:
        world.text(.5, .98, '\n'.join(omitted_intervals) + '\nFaded ends mark the omitted interval.',
                   transform=world.transAxes, ha='center', va='top', fontsize=7.5,
                   bbox=dict(facecolor='white', edgecolor='#cccccc', alpha=.95, pad=3), zorder=9)
    for index, burn in enumerate(trajectory.burn_records, 1):
        lat, lon = ground_point(np.array(burn['after_state']), epoch, burn['time_s'])
        world.scatter(lon, lat, s=25, color='black', marker='^', zorder=5)
        world.annotate(str(index), (lon, lat), xytext=(4, 5), textcoords='offset points', fontsize=8)
    target_lat, target_lon = config['target_latitude_deg'], config['target_longitude_deg']
    world.scatter(target_lon, target_lat, marker='*', s=110, color='#D55E00', edgecolors='black', linewidths=.5, zorder=6)
    world.set_title('(a) Connected execution; numbered impulses')
    world.legend(loc='lower left', fontsize=7.5, framealpha=.95)

    nearest = evidence['access']['nearest_crossing_to_prediction']
    nearest_time = nearest['time_s'] if nearest else schedule.predicted_access_s
    window_s = max(2.0, min(900.0, 4 * abs(nearest_time - schedule.predicted_access_s) + 1))
    plot_times = np.linspace(max(0, schedule.predicted_access_s - window_s),
                             min(trajectory.end_s, schedule.predicted_access_s + window_s), 1201)
    actual = np.array([ground_point(trajectory.state(t), epoch, t) for t in plot_times])
    longitude_error = np.rad2deg(wrap(np.deg2rad(actual[:, 1] - target_lon)))
    latitude_error = actual[:, 0] - target_lat
    nominal_node = solution['Omega_f_rad']
    nominal_u = solution['u_f_rad']
    node_rate, phase_rate = circular_secular_rates(planner.rom_a, planner.i, Gravity(**config['gravity']))
    nominal = []
    for t in plot_times:
        dt = t - schedule.predicted_access_s
        u, node = nominal_u + phase_rate * dt, nominal_node + node_rate * dt
        direction = np.array([cos(node)*cos(u)-sin(node)*sin(u)*cos(planner.i),
                              sin(node)*cos(u)+cos(node)*sin(u)*cos(planner.i),
                              sin(u)*sin(planner.i)])
        nominal.append(ground_point(np.concatenate((planner.rom_a * direction, np.zeros(3))), epoch, t))
    nominal = np.asarray(nominal)
    nominal_lon_error = np.rad2deg(wrap(np.deg2rad(nominal[:, 1] - target_lon)))
    nominal_lat_error = nominal[:, 0] - target_lat
    time_unit_s = 1 if window_s <= 120 else 60
    time_unit_label = 's' if time_unit_s == 1 else 'min'
    delta_times = (plot_times - schedule.predicted_access_s) / time_unit_s
    tolerance = config['longitude_tolerance_deg']

    zoom.plot(longitude_error, latitude_error, color='#0072B2', label='Connected propagation')
    zoom.plot(nominal_lon_error, nominal_lat_error, '--', color='#777777', label='Nominal regional reference')
    zoom.plot([-tolerance, tolerance], [0, 0], color='#009E73', linewidth=5, alpha=.7, label='Accepted latitude-crossing band')
    zoom.scatter(0, 0, marker='*', s=110, color='#D55E00', edgecolors='black', linewidths=.5, zorder=6)
    at_prediction = evidence['access']['at_predicted_time']
    zoom.scatter(at_prediction['longitude_residual_deg'], at_prediction['latitude_residual_deg'],
                 marker='x', s=55, color='#0072B2', label='Propagated at predicted time', zorder=7)
    zoom.scatter(solution['lon_err_deg'], 0, marker='D', s=35, facecolors='white', edgecolors='#444444', label='Nominal access', zorder=7)
    if nearest:
        color = '#009E73' if nearest['accepted'] else '#D55E00'
        zoom.scatter(nearest['longitude_residual_deg'], nearest['latitude_residual_deg'],
                     marker='o', color=color, s=40, zorder=8,
                     label='Nearest latitude crossing' + ('' if nearest['eligible_after_insertion'] else ' (before insertion)'))
    zoom.set(xlabel='Longitude offset from target [deg]', ylabel='Latitude offset from target [deg]',
             title='(b) Access neighborhood; the green segment defines acceptance')
    relevant_longitudes = [0, -tolerance, tolerance, solution['lon_err_deg'],
                           at_prediction['longitude_residual_deg']]
    if nearest:
        relevant_longitudes.append(nearest['longitude_residual_deg'])
    padding = max(.01, .08 * (max(relevant_longitudes) - min(relevant_longitudes)))
    latitude_half_width = max(.03, 2.5 * abs(at_prediction['latitude_residual_deg']))
    zoom.set_xlim(min(relevant_longitudes) - padding, max(relevant_longitudes) + padding)
    zoom.set_ylim(-latitude_half_width, latitude_half_width)
    zoom.legend(loc='best', fontsize=7)

    for ax, actual_values, reference_values, label, title in [
        (latitude_axis, latitude_error, nominal_lat_error, 'Latitude residual [deg]', '(c) Latitude crossing near the prediction'),
        (longitude_axis, longitude_error, nominal_lon_error, 'Longitude residual [deg]', '(d) Longitude acceptance at the latitude crossing')]:
        ax.plot(delta_times, actual_values, color='#0072B2', label='Connected propagation')
        ax.plot(delta_times, reference_values, '--', color='#777777', label='Nominal reference')
        ax.axvline(0, color='black', linewidth=.9, label='Predicted access')
        if nearest:
            ax.axvline(nearest['time_from_prediction_s']/time_unit_s, color='#D55E00', linestyle=':',
                       label='Nearest latitude crossing')
        ax.set(xlabel=f'Time from predicted access [{time_unit_label}]', ylabel=label, title=title)
    latitude_axis.axhline(0, color='#444444', linewidth=.8)
    longitude_axis.axhspan(-tolerance, tolerance, color='#009E73', alpha=.12, label=f'±{tolerance:g} deg band')
    for value in (-tolerance, tolerance):
        longitude_axis.axhline(value, color='#009E73', linewidth=.8)
    longitude_axis.legend(loc='best', fontsize=7)
    figure.suptitle(f'{config["case_id"]}: continuous Cartesian two-body + J2 execution', fontsize=15, y=.985)
    nearest_text = (f'nearest crossing: Δt={nearest["time_from_prediction_s"]:+.3f} s, '
                    f'Δλ={nearest["longitude_residual_deg"]:+.4f} deg'
                    + ('' if nearest['eligible_after_insertion'] else ' (before regional insertion)')) if nearest else 'no latitude crossing found'
    first = evidence['access']['first_accepted_crossing']
    access_text = (f'first accepted access at {first["time_s"]/3600:.3f} h; '
                   f'prediction error {first["time_from_prediction_s"]:+.3f} s') if first else 'no accepted access in the declared post-insertion search window'
    repeat = evidence['repeat']
    repeat_text = (f'15-node closure: {repeat["node_surface_closure_km"]:.3f} km; '
                   f'repeat-time error {repeat["repeat_interval_error_s"]:+.3f} s') if repeat['status'] == 'measured' else 'repeat closure not available'
    footer = (f'Nominal response {schedule.predicted_access_s/3600:.3f} h; {nearest_text}.\n'
              f'{access_text.capitalize()}. {repeat_text}.\n'
              'Initial state: numerical mean matched. Fixed RTN-T impulses; no orbit resets or targeting corrections. '
              'Long coasts: first/last hour displayed, with faded ends at time gaps. Radial WGS84 surface coordinates.')
    figure.text(.055, .025, footer, fontsize=8.3, va='bottom', linespacing=1.5)
    figure.subplots_adjust(left=.065, right=.98, bottom=.18, top=.93, hspace=.36, wspace=.27)
    for extension in ['png', 'svg']:
        destination = output / f'validation_example.{extension}'
        temporary = output / f'.validation_example.tmp.{extension}'
        metadata = {'Date': None} if extension == 'svg' else None
        figure.savefig(temporary, dpi=180, metadata=metadata)
        os.replace(temporary, destination)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/paper1_validation_los_angeles.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/paper1/validation-los-angeles')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    gravity, epoch, planner, solution, schedule, repeat_s, end_s, initial_state, initialization = solve_scenario(config)
    settings = IntegrationSettings(**config['integration'])
    trajectory = execute_maneuver(initial_state, schedule, end_s, gravity,
                                  config['target_latitude_deg'], settings)
    evidence = numerical_evidence(config, gravity, epoch, schedule, repeat_s, trajectory, planner.rom_a)
    tighter_settings = IntegrationSettings(**config['convergence'])
    tighter = execute_maneuver(initial_state, schedule, end_s, gravity,
                               config['target_latitude_deg'], tighter_settings)
    convergence = compare_integrations(trajectory, tighter)
    tighter_access = evaluate_access(tighter, schedule, epoch, config['target_latitude_deg'],
                                     config['target_longitude_deg'], config['longitude_tolerance_deg'])
    convergence['accepted_status_agrees'] = evidence['access']['accepted'] == tighter_access['accepted']
    convergence['tighter_settings'] = asdict(tighter_settings)
    if (convergence['maximum_position_difference_m'] > .1 or
        convergence.get('maximum_latitude_crossing_time_difference_s', float('inf')) > .001 or
        not convergence['accepted_status_agrees']):
        raise RuntimeError(f'The worked example did not meet the declared numerical convergence gate: {convergence}')
    source_files = [Path(__file__), ROOT / 'reconfig_cartesian_validation.py',
                    ROOT / 'reconfig_driftANDwaitTime_multicore.py', ROOT / 'reconfig_verification.py',
                    ROOT / 'constants.py', args.config.resolve()]
    summary = dict(
        schema='paper1_connected_validation_v1', config=config,
        force_model='Cartesian two-body plus unnormalized J2; no atmosphere, SRP, third bodies, or finite thrust',
        coordinate_model='Fixed inertial axes; planner GMST with UT1 approximated by UTC; radial WGS84 surface subpoint',
        impulse_model='Fixed signed Hohmann magnitudes at nominal times, along instantaneous RTN transverse; no state resets or target corrections',
        units=dict(position='km', velocity='km/s', time='seconds from request', elements='[a_km, e_cos_omega, e_sin_omega, i_rad, node_rad, mean_u_rad]'),
        nominal=dict(a_g_km=planner.a_G, a_d_km=planner.a_D, a_r_km=planner.rom_a,
                     predicted_response_s=schedule.predicted_access_s,
                     departure_s=schedule.departure_s, insertion_s=schedule.insertion_s,
                     wait_s=solution['T_wait'], drift_s=solution['T_d'],
                     branch=solution['branch'], longitude_residual_deg=solution['lon_err_deg'],
                     nominal_repeat_interval_s=repeat_s),
        initialization=initialization, initial_cartesian_state=initial_state.tolist(),
        integration=asdict(settings), evidence=evidence, convergence=convergence,
        source_sha256={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name:
                       hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        versions=dict(python=platform.python_version(), **{p: version(p) for p in ['numpy', 'scipy', 'matplotlib']}),
        limitations=[
            'One case; neither a multi-case validation nor a high-fidelity operational result.',
            'The numerical mean convention is not an exact Brouwer-Lyddane/SGP4 mean-to-osculating conversion.',
            'Geographic events are isolated sign-changing latitude crossings; grazing and equatorial geometries require another event formulation.',
            'No trajectory correction has been fitted to the target or desired final orbit.'])
    make_figure(config, epoch, planner, solution, schedule, trajectory, evidence, repeat_s, output)
    atomic_csv(output / 'latitude_crossings.csv', evidence['access']['crossings'])
    atomic_text(output / 'burns.json', json.dumps(trajectory.burn_records, indent=2, allow_nan=False) + '\n')
    atomic_text(output / 'summary.json', json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(output), nominal=summary['nominal'],
                          access={k: v for k, v in evidence['access'].items() if k != 'crossings'},
                          terminal=evidence['terminal'], repeat=evidence['repeat'],
                          convergence=convergence), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
