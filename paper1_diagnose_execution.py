"""Small controlled diagnostics of numerical-mean initialization and free coasts.

Six targets are chosen by coordinates before inspecting their errors. Commands
are fixed; the one- and three-orbit initial averaging conventions are compared
without fitting target error. Every execution has a tighter integration check.
These diagnostics do not establish a theory-specific mean transformation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from math import pi
from pathlib import Path
import time

import numpy as np
import pandas as pd

from paper1_validate_grid import matched_event
from paper1_validate_maneuver import compare_integrations
from reconfig_cartesian_validation import (
    Gravity, IntegrationSettings, ManeuverSchedule, averaged_elements,
    circular_secular_rates, evaluate_access, execute_maneuver, maneuver_schedule,
    state_from_elements, wrap,
)


def match_mean(target, gravity, orbit_count):
    period = 2 * pi / circular_secular_rates(target[0], target[3], gravity)[1]
    candidate = target.copy()
    scale = np.array([target[0], 1, 1, 1, 1, 1])
    for iteration in range(12):
        state = state_from_elements(candidate, gravity.mu)
        measured = averaged_elements(state, gravity, orbit_count * period, samples=512 * orbit_count + 1)
        residual = measured - target
        residual[4:] = wrap(residual[4:])
        if np.max(abs(residual / scale)) < 1e-10:
            return state, dict(orbit_count=orbit_count, samples=512 * orbit_count + 1,
                               iterations=iteration + 1, averaging_period_s=orbit_count * period,
                               initial_state=state.tolist(), residual=residual.tolist())
        candidate -= residual
    raise RuntimeError('Numerical mean matching failed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.data_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / 'manifest.json').read_text())
    experiment = next(e for e in manifest['experiments'] if e['experiment_id'] == 'h_i_da_tol_700_50_100_0p25')
    context, rows = experiment['context'], []
    cfg = context['scenario']
    archive_path = source / 'experiments' / experiment['experiment_id'] / 'per_maneuver.csv'
    archive = pd.read_csv(archive_path)
    # Two edge rows, one central row, and opposing longitudes; no outcome filter.
    selected = archive[(archive.latitude_index.isin([0, 8, 17])) & (archive.longitude_index.isin([3, 12]))].copy()
    assert len(selected) == 6
    gravity = Gravity(**cfg['gravity'])
    epoch = datetime.fromisoformat(cfg['epoch_utc'])
    target = np.array([context['a_g_km'], 0, 0, np.deg2rad(cfg['inclination_deg']),
                       np.deg2rad(cfg['node_deg']), np.deg2rad(cfg['phase_deg'])])
    half_window = .5 * context['repeat_s'] / cfg['repeat_revolutions']
    settings = IntegrationSettings(**cfg['integration'])
    tighter = IntegrationSettings(**cfg['convergence'])
    initializations = {}
    boundary_rows, free_rows = [], []
    for count in (1, 3):
        state, info = match_mean(target, gravity, count)
        initializations[count] = info
        if count == 1:
            assert np.linalg.norm(state - np.asarray(context['initial_state'])) < 1e-8
        free_end = float(selected.wait_s.max())
        free_schedule = ManeuverSchedule([], [], 0, 0, None)
        free = execute_maneuver(state, free_schedule, free_end, gravity, 0, settings)
        free_tight = execute_maneuver(state, free_schedule, free_end, gravity, 0, tighter)
        free_check = compare_integrations(free, free_tight)
        assert free_check['maximum_position_difference_m'] < .1
        for record in selected.to_dict('records'):
            index = int(record['case_index'])
            schedule = maneuver_schedule(context['a_g_km'], context['a_d_km'], context['a_r_km'],
                                         record['wait_s'], record['drift_s'], gravity.mu)
            end = schedule.predicted_access_s + 1.2 * half_window
            start = time.perf_counter()
            trajectory = execute_maneuver(state, schedule, end, gravity, record['target_latitude_deg'], settings)
            check = execute_maneuver(state, schedule, end, gravity, record['target_latitude_deg'], tighter)
            convergence = compare_integrations(trajectory, check)
            events = []
            for traj in (trajectory, check):
                access = evaluate_access(traj, schedule, epoch, record['target_latitude_deg'],
                                         record['target_longitude_deg'], cfg['longitude_tolerance_deg'])
                event = matched_event(access, record['nominal_branch'], half_window)
                assert event is not None
                events.append(event)
            event, tight_event = events
            assert convergence['maximum_position_difference_m'] < .1
            assert convergence['maximum_latitude_crossing_time_difference_s'] < .001
            assert event['accepted'] == tight_event['accepted']
            row = dict(case_index=index, averaging_orbits=count,
                       target_latitude_deg=record['target_latitude_deg'],
                       target_longitude_deg=record['target_longitude_deg'],
                       nominal_response_h=schedule.predicted_access_s / 3600,
                       matched_time_error_s=event['time_from_prediction_s'],
                       matched_longitude_error_deg=float(np.rad2deg(wrap(np.deg2rad(
                           event['longitude_residual_deg'] - record['nominal_longitude_residual_deg'])))),
                       matched_longitude_target_offset_deg=event['longitude_residual_deg'],
                       matched_accepted=event['accepted'], convergence=convergence)
            if count == 1:
                assert abs(row['matched_longitude_error_deg'] - record['matched_longitude_model_error_deg']) < 1e-7
                assert abs(row['matched_time_error_s'] - record['matched_time_error_s']) < 1e-5
            rows.append(row)
            circular_points = [
                ('end_initial_wait', record['wait_s'], 'left', context['a_g_km']),
                ('after_first_transfer', schedule.burns[1].time_s, 'right', context['a_d_km']),
                ('end_drift', schedule.burns[2].time_s, 'left', context['a_d_km']),
                ('after_final_transfer', schedule.insertion_s, 'right', context['a_r_km'])]
            for label, t, side, desired_a in circular_points:
                period = 2 * pi / circular_secular_rates(desired_a, target[3], gravity)[1]
                mean = averaged_elements(trajectory.state(t, side=side), gravity, period)
                boundary_rows.append(dict(case_index=index, averaging_orbits=count, location=label,
                                          time_s=t, numerical_mean_a_error_m=1000 * (mean[0] - desired_a),
                                          numerical_mean_eccentricity=float(np.hypot(mean[1], mean[2]))))
            free_mean = averaged_elements(free.state(record['wait_s']), gravity,
                                           2 * pi / circular_secular_rates(target[0], target[3], gravity)[1])
            node_rate, phase_rate = circular_secular_rates(target[0], target[3], gravity)
            free_rows.append(dict(case_index=index, averaging_orbits=count, wait_h=record['wait_s'] / 3600,
                                  mean_a_error_m=1000 * (free_mean[0] - target[0]),
                                  mean_phase_error_deg=float(np.rad2deg(wrap(free_mean[5] - target[5] - phase_rate * record['wait_s']))),
                                  mean_node_error_deg=float(np.rad2deg(wrap(free_mean[4] - target[4] - node_rate * record['wait_s'])))))
            (out / f'case_{index:04d}_mean{count}.json').write_text(json.dumps(dict(
                result=row, schedule=asdict(schedule), burns=trajectory.burn_records,
                matched_event=event, tighter_matched_event=tight_event), indent=2) + '\n')
            print(json.dumps(dict(case_index=index, averaging_orbits=count,
                                  elapsed_s=round(time.perf_counter() - start, 2))), flush=True)
    frame = pd.DataFrame([{k:v for k,v in r.items() if k != 'convergence'} for r in rows])
    paired = frame[frame.averaging_orbits == 1].merge(frame[frame.averaging_orbits == 3], on='case_index', suffixes=('_one', '_three'))
    boundaries = pd.DataFrame(boundary_rows)
    summary = dict(
        targets=6, executed_cases=12, averaging_orbits=[1,3],
        maximum_longitude_change_deg=float(abs(paired.matched_longitude_error_deg_three - paired.matched_longitude_error_deg_one).max()),
        maximum_crossing_time_change_s=float(abs(paired.matched_time_error_s_three - paired.matched_time_error_s_one).max()),
        changed_acceptance_count=int((paired.matched_accepted_one != paired.matched_accepted_three).sum()),
        initial_position_difference_m=float(1000 * np.linalg.norm(np.array(initializations[1]['initial_state'])[:3] - np.array(initializations[3]['initial_state'])[:3])),
        initial_velocity_difference_m_s=float(1000 * np.linalg.norm(np.array(initializations[1]['initial_state'])[3:] - np.array(initializations[3]['initial_state'])[3:])),
        maximum_convergence_position_m=max(r['convergence']['maximum_position_difference_m'] for r in rows),
        maximum_convergence_event_time_s=max(r['convergence']['maximum_latitude_crossing_time_difference_s'] for r in rows),
        boundary_mean_a_absolute_errors_m={str(c):{label:float(group.numerical_mean_a_error_m.abs().mean())
            for label,group in boundaries[boundaries.averaging_orbits == c].groupby('location')} for c in (1,3)},
        initializations=initializations,
        scope='Six prescribed coordinates at one request state; sensitivity to averaging interval, not a theory-consistent transformation or a unique causal decomposition.',
        source_sha256={str(archive_path.relative_to(source)):hashlib.sha256(archive_path.read_bytes()).hexdigest()},
        code_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    frame.to_csv(out / 'execution_comparison.csv', index=False)
    boundaries.to_csv(out / 'circular_boundary_diagnostics.csv', index=False)
    pd.DataFrame(free_rows).to_csv(out / 'free_coast_diagnostics.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'initializations'}, indent=2))


if __name__ == '__main__':
    main()
