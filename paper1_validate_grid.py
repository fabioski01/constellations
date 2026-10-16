"""Execute and checkpoint a small common-target Cartesian J2 validation grid.

The nominal planner, force model, initialization, impulse law and tolerances
are unchanged from the single-case validator. No maneuver is fitted to its
propagated target error. Report generation is in paper1_validation_report.py.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime
import gzip
import hashlib
from importlib.metadata import version
import io
import json
from math import radians
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np

from paper1_validate_maneuver import (
    atomic_text, compare_integrations, numerical_evidence, solve_scenario,
)
from reconfig_cartesian_validation import (
    Gravity, IntegrationSettings, evaluate_access, execute_maneuver,
    maneuver_schedule, surface_distance_km, wrap,
)
from reconfig_driftANDwaitTime_multicore import AnalyticRaanPhasingPlanner, Orbit

ROOT = Path(__file__).resolve().parent
_CONTEXT = None


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=json_default)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_grid(config):
    nlat, nlon = config['n_latitude'], config['n_longitude']
    if not isinstance(nlat, int) or not isinstance(nlon, int) or min(nlat, nlon) < 2:
        raise ValueError('Grid dimensions must be integers greater than one.')
    band = config['latitude_band_deg']
    if not 0 < band < 90:
        raise ValueError('The latitude band must lie strictly between 0 and 90 degrees.')
    latitudes = np.linspace(-band, band, nlat)
    longitudes = config['longitude_start_deg'] + np.arange(nlon) * 360 / nlon
    if np.any(longitudes < -180) or np.any(longitudes >= 180):
        raise ValueError('This periodic longitude grid must lie in [-180, 180).')
    edges = np.concatenate(([-band], (latitudes[:-1] + latitudes[1:]) / 2, [band]))
    areas = np.diff(np.sin(np.deg2rad(edges)))
    return [dict(case_index=i*nlon+j, latitude_index=i, longitude_index=j,
                 target_latitude_deg=float(lat), target_longitude_deg=float(lon),
                 area_weight=float(areas[i] / areas.sum() / nlon))
            for i, lat in enumerate(latitudes) for j, lon in enumerate(longitudes)]


def matched_event(access, nominal_branch, half_window_s):
    """Associate the predicted event by branch and nearest time, not by success."""
    candidates = [c for c in access['crossings']
                  if c['branch'] == nominal_branch and
                  abs(c['time_from_prediction_s']) <= half_window_s]
    return min(candidates, key=lambda c: abs(c['time_from_prediction_s'])) if candidates else None


def failure_category(matched):
    if matched is None:
        return 'no_matched_crossing'
    geographic = matched['normalized_longitude_error'] <= 1
    eligible = matched['eligible_after_insertion']
    if geographic and eligible:
        return 'accepted'
    if not geographic and not eligible:
        return 'longitude_and_before_insertion'
    return 'longitude_only' if not geographic else 'before_insertion_only'


def _initialize(context):
    global _CONTEXT
    _CONTEXT = context


def execute_target(target):
    context = _CONTEXT
    config = dict(context['scenario'],
                  target_latitude_deg=target['target_latitude_deg'],
                  target_longitude_deg=target['target_longitude_deg'])
    row = dict(target, status='numerical_failure', nominal_success=False,
               convergence_pass=False, matched_crossing_found=False,
               matched_longitude_within_tolerance=False,
               matched_access_accepted=False, post_insertion_window_accepted=False)
    start = time.perf_counter()
    try:
        gravity = Gravity(**config['gravity'])
        epoch = datetime.fromisoformat(config['epoch_utc'])
        inclination = radians(config['inclination_deg'])
        with redirect_stdout(io.StringIO()):
            planner = AnalyticRaanPhasingPlanner(
                Orbit(context['a_g_km'], inclination, radians(config['node_deg']), radians(config['phase_deg'])),
                context['a_r_km'], config['drift_offset_km'], epoch,
                target['target_latitude_deg'], target['target_longitude_deg'],
                n_jobs=1, wait_time_cap_hours=config['wait_cap_hours'],
                drift_time_cap_days=config['drift_cap_hours']/24,
                response_horizon_hours=config['response_horizon_hours'], prune_phase_wraps=True)
            solution = planner.solve_min_time([config['longitude_tolerance_deg']])
        row['nominal_success'] = bool(solution['success'])
        if not solution['success']:
            row.update(status='nominal_infeasible', failure_category='nominal_infeasible')
            return dict(row=row, nominal=solution)
        schedule = maneuver_schedule(context['a_g_km'], context['a_d_km'], context['a_r_km'],
                                     solution['T_wait'], solution['T_d'], gravity.mu)
        if abs(schedule.predicted_access_s - solution['total_time_s']) > 1e-6:
            raise RuntimeError('Nominal schedule and planner response differ.')
        row.update(nominal_branch=solution['branch'], m_u_wrap=solution['m_u_wrap'],
                   k_lon_wrap=solution['k_lon_wrap'], wait_s=solution['T_wait'],
                   drift_s=solution['T_d'], transfer1_s=solution['T1'], transfer2_s=solution['T2'],
                   nominal_response_s=schedule.predicted_access_s,
                   insertion_s=schedule.insertion_s,
                   nominal_longitude_residual_deg=solution['lon_err_deg'],
                   nominal_longitude_margin_deg=config['longitude_tolerance_deg']-abs(solution['lon_err_deg']))
        end_s = schedule.predicted_access_s + config['post_access_repeat_intervals'] * context['repeat_s']
        initial = np.asarray(context['initial_state'])
        trajectory = execute_maneuver(initial, schedule, end_s, gravity,
                                      target['target_latitude_deg'], IntegrationSettings(**config['integration']))
        evidence = numerical_evidence(config, gravity, epoch, schedule, context['repeat_s'], trajectory, context['a_r_km'])
        tighter = execute_maneuver(initial, schedule, end_s, gravity,
                                   target['target_latitude_deg'], IntegrationSettings(**config['convergence']))
        convergence = compare_integrations(trajectory, tighter)
        tighter_access = evaluate_access(tighter, schedule, epoch, target['target_latitude_deg'],
                                         target['target_longitude_deg'], config['longitude_tolerance_deg'])
        half_window = context['repeat_s']/config['repeat_revolutions'] * context['grid_config']['matched_event_half_window_nodal_periods']
        event = matched_event(evidence['access'], solution['branch'], half_window)
        tight_event = matched_event(tighter_access, solution['branch'], half_window)
        category = failure_category(event)
        convergence['post_window_status_agrees'] = evidence['access']['accepted'] == tighter_access['accepted']
        convergence['matched_status_agrees'] = category == failure_category(tight_event)
        convergence['passed'] = bool(
            convergence['maximum_position_difference_m'] <= .1 and
            convergence.get('maximum_latitude_crossing_time_difference_s', float('inf')) <= .001 and
            convergence['latitude_crossing_count_primary'] == convergence['latitude_crossing_count_tighter'] and
            convergence['post_window_status_agrees'] and convergence['matched_status_agrees'])
        row.update(convergence_pass=convergence['passed'],
                   convergence_maximum_position_difference_m=convergence['maximum_position_difference_m'],
                   convergence_maximum_crossing_time_difference_s=convergence.get('maximum_latitude_crossing_time_difference_s'),
                   propagation_end_s=end_s, failure_category=category,
                   matched_crossing_found=event is not None,
                   post_insertion_window_accepted=evidence['access']['accepted'],
                   force_evaluations_primary=trajectory.evaluations)
        first = evidence['access']['first_accepted_crossing']
        row['first_accepted_post_insertion_time_s'] = first['time_s'] if first else None
        row['first_accepted_post_insertion_delay_s'] = first['time_from_prediction_s'] if first else None
        prediction = evidence['access']['at_predicted_time']
        row.update(at_prediction_latitude_residual_deg=prediction['latitude_residual_deg'],
                   at_prediction_longitude_residual_deg=prediction['longitude_residual_deg'],
                   at_prediction_target_distance_km=prediction['spherical_ground_distance_km'])
        row['at_prediction_longitude_model_error_deg'] = float(np.rad2deg(wrap(
            np.deg2rad(prediction['longitude_residual_deg']-solution['lon_err_deg']))))
        row['at_prediction_surface_model_error_km'] = surface_distance_km(
            [prediction['latitude_deg'], prediction['longitude_deg']],
            [solution['latitude_f_deg'], solution['lambda_f_deg']])
        if event:
            model_error = float(np.rad2deg(wrap(np.deg2rad(event['longitude_residual_deg']-solution['lon_err_deg']))))
            row.update(matched_crossing_time_s=event['time_s'],
                       matched_time_error_s=event['time_from_prediction_s'],
                       matched_longitude_residual_deg=event['longitude_residual_deg'],
                       matched_longitude_model_error_deg=model_error,
                       matched_target_distance_km=surface_distance_km(
                           [event['latitude_deg'], event['longitude_deg']],
                           [target['target_latitude_deg'], target['target_longitude_deg']]),
                       matched_longitude_excess_deg=max(0., abs(event['longitude_residual_deg'])-config['longitude_tolerance_deg']),
                       matched_normalized_longitude_residual=event['normalized_longitude_error'],
                       matched_after_insertion=event['eligible_after_insertion'],
                       matched_longitude_within_tolerance=bool(event['normalized_longitude_error']<=1),
                       matched_access_accepted=event['accepted'])
        terminal = evidence['terminal']
        row.update(terminal_mean_a_error_m=terminal['numerical_mean_a_error_m'],
                   terminal_mean_eccentricity=terminal['numerical_mean_eccentricity'],
                   terminal_mean_inclination_error_deg=terminal['numerical_mean_inclination_error_deg'],
                   total_commanded_delta_v_m_s=evidence['total_commanded_delta_v_m_s'],
                   maximum_burn_position_jump_m=evidence['maximum_burn_position_jump_m'])
        repeat = evidence['repeat']
        row['repeat_status'] = repeat['status']
        if repeat['status'] == 'measured':
            row.update(repeat_node_closure_km=repeat['node_surface_closure_km'],
                       repeat_interval_error_s=repeat['repeat_interval_error_s'],
                       repeat_nominal_time_closure_km=repeat['scheduled_interval_surface_closure_km'])
        row['status'] = 'completed' if convergence['passed'] else 'convergence_failure'
        row['elapsed_wall_s'] = time.perf_counter()-start
        return dict(row=row, nominal=solution, evidence=evidence,
                    matched_event=event, convergence=convergence, burns=trajectory.burn_records)
    except Exception as exc:
        row.update(status='numerical_failure', failure_category='numerical_failure',
                   elapsed_wall_s=time.perf_counter()-start, error=str(exc))
        return dict(row=row, traceback=traceback.format_exc())


def prepare(config_path):
    grid = json.loads(config_path.read_text())
    if not grid['check_convergence_every_case']:
        raise ValueError('This campaign requires convergence checks for every case.')
    scenario_path = ROOT/grid['scenario_template']
    scenario = json.loads(scenario_path.read_text())
    with redirect_stdout(io.StringIO()):
        gravity, epoch, planner, _, _, repeat_s, _, initial, initialization = solve_scenario(scenario)
    targets = target_grid(grid)
    context = dict(grid_config=grid, scenario=scenario, a_g_km=planner.a_G,
                   a_d_km=planner.a_D, a_r_km=planner.rom_a, repeat_s=repeat_s,
                   initial_state=initial.tolist(), initialization=initialization)
    sources = [Path(__file__), ROOT/'paper1_validate_maneuver.py',
               ROOT/'reconfig_cartesian_validation.py', ROOT/'reconfig_driftANDwaitTime_multicore.py',
               ROOT/'reconfig_verification.py', ROOT/'constants.py', scenario_path, config_path]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    manifest = dict(schema='paper1_connected_grid_v1', context=context,
                    requested_count=len(targets), targets=targets, source_sha256=source_hashes,
                    versions=dict(python=platform.python_version(), **{p:version(p) for p in ['numpy','scipy','matplotlib']}),
                    acceptance=dict(
                        matched='Nearest crossing of the nominal branch within +/- half a nominal regional nodal period; original longitude band and time >= regional insertion.',
                        geographic_only='Original longitude band at the matched event, independent of whether insertion is complete.',
                        extended_window='First acceptable crossing of either branch after insertion through 1.15 nominal repeat intervals; later success does not validate the nominal response.',
                        model_error='Wrapped propagated minus nominal longitude at corresponding latitude crossings; distinct from longitude offset from the target.',
                        weights='Primary fractions count coordinates equally. Separate spherical cell weights are clipped at +/-49 degrees; neither is a demand distribution.'),
                    limitations=[
                        'One orbit configuration, longitude tolerance and request state; a deterministic coarse grid, not a random sample or a completed 360x180 validation.',
                        'Central gravity plus J2 only; fixed ideal Hohmann impulses, without targeting corrections or margins.',
                        'The one-orbit numerical mean convention is not an exact theory-specific mean-to-osculating transformation.',
                        'Radial WGS84 surface subpoints and common continuous GMST with UT1 approximated by UTC.',
                        'Convergence demonstrates numerical stability within this model, not operational truth.'])
    return context, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/paper1_validation_36x18.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/paper1/validation-36x18')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    (output/'cases').mkdir(parents=True, exist_ok=True)
    context, manifest = prepare(args.config.resolve())
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not args.resume:
            raise ValueError('Output exists; use a new directory or --resume.')
        if previous != manifest:
            raise ValueError('Manifest changed; a resume with different sources or inputs is forbidden.')
    else:
        atomic_text(manifest_path, encode(manifest)+'\n')
    index_path = output/'checkpoint.jsonl'
    checkpoints = [json.loads(s) for s in index_path.read_text().splitlines()] if index_path.exists() else []
    completed = {}
    for entry in checkpoints:
        path = output/entry['path']
        if sha256(path) != entry['sha256']:
            raise ValueError('Checkpoint checksum changed: '+str(path))
        completed[entry['case_index']] = json.loads(path.read_text())
    pending = [t for t in manifest['targets'] if t['case_index'] not in completed]
    started = time.perf_counter()
    print(encode(dict(requested=len(manifest['targets']), resumed=len(completed), pending=len(pending), workers=args.workers)), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize, initargs=(context,)) as executor:
        futures = {executor.submit(execute_target, t): t for t in pending}
        for future in as_completed(futures):
            result = future.result()
            index = result['row']['case_index']
            path = output/'cases'/f'case_{index:04d}.json'
            atomic_text(path, encode(result)+'\n')
            checkpoint = dict(case_index=index, path=str(path.relative_to(output)), sha256=sha256(path))
            with index_path.open('a', encoding='utf-8') as handle:
                handle.write(encode(checkpoint)+'\n')
                handle.flush()
                os.fsync(handle.fileno())
            completed[index] = result
            if len(completed)%18 == 0 or len(completed)==len(manifest['targets']):
                counts = {}
                for record in completed.values():
                    status = record['row']['status']
                    counts[status] = counts.get(status,0)+1
                print(encode(dict(finished=len(completed), total=len(manifest['targets']),
                                  elapsed_wall_s=round(time.perf_counter()-started,1), statuses=counts)), flush=True)
    complete = len(completed)==manifest['requested_count']
    completion = dict(complete=complete, requested_count=manifest['requested_count'],
                      finished_count=len(completed),
                      statuses={s:sum(r['row']['status']==s for r in completed.values())
                                for s in sorted({r['row']['status'] for r in completed.values()})})
    atomic_text(output/'completion.json', encode(completion)+'\n')
    # Deterministic compressed full evidence; individual files remain useful for resume.
    with (output/'case_evidence.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as archive:
            for index in sorted(completed):
                archive.write((encode(completed[index])+'\n').encode('utf-8'))
    from paper1_validation_report import write_report
    write_report(output, manifest, [completed[i]['row'] for i in sorted(completed)])
    print(encode(completion), flush=True)
    if not complete or any(r['row']['status'] in ['numerical_failure','convergence_failure'] for r in completed.values()):
        raise SystemExit('The campaign contains incomplete or numerically unresolved cases.')


if __name__ == '__main__':
    main()
