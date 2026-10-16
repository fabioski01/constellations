"""Checkpoint controlled propagation sweeps around the +100 km reference.

The physical worker is reused unchanged from paper1_validate_grid. A physical
context is prepared without requiring an arbitrarily chosen anchor target to
be nominally feasible. Identical requests may be reused only after checking
physical inputs, Cartesian initial state and the numerical source hashes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
import gzip
from importlib.metadata import version
import json
from math import pi, radians
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import paper1_validate_grid as grid_worker
from paper1_validate_grid import encode, sha256, target_grid
from paper1_validate_maneuver import atomic_text
from paper1_validation_report import summarize
from reconfig_cartesian_validation import Gravity, circular_secular_rates, mean_matched_initial_state
from reconfig_driftANDwaitTime_multicore import MU_EARTH, R_EARTH, J2, j2_rates

ROOT = Path(__file__).resolve().parent
_CONTEXTS = None
PHYSICAL_SOURCES = ['paper1_validate_grid.py', 'paper1_validate_maneuver.py',
                    'reconfig_cartesian_validation.py', 'reconfig_driftANDwaitTime_multicore.py',
                    'reconfig_verification.py', 'constants.py']
PHYSICAL_KEYS = ['epoch_utc', 'global_altitude_km', 'inclination_deg', 'node_deg', 'phase_deg',
                 'drift_offset_km', 'repeat_revolutions', 'repeat_earth_rotations',
                 'earth_rotation_rate_rad_s', 'longitude_tolerance_deg', 'wait_cap_hours',
                 'drift_cap_hours', 'response_horizon_hours', 'post_access_repeat_intervals',
                 'initialization', 'gravity', 'integration', 'convergence']


def regional_radius(scenario):
    inclination = radians(scenario['inclination_deg'])
    ratio = scenario['repeat_revolutions'] / scenario['repeat_earth_rotations']

    def residual(a):
        node, peri, mean = j2_rates(a, 0., inclination)
        return (peri + mean) / (scenario['earth_rotation_rate_rad_s'] - node) - ratio

    return brentq(residual, R_EARTH + 100, R_EARTH + 3000, xtol=1e-9)


def physical_signature(scenario):
    return {key: scenario[key] for key in PHYSICAL_KEYS}


def number_token(x):
    return format(float(x), '.12g').replace('-', 'm').replace('.', 'p').replace('+', '')


def build_experiments(config, base):
    reference = config['reference']
    experiments = {}

    def add(changes, membership, label=None):
        values = dict(reference, **changes)
        key = tuple(values[k] for k in reference)
        if key not in experiments:
            suffix = '_'.join(number_token(values[k]) for k in reference)
            identifier = 'h_i_da_tol_' + suffix
            scenario = dict(deepcopy(base), **values, case_id=identifier)
            scenario['scope'] = config['scope']
            experiments[key] = dict(experiment_id=identifier, scenario=scenario,
                                    sweeps=[], labels=[])
        experiment = experiments[key]
        if membership not in experiment['sweeps']:
            experiment['sweeps'].append(membership)
        if label and label not in experiment['labels']:
            experiment['labels'].append(label)

    # Reference first; all tolerance requests are freshly planned by the worker.
    add({}, 'tolerance', 'reference')
    for tolerance in config['tolerances_deg']:
        add(dict(longitude_tolerance_deg=tolerance), 'tolerance')
    for offset in config['drift_offsets_km']:
        add(dict(drift_offset_km=offset), 'drift_offset')
    if config.get('include_exact_regional_coincidence'):
        scenario = dict(base, **reference)
        offset = regional_radius(scenario) - (R_EARTH + reference['global_altitude_km'])
        add(dict(drift_offset_km=offset), 'drift_offset', 'exact_regional_coincidence')
    for orbit in config['orbit_cases']:
        add({key: orbit[key] for key in ['global_altitude_km', 'inclination_deg']},
            'orbit', orbit['label'])
    return list(experiments.values())


def prepare_context(scenario, grid, initial_cache):
    gravity = Gravity(**scenario['gravity'])
    if (gravity.mu, gravity.radius_km, gravity.j2) != (MU_EARTH, R_EARTH, J2):
        raise ValueError('Planner and propagator must use the same Earth constants.')
    if scenario['initialization'] != 'numerical_mean':
        raise ValueError('This controlled sweep retains numerical-mean initialization.')
    if not grid['check_convergence_every_case']:
        raise ValueError('Every propagated request requires a tighter integration.')
    if scenario['post_access_repeat_intervals'] < 1.1:
        raise ValueError('Post-access propagation must cover repeat closure.')
    for key in ['repeat_revolutions', 'repeat_earth_rotations']:
        if not isinstance(scenario[key], int) or scenario[key] <= 0:
            raise ValueError('Repeat integers must be positive.')
    datetime.fromisoformat(scenario['epoch_utc'])
    a_g = R_EARTH + scenario['global_altitude_km']
    a_d = a_g + scenario['drift_offset_km']
    a_r = regional_radius(scenario)
    if min(a_g, a_d, a_r) <= R_EARTH:
        raise ValueError('All nominal radii must lie above the reference surface.')
    mean = np.array([a_g, 0., 0., radians(scenario['inclination_deg']),
                     radians(scenario['node_deg']), radians(scenario['phase_deg'])])
    key = tuple(mean)
    if key not in initial_cache:
        state, initialization = mean_matched_initial_state(mean, gravity)
        initial_cache[key] = (state.tolist(), initialization)
    state, initialization = initial_cache[key]
    _, gamma = circular_secular_rates(a_r, mean[3], gravity)
    repeat_s = 2*pi*scenario['repeat_revolutions']/gamma
    return dict(grid_config=grid, scenario=scenario, a_g_km=a_g, a_d_km=a_d,
                a_r_km=a_r, repeat_s=repeat_s, initial_state=state,
                initialization=initialization)


def _initialize(contexts):
    global _CONTEXTS
    _CONTEXTS = contexts


def execute_request(item):
    identifier, target = item
    grid_worker._initialize(_CONTEXTS[identifier])
    record = grid_worker.execute_target(target)
    record['experiment_id'] = identifier
    return identifier, record


def read_source_records(path):
    """Read a previous single-grid or sensitivity campaign without mutating it."""
    manifest = json.loads((path/'manifest.json').read_text())
    if 'context' in manifest:
        entries = [(manifest['context'], path/'case_evidence.jsonl.gz')]
    else:
        entries = [(entry['context'], path/'experiments'/entry['experiment_id']/'case_evidence.jsonl.gz')
                   for entry in manifest['experiments']]
    for context, archive in entries:
        if not archive.exists():
            continue
        provenance = dict(archive=str(archive.relative_to(ROOT)), sha256=sha256(archive))
        with gzip.open(archive, 'rt') as handle:
            records = [json.loads(line) for line in handle]
        yield manifest, context, provenance, records


def reusable_records(paths, manifest):
    reusable = {}
    for path in paths:
        for old_manifest, context, provenance, records in read_source_records(path):
            if any(old_manifest['source_sha256'].get(name) != manifest['source_sha256'][name]
                   for name in PHYSICAL_SOURCES):
                raise ValueError('Numerical source changed; refusing reuse from '+str(path))
            signature = physical_signature(context['scenario'])
            for entry in manifest['experiments']:
                if signature != physical_signature(entry['context']['scenario']):
                    continue
                if not np.array_equal(context['initial_state'], entry['context']['initial_state']):
                    raise ValueError('Physical inputs agree but Cartesian initial states differ.')
                by_coordinate = {(r['row']['target_latitude_deg'], r['row']['target_longitude_deg']):r
                                 for r in records}
                for target in manifest['targets']:
                    key = (target['target_latitude_deg'], target['target_longitude_deg'])
                    if key not in by_coordinate:
                        continue
                    original = by_coordinate[key]
                    # A failed calculation is evidence to retain, never a reusable success.
                    if original['row']['status'] not in ['completed', 'nominal_infeasible']:
                        continue
                    record = deepcopy(original)
                    record['row'].update(target)
                    record['experiment_id'] = entry['experiment_id']
                    record['reused_from'] = dict(provenance, case_index=original['row']['case_index'])
                    reusable[(entry['experiment_id'], target['case_index'])] = record
    return reusable


def save_record(output, identifier, record):
    index = record['row']['case_index']
    path = output/'experiments'/identifier/'cases'/f'case_{index:04d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(path, encode(record)+'\n')
    checkpoint = dict(experiment_id=identifier, case_index=index,
                      path=str(path.relative_to(output)), sha256=sha256(path))
    with (output/'checkpoint.jsonl').open('a') as handle:
        handle.write(encode(checkpoint)+'\n'); handle.flush(); os.fsync(handle.fileno())


def finalize_experiment(output, experiment, targets):
    identifier = experiment['experiment_id']
    folder = output/'experiments'/identifier
    rows = []
    with (folder/'case_evidence.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as archive:
            for target in targets:
                content = (folder/'cases'/f'case_{target["case_index"]:04d}.json').read_bytes()
                archive.write(content)
                rows.append(json.loads(content)['row'])
    frame = pd.DataFrame(rows).sort_values('case_index')
    frame.to_csv(folder/'per_maneuver.csv', index=False, float_format='%.15g')
    result = dict(experiment_id=identifier, **summarize(frame))
    atomic_text(folder/'summary.json', encode(result)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/paper1_validation_sensitivity.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/paper1/validation-sensitivity-plus100')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reuse', type=Path, action='append', default=[])
    parser.add_argument('--only', nargs='+', help='Selected experiment identifiers for a separately saved refinement.')
    parser.add_argument('--n-longitude', type=int)
    parser.add_argument('--probe', action='store_true', help='Six geographic controls per setting; separate output required.')
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if args.n_longitude:
        config['n_longitude'] = args.n_longitude
    base_path = ROOT/config['scenario_template']
    experiments = build_experiments(config, json.loads(base_path.read_text()))
    if args.only:
        experiments = [entry for entry in experiments if entry['experiment_id'] in args.only]
        if len(experiments) != len(args.only):
            raise ValueError('Unknown or duplicated experiment selection.')
    targets = target_grid(config)
    if args.probe:
        choices = [0, config['n_longitude']//2, config['n_longitude']-1,
                   len(targets)//2, len(targets)//2+config['n_longitude']//2, len(targets)-1]
        targets = [targets[i] for i in choices]
    initial_cache = {}
    for experiment in experiments:
        experiment['context'] = prepare_context(experiment.pop('scenario'), config, initial_cache)
    sources = PHYSICAL_SOURCES + [Path(__file__).name, str(config_path.relative_to(ROOT)), str(base_path.relative_to(ROOT))]
    manifest = dict(schema='paper1_connected_sensitivity_v1', config=config,
                    probe=args.probe, requested_count=len(targets)*len(experiments),
                    target_count_per_experiment=len(targets), targets=targets, experiments=experiments,
                    source_sha256={name:sha256(ROOT/name) for name in sources},
                    versions=dict(python=platform.python_version(), **{p:version(p) for p in ['numpy','scipy','pandas']}),
                    scope=config['scope'],
                    acceptance=dict(
                        matched='Nearest same-branch latitude crossing within half a nominal regional nodal period; original longitude tolerance and crossing at or after actual scheduled insertion.',
                        longitude_only='Longitude tolerance at the matched event, without insertion eligibility.',
                        extended='Either crossing branch after insertion, through 1.15 nominal repeat intervals after predicted access; delayed access does not validate predicted response.',
                        denominators='Every requested coordinate is retained. Conditional rates use nominally accepted requests; descriptive errors use numerically completed requests.',
                        weighting='Primary coordinate fractions; spherical cell-area fractions are separate and clipped to the common latitude band.'),
                    limitations=['Fixed request epoch, mean initial node and phase; descriptive deterministic grid statistics.',
                                 'Numerical averaging convention, central gravity plus J2 and ideal fixed RTN transverse impulses; no correction or added force.',
                                 'Five orbit configurations give separate altitude and inclination slices, not a full factorial interaction study.'])
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Existing output requires --resume with identical numerical sources and inputs.')
    else:
        atomic_text(manifest_path, encode(manifest)+'\n')
    completed = {}
    checkpoint_path = output/'checkpoint.jsonl'
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text().splitlines():
            item = json.loads(line); path = output/item['path']
            if sha256(path) != item['sha256']:
                raise ValueError('Checkpoint hash mismatch: '+str(path))
            record = json.loads(path.read_text())
            key = (item['experiment_id'], item['case_index'])
            if key in completed:
                raise ValueError('Duplicate checkpoint entry.')
            completed[key] = record['row']
    reused = 0
    for key, record in reusable_records([p.resolve() for p in args.reuse], manifest).items():
        if key not in completed:
            save_record(output, key[0], record); completed[key] = record['row']; reused += 1
    finalized = {}
    for experiment in experiments:
        identifier = experiment['experiment_id']
        if all((identifier, t['case_index']) in completed for t in targets):
            finalized[identifier] = finalize_experiment(output, experiment, targets)
    pending = [(entry['experiment_id'], target) for entry in experiments for target in targets
               if (entry['experiment_id'], target['case_index']) not in completed]
    print(encode(dict(requested=manifest['requested_count'], settings=len(experiments),
                      targets_per_setting=len(targets), retained=len(completed), reused=reused,
                      pending=len(pending), workers=args.workers)), flush=True)
    started = time.perf_counter()
    contexts = {entry['experiment_id']:entry['context'] for entry in experiments}
    by_id = {entry['experiment_id']:entry for entry in experiments}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize, initargs=(contexts,)) as pool:
        futures = {pool.submit(execute_request, item):item for item in pending}
        for future in as_completed(futures):
            identifier, record = future.result()
            save_record(output, identifier, record)
            completed[(identifier, record['row']['case_index'])] = record['row']
            experiment_count = sum(key[0] == identifier for key in completed)
            if experiment_count == len(targets):
                finalized[identifier] = finalize_experiment(output, by_id[identifier], targets)
                s = finalized[identifier]
                print(encode(dict(experiment_complete=identifier, counts=s['counts'],
                                  longitude_mae_deg=s['statistics']['matched_longitude_model_error_deg'].get('mean_absolute'),
                                  full_acceptance_percent=s['rates']['matched_access_accepted_percent_of_requested'])), flush=True)
            if len(completed)%36 == 0 or len(completed)==manifest['requested_count']:
                counts = {}
                for row in completed.values():
                    counts[row['status']] = counts.get(row['status'], 0)+1
                progress = dict(finished=len(completed), total=manifest['requested_count'],
                                settings_complete=len(finalized), elapsed_wall_s=round(time.perf_counter()-started,1), statuses=counts)
                atomic_text(output/'progress.json', encode(progress)+'\n')
                print(encode(progress), flush=True)
    rows = []
    for experiment in experiments:
        scenario = experiment['context']['scenario']
        for target in targets:
            row = dict(completed[(experiment['experiment_id'],target['case_index'])],
                       experiment_id=experiment['experiment_id'], sweeps=';'.join(experiment['sweeps']),
                       global_altitude_km=scenario['global_altitude_km'], inclination_deg=scenario['inclination_deg'],
                       drift_offset_km=scenario['drift_offset_km'], longitude_tolerance_deg=scenario['longitude_tolerance_deg'])
            rows.append(row)
    pd.DataFrame(rows).to_csv(output/'all_maneuvers.csv', index=False, float_format='%.15g')
    completion = dict(complete=len(completed)==manifest['requested_count'], requested=manifest['requested_count'],
                      finished=len(completed), settings_complete=len(finalized), reused_this_invocation=reused,
                      statuses=pd.DataFrame(rows).status.value_counts().to_dict())
    atomic_text(output/'completion.json', encode(completion)+'\n')
    atomic_text(output/'summary.json', encode(dict(completion=completion, experiments=finalized))+'\n')
    print(encode(completion), flush=True)
    if any(row['status'] in ['numerical_failure','convergence_failure'] for row in completed.values()):
        raise SystemExit('Numerical failures are retained and require investigation.')


if __name__ == '__main__':
    main()
