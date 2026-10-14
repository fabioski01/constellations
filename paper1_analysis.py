"""Reproducible small runs and exports for the Earth wait-and-drift paper.

Run from the repository root. Results are nominal secular-model predictions;
continuous Cartesian maneuver and repeat-closure validation is a separate task.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from math import isfinite, radians, sqrt
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from reconfig_driftANDwaitTime_multicore import AnalyticRaanPhasingPlanner, Orbit, MU_EARTH, J2, R_EARTH
from reconfig_phasing_RGT import PhasingRGT
from reconfig_linear_reference import solve_point

OUTPUT_COLUMNS = {
    'total_time_s': 'response_s', 'T_wait': 'wait_s', 'T1': 'transfer1_s',
    'T_d': 'drift_s', 'T2': 'transfer2_s', 'insertion_time_s': 'insertion_s',
    'departure_time_s': 'departure_s', 'lon_err_deg': 'longitude_error_deg',
    'lat_err_deg': 'latitude_error_deg', 'm_u_wrap': 'phase_wrap',
    'k_lon_wrap': 'longitude_wrap',
}


def outbound_cost_m_s(a_g, a_d, a_r):
    def transfer(a, b):
        t = (a + b) / 2.0
        return abs(sqrt(MU_EARTH * (2/a - 1/t)) - sqrt(MU_EARTH/a)) + abs(sqrt(MU_EARTH/b) - sqrt(MU_EARTH * (2/b - 1/t)))
    return 1000.0 * (transfer(a_g, a_d) + transfer(a_d, a_r))


def run_request(task):
    row = dict(task)
    row.update({name: np.nan for name in OUTPUT_COLUMNS.values()})
    row.update(branch='', diagnostic_longitude_error_deg=np.nan,
               outbound_cost_m_s=np.nan)
    start = time.perf_counter()
    drift_hours = task['drift_cap_hours']
    if task['policy'] == 'immediate' and task['offset_km'] == 0.0:
        # With no first transfer, any initial drift is still delayed departure.
        drift_hours = 0.0
    try:
        p = AnalyticRaanPhasingPlanner(
            Orbit(task['a_g_km'], radians(task['inclination_deg']), radians(task['node_deg']), radians(task['phase_deg'])),
            task['a_r_km'], task['offset_km'], datetime.fromisoformat(task['epoch_utc']),
            task['target_latitude_deg'], task['target_longitude_deg'],
            drift_time_cap_days=drift_hours / 24.0, n_jobs=1,
            wait_time_cap_hours=0.0 if task['policy'] == 'immediate' else task['wait_cap_hours'],
            response_horizon_hours=task['horizon_hours'],
            scan_discovery=task['scan_discovery'], scan_bracket=task['scan_bracket'],
            longitude_neighbor_pad=task['longitude_neighbor_pad'])
        s = p.solve_min_time([task['tolerance_deg']])
        row.update(success=bool(s['success']), status=s.get('status', 'infeasible'),
                   failure_note=s.get('note', ''), outbound_cost_m_s=outbound_cost_m_s(p.a_G, p.a_D, p.rom_a))
        row.update({out: s.get(key, np.nan) for key, out in OUTPUT_COLUMNS.items()})
        row['branch'] = s.get('branch', '')
        diagnostic = [x.get('diagnostic') for x in (s.get('ascending', {}), s.get('descending', {}))]
        errors = [abs(x['lon_err_deg']) for x in diagnostic if x]
        row['diagnostic_longitude_error_deg'] = min(errors) if errors else np.nan
        if s['success'] and (abs(s['lon_err_deg']) > task['tolerance_deg'] or abs(s['lat_err_deg']) > 1e-7):
            raise RuntimeError('Final exported candidate violates geographic acceptance.')
        if s['success'] and task['policy'] == 'immediate' and abs(s['departure_time_s']) > 1e-8:
            raise RuntimeError('Immediate-departure candidate contains an initial coast.')
    except Exception as exc:
        row.update(success=False, status='numerical_failure', failure_note=f'{type(exc).__name__}: {exc}')
        row.update({name: np.nan for name in OUTPUT_COLUMNS.values()})
    row['runtime_s'] = time.perf_counter() - start
    return row


def summarize(records):
    keys = ['study', 'case_id', 'policy', 'offset_km', 'tolerance_deg']
    rows = []
    for values, group in records.groupby(keys, dropna=False, sort=True):
        accepted = group[group.success]
        row = dict(zip(keys, values))
        row.update(requested_count=len(group), accepted_count=len(accepted),
                   feasible_fraction=len(accepted) / len(group),
                   numerical_failure_count=int((group.status == 'numerical_failure').sum()))
        for key in ['response_s', 'wait_s', 'drift_s', 'outbound_cost_m_s']:
            row['mean_' + key] = float(accepted[key].mean()) if len(accepted) else np.nan
        for name, fn in [('median', np.median), ('p90', lambda x: np.quantile(x, .9)), ('maximum', np.max)]:
            row[name + '_response_s'] = float(fn(accepted.response_s)) if len(accepted) else np.nan
        row['maximum_accepted_longitude_error_deg'] = float(accepted.longitude_error_deg.abs().max()) if len(accepted) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def departure_comparison(records, case_id, budgets):
    records = records[(records.study == 'altitude') & (records.case_id == case_id)]
    rows = []
    for budget in budgets:
        for target in sorted(records.target_id.unique()):
            row = {'case_id': case_id, 'target_id': target, 'budget_m_s': budget}
            for policy in ['joint', 'immediate', 'direct']:
                valid = records[(records.target_id == target) & (records.policy == policy)
                                & records.success & (records.outbound_cost_m_s <= budget + 1e-9)]
                if policy == 'direct':
                    # Immediate G -> R, allowing a coast after regional insertion.
                    valid = records[(records.target_id == target) & (records.policy == 'immediate')
                                    & records.success & (records.outbound_cost_m_s <= budget + 1e-9)
                                    & ((records.offset_km - (records.a_r_km - records.a_g_km)).abs() < 1e-8)]
                if len(valid):
                    best = valid.loc[valid.response_s.idxmin()]
                    row[policy + '_response_s'] = best.response_s
                    row[policy + '_offset_km'] = best.offset_km
                else:
                    row[policy + '_response_s'] = np.nan
                    row[policy + '_offset_km'] = np.nan
            both = isfinite(row['joint_response_s']) and isfinite(row['immediate_response_s'])
            row['both_feasible'] = both
            row['paired_time_saved_s'] = row['immediate_response_s'] - row['joint_response_s'] if both else np.nan
            both_direct = isfinite(row['joint_response_s']) and isfinite(row['direct_response_s'])
            row['joint_direct_both_feasible'] = both_direct
            row['paired_direct_time_saved_s'] = row['direct_response_s'] - row['joint_response_s'] if both_direct else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def check_tolerance_nesting(records):
    errors = []
    data = records[(records.study == 'tolerance') & (records.policy == 'joint')]
    for keys, group in data.groupby(['case_id', 'target_id', 'offset_km']):
        previous = None
        for row in group.sort_values('tolerance_deg').itertuples():
            if previous is not None and (not row.success or row.response_s > previous + .01):
                errors.append({'case_id': str(keys[0]), 'target_id': int(keys[1]), 'offset_km': float(keys[2]),
                               'tolerance_deg': float(row.tolerance_deg), 'status': str(row.status)})
            if row.success:
                previous = row.response_s
    return errors


def reference_checks(tasks):
    rows = []
    for task in tasks:
        if task['study'] != 'altitude' or task['policy'] != 'joint' or task['case_id'] != 'E2':
            continue
        p = AnalyticRaanPhasingPlanner(Orbit(task['a_g_km'], radians(task['inclination_deg']),
            radians(task['node_deg']), radians(task['phase_deg'])),
            task['a_r_km'], task['offset_km'], datetime.fromisoformat(task['epoch_utc']),
            task['target_latitude_deg'], task['target_longitude_deg'],
            drift_time_cap_days=task['drift_cap_hours']/24, n_jobs=1,
            wait_time_cap_hours=task['wait_cap_hours'], response_horizon_hours=task['horizon_hours'])
        a = solve_point(p, task['horizon_hours']*3600)
        b = solve_point(p, task['horizon_hours']*3600, wrap_padding=3)
        difference = abs(a['total_time_s'] - b['total_time_s']) if a['success'] and b['success'] else np.nan
        agreement = a['success'] == b['success'] and (not a['success'] or difference < .001)
        rows.append({'target_id': task['target_id'], 'offset_km': task['offset_km'],
                     'model': 'linear_constant_rotation_exact_point', 'success': a['success'],
                     'bounded_enumeration_agrees': agreement, 'time_difference_s': difference,
                     'response_s': a.get('total_time_s', np.nan),
                     'equation_residual_rad': a.get('max_equation_residual_rad', np.nan),
                     'condition_number': a.get('condition_number', np.nan)})
    return pd.DataFrame(rows)


def plots(summary, paired, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for case, group in summary[(summary.study == 'altitude') & (summary.policy == 'joint')].groupby('case_id'):
        group = group.sort_values('offset_km')
        axes[0].plot(group.offset_km, group.mean_response_s / 3600, 'o-', label=case)
        axes[1].plot(group.offset_km, group.feasible_fraction, 'o-')
    axes[0].set_ylabel('Conditional mean response [h]'); axes[0].legend()
    axes[1].set_ylabel('Accepted fraction'); axes[1].set_xlabel('Drift offset [km]')
    axes[1].set_ylim(-.03, 1.03)
    for ax in axes: ax.grid(alpha=.25)
    figure.suptitle('Declared common-target smoke study, nominal model')
    figure.tight_layout(); figure.savefig(output/'altitude_smoke.png', dpi=180); plt.close(figure)
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for (case, offset), group in summary[(summary.study == 'tolerance') & (summary.policy == 'joint')].groupby(['case_id','offset_km']):
        group = group.sort_values('tolerance_deg')
        axes[0].semilogx(group.tolerance_deg, group.mean_response_s/3600, 'o-', label=f'{case}, {offset:g} km')
        axes[1].semilogx(group.tolerance_deg, group.feasible_fraction, 'o-')
    axes[0].set_ylabel('Conditional mean response [h]'); axes[0].legend(fontsize=8)
    axes[1].set_xlabel('Longitude tolerance [deg]'); axes[1].set_ylabel('Accepted fraction'); axes[1].set_ylim(-.03,1.03)
    for ax in axes: ax.grid(alpha=.25)
    figure.suptitle('Fixed targets and horizon, nominal longitude-band study')
    figure.tight_layout(); figure.savefig(output/'tolerance_smoke.png', dpi=180); plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10,4))
    for policy in ('joint', 'immediate', 'direct'):
        grouped = paired.groupby('budget_m_s')[policy+'_response_s']
        axes[0].plot(grouped.mean().index, grouped.mean()/3600, 'o-', label=policy)
        axes[1].plot(grouped.count().index, grouped.count()/paired.target_id.nunique(), 'o-', label=policy)
    axes[0].set_ylabel('Conditional mean response [h]'); axes[1].set_ylabel('Accepted fraction'); axes[1].set_ylim(-.03,1.03)
    for ax in axes: ax.set_xlabel('Outbound budget [m/s]'); ax.grid(alpha=.25); ax.legend()
    figure.suptitle('E2 paired policies over the same sampled offsets')
    figure.tight_layout(); figure.savefig(output/'departure_policies_smoke.png', dpi=180); plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/paper1_smoke.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.workers < 1: parser.error('--workers must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use an empty output directory to preserve existing runs.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks, cases = [], []
    for definition in config['cases']:
        case = dict(definition)
        a_r, ok, _ = PhasingRGT().solve_for_semi_major_axis(i=case['inclination_deg'], e=0.0,
            No=config['repeat_revolutions'], Nd=config['repeat_days'])
        if not ok: raise RuntimeError(f'Regional repeat root failed for {case["case_id"]}.')
        case.update(a_g_km=R_EARTH+case['global_altitude_km'], a_r_km=float(a_r))
        cases.append(case)
        offsets = sorted(set(float(a_r)-case['a_g_km'] if x == 'regional' else float(x) for x in config['offsets_km']))
        for target_id, (lat, lon) in enumerate(config['targets'],1):
            for policy in (['joint','immediate'] if case['case_id'] == config['departure_case_id'] else ['joint']):
                for offset in offsets:
                    tasks.append(dict(case, target_id=target_id, target_latitude_deg=lat, target_longitude_deg=lon,
                        study='altitude', policy=policy, offset_km=offset, tolerance_deg=config['tolerance_deg'],
                        epoch_utc=config['epoch_utc'], node_deg=config['node_deg'], phase_deg=config['phase_deg'],
                        wait_cap_hours=config['wait_cap_hours'], drift_cap_hours=config['drift_cap_hours'], horizon_hours=config['horizon_hours'],
                        scan_discovery=config['scan_discovery'], scan_bracket=config['scan_bracket'],
                        longitude_neighbor_pad=config['longitude_neighbor_pad']))
            if case['case_id'] in config.get('tolerance_cases',[]):
                for tol in config['tolerances_deg']:
                    tasks.append(dict(case, target_id=target_id, target_latitude_deg=lat, target_longitude_deg=lon,
                        study='tolerance', policy='joint', offset_km=config['tolerance_offset_km'], tolerance_deg=tol,
                        epoch_utc=config['epoch_utc'], node_deg=config['node_deg'], phase_deg=config['phase_deg'],
                        wait_cap_hours=config['wait_cap_hours'], drift_cap_hours=config['drift_cap_hours'], horizon_hours=config['horizon_hours'],
                        scan_discovery=config['scan_discovery'], scan_bracket=config['scan_bracket'],
                        longitude_neighbor_pad=config['longitude_neighbor_pad']))
    sources = ['constants.py','reconfig_driftANDwaitTime_multicore.py','reconfig_phasing_RGT.py',
               'reconfig_verification.py','reconfig_linear_reference.py','paper1_analysis.py']
    manifest = dict(config=config, actual_cases=cases, generated_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), workers=args.workers, requested_records=len(tasks),
        package_versions={name:version(name) for name in ('numpy', 'pandas', 'matplotlib')},
        constants=dict(mu_km3_s2=MU_EARTH,J2=J2,spherical_and_gravity_radius_km=R_EARTH),
        code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        scope='Small declared nominal-model study, not historical reproduction or continuous maneuver validation.')
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows = []
    if args.workers == 1:
        iterator = map(run_request, tasks)
        for i, row in enumerate(iterator,1):
            rows.append(row)
            if i % 25 == 0: print(f'Completed {i}/{len(tasks)} requests',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i,row in enumerate(pool.map(run_request,tasks,chunksize=1),1):
                rows.append(row)
                if i % 25 == 0: print(f'Completed {i}/{len(tasks)} requests',flush=True)
    records = pd.DataFrame(rows)
    records.to_csv(args.output_dir/'per_target.csv',index=False)
    summary = summarize(records); summary.to_csv(args.output_dir/'summary.csv',index=False)
    paired = departure_comparison(records,config['departure_case_id'],config['budgets_m_s'])
    paired.to_csv(args.output_dir/'departure_paired.csv',index=False)
    reference = reference_checks(tasks); reference.to_csv(args.output_dir/'linear_reference_checks.csv',index=False)
    violations = check_tolerance_nesting(records)
    checks = dict(requested=len(records),accepted=int(records.success.sum()),
                  numerical_failures=int((records.status=='numerical_failure').sum()),
                  tolerance_nesting_violations=violations,
                  reference_cases=len(reference), reference_disagreements=int((~reference.bounded_enumeration_agrees).sum()))
    (args.output_dir/'checks.json').write_text(json.dumps(checks,indent=2))
    plots(summary,paired,args.output_dir)
    print(json.dumps(checks),flush=True)
    if checks['numerical_failures'] or violations or checks['reference_disagreements']:
        raise SystemExit('The run exposed failed checks; inspect per_target.csv and checks.json.')


if __name__ == '__main__':
    main()
