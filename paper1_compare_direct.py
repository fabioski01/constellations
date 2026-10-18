"""Compare the archived +100 km maneuver with an independent direct reference."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reconfig_direct_reference import solve_direct
from reconfig_driftANDwaitTime_multicore import AnalyticRaanPhasingPlanner, Orbit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.data_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / 'manifest.json').read_text())
    entries = {e['experiment_id']: e for e in manifest['experiments']}
    reference_id = 'h_i_da_tol_700_50_100_0p25'
    context = entries[reference_id]['context']
    direct_id = next(k for k, e in entries.items()
                     if abs(e['context']['a_d_km'] - context['a_r_km']) < 1e-8
                     and e['context']['scenario']['global_altitude_km'] == 700
                     and e['context']['scenario']['inclination_deg'] == 50)
    paths = [source / 'experiments' / tag / 'per_maneuver.csv' for tag in (reference_id, direct_id)]
    drift, direct = [pd.read_csv(path) for path in paths]
    cfg = context['scenario']
    direct_context = entries[direct_id]['context']
    matched_keys = ['epoch_utc', 'global_altitude_km', 'inclination_deg', 'node_deg',
                    'phase_deg', 'longitude_tolerance_deg', 'wait_cap_hours', 'drift_cap_hours',
                    'response_horizon_hours', 'gravity', 'integration', 'convergence', 'initialization']
    assert all(cfg[k] == direct_context['scenario'][k] for k in matched_keys)
    assert context['initial_state'] == direct_context['initial_state']
    independent = []
    with redirect_stdout(io.StringIO()):
        for row in direct.to_dict('records'):
            planner = AnalyticRaanPhasingPlanner(
                Orbit(context['a_g_km'], np.deg2rad(cfg['inclination_deg']),
                      np.deg2rad(cfg['node_deg']), np.deg2rad(cfg['phase_deg'])),
                context['a_r_km'], cfg['drift_offset_km'], datetime.fromisoformat(cfg['epoch_utc']),
                row['target_latitude_deg'], row['target_longitude_deg'], n_jobs=1,
                wait_time_cap_hours=cfg['wait_cap_hours'],
                drift_time_cap_days=cfg['drift_cap_hours'] / 24,
                response_horizon_hours=cfg['response_horizon_hours'])
            result = solve_direct(planner, cfg['longitude_tolerance_deg'])
            assert result['success']
            independent.append(dict(case_index=row['case_index'], **result))
    verified = pd.DataFrame(independent)
    max_difference = float(np.max(np.abs(verified.total_time_s.to_numpy() - direct.nominal_response_s.to_numpy())))
    assert max_difference < .01, max_difference
    keys = ['case_index', 'target_latitude_deg', 'target_longitude_deg']
    paired = drift.merge(direct, on=keys, suffixes=('_drift', '_direct'), validate='one_to_one')
    assert len(paired) == len(drift) == len(direct) == 324
    paired['direct_minus_drift_response_h'] = (paired.nominal_response_s_direct - paired.nominal_response_s_drift) / 3600
    difference = paired.direct_minus_drift_response_h
    summary = dict(
        requested=324, independent_direct_max_time_difference_s=max_difference,
        direct_mean_response_h=float(direct.nominal_response_s.mean() / 3600),
        drift_mean_response_h=float(drift.nominal_response_s.mean() / 3600),
        drift_earlier_count=int((difference > .01 / 3600).sum()),
        direct_earlier_count=int((difference < -.01 / 3600).sum()),
        tied_count=int((abs(difference) <= .01 / 3600).sum()),
        mean_direct_minus_drift_h=float(difference.mean()),
        median_direct_minus_drift_h=float(difference.median()),
        direct_delta_v_m_s=float(direct.total_commanded_delta_v_m_s.iloc[0]),
        drift_delta_v_m_s=float(drift.total_commanded_delta_v_m_s.iloc[0]),
        direct_full_acceptance_pct=float(100 * direct.matched_access_accepted.mean()),
        drift_full_acceptance_pct=float(100 * drift.matched_access_accepted.mean()),
        direct_longitude_mae_deg=float(direct.matched_longitude_model_error_deg.abs().mean()),
        drift_longitude_mae_deg=float(drift.matched_longitude_model_error_deg.abs().mean()),
        direct_extended_acceptance_pct=float(100 * direct.post_insertion_window_accepted.mean()),
        drift_extended_acceptance_pct=float(100 * drift.post_insertion_window_accepted.mean()),
        scope='One common deterministic target grid and request state; fixed +100 km is not an altitude-optimized strategy.',
        model='The direct reference is a separately searched two-impulse strategy under the same secular model and coast caps.',
        source_experiments=[reference_id, direct_id],
        source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [source / 'manifest.json', *paths]},
        code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [Path(__file__), Path(__file__).with_name('reconfig_direct_reference.py')]})
    verified.to_csv(out / 'independent_direct.csv', index=False)
    paired.to_csv(out / 'paired_comparison.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    make_figure(paired, out)
    print(json.dumps(summary, indent=2))


def make_figure(paired, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout='constrained')
    x = paired.nominal_response_s_direct / 3600
    y = paired.nominal_response_s_drift / 3600
    points = axes[0].scatter(x, y, c=paired.target_latitude_deg, s=17, cmap='coolwarm', vmin=-49, vmax=49)
    limit = max(x.max(), y.max()) * 1.04
    axes[0].plot([0, limit], [0, limit], '--', color='.45', lw=1)
    axes[0].set(xlabel='Direct-transfer response [h]', ylabel='+100 km drift response [h]',
                xlim=(0, limit), ylim=(0, limit), title='(a) Paired nominal responses')
    fig.colorbar(points, ax=axes[0], label='Target latitude [deg]', shrink=.85)
    values = np.sort(paired.direct_minus_drift_response_h)
    axes[1].step(values, np.arange(1, len(values) + 1) / len(values), where='post', color='#0072B2')
    axes[1].axvline(0, linestyle='--', color='.45', lw=1)
    axes[1].set(xlabel='Direct minus drift response [h]', ylabel='Fraction of requested targets',
                ylim=(0, 1.02), title='(b) Distribution of paired differences')
    for ax in axes:
        ax.grid(alpha=.2)
    for extension in ('png', 'svg'):
        fig.savefig(out / ('direct_comparison.' + extension), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
