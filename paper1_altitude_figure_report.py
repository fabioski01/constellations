"""Recover the nominal inputs for Figures 2 and 4 and plot explicit samples.

The original study contains 70 common-grid, 0.25-degree configurations.
An optional extension samples six further negative offsets for E2--E5,
bringing the total to 94. Compatible raw chunks can be reused after checking
all numerical inputs and core source hashes. This does not rerun the
historical implementation or Cartesian validation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper1_dense_analysis import atomic_json, chunk_worker, digest, summarize_experiment
from paper1_dense_checks import CASES
from reconfig_plotter_multicore import plot_saved_altitude_summary

ROOT = Path(__file__).resolve().parent
OFFSETS = [-200., -150., -100., -50., -10., -5., 0., 5., 10., 50., 100., 150., 200.]
EXTENDED_NEGATIVE_OFFSETS = [-500., -450., -400., -350., -300., -250.]
NUMERICAL_SOURCES = [
    'paper1_dense_solver.py', 'paper1_dense_checks.py', 'paper1_dense_analysis.py',
    'paper1_dense_audit.py', 'paper1_analysis.py',
    'reconfig_driftANDwaitTime_multicore.py', 'reconfig_verification.py',
    'reconfig_phasing_RGT.py', 'reconfig_linear_reference.py', 'constants.py',
    'paper1_altitude_figure_report.py',
]


def study(extend_negative_offsets=False):
    config = dict(epoch_utc='2025-05-05T12:00:00+00:00', node_deg=0., phase_deg=0.,
                  n_latitude=180, n_longitude=360, longitude_start_deg=-180.,
                  common_latitude_band_deg=49., wait_cap_hours=200.,
                  drift_cap_hours=300., latitude_chunk_size=180,
                  tolerance_deg=.25, matched_offsets_km=OFFSETS + ['regional'])
    cases = [dict(case, node_deg=0., phase_deg=0.) for case in CASES]
    if extend_negative_offsets:
        config['additional_offsets_by_case_km'] = {
            case['case_id']: ([] if case['case_id'] == 'E1' else EXTENDED_NEGATIVE_OFFSETS)
            for case in cases}
    experiments = []
    for case in cases:
        regional = case['a_r_km'] - case['a_g_km']
        extra = (EXTENDED_NEGATIVE_OFFSETS
                 if extend_negative_offsets and case['case_id'] != 'E1' else [])
        for offset in extra + OFFSETS + [regional]:
            slug = f'{offset:+.8f}'.replace('+', 'p').replace('-', 'm').replace('.', 'd')
            experiments.append(dict(experiment_id=f'common_{case["case_id"]}_{slug}',
                profile='common', case=case, offset_km=offset, n_lat=180, n_lon=360,
                latitude_band_deg=49., longitude_shift_deg=0.,
                tolerances_deg=[.25], original=False))
    return config, cases, experiments


def reuse_chunks(source, output, config, experiments, sources, completed):
    """Copy verified compatible whole-grid chunks, preserving their provenance."""
    prior = json.loads((source / 'manifest.json').read_text())
    for key in ('epoch_utc', 'node_deg', 'phase_deg', 'n_latitude', 'n_longitude',
                'longitude_start_deg', 'common_latitude_band_deg', 'wait_cap_hours',
                'drift_cap_hours', 'latitude_chunk_size', 'tolerance_deg'):
        if prior['config'][key] != config[key]:
            raise ValueError('Saved chunks use different inputs: ' + key)
    # The orchestration file changes when a study is extended; every numerical
    # dependency that actually produces or summarizes the arrays must match.
    for name, expected in sources.items():
        if name != Path(__file__).name and prior['source_sha256'].get(name) != expected:
            raise ValueError('Saved chunks use a different numerical source: ' + name)
    prior_experiments = {item['experiment_id']: item for item in prior['experiments']}
    indexed = {}
    for line in (source / 'checkpoint.jsonl').read_text().splitlines():
        item = json.loads(line)
        indexed.setdefault(item['experiment_id'], []).append(item)
    manifest_hash = digest(source / 'manifest.json')
    for experiment in experiments:
        identifier = experiment['experiment_id']
        if identifier in completed or identifier not in prior_experiments:
            continue
        if prior_experiments[identifier] != experiment:
            raise ValueError('Saved chunks use a different experiment: ' + identifier)
        entries = indexed.get(identifier, [])
        if (len(entries) != 1 or entries[0]['latitude_start'] != 0 or
                entries[0]['latitude_stop'] != experiment['n_lat']):
            raise ValueError('Saved experiment needs one complete grid chunk: ' + identifier)
        entry = entries[0]
        name = entry['relative_path']
        if Path(name).name != name:
            raise ValueError('Saved chunk path is not a basename.')
        original = source / 'per_target' / name
        if digest(original) != entry['sha256']:
            raise ValueError('A reused nominal chunk failed its checksum: ' + identifier)
        target = output / 'per_target' / name
        temporary = target.with_name(target.name + '.tmp')
        shutil.copyfile(original, temporary)
        if digest(temporary) != entry['sha256']:
            raise ValueError('A copied nominal chunk failed its checksum: ' + identifier)
        temporary.replace(target)
        yield dict(entry, reused_from=dict(directory=str(source),
                   manifest_sha256=manifest_hash, chunk_sha256=entry['sha256']))


def validate_existing_summary(summary, experiments):
    """Select only complete, unique common-grid records from a saved study."""
    rows = []
    for experiment in experiments:
        selected = summary[
            summary.profile.eq('common') & summary.version.eq('corrected') &
            summary.case_id.eq(experiment['case']['case_id']) &
            np.isclose(summary.offset_km, experiment['offset_km'], rtol=0, atol=1e-7) &
            np.isclose(summary.tolerance_deg, .25, rtol=0, atol=1e-12)]
        if len(selected) != 1:
            raise ValueError('Missing or duplicate saved record: ' + experiment['experiment_id'])
        row = selected.iloc[0]
        if (row.n_lat != 180 or row.n_lon != 360 or row.requested_count != 64800 or
                row.latitude_band_deg != 49 or row.longitude_shift_deg != 0):
            raise ValueError('Saved record uses a different target grid.')
        rows.append(row.to_dict())
    return pd.DataFrame(rows)


def plot_figures(summary, cases, output):
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
        'axes.titlesize': 10, 'legend.fontsize': 9, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'axes.spines.top': False,
        'axes.spines.right': False, 'savefig.facecolor': 'white',
        'svg.fonttype': 'none'})
    groups = [
        ('paper_figure_2_altitude', cases[:3],
         'Initial altitude comparison | inclination 50°\n'
         '64,800 common targets | longitude tolerance 0.25°'),
        ('paper_figure_4_inclination', [cases[index] for index in (1, 3, 4)],
         'Inclination comparison | initial altitude 700 km\n'
         '64,800 common targets | longitude tolerance 0.25°'),
    ]
    outputs = []
    for name, selected_cases, title in groups:
        figure = plot_saved_altitude_summary(summary, selected_cases, title=title)
        for extension in ('png', 'pdf', 'svg'):
            target = figures / (name + '.' + extension)
            temporary = target.with_name(target.name + '.tmp')
            figure.savefig(temporary, format=extension, dpi=250)
            temporary.replace(target)
            outputs.append(dict(path=str(target.relative_to(output)), sha256=digest(target)))
        plt.close(figure)
    offsets_by_case = {}
    for case in cases:
        selected = summary[summary.case_id.eq(case['case_id'])]
        regional = case['a_r_km'] - case['a_g_km']
        regular = selected[~np.isclose(selected.offset_km, regional, rtol=0, atol=1e-7)]
        offsets_by_case[case['case_id']] = sorted(regular.offset_km.unique().tolist())
    atomic_json(figures / 'figure_manifest.json', dict(
        numerical_summary_sha256=digest(output / 'summary.csv'),
        plot_source_sha256=digest(ROOT / 'reconfig_plotter_multicore.py'),
        outputs=outputs, common_regular_offsets_km=OFFSETS,
        regular_offsets_by_case_km=offsets_by_case,
        isolated_limits='Each exact a_D=a_R point is categorical in the right column.',
        scope='Nominal secular predictions; no Cartesian execution or interpolation.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--input-dir', type=Path,
                        help='Read a complete saved dense study instead of evaluating nominal cases.')
    inputs.add_argument('--reuse-dir', type=Path,
                        help='Reuse compatible saved raw chunks and evaluate only missing configurations.')
    parser.add_argument('--extend-negative-offsets', action='store_true',
                        help='Add -500 through -250 km at 50 km intervals for E2--E5.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--source-commit', default=None)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config, cases, experiments = study(args.extend_negative_offsets)
    sources = {name: digest(ROOT / name) for name in NUMERICAL_SOURCES}
    identity = dict(config=config, cases=cases, source_sha256=sources,
                    experiments=experiments)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        if not args.resume:
            raise ValueError('Output exists; use --resume or a new output directory.')
        old = json.loads(manifest_path.read_text())
        if any(old[key] != value for key, value in identity.items()):
            raise ValueError('Numerical inputs or sources changed; use a new output directory.')
    else:
        atomic_json(manifest_path, dict(identity,
            generated_utc=datetime.now(timezone.utc).isoformat(),
            source_commit=args.source_commit, python=platform.python_version(),
            numpy=np.__version__, pandas=pd.__version__,
            scope=('Common-grid nominal configurations for Figures 2 and 4; '
                   f'{len(experiments)} orbit-drift settings. No Cartesian execution.')))
    if args.input_dir:
        source = args.input_dir.resolve()
        prior = json.loads((source / 'manifest.json').read_text())
        for key in ('epoch_utc', 'node_deg', 'phase_deg', 'wait_cap_hours', 'drift_cap_hours'):
            if prior['config'][key] != config[key]:
                raise ValueError('Saved study uses different inputs: ' + key)
        saved_cases = {item['case']['case_id']: item['case'] for item in prior['experiments']}
        for case in cases:
            if any(saved_cases[case['case_id']][key] != case[key]
                   for key in ('a_g_km', 'a_r_km', 'inclination_deg')):
                raise ValueError('Saved study uses a different orbit: ' + case['case_id'])
        summary = validate_existing_summary(pd.read_csv(source / 'summary.csv'), experiments)
        atomic_json(output / 'input_provenance.json', dict(
            source_directory=str(source), source_manifest_sha256=digest(source / 'manifest.json'),
            source_summary_sha256=digest(source / 'summary.csv'),
            scope='Saved summary reuse; per-target audit remains in the source study.'))
    else:
        shards = output / 'per_target'
        shards.mkdir(exist_ok=True)
        checkpoint = output / 'checkpoint.jsonl'
        completed = {}
        if checkpoint.exists():
            for line in checkpoint.read_text().splitlines():
                item = json.loads(line)
                if digest(shards / item['relative_path']) != item['sha256']:
                    raise ValueError('A nominal checkpoint failed its checksum.')
                completed[item['experiment_id']] = item
        if args.reuse_dir:
            source = args.reuse_dir.resolve()
            with checkpoint.open('a', buffering=1) as stream:
                for item in reuse_chunks(source, output, config, experiments, sources, completed):
                    stream.write(json.dumps(item) + '\n')
                    stream.flush()
                    completed[item['experiment_id']] = item
            atomic_json(output / 'reuse_provenance.json', dict(
                source_directory=str(source), source_manifest_sha256=digest(source / 'manifest.json'),
                reused_experiments=[identifier for identifier, item in completed.items()
                                    if 'reused_from' in item],
                scope='Raw chunks reused only with identical numerical sources, inputs and checksums.'))
        tasks = [(experiment, 0, 180, config,
                  str(shards / (experiment['experiment_id'] + '_lat000.npz')))
                 for experiment in experiments if experiment['experiment_id'] not in completed]
        print(json.dumps(dict(configurations=len(experiments), pending=len(tasks),
                              requested_per_configuration=64800)), flush=True)
        with checkpoint.open('a', buffering=1) as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(chunk_worker, task) for task in tasks]
            for count, future in enumerate(as_completed(futures), 1):
                item = future.result()
                stream.write(json.dumps(item) + '\n')
                stream.flush()
                completed[item['experiment_id']] = item
                if count % 5 == 0 or count == len(tasks):
                    print(json.dumps(dict(completed=len(completed), total=len(experiments),
                                          latest_runtime_s=item['runtime_s'])), flush=True)
        records = []
        for experiment in experiments:
            entry = completed[experiment['experiment_id']]
            records.extend(summarize_experiment(experiment,
                [shards / entry['relative_path']], config))
        summary = pd.DataFrame(records)
        atomic_json(output / 'completion.json', dict(
            complete=len(completed) == len(experiments),
            finished_experiments=list(completed), total_experiments=len(experiments),
            finished_chunks=len(completed),
            requested_records=sum(e['n_lat'] * e['n_lon'] * len(e['tolerances_deg']) for e in experiments),
            bytes_in_raw_shards=sum(path.stat().st_size for path in shards.glob('*.npz'))))
    summary.to_csv(output / 'summary.csv', index=False)
    atomic_json(output / 'cases.json', cases)
    plot_figures(summary, cases, output)
    print(json.dumps(dict(output=str(output), summary_rows=len(summary), figures=2)), flush=True)


if __name__ == '__main__':
    main()
