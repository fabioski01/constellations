"""Check that plot presentation preserves the evaluated domain and cohort."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper1_altitude_figure_report import study, validate_existing_summary, reuse_chunks
from paper1_dense_analysis import digest
from reconfig_plotter_multicore import plot_saved_altitude_summary


class AltitudeFigureTests(unittest.TestCase):
    def setUp(self):
        _, self.cases, self.experiments = study()
        self.cases = self.cases[:3]
        records = []
        for case in self.cases:
            for offset in (-200., 0., 200., case['a_r_km'] - case['a_g_km']):
                records.append(dict(case_id=case['case_id'], version='corrected',
                    offset_km=offset, tolerance_deg=.25, reported_mean_response_h=20.,
                    reported_accepted_count=10, requested_count=10,
                    outbound_cost_m_s=100.))
        self.summary = pd.DataFrame(records)

    def tearDown(self):
        plt.close('all')

    def test_isolated_limit_does_not_expand_sweep_axis(self):
        figure = plot_saved_altitude_summary(self.summary, self.cases)
        sweep, limits = figure.axes[:2]
        self.assertGreater(sweep.get_xlim()[0], -250.)
        self.assertLess(sweep.get_xlim()[1], 250.)
        self.assertEqual(len(limits.collections), 3)
        self.assertEqual(limits.collections[2].get_offsets()[0, 0], 2.)
        labels = [item.get_text() for item in figure.axes[-1].get_xticklabels()]
        self.assertIn('E3\n-496.684', labels)

    def test_case_labels_and_markers_reveal_samples(self):
        figure = plot_saved_altitude_summary(self.summary, self.cases)
        lines = [line for line in figure.axes[0].lines if line.get_label().startswith('E')]
        self.assertEqual(len(lines), 3)
        self.assertEqual(len({line.get_marker() for line in lines}), 3)
        for line in lines:
            np.testing.assert_array_equal(line.get_xdata(), [-200., 0., 200.])
        labels = [label.get_text() for label in figure.legends[0].get_texts()]
        self.assertEqual(labels, ['E1: 500 km', 'E2: 700 km', 'E3: 1000 km'])
        self.assertFalse(any('corrected' in label for label in labels))

    def test_inclination_comparison_names_the_inclinations(self):
        _, cases, _ = study()
        cases = [cases[index] for index in (1, 3, 4)]
        rows = []
        for case in cases:
            for offset in (-200., 200., case['a_r_km'] - case['a_g_km']):
                rows.append(dict(self.summary.iloc[0], case_id=case['case_id'], offset_km=offset))
        figure = plot_saved_altitude_summary(pd.DataFrame(rows), cases)
        labels = [label.get_text() for label in figure.legends[0].get_texts()]
        self.assertEqual(labels, ['E2: 50°', 'E4: 60°', 'E5: 98°'])

    def test_comparison_keeps_historical_values_identifiable(self):
        original = self.summary.assign(version='original')
        figure = plot_saved_altitude_summary(pd.concat([self.summary, original]), self.cases)
        labels = [label.get_text() for label in figure.legends[0].get_texts()]
        self.assertEqual(len(labels), 6)
        self.assertEqual(sum('original reported' in label for label in labels), 3)

    def test_duplicate_grids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'single grid/profile'):
            plot_saved_altitude_summary(pd.concat([self.summary, self.summary.iloc[:1]]), self.cases)

    def test_saved_summary_reuse_requires_all_common_grid_records(self):
        rows = [dict(profile='common', version='corrected',
                     case_id=e['case']['case_id'], offset_km=e['offset_km'],
                     tolerance_deg=.25, n_lat=180, n_lon=360, requested_count=64800,
                     latitude_band_deg=49., longitude_shift_deg=0.) for e in self.experiments]
        saved = pd.DataFrame(rows)
        self.assertEqual(len(validate_existing_summary(saved, self.experiments)), 70)
        with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
            validate_existing_summary(saved.iloc[1:], self.experiments)
        saved.loc[0, 'n_lon'] = 18
        with self.assertRaisesRegex(ValueError, 'different target grid'):
            validate_existing_summary(saved, self.experiments)

    def test_extension_adds_only_the_requested_higher_orbit_offsets(self):
        _, cases, experiments = study(True)
        self.assertEqual(len(experiments), 94)
        base_ids = {e['experiment_id'] for e in self.experiments}
        additions = [e for e in experiments if e['experiment_id'] not in base_ids]
        self.assertEqual(len(additions), 24)
        self.assertEqual({e['case']['case_id'] for e in additions}, {'E2', 'E3', 'E4', 'E5'})
        for case in cases[1:]:
            self.assertEqual(sorted(e['offset_km'] for e in additions if e['case'] == case),
                             [-500., -450., -400., -350., -300., -250.])
        self.assertTrue(all(e['n_lat'] * e['n_lon'] == 64800 and e['tolerances_deg'] == [.25]
                            for e in additions))

    def test_extended_samples_appear_on_the_sweep(self):
        additions = pd.DataFrame([dict(self.summary.iloc[0], case_id=case['case_id'],
                                      offset_km=offset)
                                  for case in self.cases[1:]
                                  for offset in (-500., -400., -300.)])
        figure = plot_saved_altitude_summary(pd.concat([self.summary, additions]), self.cases)
        lines = [line for line in figure.axes[0].lines if line.get_label().startswith('E')]
        self.assertEqual(min(lines[0].get_xdata()), -200.)
        self.assertEqual(min(lines[1].get_xdata()), -500.)
        self.assertEqual(min(lines[2].get_xdata()), -500.)
        self.assertLess(figure.axes[0].get_xlim()[0], -500.)
        self.assertGreater(figure.axes[0].get_xlim()[1], 200.)

    def make_reuse_fixture(self, root):
        config, _, experiments = study()
        source, output = root / 'source', root / 'output'
        for directory in (source, output):
            (directory / 'per_target').mkdir(parents=True)
        experiment = experiments[0]
        path = source / 'per_target' / 'complete.npz'
        np.savez_compressed(path, latitude_deg=np.linspace(-49, 49, 180))
        sources = {'constants.py': 'same-core', 'paper1_altitude_figure_report.py': 'new-runner'}
        manifest = dict(config=config, experiments=[experiment],
                        source_sha256=dict(sources, **{'paper1_altitude_figure_report.py': 'old-runner'}))
        (source / 'manifest.json').write_text(json.dumps(manifest))
        checkpoint = dict(experiment_id=experiment['experiment_id'], latitude_start=0,
                          latitude_stop=180, relative_path=path.name, sha256=digest(path))
        (source / 'checkpoint.jsonl').write_text(json.dumps(checkpoint) + '\n')
        return source, output, config, [experiment], sources, path

    def test_compatible_raw_chunks_reuse_exact_bytes_with_provenance(self):
        with TemporaryDirectory() as directory:
            source, output, config, experiments, sources, path = self.make_reuse_fixture(Path(directory))
            records = list(reuse_chunks(source, output, config, experiments, sources, {}))
            self.assertEqual(len(records), 1)
            self.assertEqual(digest(output / 'per_target' / path.name), digest(path))
            self.assertEqual(records[0]['reused_from']['manifest_sha256'], digest(source / 'manifest.json'))
            completed = {records[0]['experiment_id']: records[0]}
            self.assertEqual(list(reuse_chunks(source, output, config, experiments, sources, completed)), [])

    def test_reuse_rejects_changed_physics_or_numerical_source(self):
        with TemporaryDirectory() as directory:
            source, output, config, experiments, sources, _ = self.make_reuse_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, 'different inputs: wait_cap_hours'):
                list(reuse_chunks(source, output, dict(config, wait_cap_hours=100.), experiments, sources, {}))
            with self.assertRaisesRegex(ValueError, 'different numerical source: constants.py'):
                list(reuse_chunks(source, output, config, experiments, dict(sources, **{'constants.py': 'changed'}), {}))

    def test_reuse_rejects_corrupted_raw_chunk(self):
        with TemporaryDirectory() as directory:
            source, output, config, experiments, sources, path = self.make_reuse_fixture(Path(directory))
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'failed its checksum'):
                list(reuse_chunks(source, output, config, experiments, sources, {}))
            self.assertFalse((output / 'per_target' / path.name).exists())


if __name__ == '__main__':
    unittest.main()
