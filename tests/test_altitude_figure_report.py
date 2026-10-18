"""Check that plot presentation preserves the evaluated domain and cohort."""
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper1_altitude_figure_report import study, validate_existing_summary
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


if __name__ == '__main__':
    unittest.main()
