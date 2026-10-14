"""Checks that analysis exports preserve failures and compare the same requests."""
import json
import math
import unittest

import pandas as pd

from paper1_analysis import check_tolerance_nesting, departure_comparison, run_request, summarize


class AnalysisExportTests(unittest.TestCase):
    def test_all_failed_group_keeps_schema_and_has_no_conditional_mean(self):
        task = dict(a_g_km=7071, a_r_km=6874.316367790103,
                    inclination_deg=50, node_deg=0, phase_deg=0,
                    offset_km=50, epoch_utc='2025-05-05T12:00:00+00:00',
                    target_latitude_deg=0, target_longitude_deg=0,
                    drift_cap_hours=4, wait_cap_hours=4, horizon_hours=-1,
                    policy='joint', study='altitude', case_id='E2', target_id=1,
                    tolerance_deg=.25, scan_discovery=160, scan_bracket=160,
                    longitude_neighbor_pad=8)
        row = run_request(task)
        self.assertFalse(row['success'])
        self.assertEqual(row['status'], 'numerical_failure')
        self.assertTrue(math.isnan(row['response_s']))
        total = summarize(pd.DataFrame([row])).iloc[0]
        self.assertEqual(total.accepted_count, 0)
        self.assertEqual(total.numerical_failure_count, 1)
        self.assertTrue(math.isnan(total.mean_response_s))

    def test_tolerance_failure_after_success_is_reported_and_serializable(self):
        common = dict(study='tolerance', policy='joint', case_id='E2',
                      target_id=1, offset_km=50)
        rows = [dict(common, tolerance_deg=.01, success=True, response_s=100, status='accepted'),
                dict(common, tolerance_deg=.25, success=False, response_s=float('nan'), status='infeasible')]
        violations = check_tolerance_nesting(pd.DataFrame(rows))
        self.assertEqual(len(violations), 1)
        json.dumps(violations)

    def test_budget_comparison_keeps_unmatched_targets_and_direct_coast(self):
        common = dict(study='altitude', case_id='E2', a_g_km=7071,
                      a_r_km=6871, success=True)
        rows = [dict(common, target_id=1, policy='joint', offset_km=50,
                     response_s=100, outbound_cost_m_s=90),
                dict(common, target_id=1, policy='immediate', offset_km=-200,
                     response_s=200, outbound_cost_m_s=100),
                dict(common, target_id=2, policy='joint', offset_km=50,
                     response_s=110, outbound_cost_m_s=90)]
        result = departure_comparison(pd.DataFrame(rows), 'E2', [110])
        self.assertEqual(len(result), 2)
        first, second = result.iloc[0], result.iloc[1]
        self.assertEqual(first.paired_time_saved_s, 100)
        self.assertEqual(first.direct_response_s, 200)
        self.assertFalse(second.both_feasible)
        self.assertTrue(math.isnan(second.paired_time_saved_s))


if __name__ == '__main__':
    unittest.main()
