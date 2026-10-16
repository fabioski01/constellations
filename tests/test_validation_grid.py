"""Event association, failure denominators and grid-geometry controls."""
import unittest

import numpy as np
import pandas as pd

from paper1_validate_grid import failure_category, matched_event, target_grid
from paper1_validation_report import distribution, summarize


class ValidationGridTests(unittest.TestCase):
    def test_periodic_grid_has_all_coordinates_and_normalized_cell_areas(self):
        grid=target_grid(dict(n_latitude=18,n_longitude=36,latitude_band_deg=49,
                              longitude_start_deg=-180))
        self.assertEqual(len(grid),648)
        self.assertEqual(len({(r['target_latitude_deg'],r['target_longitude_deg']) for r in grid}),648)
        self.assertEqual(min(r['target_longitude_deg'] for r in grid),-180)
        self.assertEqual(max(r['target_longitude_deg'] for r in grid),170)
        self.assertAlmostEqual(sum(r['area_weight'] for r in grid),1)
        self.assertTrue(all(r['area_weight']>0 for r in grid))
        self.assertAlmostEqual(grid[0]['area_weight'],grid[-1]['area_weight'])

    def test_matching_keeps_branch_and_time_window_without_favoring_success(self):
        events=[dict(branch='descending',time_from_prediction_s=.01,accepted=True),
                dict(branch='ascending',time_from_prediction_s=-.1,accepted=False),
                dict(branch='ascending',time_from_prediction_s=6000,accepted=True)]
        self.assertIs(matched_event(dict(crossings=events),'ascending',3000),events[1])
        self.assertIsNone(matched_event(dict(crossings=events),'ascending',.05))

    def test_geographic_success_does_not_override_insertion(self):
        event=dict(normalized_longitude_error=1,eligible_after_insertion=False)
        self.assertEqual(failure_category(event),'before_insertion_only')
        event['eligible_after_insertion']=True
        self.assertEqual(failure_category(event),'accepted')
        event.update(normalized_longitude_error=1.000001,eligible_after_insertion=False)
        self.assertEqual(failure_category(event),'longitude_and_before_insertion')

    def test_unresolved_requests_remain_in_success_denominators(self):
        records=[]
        for status,nominal,accepted in [('completed',True,True),('completed',True,False),
                                        ('numerical_failure',True,False),('nominal_infeasible',False,False)]:
            records.append(dict(status=status,nominal_success=nominal,area_weight=.25,
                matched_crossing_found=status=='completed',matched_after_insertion=True,
                matched_longitude_within_tolerance=accepted,matched_access_accepted=accepted,
                post_insertion_window_accepted=accepted,failure_category='accepted' if accepted else 'longitude_only'))
        result=summarize(pd.DataFrame(records))
        self.assertEqual(result['counts']['requested'],4)
        self.assertEqual(result['counts']['nominal_accepted'],3)
        self.assertEqual(result['counts']['numerical_failures'],1)
        self.assertEqual(result['rates']['matched_access_accepted_percent_of_requested'],25)
        self.assertAlmostEqual(result['rates']['matched_access_accepted_percent_of_nominal_accepted'],100/3)

    def test_error_statistics_do_not_hide_cancellation(self):
        result=distribution([-2,2,np.nan])
        self.assertEqual(result['mean'],0)
        self.assertEqual(result['mean_absolute'],2)
        self.assertEqual(result['rms'],2)


if __name__=='__main__':unittest.main()
