"""Physical controls for continuous propagation, impulses and access events."""
from datetime import datetime, timezone
from math import pi, radians
import unittest

import numpy as np

from reconfig_cartesian_validation import (
    Gravity, IntegrationSettings, averaged_elements, conserved_energy,
    elements_from_state, evaluate_access, evaluate_repeat, execute_maneuver,
    ground_point, maneuver_schedule, mean_matched_initial_state,
    state_from_elements,
)

EPOCH = datetime(2025, 5, 5, 12, tzinfo=timezone.utc)


class CartesianValidationTests(unittest.TestCase):
    def setUp(self):
        self.gravity = Gravity(j2=0)
        self.settings = IntegrationSettings(rtol=2e-12, max_step_s=30)
        self.initial = np.array([7000.0, 0, 0, radians(50), .3, .7])

    def test_nonsingular_element_roundtrip_including_retrograde(self):
        for inclination in [50, 98]:
            for ex, ey in [(0, 0), (.01, -.004)]:
                elements = np.array([7000, ex, ey, radians(inclination), -.7, 2.8])
                state = state_from_elements(elements, self.gravity.mu)
                recovered = elements_from_state(state, self.gravity.mu)
                np.testing.assert_allclose(recovered, elements, rtol=0, atol=3e-11)

    def test_two_body_hohmann_chain_matches_closed_form(self):
        g = self.gravity
        schedule = maneuver_schedule(7000, 7200, 7100, 600, 1200, g.mu)
        initial = state_from_elements(self.initial, g.mu)
        end_s = schedule.predicted_access_s
        trajectory = execute_maneuver(initial, schedule, end_s, g, 20, self.settings)
        phase = (.7 + np.sqrt(g.mu/7000**3)*600 + pi +
                 np.sqrt(g.mu/7200**3)*1200 + pi)
        expected = state_from_elements([7100, 0, 0, radians(50), .3, phase], g.mu)
        np.testing.assert_allclose(trajectory.state(end_s), expected, rtol=0, atol=3e-6)
        drift_state = np.array(trajectory.burn_records[1]['after_state'])
        drift = elements_from_state(drift_state, g.mu)
        self.assertAlmostEqual(drift[0], 7200, delta=2e-6)
        self.assertLess(np.hypot(drift[1], drift[2]), 1e-9)
        self.assertEqual(len(trajectory.burn_records), 4)
        self.assertEqual(max(b['position_jump_m'] for b in trajectory.burn_records), 0)
        self.assertLess(max(b['impulse_residual_m_s'] for b in trajectory.burn_records), 1e-11)

    def test_omitted_transfers_preserve_physical_event_definitions(self):
        first_omitted = maneuver_schedule(7000, 7000, 7100, 600, 1200, self.gravity.mu)
        self.assertEqual(first_omitted.departure_s, 1800)
        self.assertEqual(len(first_omitted.burns), 2)
        second_omitted = maneuver_schedule(7000, 7100, 7100, 600, 1200, self.gravity.mu)
        self.assertEqual(len(second_omitted.burns), 2)
        self.assertAlmostEqual(second_omitted.predicted_access_s - second_omitted.insertion_s, 1200)
        no_transfer = maneuver_schedule(7000, 7000, 7000, 600, 1200, self.gravity.mu)
        self.assertEqual(no_transfer.burns, [])
        self.assertEqual(no_transfer.insertion_s, 0)

    def test_zero_drift_coincident_burns_still_execute_two_transfers(self):
        schedule = maneuver_schedule(7000, 7200, 7100, 0, 0, self.gravity.mu)
        y = state_from_elements(self.initial, self.gravity.mu)
        trajectory = execute_maneuver(y, schedule, schedule.predicted_access_s,
                                      self.gravity, 20, self.settings)
        self.assertEqual(len(trajectory.burn_records), 4)
        self.assertEqual(trajectory.burn_records[1]['time_s'], trajectory.burn_records[2]['time_s'])
        elements = elements_from_state(trajectory.state(trajectory.end_s), self.gravity.mu)
        self.assertAlmostEqual(elements[0], 7100, delta=3e-6)
        self.assertLess(np.hypot(elements[1], elements[2]), 1e-9)

    def test_event_search_recovers_independent_circular_solution(self):
        g = self.gravity
        mean_motion = np.sqrt(g.mu/7000**3)
        period = 2*pi/mean_motion
        target_time = 1234.5
        target_state = state_from_elements([7000, 0, 0, radians(50), .3,
                                            .7 + mean_motion*target_time], g.mu)
        lat, lon = ground_point(target_state, EPOCH, target_time)
        schedule = maneuver_schedule(7000, 7000, 7000, 0, 0, g.mu)
        trajectory = execute_maneuver(state_from_elements(self.initial, g.mu), schedule,
                                      1.2*period, g, float(lat), self.settings)
        access = evaluate_access(trajectory, schedule, EPOCH, float(lat), float(lon), 1e-5)
        self.assertTrue(access['accepted'])
        self.assertAlmostEqual(access['first_accepted_crossing']['time_s'], target_time, delta=2e-6)
        failure = evaluate_access(trajectory, schedule, EPOCH, float(lat), float(lon+1), 1e-5)
        self.assertFalse(failure['accepted'])
        self.assertIsNone(failure['first_accepted_crossing'])
        self.assertIsNotNone(failure['nearest_crossing_to_prediction'])

    def test_j2_energy_and_axial_momentum_conservation(self):
        gravity = Gravity()
        schedule = maneuver_schedule(7000, 7000, 7000, 0, 0, gravity.mu)
        y = state_from_elements(self.initial, gravity.mu)
        trajectory = execute_maneuver(y, schedule, 7000, gravity, 20, self.settings)
        samples = [trajectory.state(t) for t in np.linspace(0, 7000, 101)]
        energy = np.array([conserved_energy(s, gravity) for s in samples])
        axial = np.array([np.cross(s[:3], s[3:])[2] for s in samples])
        self.assertLess(np.ptp(energy)/abs(energy[0]), 5e-11)
        self.assertLess(np.ptp(axial)/abs(axial[0]), 5e-11)

    def test_access_longitude_wraps_across_the_antimeridian(self):
        g = self.gravity
        mean_motion = np.sqrt(g.mu/7000**3)
        target_time = 1234.5
        elements = self.initial.copy()
        future = elements.copy()
        future[5] += mean_motion * target_time
        lat, lon = ground_point(state_from_elements(future, g.mu), EPOCH, target_time)
        elements[4] += radians(179.99 - lon)
        schedule = maneuver_schedule(7000, 7000, 7000, 0, 0, g.mu)
        trajectory = execute_maneuver(state_from_elements(elements, g.mu), schedule,
                                      1.2*2*pi/mean_motion, g, float(lat), self.settings)
        access = evaluate_access(trajectory, schedule, EPOCH, float(lat), -179.99, .025)
        self.assertTrue(access['accepted'])
        self.assertAlmostEqual(access['first_accepted_crossing']['longitude_residual_deg'], -.02, delta=1e-7)

    def test_numerical_mean_initialization_recovers_declared_mean(self):
        state, metadata = mean_matched_initial_state(self.initial, Gravity())
        independently_averaged = averaged_elements(state, Gravity(), metadata['averaging_period_s'])
        self.assertLess(abs(independently_averaged[0]-7000), 1e-5)
        self.assertLess(np.hypot(independently_averaged[1], independently_averaged[2]), 1e-9)
        self.assertLess(abs(independently_averaged[3]-self.initial[3]), 1e-9)
        state_zero_j2, _ = mean_matched_initial_state(self.initial, self.gravity)
        np.testing.assert_allclose(state_zero_j2, state_from_elements(self.initial, self.gravity.mu),
                                   rtol=0, atol=1e-6)

    def test_node_indexed_repeat_in_two_body_control(self):
        g = self.gravity
        period = 2*pi*np.sqrt(7000**3/g.mu)
        schedule = maneuver_schedule(7000, 7000, 7000, 0, 0, g.mu)
        trajectory = execute_maneuver(state_from_elements(self.initial, g.mu), schedule,
                                      3.2*period, g, 20, self.settings)
        repeat = evaluate_repeat(trajectory, schedule, EPOCH, 2, 2*period)
        self.assertEqual(repeat['status'], 'measured')
        self.assertAlmostEqual(repeat['achieved_repeat_interval_s'], 2*period, delta=2e-5)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            IntegrationSettings(max_step_s=0)
        with self.assertRaises(ValueError):
            maneuver_schedule(7000, 7200, 7100, -1, 0, self.gravity.mu)
        with self.assertRaises(ValueError):
            Gravity(mu=float('nan'))


if __name__ == '__main__':
    unittest.main()
