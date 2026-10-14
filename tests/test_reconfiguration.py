"""Regression checks for scientific acceptance, time, and omitted-leg behavior."""
from datetime import datetime, timedelta, timezone
import json
import math
import unittest

import reconfig_driftANDwaitTime as serial
import reconfig_driftANDwaitTime_multicore as multicore
from reconfig_verification import (
    TargetGeometryError, earth_angle_elapsed, gmst_rad,
    reachable_geodetic_latitude_deg, select_branch_candidate, target_grid,
)

EPOCH = datetime(2025, 5, 5, 12, tzinfo=timezone.utc)
REGIONAL = 6874.316367790103


def planner(module=multicore, offset=50.0, latitude=0.0, longitude=0.0,
            wait_hours=4.0, drift_hours=4.0, horizon_hours=6.0, n_jobs=1):
    kw = {'n_jobs': n_jobs} if module is multicore else {}
    return module.AnalyticRaanPhasingPlanner(
        module.Orbit(7071.0, math.radians(50.0), 0.0, 0.0), REGIONAL, offset,
        EPOCH, latitude, longitude, drift_time_cap_days=drift_hours / 24.0,
        wait_time_cap_hours=wait_hours, response_horizon_hours=horizon_hours, **kw)


def manufactured_target(p, tw, td):
    """Ground point obtained by independently evaluating the declared sequence."""
    u = p.u0 + p.du_G * tw + p.du_T1 * p.T1 + p.du_D * td + p.du_T2 * p.T2
    node = p.Omega0 + p.dOm_G * tw + p.dOm_T1 * p.T1 + p.dOm_D * td + p.dOm_T2 * p.T2
    total = tw + p.T1 + td + p.T2
    phi = math.asin(math.sin(p.i) * math.sin(u))
    latitude = math.degrees(math.atan(math.tan(phi) / (1.0 - p.WGS84_e2))) if hasattr(p, 'WGS84_e2') else math.degrees(math.atan(math.tan(phi) / (1.0 - serial.WGS84_e2)))
    q = math.atan2(math.cos(p.i) * math.sin(u), math.cos(u))
    longitude = math.degrees(serial.wrap_pm_pi(node + q - earth_angle_elapsed(EPOCH, total)))
    return latitude, longitude


class TimeAndGeometryTests(unittest.TestCase):
    def test_fractional_seconds_and_timezone(self):
        expected = math.radians(360.98564736629) * 0.5 / 86400.0
        self.assertAlmostEqual(gmst_rad(EPOCH + timedelta(seconds=0.5)) - gmst_rad(EPOCH), expected, delta=1e-10)
        self.assertEqual(gmst_rad(EPOCH), gmst_rad(EPOCH.astimezone(timezone(timedelta(hours=2)))))
        self.assertGreater(serial.datetime_to_jd(EPOCH + timedelta(seconds=0.5)), serial.datetime_to_jd(EPOCH))

    def test_unwrapped_angle_continuity_and_consistency(self):
        values = [earth_angle_elapsed(EPOCH, t) for t in (0, 75000, 76000, 90000, 200000)]
        self.assertTrue(all(a < b for a, b in zip(values, values[1:])))
        for t in (0.5, 75726, 200000):
            residual = serial.wrap_pm_pi(earth_angle_elapsed(EPOCH, t) - gmst_rad(EPOCH + timedelta(seconds=t)))
            self.assertLess(abs(residual), 2e-10)

    def test_discontinuity_is_not_a_root(self):
        discontinuity = lambda t: -math.pi if t >= 0.5 else math.pi
        self.assertIsNone(multicore._bisection_root(discontinuity, 0.0, 1.0))
        self.assertAlmostEqual(multicore._bisection_root(lambda x: x - .25, 0.0, 1.0), .25)

    def test_invalid_and_unreachable_retrograde_latitude(self):
        for module in (serial, multicore):
            for lat, status in [(85.0, 'unreachable'), (97.0, 'invalid_coordinate')]:
                kw = {'n_jobs': 1} if module is multicore else {}
                p = module.AnalyticRaanPhasingPlanner(module.Orbit(7071, math.radians(98), 0, 0),
                    6939.753, 50, EPOCH, lat, 0, drift_time_cap_days=1, **kw)
                s = p.solve_min_time([.25])
                self.assertFalse(s['success']); self.assertEqual(s['status'], status)

    def test_reachable_grid_and_boundary(self):
        limit = reachable_geodetic_latitude_deg(98.0)
        self.assertAlmostEqual(limit, 82.052869, places=5)
        with self.assertRaises(ValueError):
            target_grid(98.0, 97.0, 120, 240)
        latitudes, longitudes = target_grid(98.0, 81.0, 3, 4)
        self.assertEqual(len(latitudes) * len(longitudes), 12)
        u1, u2 = serial.u_required_branches(limit, math.radians(98.0))
        self.assertAlmostEqual(u1, u2, delta=1e-7)

    def test_omitted_transfer_has_no_artificial_coast(self):
        for module in (serial, multicore):
            self.assertEqual(module.hohmann_half(7071.0, 7071.0), (7071.0, 0.0, 0.0))
            self.assertGreater(module.hohmann_half(7071.0, REGIONAL)[2], 0.0)


class SolverRegressionTests(unittest.TestCase):
    def assert_accepted(self, p, s, tolerance):
        self.assertTrue(s['success'], s)
        self.assertLessEqual(abs(s['lon_err_deg']), tolerance)
        self.assertLess(abs(s['lat_err_deg']), 1e-7)
        self.assertGreaterEqual(s['T_wait'], 0.0); self.assertGreaterEqual(s['T_d'], 0.0)
        self.assertAlmostEqual(s['total_time_s'], sum(s[k] for k in ('T_wait', 'T1', 'T_d', 'T2')), places=6)
        if p.response_horizon is not None:
            self.assertLessEqual(s['total_time_s'], p.response_horizon + 1e-8)

    def test_los_angeles_invalid_earlier_branch_cannot_win(self):
        p = planner(offset=0.0, latitude=34.0522, longitude=-118.2437,
                    wait_hours=200, drift_hours=24, horizon_hours=None)
        s = p.solve_min_time([.25])
        self.assert_accepted(p, s, .25)
        self.assertEqual(s['branch'], 'ascending')
        self.assertFalse(s['descending']['success'])
        self.assertEqual(s['T1'], 0.0)
        self.assertAlmostEqual(s['departure_time_s'], s['initial_coast_s'])
        json.dumps(s, default=str)  # No circular ascending/selected-result references.

    def test_near_misses_are_diagnostics(self):
        invalid = {'success': True, 'T_wait': 1, 'T_d': 0, 'total_time_s': 1,
                   'lon_err_deg': .883, 'lat_err_deg': 0, 'm_u_wrap': 0, 'k_lon_wrap': 0}
        s = select_branch_candidate([invalid], [.25], 'descending', 10, 10)
        self.assertFalse(s['success']); self.assertFalse(s['diagnostic']['success'])

    def test_zero_duration_endpoints_are_included(self):
        for module in (serial, multicore):
            seed = planner(module, wait_hours=0, drift_hours=0)
            lat, lon = manufactured_target(seed, 0, 0)
            p = planner(module, latitude=lat, longitude=lon, wait_hours=0, drift_hours=0)
            s = p.solve_min_time([.001]); self.assert_accepted(p, s, .001)
            self.assertEqual(s['T_wait'], 0.0); self.assertEqual(s['T_d'], 0.0)

    def test_regional_coast_follows_insertion(self):
        offset = REGIONAL - 7071.0
        seed = planner(offset=offset)
        lat, lon = manufactured_target(seed, 600.0, 3600.0)
        p = planner(offset=offset, latitude=lat, longitude=lon)
        s = p.solve_min_time([.001]); self.assert_accepted(p, s, .001)
        self.assertEqual(s['T2'], 0.0)
        self.assertGreater(s['regional_coast_s'], 0.0)
        self.assertLess(s['insertion_time_s'], s['access_time_s'])

    def test_deadline_excludes_all_transfers(self):
        p = planner(horizon_hours=0.0)
        self.assertFalse(p.solve_min_time([.25])['success'])

    def test_serial_parallel_and_tolerance_nesting(self):
        seed = planner()
        lat, lon = manufactured_target(seed, 600.0, 3600.0)
        solutions = []
        for module, n_jobs in [(serial, 1), (multicore, 1), (multicore, 2)]:
            p = planner(module, latitude=lat, longitude=lon, n_jobs=n_jobs)
            s = p.solve_min_time([.001]); self.assert_accepted(p, s, .001)
            solutions.append(s['total_time_s'])
        self.assertLess(max(solutions) - min(solutions), .01)
        p = planner(latitude=lat, longitude=lon)
        times = [p.solve_min_time([tol])['total_time_s'] for tol in (.001, .01, .25)]
        self.assertTrue(all(a + 1e-6 >= b for a, b in zip(times, times[1:])), times)

    def test_wider_bands_retain_earlier_boundary_candidates(self):
        # These cases previously selected the exact center after rejecting a
        # computed band edge by a few floating-point ulps.
        for module in (serial, multicore):
            p = planner(module, offset=50, latitude=-40, longitude=-120,
                        wait_hours=200, drift_hours=300, horizon_hours=72)
            previous = float('inf')
            for tolerance in (.001, .005, .01, .25, 1.0, 5.0):
                s = p.solve_min_time([tolerance])
                self.assert_accepted(p, s, tolerance)
                self.assertLessEqual(s['total_time_s'], previous + .01)
                previous = s['total_time_s']


if __name__ == '__main__':
    unittest.main()
