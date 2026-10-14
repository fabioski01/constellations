"""Check the horizon bounds against oversized integer enumeration."""
import math
import unittest
import numpy as np
from reconfig_linear_reference import row_wrap_bounds, solve_point
from test_reconfiguration import EPOCH, REGIONAL, planner, manufactured_target


class LinearReferenceTests(unittest.TestCase):
    def test_row_bounds_include_random_feasible_coasts(self):
        rng = np.random.default_rng(20261014)
        a = np.array([[.0011, .0012], [-.000073, -.000074]])
        horizon = 72 * 3600.0
        for _ in range(100):
            tw, td = rng.dirichlet([1, 1, 1])[:2] * horizon
            wraps = rng.integers(-30, 30, 2)
            c = a @ [tw, td] - 2*math.pi*wraps
            bounds = row_wrap_bounds(a, c, horizon)
            for k, (lo, hi) in zip(wraps, bounds):
                self.assertLessEqual(lo, k); self.assertLessEqual(k, hi)

    def test_bounded_and_oversized_enumeration_agree(self):
        for offset in [-300.0, REGIONAL - 7071.0, 0.0, 50.0, 200.0]:
            for lat, lon in [(0.0, 0.0), (34.0522, -118.2437), (-40.0, 120.0)]:
                p = planner(offset=offset, latitude=lat, longitude=lon,
                            wait_hours=200, drift_hours=300, horizon_hours=72)
                bounded = solve_point(p, 72 * 3600)
                oversized = solve_point(p, 72 * 3600, wrap_padding=3)
                self.assertEqual(bounded['success'], oversized['success'])
                if bounded['success']:
                    self.assertAlmostEqual(bounded['total_time_s'], oversized['total_time_s'], places=6)
                    self.assertLessEqual(bounded['max_equation_residual_rad'], 1e-9)
                    self.assertLessEqual(bounded['total_time_s'], 72 * 3600 + 1e-8)

    def test_coincident_orbits_reduce_without_pseudoinverse(self):
        p = planner(offset=0.0, wait_hours=4, drift_hours=4, horizon_hours=6)
        u = p.u0 + p.du_G*600 + p.du_T2*p.T2
        node = p.Omega0 + p.dOm_G*600 + p.dOm_T2*p.T2
        from reconfig_verification import gmst_rad
        from reconfig_linear_reference import EARTH_OMEGA_RAD_S
        phi = math.asin(math.sin(p.i)*math.sin(u))
        lat = math.degrees(math.atan(math.tan(phi)/(1 - 0.0066943799901413165)))
        lon = math.degrees(node + math.atan2(math.cos(p.i)*math.sin(u), math.cos(u))
                           - gmst_rad(EPOCH) - EARTH_OMEGA_RAD_S*(600+p.T2))
        p.target_lat_deg, p.target_lon_rad = lat, math.radians(lon)
        s = solve_point(p, 6*3600)
        self.assertTrue(s['success'], s)
        self.assertAlmostEqual(s['T_wait'] + s['T_d'], 600, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
