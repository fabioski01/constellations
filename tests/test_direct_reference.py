"""Physical and constraint checks for the independently searched direct strategy."""
import contextlib
from datetime import datetime, timezone
import io
from math import cos, pi, sin
import unittest

import numpy as np

from reconfig_direct_reference import solve_direct
from reconfig_driftANDwaitTime_multicore import AnalyticRaanPhasingPlanner, Orbit, j2_rates
from reconfig_cartesian_validation import Gravity, maneuver_schedule
from reconfig_verification import earth_angle_elapsed, WGS84_E2


class DirectReferenceTests(unittest.TestCase):
    def planner(self, latitude=-35, longitude=40, wait_cap=200, coast_cap=300):
        with contextlib.redirect_stdout(io.StringIO()):
            return AnalyticRaanPhasingPlanner(
                Orbit(7071, np.deg2rad(50), .2, .3), 6874.316367539429, 100,
                datetime(2025,5,5,12,tzinfo=timezone.utc), latitude, longitude,
                n_jobs=1, wait_time_cap_hours=wait_cap, drift_time_cap_days=coast_cap/24)

    def test_two_burns_and_forward_geography(self):
        planner=self.planner()
        solution=solve_direct(planner,.25)
        self.assertTrue(solution['success'])
        schedule=maneuver_schedule(planner.a_G,planner.rom_a,planner.rom_a,
                                   solution['T_wait'],solution['T_d'],Gravity().mu)
        self.assertEqual(len(schedule.burns),2)
        self.assertAlmostEqual(schedule.predicted_access_s,solution['total_time_s'],places=8)
        self.assertLessEqual(schedule.insertion_s,schedule.predicted_access_s)
        a_t=(planner.a_G+planner.rom_a)/2
        e_t=abs(planner.a_G-planner.rom_a)/(planner.a_G+planner.rom_a)
        phase=planner.u0;node=planner.Omega0
        for a,e,t in [(planner.a_G,0,solution['T_wait']),
                      (a_t,e_t,solution['T1']),(planner.rom_a,0,solution['T_d'])]:
            rate,w,m=j2_rates(a,e,planner.i)
            phase+=(w+m)*t;node+=rate*t
        x=cos(node)*cos(phase)-sin(node)*cos(planner.i)*sin(phase)
        y=sin(node)*cos(phase)+cos(node)*cos(planner.i)*sin(phase)
        z=sin(planner.i)*sin(phase)
        latitude=np.rad2deg(np.arctan2(z,(1-WGS84_E2)*np.hypot(x,y)))
        longitude=np.arctan2(y,x)-earth_angle_elapsed(planner.epoch,solution['total_time_s'])
        error=(longitude-planner.target_lon_rad+pi)%(2*pi)-pi
        self.assertAlmostEqual(latitude,planner.target_lat_deg,places=8)
        self.assertLessEqual(abs(np.rad2deg(error)),.25+1e-9)

    def test_larger_band_cannot_delay_reference(self):
        planner=self.planner(latitude=20,longitude=-100)
        narrow=solve_direct(planner,.01);wide=solve_direct(planner,2)
        self.assertTrue(narrow['success'] and wide['success'])
        self.assertLessEqual(wide['total_time_s'],narrow['total_time_s']+1e-5)

    def test_zero_coast_caps_preserve_infeasibility(self):
        solution=solve_direct(self.planner(wait_cap=0,coast_cap=0),.25)
        self.assertFalse(solution['success'])


if __name__=='__main__':
    unittest.main()
