"""Analytic J2 Reconfiguration Planner (single-core)

Phases: GOM wait → Hohmann (T1) → Drift → Hohmann (T2) → ROM pass.
Closed-form secular J2 propagation (Ω, ω, M) per segment; uses mean longitude
ℓ = Ω + ω + M for clean handling of circular portions. Enforces final pass
latitude and longitude simultaneously by solving for (T_wait, T_d) across
ascending & descending branches and wrap integers (m,k) with finite transfer
durations included.

Core Inputs (AnalyticRaanPhasingPlanner)
    gom (Orbit), rom_a_km, delta_a_km, epoch_gom (UTC), target_lat_geod_deg, target_lon_deg,
    max_wrap (legacy bounding), drift_time_cap_days (search cap)

Output Dict (success path)
    T_wait, T1, T_d, T2, total_time_s, branch, m_u_wrap, k_lon_wrap,
    lon_err_deg, Omega_f_rad, u_f_rad, final_orbit, selection_meta

Helper Exports
    build_maneuver_sequence_for_visualizer(...)  -> plotting package
    compute_reconfig_times_vs_delta_a(...)       -> Δa sweep timing arrays
    print_solution_summary(sol)                  -> human readable summary

See reconfig_driftANDwaitTime_multicore.py for multiprocessing variant.

Author
Fabio Meloni - TU Delft MSc Student - NASA Jet Propulsion Laboratory Visiting Researcher
"""
from __future__ import annotations
from dataclasses import dataclass
from math import (
    pi, sin, cos, tan, atan, atan2, asin, acos, sqrt, copysign,
    fmod, floor, isfinite
)
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any, List
import constants
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# ---------------------------
# Constants and small helpers
# ---------------------------

R_EARTH = constants.EARTH_RADIUS            # km (WGS-84 equatorial radius)
MU_EARTH = constants.GRAVITATIONAL_CONSTANT      # km^3/s^2 (Earth μ)
J2 = constants.J2_EARTH            # Earth's J2
WGS84_f = 1.0 / 298.257223563
WGS84_e2 = WGS84_f * (2.0 - WGS84_f)

DEG = pi / 180.0
RAD = 1.0

def rad2deg(x: float) -> float:
    return x / DEG

def deg2rad(x: float) -> float:
    return x * DEG

def wrap_2pi(x: float) -> float:
    """Wrap to [0, 2π)."""
    y = fmod(x, 2.0 * pi)
    return y + (2.0 * pi if y < 0.0 else 0.0)

def wrap_pm_pi(x: float) -> float:
    """Wrap to (-π, π]."""
    y = fmod(x + pi, 2.0 * pi)
    if y < 0.0:
        y += 2.0 * pi
    return y - pi

def clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))

def sign_nonzero(x: float) -> float:
    return 1.0 if x >= 0.0 else -1.0

# ---------------------------
# Time / GMST utilities
# ---------------------------

def datetime_to_jd(dt: datetime) -> float:
    """UTC datetime -> Julian Date (UT)."""
    # Algorithm from Vallado (consistent with SGP4)
    y = dt.year
    m = dt.month
    D = dt.day + (dt.hour + (dt.minute + dt.second / 60.0) / 60.0) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    A = floor(y / 100)
    B = 2 - A + floor(A / 4)
    JD = floor(365.25 * (y + 4716)) + floor(30.6001 * (m + 1)) + D + B - 1524.5
    return JD

def gmst_rad(dt: datetime) -> float:
    """
    IAU-82 style GMST at UT1≈UTC (sufficient for planning):
      GMST [deg] = 280.46061837 + 360.98564736629*(JD-2451545.0)
                   + 0.000387933*T^2 - T^3/38710000
      where T = (JD-2451545.0)/36525
    Return wrapped radians in [0,2π).
    """
    JD = datetime_to_jd(dt)
    D = JD - 2451545.0
    T = D / 36525.0
    gmst_deg = 280.46061837 + 360.98564736629 * D + 0.000387933 * T * T - (T**3) / 38710000.0
    return wrap_2pi(deg2rad(gmst_deg))

# ---------------------------
# Geodetic ↔ geocentric latitude
# ---------------------------

def geodetic_to_geocentric_latitude(phi_gd: float) -> float:
    """
    Convert geodetic latitude (rad) to geocentric latitude (rad) using WGS-84:
      tan φ_c = (1 - e^2) * tan φ_gd
    """
    return atan((1.0 - WGS84_e2) * tan(phi_gd))

# ---------------------------
# Kepler / orbital utilities
# ---------------------------

def mean_motion(a_km: float) -> float:
    return sqrt(MU_EARTH / (a_km**3))  # rad/s

def j2_rates(a_km: float, e: float, i_rad: float) -> Tuple[float, float, float]:
    """
    First-order secular J2 rates (Brouwer-Lyddane, LEO-valid):
      p = a(1-e^2)
      n = sqrt(μ/a^3)
      dΩ = -3/2 J2 n (R/p)^2 cos i
      dω =  3/4 J2 n (R/p)^2 (5 cos^2 i - 1)
      dM =  n + 3/4 J2 n (R/a)^2 (1-e^2)^(-3/2) (3 cos^2 i - 1)
    All returned in rad/s.
    """
    a = a_km
    n = mean_motion(a)
    p = a * (1.0 - e * e)
    ci = cos(i_rad)
    ci2 = ci * ci
    R_over_p2 = (R_EARTH / p) ** 2
    R_over_a2 = (R_EARTH / a) ** 2
    dOmega = -1.5 * J2 * n * R_over_p2 * ci
    domega =  0.75 * J2 * n * R_over_p2 * (5.0 * ci2 - 1.0)
    dM = n + 0.75 * J2 * n * R_over_a2 * ((1.0 - e * e) ** -1.5) * (3.0 * ci2 - 1.0)
    return dOmega, domega, dM

def hohmann_half(a1: float, a2: float) -> Tuple[float, float, float]:
    """
    Half-ellipse (one burn to the other burn) Hohmann coast:
      a_tr = (a1 + a2)/2
      e_tr = |a2 - a1| / (a1 + a2)
      Δt_half = π / sqrt(μ / a_tr^3)
    Returns (a_tr, e_tr, Δt_half).
    """
    a_tr = 0.5 * (a1 + a2)
    e_tr = abs(a2 - a1) / (a1 + a2)
    dt_half = pi / sqrt(MU_EARTH / (a_tr**3))
    return a_tr, e_tr, dt_half

# ---------------------------
# Geometry helpers
# ---------------------------

def u_required_branches(phi_t_geod_deg: float, i_rad: float) -> Tuple[float, float]:
    """
    Target geodetic latitude -> geocentric -> required u at overflight.
      sin φ = sin i * sin u  =>  u = arcsin(sin φ / sin i) or π - that (desc)
    Returns (u_asc, u_desc) in radians, wrapped to [0, 2π).
    """
    phi_gd = deg2rad(phi_t_geod_deg)
    # convert to geocentric latitude for the spherical-geometry relation
    phi_gc = geodetic_to_geocentric_latitude(phi_gd)
    s = clamp(sin(phi_gc) / max(1e-12, sin(i_rad)), -1.0, 1.0)
    u_asc = asin(s)                   # in [-π/2, π/2]
    if u_asc < 0.0:
        u_asc += 2.0 * pi             # wrap to [0,2π)
    u_desc = pi - (u_asc % (2.0 * pi))
    return (wrap_2pi(u_asc), wrap_2pi(u_desc))

def ra_of_position(i_rad: float, u_rad: float, Omega_rad: float) -> float:
    """
    Right ascension (inertial) of satellite position vector:
      α = Ω + atan2( cos i * sin u, cos u )
    """
    return wrap_2pi(Omega_rad + atan2(cos(i_rad) * sin(u_rad), cos(u_rad)))

# ---------------------------
# Orbit container
# ---------------------------

@dataclass
class Orbit:
    a_km: float
    i_rad: float
    Omega_rad: float
    u_rad: float           # initial u (for circular: u = M when ω=0)
    e: float = 0.0         # allow nonzero if you want; GOM/ROM are circular here

# ---------------------------
# Planner
# ---------------------------

class AnalyticRaanPhasingPlanner:
    def __init__(
        self,
        gom: Orbit,
        rom_a_km: float,
        delta_a_km: float,
        epoch_gom: datetime,
        target_lat_geod_deg: float,
        target_lon_deg: float,
        max_wrap: int = 50,
        drift_time_cap_days: float = 30.0
    ):
        self.gom = gom
        self.rom_a = rom_a_km
        self.delta_a = delta_a_km
        self.epoch = epoch_gom
        self.target_lat_deg = target_lat_geod_deg
        self.target_lon_rad = deg2rad(target_lon_deg)
        self.max_wrap = max_wrap
        self.drift_time_cap = drift_time_cap_days * 86400.0  # s

        # Derived orbits
        self.a_G = gom.a_km
        self.i = gom.i_rad
        self.Omega0 = wrap_2pi(gom.Omega_rad)
        self.u0 = wrap_2pi(gom.u_rad)

        # Drift orbit is circular with fixed Δa
        self.a_D = self.a_G + self.delta_a
        if self.a_D <= 0.0:
            raise ValueError("Drift semi-major axis must be positive.")

        # Transfers
        self.aT1, self.eT1, self.T1 = hohmann_half(self.a_G, self.a_D)
        self.aT2, self.eT2, self.T2 = hohmann_half(self.a_D, self.rom_a)

        # J2 rates per segment
        self.dOm_G, self.dom_G, self.dM_G = j2_rates(self.a_G, 0.0, self.i)
        self.dOm_D, self.dom_D, self.dM_D = j2_rates(self.a_D, 0.0, self.i)
        self.dOm_T1, self.dom_T1, self.dM_T1 = j2_rates(self.aT1, self.eT1, self.i)
        self.dOm_T2, self.dom_T2, self.dM_T2 = j2_rates(self.aT2, self.eT2, self.i)

        # Mean-longitude / mean-u rates
        self.dell_G = self.dOm_G + self.dom_G + self.dM_G
        self.dell_D = self.dOm_D + self.dom_D + self.dM_D
        self.du_G = self.dom_G + self.dM_G
        self.du_D = self.dom_D + self.dM_D
        self.dell_T1 = self.dOm_T1 + self.dom_T1 + self.dM_T1
        self.dell_T2 = self.dOm_T2 + self.dom_T2 + self.dM_T2
        self.du_T1 = self.dom_T1 + self.dM_T1
        self.du_T2 = self.dom_T2 + self.dM_T2

        # Starting mean longitude ℓ0 = Ω0 + u0 (ω0=0 for circular)
        self.ell0 = wrap_2pi(self.Omega0 + self.u0)

    # ------------- core solve -------------

    def solve_min_time(self, lon_tol_schedule: list[float] | None = None) -> Dict[str, Any]:
        u_asc, u_desc = u_required_branches(self.target_lat_deg, self.i)

        asc_sol = self._solve_branch('ascending', u_asc, lon_tol_schedule=lon_tol_schedule)
        desc_sol = self._solve_branch('descending', u_desc, lon_tol_schedule=lon_tol_schedule)

        # Choose the best feasible
        candidates = [s for s in [asc_sol, desc_sol] if s.get('success')]
        if not candidates:
            return {
                'success': False,
                'note': 'No feasible analytic solution; try another Δa or widen max_wrap.',
                'ascending': asc_sol,
                'descending': desc_sol
            }

        best = min(candidates, key=lambda s: s['total_time_s'])
        best['ascending'] = asc_sol
        best['descending'] = desc_sol
        best['branch'] = best['branch']  # already set
        return best

    ############### Robust branch solve with wait-cap aware search ###############
    def _solve_branch(
        self,
        branch_name: str,
        u_req: float,
        lon_tol_schedule: list[float] | None = None
    ) -> Dict[str, Any]:
        """
        Robust branch solve with wait-cap aware search + tolerance-aware band bracketing.
        - Build a tight m-range implied by (Tw,Td) caps (no blind max_wrap scan).
        - For each m, restrict Td to the sub-interval where 0<=Tw<=WAIT_CAP.
        - Discover k over that capped interval (coarse sweep), add ±neighbors.
        - For each (m,k), bracket/solve:
            * exact root r(Td)=0
            * band edges |r(Td)| = tol for each tol in LON_TOL_SCHEDULE
        If nothing brackets, add a near-miss (best |r| within the interval).
        - Select the earliest candidate meeting the tolerance ladder.
        """
        # -------- knobs --------
        NSCAN_KDISC    = 160         # discover k across Td interval
        NSCAN_BRACKET  = 192         # bracket across Td for (r==0) and band edges
        K_NEIGHBOR_PAD = 5
        LON_TOL_SCHEDULE = lon_tol_schedule or [0.25, 0.5, 0.8, 1.0, 1.5, 2.0]  # deg (tune as you like)
        WAIT_CAP_HR    = 150.0       # set None to disable wait cap. Usually is never more than 150 hr
        WAIT_CAP_S     = None if WAIT_CAP_HR is None else WAIT_CAP_HR * 3600.0
        # -----------------------

        all_candidates: List[Dict[str, Any]] = []

        # Branch-fixed geometry
        psi = atan2(cos(self.i) * sin(u_req), cos(u_req))  # α-Ω offset at overflight

        # Eq A: du_G*Tw + du_D*Td = const_u + 2π m
        const_u = -(self.du_T1 * self.T1 + self.du_T2 * self.T2) + (u_req - self.u0)

        # Caps
        td_base_lo = 0.0
        td_base_hi = self.drift_time_cap
        if td_base_hi <= td_base_lo + 1e-6:
            return {'success': False, 'branch': branch_name, 'note': 'Invalid drift-time bounds.'}
        Tw_cap = float('inf') if WAIT_CAP_S is None else WAIT_CAP_S

        # Tight m-range from caps: A = du_D*Td + du_G*Tw must hit const_u+2πm
        corners = [(0.0, 0.0), (td_base_hi, 0.0)]
        if Tw_cap != float('inf'):
            corners += [(0.0, Tw_cap), (td_base_hi, Tw_cap)]
        A_vals = [self.du_D*Td + self.du_G*Tw for (Td, Tw) in corners]
        A_min, A_max = min(A_vals), max(A_vals)
        m_min = int((A_min - const_u) // (2.0*pi))
        m_max = int((A_max - const_u) // (2.0*pi))
        PAD   = 2
        m_range = range(m_min - PAD, m_max + PAD + 1)

        # --- helpers ---
        def Tw_of_Td(Td: float, A_m: float) -> float:
            return (A_m - self.du_D * Td) / self.du_G

        def Omega_f_of_Td(Td: float, Tw: float) -> float:
            return (self.Omega0 +
                    self.dOm_G * Tw +
                    self.dOm_T1 * self.T1 +
                    self.dOm_D * Td +
                    self.dOm_T2 * self.T2)

        def total_time(Td: float, Tw: float) -> float:
            return Tw + self.T1 + Td + self.T2

        def phi_of(Td: float, A_m: float) -> Tuple[float, float, datetime]:
            Tw = Tw_of_Td(Td, A_m)
            tf = self.epoch + timedelta(seconds=total_time(Td, Tw))
            thetaG = gmst_rad(tf)
            phi = Omega_f_of_Td(Td, Tw) + psi - thetaG  # UNWRAPPED inertial-minus-Earth longitude
            return phi, Tw, tf

        def package_candidate(Td_sol: float, Tw_sol: float, m: int, k_fix: int, tag: str) -> None:
            Ttot = total_time(Td_sol, Tw_sol)
            tf   = self.epoch + timedelta(seconds=Ttot)
            Omega_f = wrap_2pi(Omega_f_of_Td(Td_sol, Tw_sol))
            u_f = wrap_2pi(u_req)          # circular ROM: ω=0 ⇒ M=u
            M_f = u_f
            thetaG_tf = gmst_rad(tf)
            psi_final = atan2(cos(self.i) * sin(u_f), cos(u_f))
            lam_f     = wrap_pm_pi(Omega_f + psi_final - thetaG_tf)
            lon_err   = wrap_pm_pi(lam_f - self.target_lon_rad)

            all_candidates.append({
                'success': True, 'branch': branch_name,
                'm_u_wrap': m, 'k_lon_wrap': k_fix,
                'T_wait': Tw_sol, 'T1': self.T1, 'T_d': Td_sol, 'T2': self.T2,
                'total_time_s': Ttot, 'tf': tf,
                'Omega_f_rad': Omega_f, 'u_f_rad': u_f,
                'alpha_f_rad': ra_of_position(self.i, u_f, Omega_f),
                'lambda_f_rad': lam_f, 'lambda_f_deg': rad2deg(lam_f),
                'lon_err_deg': rad2deg(lon_err), 'hit_type': tag,
                'final_orbit': {
                    'a_km': self.rom_a, 'e': 0.0, 'i_rad': self.i,
                    'Omega_rad': Omega_f, 'omega_rad': 0.0, 'M_rad': M_f, 'epoch': tf
                },
                'final_orbit_deg': {
                    'a_km': self.rom_a, 'e': 0.0, 'i_deg': rad2deg(self.i),
                    'RAAN_deg': rad2deg(Omega_f), 'arg_perigee_deg': 0.0,
                    'M_deg': rad2deg(M_f), 'epoch': tf
                }
            })

        def bisection(f, a, b):
            fa = f(a); fb = f(b)
            if fa == 1e9 or fb == 1e9 or fa*fb > 0.0:
                return None
            for _ in range(80):
                m = 0.5*(a+b)
                fm = f(m)
                if fm == 1e9:
                    a = m; fa = f(a); continue
                if abs(fm) < 1e-12 or (b-a) < 1e-6:
                    return m
                if fa*fm <= 0.0:
                    b = m; fb = fm
                else:
                    a = m; fa = fm
            return 0.5*(a+b)

        tol_rads = [deg2rad(t) for t in LON_TOL_SCHEDULE]

        # ---- main loop over m ----
        for m in m_range:
            A_m = const_u + 2.0*pi*m

            # Feasible Td interval where 0 ≤ Tw ≤ WAIT_CAP
            if abs(self.du_D) < 1e-16 or abs(self.du_G) < 1e-16:
                continue  # pathological

            # Root for Tw=0
            t0 = A_m / self.du_D

            if WAIT_CAP_S is None:
                # Tw >= 0  →  depending on slope, it is one side of t0
                s = -self.du_D / self.du_G  # dTw/dTd
                if s < 0.0:
                    td_lo = td_base_lo
                    td_hi = min(td_base_hi, t0)
                else:
                    td_lo = max(td_base_lo, t0)
                    td_hi = td_base_hi
            else:
                # Tw in [0, cap] → between the two roots (monotone function)
                tc = (A_m - self.du_G * WAIT_CAP_S) / self.du_D
                td_lo = max(td_base_lo, min(t0, tc))
                td_hi = min(td_base_hi, max(t0, tc))

            if td_hi <= td_lo + 1e-6:
                continue

            # Discover k across [td_lo, td_hi]
            kset: set[int] = set()
            for j in range(NSCAN_KDISC + 1):
                t_cur = td_lo + (td_hi - td_lo) * (j / NSCAN_KDISC)
                phi_cur, Tw_cur, _ = phi_of(t_cur, A_m)
                if Tw_cur < -1e-9 or (WAIT_CAP_S is not None and Tw_cur - WAIT_CAP_S > 1e-9):
                    continue
                k_near = int(round((phi_cur - self.target_lon_rad) / (2.0*pi)))
                kset.add(k_near)
            if not kset:
                continue
            kmin, kmax = min(kset), max(kset)
            for dk in range(1, K_NEIGHBOR_PAD + 1):
                kset.add(kmin - dk); kset.add(kmax + dk)

            # For each fixed k, bracket/solve
            for k_fix in sorted(kset):

                def r_of_Td(Td: float) -> Tuple[float, float]:
                    phi, Tw, _ = phi_of(Td, A_m)
                    if Tw < -1e-12 or (WAIT_CAP_S is not None and Tw - WAIT_CAP_S > 1e-12):
                        return 1e9, Tw
                    return phi - (self.target_lon_rad + 2.0*pi*k_fix), Tw

                # Coarse scan to build segment list, keep best |r|
                segments: List[Tuple[float, float, float, float]] = []  # (a,b,ra,rb)
                best_sample = (float('inf'), None)
                t_prev = td_lo
                r_prev, _ = r_of_Td(t_prev)
                if r_prev != 1e9:
                    best_sample = (abs(r_prev), t_prev)

                for j in range(1, NSCAN_BRACKET + 1):
                    t_cur = td_lo + (td_hi - td_lo) * (j / NSCAN_BRACKET)
                    r_cur, _ = r_of_Td(t_cur)
                    if r_cur != 1e9 and abs(r_cur) < best_sample[0]:
                        best_sample = (abs(r_cur), t_cur)
                    segments.append((t_prev, t_cur, r_prev, r_cur))
                    t_prev, r_prev = t_cur, r_cur

                # First: exact roots r==0
                for (a, b, ra, rb) in segments:
                    if (ra == 1e9) or (rb == 1e9):
                        continue
                    if ra * rb <= 0.0:
                        Td0 = bisection(lambda t: r_of_Td(t)[0], a, b)
                        if Td0 is not None:
                            r0, Tw0 = r_of_Td(Td0)
                            if Tw0 >= 0.0 and (WAIT_CAP_S is None or Tw0 <= WAIT_CAP_S):
                                package_candidate(Td0, Tw0, m, k_fix, 'r==0')

                # Second: band edges |r|=tol for every tol rung
                for tol_rad in tol_rads:
                    for (a, b, ra, rb) in segments:
                        if (ra == 1e9) or (rb == 1e9):
                            continue
                        rma, rmb = ra - tol_rad, rb - tol_rad  # r - tol
                        rpa, rpb = ra + tol_rad, rb + tol_rad  # r + tol
                        # cross either band edge
                        if rma * rmb <= 0.0:
                            Td1 = bisection(lambda t, tr=tol_rad: r_of_Td(t)[0] - tr, a, b)
                            if Td1 is not None:
                                r1, Tw1 = r_of_Td(Td1)
                                if Tw1 >= 0.0 and (WAIT_CAP_S is None or Tw1 <= WAIT_CAP_S):
                                    package_candidate(Td1, Tw1, m, k_fix, '|r|=tol')
                        if rpa * rpb <= 0.0:
                            Td2 = bisection(lambda t, tr=tol_rad: r_of_Td(t)[0] + tr, a, b)
                            if Td2 is not None:
                                r2, Tw2 = r_of_Td(Td2)
                                if Tw2 >= 0.0 and (WAIT_CAP_S is None or Tw2 <= WAIT_CAP_S):
                                    package_candidate(Td2, Tw2, m, k_fix, '|r|=tol')

                # Finally: near-miss fallback if we found no brackets at all
                if best_sample[1] is not None and not any(c for c in all_candidates if c['m_u_wrap']==m and c['k_lon_wrap']==k_fix):
                    Td_nm = float(best_sample[1])
                    r_nm, Tw_nm = r_of_Td(Td_nm)
                    if Tw_nm >= 0.0 and (WAIT_CAP_S is None or Tw_nm <= WAIT_CAP_S):
                        package_candidate(Td_nm, Tw_nm, m, k_fix, 'near_miss')

        # -------- selection: earliest within tolerance ladder --------
        if not all_candidates:
            return {'success': False, 'branch': branch_name,
                    'note': 'No solution within wait/drift caps.'}

        tol_used = None
        best = None
        for tol_deg in LON_TOL_SCHEDULE:
            feas = [c for c in all_candidates if abs(c['lon_err_deg']) <= tol_deg]
            if feas:
                best = min(feas, key=lambda s: s['total_time_s'])
                tol_used = tol_deg
                break
        if best is None:
            # No candidate inside the loosest tol: pick smallest error, then earliest
            best = min(all_candidates, key=lambda s: (abs(s['lon_err_deg']), s['total_time_s']))

        best['selection_meta'] = {
            'policy': 'earliest_within_tol' if tol_used is not None else 'fallback_tradeoff',
            'tolerance_used_deg': tol_used,
            'num_candidates_considered': len(all_candidates),
            'wait_cap_s': WAIT_CAP_S,
            'wait_cap_hr': (None if WAIT_CAP_S is None else WAIT_CAP_S/3600.0),
            'drift_cap_s': self.drift_time_cap,
            'drift_cap_hr': self.drift_time_cap/3600.0,
            'chosen_m': best['m_u_wrap'],
            'chosen_k': best['k_lon_wrap'],
        }
        return best


# ---------------------------
# Helper to print details
# ---------------------------

def print_solution_summary(sol: dict) -> None:
    meta = sol.get('selection_meta', {})
    wait_hr  = sol['T_wait'] / 3600.0
    drift_hr = sol['T_d']    / 3600.0
    tot_hr   = sol['total_time_s'] / 3600.0
    tol      = meta.get('tolerance_used_deg')
    tol_str  = f"{tol:.2f}°" if tol is not None else "— (fallback)"
    wcap_str = ("∞" if meta.get('wait_cap_hr') is None
                else f"{meta['wait_cap_hr']:.1f} h")
    dcap_str = f"{meta['drift_cap_hr']:.1f} h"

    print(
        "[selected] "
        f"branch={sol['branch']}  m={meta.get('chosen_m')}  k={meta.get('chosen_k')}  |  "
        f"wait={wait_hr:.2f} h  drift={drift_hr:.2f} h  total={tot_hr:.2f} h  |  "
        f"lon_err={sol['lon_err_deg']:.3f}°  tol_used={tol_str}  |  "
        f"wait_cap={wcap_str}  drift_cap={dcap_str}  |  "
        f"candidates={meta.get('num_candidates_considered')}"
    )

# ---------------------------
# Glue for maneuver visualizer
# ---------------------------

def build_maneuver_sequence_for_visualizer(
    gom: Orbit,
    rom_a_km: float,
    delta_a_km: float,
    epoch: datetime,
    target_lat_geod_deg: float,
    target_lon_deg: float,
    max_wrap: int = 300,
    drift_time_cap_days: float = 30.0
) -> Dict[str, Any]:
    """
    Run the analytic planner once and return a compact package with:
      - times: T_wait, T1, T_d, T2, total_time_s
      - epochs: base, transfer1, drift, transfer2, rom (tf)
      - angles: initial RAAN (deg), final RAAN (deg), final M/true anomaly (deg)
      - orbit dicts for each segment in the visualizer's expected format

    The visualizer sets segment "true_anomaly" to the cumulative mean-anomaly
    progress at the start of each segment (circular uses ν≈M; for ellipses we
    still use M increment from mean motion).
    """
    planner = AnalyticRaanPhasingPlanner(
        gom=gom,
        rom_a_km=rom_a_km,
        delta_a_km=delta_a_km,
        epoch_gom=epoch,
        target_lat_geod_deg=target_lat_geod_deg,
        target_lon_deg=target_lon_deg,
        max_wrap=max_wrap,
        drift_time_cap_days=drift_time_cap_days,
    )
    sol = planner.solve_min_time()
    if not sol.get('success'):
        return {
            'success': False,
            'note': sol.get('note', 'Planner failed to find a solution.')
        }

    # Times
    T_wait = float(sol['T_wait'])
    T1     = float(sol['T1'])
    T_d    = float(sol['T_d'])
    T2     = float(sol['T2'])
    Ttot   = float(sol['total_time_s'])

    # Orbits (semi-major axes of segments)
    aG   = planner.a_G
    aD   = planner.a_D
    aT1  = planner.aT1
    eT1  = planner.eT1
    aT2  = planner.aT2
    eT2  = planner.eT2

    # Mean anomaly increments by segment (rad)
    dM_G   = mean_motion(aG)  * T_wait
    dM_T1  = mean_motion(aT1) * T1
    dM_D   = mean_motion(aD)  * T_d
    dM_T2  = mean_motion(aT2) * T2

    # Cumulative starts for each segment (deg), wrapped to [0,360)
    M_waiting_deg   = (rad2deg(dM_G)) % 360.0
    M_T1_start_deg  = (rad2deg(dM_G + dM_T1)) % 360.0
    M_D_start_deg   = M_T1_start_deg
    M_T2_start_deg  = (rad2deg(dM_G + dM_T1 + dM_D)) % 360.0
    M_ROM_start_deg = (rad2deg(dM_G + dM_T1 + dM_D + dM_T2)) % 360.0

    # Epochs per segment start
    epoch_T1 = epoch + timedelta(seconds=T_wait)
    epoch_D  = epoch_T1 + timedelta(seconds=T1)
    epoch_T2 = epoch_D  + timedelta(seconds=T_d)
    epoch_ROM= epoch_T2 + timedelta(seconds=T2)  # should match sol['tf']

    # Angles/RAANs
    RAAN0_deg = rad2deg(planner.Omega0)
    RAANf_deg = rad2deg(sol['Omega_f_rad'])
    i_deg     = rad2deg(planner.i)

    # Package orbit dictionaries for the visualizer API
    gom_orbit = {
        'semi_major_axis': aG,
        'eccentricity': 0.0,
        'inclination': i_deg,
        'RAAN': RAAN0_deg,
        'argument_of_periapsis': 0.0,
        'true_anomaly': 0.0,
    }
    transfer1_orbit = {
        'semi_major_axis': aT1,
        'eccentricity': eT1,
        'inclination': i_deg,
        'RAAN': RAAN0_deg,  # small ΔΩ over T1 neglected for this visualizer
        'argument_of_periapsis': 0.0,
        'true_anomaly': M_waiting_deg,
    }
    drift_orbit = {
        'semi_major_axis': aD,
        'eccentricity': 0.0,
        'inclination': i_deg,
        'RAAN': RAAN0_deg,  # small ΔΩ over drift ignored here; use sgp4/J2 if needed
        'argument_of_periapsis': 0.0,
        'true_anomaly': M_T1_start_deg,
    }
    transfer2_orbit = {
        'semi_major_axis': aT2,
        'eccentricity': eT2,
        'inclination': i_deg,
        'RAAN': RAANf_deg,  # match ROM RAAN at tf for continuity in viz
        'argument_of_periapsis': 0.0,
        'true_anomaly': M_T2_start_deg,
    }
    rom_orbit = {
        'semi_major_axis': rom_a_km,
        'eccentricity': 0.0,
        'inclination': i_deg,
        'RAAN': RAANf_deg,
        'argument_of_periapsis': 0.0,
        # 'true_anomaly': rad2deg(sol['final_orbit']['M_rad']),  # previously used
    # Use mean anomaly at epoch (SGP4 expects M; for circular ROM, ν≈M)
    'mean_anomaly': rad2deg(sol['final_orbit']['M_rad']),
    }
    rom_orbit_test = {
        'semi_major_axis': rom_a_km,
        'eccentricity': 0.0,
        'inclination': i_deg,
        'RAAN': RAANf_deg,
        'argument_of_periapsis': 0.0,\
        # 'true_anomaly': M_ROM_start_deg,  # previously used
    'mean_anomaly': M_ROM_start_deg,
    }

    return {
        'success': True,
        'times': {
            'T_wait_s': T_wait,
            'T1_s': T1,
            'T_d_s': T_d,
            'T2_s': T2,
            'total_time_s': Ttot,
        },
        'epochs': {
            'gom_epoch': epoch,
            'transfer1_epoch': epoch_T1,
            'drift_epoch': epoch_D,
            'transfer2_epoch': epoch_T2,
            'rom_epoch': epoch_ROM,
        },
        'angles': {
            'RAAN0_deg': RAAN0_deg,
            'RAANf_deg': RAANf_deg,
            'i_deg': i_deg,
        },
        'orbits': {
            'gom': gom_orbit,
            'transfer1': transfer1_orbit,
            'drift': drift_orbit,
            'transfer2': transfer2_orbit,
            'rom': rom_orbit,
            'rom_test': rom_orbit_test,
        },
        'solution': sol,
        'planner': planner,
    }

# # ---------------------------
# # Sweep Δa and plot times vs Δa
# # ---------------------------

def compute_reconfig_times_vs_delta_a(
    gom: Orbit,
    rom_a_km: float,
    epoch: datetime,
    target_lat_geod_deg: float,
    target_lon_deg: float,
    delta_min_km: float = -200.0,
    delta_max_km: float =  200.0,
    delta_step_km: float =  5.0,
    max_wrap: int = 40,
    drift_time_cap_days: float = 15.0,
    verbose: bool = False,
):
    """
    Returns (delta_a_km, total_time_s, wait_time_s, drift_time_s), using NaN where no solution exists.
    """
    dAs = np.arange(delta_min_km, delta_max_km + 1e-9, delta_step_km, dtype=float)
    totals = np.full(dAs.shape, np.nan, dtype=float)
    waits  = np.full(dAs.shape, np.nan, dtype=float)
    drifts = np.full(dAs.shape, np.nan, dtype=float)
    if verbose:
        print(f"[Δa sweep] {len(dAs)} steps from {delta_min_km} to {delta_max_km} km (step {delta_step_km})...", flush=True)
    # Determine progress granularity (~10% of steps)
    progress_step = max(1, len(dAs)//10)

    for idx, dA in enumerate(dAs):
        if verbose and ((idx % progress_step) == 0 or idx == 0):
            print(f"[Δa sweep] starting {idx+1}/{len(dAs)}: Δa={dA:g} km", flush=True)
        # Skip unphysical drift SMA
        if gom.a_km + dA <= 0.0:
            continue
        planner = AnalyticRaanPhasingPlanner(
            gom=gom,
            rom_a_km=rom_a_km,
            delta_a_km=float(dA),
            epoch_gom=epoch,
            target_lat_geod_deg=target_lat_geod_deg,
            target_lon_deg=target_lon_deg,
            max_wrap=max_wrap,
            drift_time_cap_days=drift_time_cap_days
        )
        sol = planner.solve_min_time()
        if sol.get('success'):
            totals[idx] = sol['total_time_s']
            waits[idx]  = sol['T_wait']
            drifts[idx] = sol['T_d']
        if verbose:
            # Print every ~10% progress
            if (idx + 1) % progress_step == 0 or (idx + 1) == len(dAs):
                pct = 100.0 * (idx + 1) / len(dAs)
                print(f"[Δa sweep] {idx+1}/{len(dAs)} ({pct:.0f}%)", flush=True)
    if verbose:
        print("[Δa sweep] done.", flush=True)
    return dAs, totals, waits, drifts

# ---------------------------
# Usage
# ---------------------------

if __name__ == "__main__":
    # Import the new plotter module
    from reconfig_plotter import run_reconfig_plots, PlotFlags
    
    # Example inputs (your sample)
    epoch = datetime(2025, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    i_deg = 51.4476847948

    gom = Orbit(
        a_km = R_EARTH + 500.0,
        i_rad= deg2rad(i_deg),
        Omega_rad = deg2rad(0.0),
        u_rad     = deg2rad(0.0)
    )

    # ROM altitude (semi-major axis)
    rom_a_km = R_EARTH + 504.59534097

    # Targets list: add as many (name, lat_deg, lon_deg) tuples as you like
    targets = [
        ("LosAngeles", 34.0522, -118.2437),
        # ("Sumatra_2004", 3.295, 95.982),
        # ("Zero_0_0", 0.0, 0.0),
        # ("Rome", 41.9028, 12.4964),
        # ("Hawaii", 21.3069, -157.8583),
        # ("SanFrancisco", 37.7749, -122.4194),
        # ("SanDiego", 32.7157, -117.1611),
        # ("Vancouver", 49.2827, -123.1207),
        # ("Sydney", -33.8688, 151.2093),
        # ("Beijing", 39.9042, 116.4074),
        # ("Tokyo", 35.6895, 139.6917),
    ]

    FLAG_TIMES_VS_DELTA_A = False
    FLAG_TIMES_VS_TOLERANCE = False
    FLAG_AVG_VS_DELTA_A = True
    FLAG_AVG_VS_TOLERANCE = True
    FLAG_HEATMAP = True

    # Diagnostics
    FLAG_PARETO_TIME_VS_DV    = False # works
    FLAG_ISO_TIME_CONTOURS    = False # works, to be detailed and expand with avg
    FLAG_CHOSEN_VS_TOL        = False # works
    FLAG_TOL_SENSITIVITY      = False # doesnt work, type error
    FLAG_TARGET_JITTER        = False # doesnt work, type error
    FLAG_EPOCH_SENSITIVITY    = False # works, need to expand with avg and detail
    FLAG_RESIDUAL_CURVES      = False # doesnt work, name error


    # Common knobs
    max_wrap = 200
    drift_time_cap_days = 200.0/24.0  # ~200 hr
    verbose = True

    # Plot parameter sets
    delta_range = dict(delta_min_km=-100, delta_max_km=200, delta_step_km=10)
    tol_ladder  = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    delta_a_fixed_km = 50.0

    # Diagnostics params
    iso_levels_hours = [12.0, 24.0, 48.0, 72.0, 96.0]
    tol_sens_series  = (True, True, True)  # plot total/wait/drift
    jitter_radius_deg = 1.0
    jitter_samples    = 200
    jitter_seed       = 7
    jitter_tol_deg    = 0.25
    epoch_window_hours = 24.0
    epoch_samples      = 49
    epoch_tol_deg      = 0.25
    # residuals: leave branch/m/k None to auto-pick best solution, or set explicitly
    residual_branch    = None   # 'ascending' or 'descending' or None
    residual_m         = None
    residual_k         = None
    residual_nsamples  = 600

    # ===============================
    # 1) Per-target case study plots
    # ===============================
    if any([FLAG_TIMES_VS_DELTA_A, FLAG_TIMES_VS_TOLERANCE,
            FLAG_PARETO_TIME_VS_DV, FLAG_CHOSEN_VS_TOL,
            FLAG_TOL_SENSITIVITY, FLAG_TARGET_JITTER,
            FLAG_EPOCH_SENSITIVITY, FLAG_RESIDUAL_CURVES]):
        for (target_name, tlat, tlon) in targets:
            print(f"\n[CASE] {target_name} ({tlat:+.3f}°, {tlon:+.3f}°)")

            run_reconfig_plots(
                gom=gom,
                rom_a_km=rom_a_km,
                epoch=epoch,
                target_name=target_name,
                target_lat_geod_deg=tlat,
                target_lon_deg=tlon,

                flags=PlotFlags(
                    times_vs_delta_a=FLAG_TIMES_VS_DELTA_A,
                    times_vs_tolerance=FLAG_TIMES_VS_TOLERANCE,
                    avg_vs_delta_a=False,
                    avg_vs_tolerance=False,
                    heatmap=False,
                    # diagnostics
                    pareto_time_vs_dv=FLAG_PARETO_TIME_VS_DV,
                    iso_time_contours=False,              # iso-time is grid-wide, see section 3
                    chosen_vs_tolerance=FLAG_CHOSEN_VS_TOL,
                    tolerance_sensitivity=FLAG_TOL_SENSITIVITY,
                    target_jitter=FLAG_TARGET_JITTER,
                    epoch_sensitivity=FLAG_EPOCH_SENSITIVITY,
                    residual_curves=FLAG_RESIDUAL_CURVES,
                ),

                # Δa sweep (case + pareto)
                delta_min_km=delta_range["delta_min_km"],
                delta_max_km=delta_range["delta_max_km"],
                delta_step_km=delta_range["delta_step_km"],

                # tolerance sweep (case + chosen_vs_tolerance + tol sensitivity)
                delta_a_fixed_km=delta_a_fixed_km,
                tolerances_deg=tol_ladder,

                # diagnostics knobs
                pareto_include_wait=True,
                tol_sens_include_series=tol_sens_series,
                jitter_radius_deg=jitter_radius_deg,
                jitter_samples=jitter_samples,
                jitter_seed=jitter_seed,
                jitter_tolerance_deg=jitter_tol_deg,
                epoch_window_hours=epoch_window_hours,
                epoch_samples=epoch_samples,
                epoch_tolerance_deg=epoch_tol_deg,
                residual_branch=residual_branch,
                residual_m=residual_m,
                residual_k=residual_k,
                residual_nsamples=residual_nsamples,

                # solver
                max_wrap=max_wrap,
                drift_time_cap_days=drift_time_cap_days,
                verbose=verbose,
            )


    # ---------------------------------------------------------
    # 2) Averaged plots across a latitude band & all longitudes
    #    (no target loop; this averages over many targets)
    # ---------------------------------------------------------
    if FLAG_AVG_VS_DELTA_A or FLAG_AVG_VS_TOLERANCE:
        # Latitude band (deg): must be < inclination to ensure access
        lat_band_deg = i_deg -1.0         # must be < inclination to ensure access
        n_lat = 5                   # number of latitude samples in [-band,+band]
        n_lon = 18                   # number of longitudes in [-180,180)

        print("\n[AVG] Generating averaged plots across target grid...")
        run_reconfig_plots(
            gom=gom,
            rom_a_km=rom_a_km,
            epoch=epoch,
            target_name="AVG",               # not used by averaged plots
            target_lat_geod_deg=0.0,         # not used by averaged plots
            target_lon_deg=0.0,              # not used by averaged plots
            flags=PlotFlags(times_vs_delta_a=False, times_vs_tolerance=False,
                            avg_vs_delta_a=FLAG_AVG_VS_DELTA_A, avg_vs_tolerance=FLAG_AVG_VS_TOLERANCE, heatmap=False),
            delta_min_km=delta_range["delta_min_km"],
            delta_max_km=delta_range["delta_max_km"],
            delta_step_km=delta_range["delta_step_km"],
            delta_a_fixed_km=delta_a_fixed_km,
            tolerances_deg=tol_ladder,
            inclination_deg=i_deg,
            lat_band_deg=lat_band_deg,
            n_lat=n_lat, n_lon=n_lon,
            max_wrap=max_wrap,
            drift_time_cap_days=drift_time_cap_days,
            verbose=verbose,
        )

    # ==========================
    # 3) Heatmaps & Iso-time map
    # ==========================
    if FLAG_HEATMAP or FLAG_ISO_TIME_CONTOURS:
        lat_band_deg = i_deg - 1.0
        n_lat_heatmap = 12           # across inclination band
        n_long_heatmap = 42          # around Earth

        print("\n[GRID] Generating world maps...")
        run_reconfig_plots(
            gom=gom,
            rom_a_km=rom_a_km,
            epoch=epoch,
            target_name="GRID",               # not used by grid plots
            target_lat_geod_deg=0.0,
            target_lon_deg=0.0,
            flags=PlotFlags(
                times_vs_delta_a=False,
                times_vs_tolerance=False,
                avg_vs_delta_a=False,
                avg_vs_tolerance=False,
                heatmap=FLAG_HEATMAP,
                iso_time_contours=FLAG_ISO_TIME_CONTOURS
            ),
            delta_a_fixed_km=delta_a_fixed_km,
            tolerances_deg=[0.25],            # a single tolerance for the heatmap
            inclination_deg=i_deg,
            lat_band_deg=lat_band_deg,
            n_lat=n_lat_heatmap,
            n_lon=n_long_heatmap,
            iso_levels_hours=iso_levels_hours,
            max_wrap=max_wrap,
            drift_time_cap_days=drift_time_cap_days,
            verbose=verbose,
        )

        print("\nAll plots saved under results/reconfig")
