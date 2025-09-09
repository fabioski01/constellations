# -*- coding: utf-8 -*-
"""
Analytic J2 phasing with wait + Hohmann-to-drift + drift + Hohmann-to-ROM.

- Propagates secular J2 for Ω, ω, M across each segment (GOM wait, T1, drift, T2)
- Uses mean longitude ℓ = Ω + ω + M so circular segments are handled cleanly
- Enforces final-pass geometry exactly:
    latitude:  sin φ_t = sin i * sin u_f  (two branches: asc/desc)
    longitude: Ω_f + atan2(cos i sin u_f, cos u_f) - θ_G(tf) = λ_t + 2π k
- Solves for T_wait and T_d (drift time) per branch and wrap integers (m,k)
- Includes the finite durations of both Hohmann halves (T1 and T2)
- Reports final ROM elements at tf (circular: ω=0, M = u_f mod 2π)

No external deps beyond Python stdlib.
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

    def solve_min_time(self) -> Dict[str, Any]:
        u_asc, u_desc = u_required_branches(self.target_lat_deg, self.i)

        asc_sol = self._solve_branch('ascending', u_asc)
        desc_sol = self._solve_branch('descending', u_desc)

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

    def _solve_branch(self, branch_name: str, u_req: float) -> Dict[str, Any]:
        """
        For a given branch (asc/desc) and required u at overflight, search integer
        wraps (m for u, k for longitude) and solve for Td (bisection) then Twait from linear eq.
        """
        best: Optional[Dict[str, Any]] = None

        # Precompute constants from fixed segments
        const_u = - (self.du_T1 * self.T1 + self.du_T2 * self.T2) + (u_req - self.u0)
        # Equation A: du_G * Tw + du_D * Td = const_u + 2π m
        # -> Tw(Td; m) = [const_u + 2π m - du_D*Td] / du_G

        # We'll try m and k in a symmetric range (0..max_wrap is fine too)
        m_candidates = list(range(-self.max_wrap, self.max_wrap + 1))
        k_candidates = list(range(-self.max_wrap, self.max_wrap + 1))

        for m in m_candidates:
            A = const_u + 2.0 * pi * m

            # Construct a safe Td search interval from wait>=0 and a global cap
            Td_upper_by_wait = float('inf')
            if self.du_D != 0.0:
                # Tw >= 0 => A - du_D*Td >= 0 (assuming du_G > 0; if du_G<0, inequality flips)
                if self.du_G > 0.0:
                    Td_upper_by_wait = A / self.du_D
                elif self.du_G < 0.0:
                    Td_upper_by_wait = -float('inf')  # rarely the case for prograde LEO
            # We want Td ≥ 0 and Tw ≥ 0
            td_lo = 0.0
            td_hi = self.drift_time_cap
            if isfinite(Td_upper_by_wait):
                td_hi = min(td_hi, max(0.0, Td_upper_by_wait))

            if td_hi <= td_lo + 1e-6:
                # This m cannot yield nonnegative Tw over any Td≥0 within cap
                continue

            # Build f(Td; k) for longitude equation B:
            # Ω_f(Td) + ψ(u_req) - θ_G(tf) = λ_t + 2π k
            psi = atan2(cos(self.i) * sin(u_req), cos(u_req))  # constant for this branch

            # Ω_f = Ω0 + dΩ_G * Tw + dΩ_T1*T1 + dΩ_D*Td + dΩ_T2*T2
            # Tw = (A - du_D*Td)/du_G

            def Omega_f_of_Td(Td: float) -> float:
                Tw = (A - self.du_D * Td) / self.du_G
                return (self.Omega0 +
                        self.dOm_G * Tw +
                        self.dOm_T1 * self.T1 +
                        self.dOm_D * Td +
                        self.dOm_T2 * self.T2)

            def total_time(Td: float) -> float:
                Tw = (A - self.du_D * Td) / self.du_G
                return Tw + self.T1 + Td + self.T2

            def f_of_Td_k(Td: float, k_wrap: int) -> float:
                Tw = (A - self.du_D * Td) / self.du_G
                if Tw < 0.0:
                    # Penalize outside feasible wait
                    return 1e9
                tf = self.epoch + timedelta(seconds=total_time(Td))
                thetaG = gmst_rad(tf)
                lhs = wrap_pm_pi(Omega_f_of_Td(Td) + psi - thetaG)
                rhs = wrap_pm_pi(self.target_lon_rad + 2.0 * pi * k_wrap)
                return wrap_pm_pi(lhs - rhs)

            # Now scan k and attempt to bracket Td
            for k in k_candidates:
                # Coarse scan on Td to find a sign change
                Nscan = 64
                tlo = td_lo
                thi = td_hi
                if thi <= tlo + 1e-6:
                    continue

                f_prev = f_of_Td_k(tlo, k)
                bracket = None
                for j in range(1, Nscan + 1):
                    tcur = tlo + (thi - tlo) * j / Nscan
                    f_cur = f_of_Td_k(tcur, k)
                    if not (abs(f_prev) > 1e-12 and abs(f_cur) > 1e-12):
                        # Avoid zero at endpoints triggering issues
                        pass
                    if f_prev == 1e9 or f_cur == 1e9:
                        f_prev = f_cur
                        continue
                    if f_prev * f_cur <= 0.0:
                        bracket = (tlo + (thi - tlo) * (j - 1) / Nscan, tcur)
                        break
                    f_prev = f_cur

                if bracket is None:
                    continue  # no sign change for this (m,k) within bounds

                # Bisection
                a, b = bracket
                fa = f_of_Td_k(a, k)
                fb = f_of_Td_k(b, k)
                if fa == 1e9 or fb == 1e9:
                    continue
                if fa * fb > 0:
                    continue

                for _ in range(80):
                    mid = 0.5 * (a + b)
                    fm = f_of_Td_k(mid, k)
                    if abs(fm) < 1e-12 or (b - a) < 1e-6:
                        Td_sol = mid
                        break
                    if fa * fm <= 0.0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm
                else:
                    Td_sol = 0.5 * (a + b)

                # Compute the rest
                Tw_sol = (A - self.du_D * Td_sol) / self.du_G
                if Tw_sol < -1e-6:
                    continue

                Ttot = total_time(Td_sol)
                tf = self.epoch + timedelta(seconds=Ttot)

                # Final angles
                Omega_f = wrap_2pi(Omega_f_of_Td(Td_sol))
                # From eq A, u_f = u_req + 2π m exactly
                u_f = wrap_2pi(u_req)
                # For ROM circular: choose ω_f=0, then M_f = u_f
                M_f = u_f

                # ROM elements
                final_orbit = {
                    'a_km': self.rom_a,
                    'e': 0.0,
                    'i_rad': self.i,
                    'Omega_rad': Omega_f,
                    'omega_rad': 0.0,
                    'M_rad': M_f,
                    'epoch': tf
                }
                final_orbit_deg = {
                    'a_km': self.rom_a,
                    'e': 0.0,
                    'i_deg': rad2deg(self.i),
                    'RAAN_deg': rad2deg(Omega_f),
                    'arg_perigee_deg': 0.0,
                    'M_deg': rad2deg(M_f),
                    'epoch': tf
                }

                sol = {
                    'success': True,
                    'branch': branch_name,
                    'm_u_wrap': m,
                    'k_lon_wrap': k,
                    'T_wait': Tw_sol,
                    'T1': self.T1,
                    'T_d': Td_sol,
                    'T2': self.T2,
                    'total_time_s': Ttot,
                    'tf': tf,
                    'Omega_f_rad': Omega_f,
                    'u_f_rad': u_f,
                    'alpha_f_rad': ra_of_position(self.i, u_f, Omega_f),
                    'final_orbit': final_orbit,
                    'final_orbit_deg': final_orbit_deg
                }

                # Keep best (minimum total time)
                if (best is None) or (sol['total_time_s'] < best['total_time_s']):
                    best = sol

        if best is None:
            return {'success': False, 'branch': branch_name,
                    'note': 'No solution found for this branch within search limits.'}
        return best

# ---------------------------
# Optional: minimal stub plotter (to avoid NameError if you call it)
# ---------------------------

def simple_plotter_GOM_ROM(planner: AnalyticRaanPhasingPlanner, sol: Dict[str, Any],
                           tgt_lat_deg: float, tgt_lon_deg: float) -> None:
    """
    Minimal stub: prints key waypoints and a couple of sanity angles.
    (Replace with your own plotting if desired.)
    """
    print("\n[Plotter stub] Use your existing visualizer for ground tracks.")
    print(f"  Final epoch (tf): {sol['tf'].isoformat()}")
    print(f"  Final RAAN (deg): {rad2deg(sol['Omega_f_rad']):.6f}")
    print(f"  Final M (deg):    {sol['final_orbit_deg']['M_deg']:.6f}")
    print(f"  Target (lat,lon): ({tgt_lat_deg:.3f}°, {tgt_lon_deg:.3f}°)")
    print("  (This stub avoids external deps. Replace with your Cartopy/SGP4 plotter.)\n")
# ---------------------------
# Simple ground-track plotter
# ---------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

def _R3_ccw(theta):
    # Active rotation about +Z by +theta (CCW looking down +Z)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def _R3_ecef_from_eci(theta):
    # ECI -> ECEF: rotate by +theta (this is R3_ccw(theta).T)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c,  s, 0.0],
                     [-s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def _R1_ccw(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,   c, -s ],
                     [0.0,   s,  c ]])

def _eci_from_circular(i_rad, Omega_rad, u_rad, r_km=1.0):
    # r̂_PF = [cos u, sin u, 0], then rotate: R3(Ω) * R1(i)
    r_pf = np.array([np.cos(u_rad), np.sin(u_rad), 0.0], dtype=float)
    r_eci = _R3_ccw(Omega_rad) @ _R1_ccw(i_rad) @ r_pf
    return r_eci * r_km

def _groundtrack_for_circular(a_km, i_rad,
                              Omega0_rad, u0_rad,
                              dOmega_dt, du_dt,
                              epoch_dt,
                              hours=1.0, dt_s=10.0):
    """
    Propagate a circular orbit under secular J2:
        Ω(t) = Ω0 + dΩ t,     u(t) = u0 + (dω+dM) t
    Convert ECI->ECEF with GMST(epoch + t), then to geocentric lat/lon.
    Returns arrays (times_s, lat_deg, lon_deg).
    """
    N = int(max(1, round(hours * 3600.0 / dt_s))) + 1
    t_arr = np.linspace(0.0, hours * 3600.0, N)
    lat = np.zeros(N)
    lon = np.zeros(N)

    for k, t in enumerate(t_arr):
        Om = Omega0_rad + dOmega_dt * t
        uu = u0_rad     + du_dt     * t

        r_eci = _eci_from_circular(i_rad, Om, uu, r_km=a_km)  # radius magnitude not needed for lon/lat
        thetaG = gmst_rad(epoch_dt + timedelta(seconds=float(t)))
        r_ecef = _R3_ecef_from_eci(thetaG) @ r_eci

        x, y, z = r_ecef
        rnorm = np.linalg.norm(r_ecef)
        lat[k] = np.degrees(np.arcsin(z / rnorm))                 # geocentric latitude
        lon_raw = np.degrees(np.arctan2(y, x))                    # [-180,180]
        lon[k] = ((lon_raw + 180.0) % 360.0) - 180.0              # wrap to [-180,180]
    return t_arr, lat, lon

def _plot_segments(ax, lons_deg, lats_deg, **plot_kwargs):
    lons = np.asarray(lons_deg)
    lats = np.asarray(lats_deg)
    jump = np.where(np.abs(np.diff(lons)) > 180.0)[0]
    starts = np.r_[0, jump + 1]
    ends   = np.r_[jump + 1, len(lons)]
    for s, e in zip(starts, ends):
        if e - s >= 2:
            ax.plot(lons[s:e], lats[s:e], **plot_kwargs)

def plot_ground_tracks_simple(planner: AnalyticRaanPhasingPlanner,
                              sol: dict,
                              hours_gom=1.0, hours_rom=1.0, dt_s=10.0,
                              target_lat_deg=None, target_lon_deg=None):
    """
    Draws GOM track from its epoch and ROM track from final tf (solution).
    Longitude axis uses a dense grid (major 30°, minor 10°).
    """
    # Rates for GOM and ROM
    dOm_G, dom_G, dM_G = planner.dOm_G, planner.dom_G, planner.dM_G
    du_G = dom_G + dM_G

    dOm_R, dom_R, dM_R = j2_rates(planner.rom_a, 0.0, planner.i)
    du_R = dom_R + dM_R

    # Initial angles
    OmG0 = planner.Omega0
    uG0  = planner.u0

    OmR0 = sol['Omega_f_rad']
    uR0  = sol['final_orbit']['M_rad']  # ω=0 ⇒ u = M for circular ROM

    # Propagate both
    tG, latG, lonG = _groundtrack_for_circular(planner.a_G, planner.i, OmG0, uG0,
                                               dOm_G, du_G, planner.epoch,
                                               hours=hours_gom, dt_s=dt_s)
    tR, latR, lonR = _groundtrack_for_circular(planner.rom_a, planner.i, OmR0, uR0,
                                               dOm_R, du_R, sol['tf'],
                                               hours=hours_rom, dt_s=dt_s)
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])

    # Dense grid on longitude axis
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.grid(which='major', linestyle='-', alpha=0.35)
    ax.grid(which='minor', linestyle=':', alpha=0.2)

    _plot_segments(ax, lonG, latG, color='tab:blue',  linewidth=1.8, label='GOM (from epoch)')
    _plot_segments(ax, lonR, latR, color='tab:red',   linewidth=1.8, label='ROM (from tf)')

    # Optional: mark target
    if target_lat_deg is not None and target_lon_deg is not None:
        # Wrap target lon to [-180,180]
        lam = ((target_lon_deg + 180.0) % 360.0) - 180.0
        ax.plot([lam], [target_lat_deg], marker='*', markersize=10,
                color='gold', markeredgecolor='k', label='Target')

    ax.legend(loc='upper right')
    ax.set_title("Simple ground tracks (analytic J2 secular propagation)")
    plt.tight_layout()
    plt.show()

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
        'true_anomaly': rad2deg(sol['final_orbit']['M_rad']),
    }
    rom_orbit_test = {
        'semi_major_axis': rom_a_km,
        'eccentricity': 0.0,
        'inclination': i_deg,
        'RAAN': RAANf_deg,
        'argument_of_periapsis': 0.0,
        'true_anomaly': M_ROM_start_deg,
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

def plot_ground_tracks_simple(planner: AnalyticRaanPhasingPlanner,
                              sol: dict,
                              hours_gom=1.0, hours_rom=1.0, dt_s=10.0,
                              target_lat_deg=None, target_lon_deg=None):
    """
    Draws GOM track from its epoch and ROM track from final tf (solution).
    Longitude axis uses a dense grid (major 30°, minor 10°).
    """
    # ...existing code...
    plt.show()

# ---------------------------
# Sweep Δa and plot times vs Δa
# ---------------------------

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

def plot_reconfig_times_vs_delta_a(
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
    dAs, totals, waits, drifts = compute_reconfig_times_vs_delta_a(
        gom, rom_a_km, epoch, target_lat_geod_deg, target_lon_deg,
        delta_min_km, delta_max_km, delta_step_km, max_wrap, drift_time_cap_days, verbose
    )
    if verbose:
        print("[plot] Rendering figure...", flush=True)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 9))

    axs[0].plot(dAs, totals, color='k', lw=1.8)
    axs[0].set_ylabel("Total time [s]")
    axs[0].grid(True, which='both', alpha=0.3)

    axs[1].plot(dAs, waits, color='tab:blue', lw=1.8)
    axs[1].set_ylabel("Wait time [s]")
    axs[1].grid(True, which='both', alpha=0.3)

    axs[2].plot(dAs, drifts, color='tab:orange', lw=1.8)
    axs[2].set_ylabel("Drift time [s]")
    axs[2].set_xlabel("Δa [km]")
    axs[2].grid(True, which='both', alpha=0.3)

    axs[2].set_xlim([delta_min_km, delta_max_km])

    # Optional reference at Δa=0
    for ax in axs:
        ax.axvline(0.0, color='0.7', lw=1, ls='--')

    # Title and subtitle with scenario details
    inc_deg = rad2deg(gom.i_rad)
    alt_gom_km = gom.a_km - R_EARTH
    alt_rom_km = rom_a_km - R_EARTH
    dA_points = len(dAs)
    title_main = "Reconfiguration times vs Δa (best branch, J2 analytic planner)"
    title_line1 = (
        f"GOM alt {alt_gom_km:.1f} km, ROM alt {alt_rom_km:.1f} km, inc {inc_deg:.3f}°, "
        f"target ({target_lat_geod_deg:+.3f}°, {target_lon_deg:+.3f}°), "
        f"epoch {epoch.strftime('%Y%m%dT%H%M%SZ')}"
    )
    title_line2 = (
        f"Δa {delta_min_km:g}..{delta_max_km:g} km, step {delta_step_km:g} km (N={dA_points}), "
        f"max_wrap {max_wrap}, drift_cap {drift_time_cap_days:g} days"
    )
    fig.suptitle(f"{title_main}\n{title_line1}\n{title_line2}")
    plt.tight_layout()

    # Dynamic filename based on scenario
    try:
        import os
        tgt_lat_fmt = f"{target_lat_geod_deg:+.3f}".replace("+", "p").replace("-", "m")
        tgt_lon_fmt = f"{target_lon_deg:+.3f}".replace("+", "p").replace("-", "m")
        inc_deg = rad2deg(gom.i_rad)
        fname = (
            f"reconfigTimes_GOM{gom.a_km:.1f}km_"
            f"ROM{rom_a_km:.1f}km_"
            f"inc{inc_deg:.3f}deg_"
            f"tgtLat{tgt_lat_fmt}_tgtLon{tgt_lon_fmt}_"
            f"epoch{epoch.strftime('%Y%m%dT%H%M%SZ')}_"
            f"dAstep{delta_step_km:g}"
            f".png"
        )
        outdir = "results"
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, fname), dpi=200, bbox_inches='tight')
    except Exception:
        pass

    plt.show()

# ---------------------------
# Averaged sweep over altitude and phases
# ---------------------------

def plot_reconfig_times_vs_delta_a_averaged(
    epoch: datetime,
    target_lat_geod_deg: float,
    target_lon_deg: float,
    rom_a_km: float,
    # Δa sweep
    delta_min_km: float = -200.0,
    delta_max_km: float = 200.0,
    delta_step_km: float = 5.0,
    # Averaging over GOM altitude
    gom_alt_min_km: float = 400.0,
    gom_alt_max_km: float = 800.0,
    n_alt_samples: int = 9,                # e.g., 400..800 every 50 km
    # Averaging over initial RAAN/true anomaly
    n_phase_samples: int = 16,             # number of (Ω,u) samples per altitude/inclination
    phase_seed: int = 42,                  # RNG seed for reproducibility
    # Inclination handling (set one fixed value, or average over a range)
    i_deg: float | None = None,            # if None and no range given, defaults to 51.4476847948°
    inc_min_deg: float | None = None,
    inc_max_deg: float | None = None,
    inc_samples: int = 1,
    # Planner knobs
    max_wrap: int = 40,
    drift_time_cap_days: float = 15.0,
    verbose: bool = False,
):
    """
    Produce the same 3-panel plot (total/wait/drift vs Δa), but with values
    averaged over:
      - GOM altitude in [gom_alt_min_km, gom_alt_max_km]
      - random initial RAAN and true anomaly (uniform on [0, 2π))
      - optional inclination range (if inc_samples > 1 and bounds provided)

    Notes on efficiency:
      - For each sample, we reuse the existing vectorized Δa sweep function.
      - Use a moderate number of samples (e.g., 9 altitudes × 16 phases = 144 sweeps).
      - For faster convergence with fewer samples, consider quasi-Monte Carlo
        (Sobol/Latin Hypercube). Here we use a reproducible RNG by default.
    """
    rng = np.random.default_rng(phase_seed)

    # Δa grid
    dAs = np.arange(delta_min_km, delta_max_km + 1e-9, delta_step_km, dtype=float)
    n_pts = dAs.size

    # Decide which inclinations to use
    if inc_samples > 1 and (inc_min_deg is not None) and (inc_max_deg is not None):
        inc_list_deg = np.linspace(inc_min_deg, inc_max_deg, inc_samples)
    else:
        base_i_deg = 51.4476847948 if i_deg is None else i_deg
        inc_list_deg = np.array([base_i_deg], dtype=float)

    # Altitude samples (uniform grid)
    altitudes_km = np.linspace(gom_alt_min_km, gom_alt_max_km, n_alt_samples)

    # Accumulators
    sum_totals = np.zeros(n_pts, dtype=float)
    sum_waits  = np.zeros(n_pts, dtype=float)
    sum_drifts = np.zeros(n_pts, dtype=float)
    counts     = np.zeros(n_pts, dtype=int)

    total_cases = len(inc_list_deg) * n_alt_samples * n_phase_samples
    if verbose:
        print(
            f"[AVG] Starting averaged sweep: incl {len(inc_list_deg)} × alt {n_alt_samples} × phases {n_phase_samples} = {total_cases} sweeps; Δa points per sweep: {n_pts}.",
            flush=True,
        )

    # Loop over inclination, altitude, and phase samples
    cases_done = 0
    for inc_deg in inc_list_deg:
        i_rad_val = deg2rad(inc_deg)

        for alt_km in altitudes_km:
            a_gom = R_EARTH + float(alt_km)

            # Sample (Ω,u) pairs uniformly on [0, 2π)
            phases = rng.random((n_phase_samples, 2)) * (2.0 * pi)

            for k in range(n_phase_samples):
                Om0 = float(phases[k, 0])
                u0  = float(phases[k, 1])

                gom = Orbit(a_km=a_gom, i_rad=i_rad_val, Omega_rad=Om0, u_rad=u0)

                _, totals, waits, drifts = compute_reconfig_times_vs_delta_a(
                    gom=gom,
                    rom_a_km=rom_a_km,
                    epoch=epoch,
                    target_lat_geod_deg=target_lat_geod_deg,
                    target_lon_deg=target_lon_deg,
                    delta_min_km=delta_min_km,
                    delta_max_km=delta_max_km,
                    delta_step_km=delta_step_km,
                    max_wrap=max_wrap,
                    drift_time_cap_days=drift_time_cap_days,
                    verbose=verbose,
                )

                # Accumulate where we have valid solutions
                mask = ~np.isnan(totals)
                sum_totals[mask] += totals[mask]
                sum_waits[mask]  += waits[mask]
                sum_drifts[mask] += drifts[mask]
                counts[mask]     += 1

                cases_done += 1
                if verbose:
                    # Update every ~5% or on completion
                    step = max(1, total_cases // 20)
                    if (cases_done % step == 0) or (cases_done == total_cases):
                        pct = 100.0 * cases_done / total_cases
                        print(f"[AVG] Completed {cases_done}/{total_cases} sweeps ({pct:.0f}%)", flush=True)

    # Compute averages (avoid division by zero)
    with np.errstate(invalid='ignore', divide='ignore'):
        avg_totals = np.where(counts > 0, sum_totals / counts, np.nan)
        avg_waits  = np.where(counts > 0, sum_waits  / counts, np.nan)
        avg_drifts = np.where(counts > 0, sum_drifts / counts, np.nan)

    # Plot
    if verbose:
        print("[AVG] Computing averages and plotting...", flush=True)
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 9))

    axs[0].plot(dAs, avg_totals, color='k', lw=1.8)
    axs[0].set_ylabel("Total time [s]")
    axs[0].grid(True, which='both', alpha=0.3)

    axs[1].plot(dAs, avg_waits, color='tab:blue', lw=1.8)
    axs[1].set_ylabel("Wait time [s]")
    axs[1].grid(True, which='both', alpha=0.3)

    axs[2].plot(dAs, avg_drifts, color='tab:orange', lw=1.8)
    axs[2].set_ylabel("Drift time [s]")
    axs[2].set_xlabel("Δa [km]")
    axs[2].grid(True, which='both', alpha=0.3)
    axs[2].set_xlim([delta_min_km, delta_max_km])

    for ax in axs:
        ax.axvline(0.0, color='0.7', lw=1, ls='--')

    title_suffix = f"avg over alt {gom_alt_min_km:.0f}-{gom_alt_max_km:.0f} km, phases N={n_phase_samples}"
    if len(inc_list_deg) > 1:
        title_suffix += f", incl {inc_min_deg:.1f}-{inc_max_deg:.1f}° (N={len(inc_list_deg)})"
    else:
        title_suffix += f", incl {inc_list_deg[0]:.3f}°"

    fig.suptitle(f"Reconfiguration times vs Δa (averaged)\n{title_suffix}")
    plt.tight_layout()

    # Save under results with a descriptive name
    try:
        import os
        tgt_lat_fmt = f"{target_lat_geod_deg:+.3f}".replace("+", "p").replace("-", "m")
        tgt_lon_fmt = f"{target_lon_deg:+.3f}".replace("+", "p").replace("-", "m")

        if len(inc_list_deg) > 1:
            inc_tag = f"incAvg{inc_min_deg:.0f}-{inc_max_deg:.0f}"
        else:
            inc_tag = f"inc{inc_list_deg[0]:.3f}"

        fname = (
            f"reconfigTimes_AVG_"
            f"GOMalt{gom_alt_min_km:.0f}-{gom_alt_max_km:.0f}km_"
            f"{inc_tag}_"
            f"phN{n_phase_samples}_altN{n_alt_samples}_"
            f"ROM{rom_a_km:.1f}km_"
            f"tgtLat{tgt_lat_fmt}_tgtLon{tgt_lon_fmt}_"
            f"epoch{epoch.strftime('%Y%m%dT%H%M%SZ')}_"
            f"dAstep{delta_step_km:g}"
            f".png"
        )
        outdir = "results"
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, fname), dpi=200, bbox_inches='tight')
    except Exception:
        pass

    plt.show()

# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # Example inputs (your sample)
    epoch = datetime(2025, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    i_deg = 51.4476847948

    gom = Orbit(
        a_km = R_EARTH + 500.0,
        i_rad= deg2rad(i_deg),
        Omega_rad = deg2rad(0.0),
        u_rad     = deg2rad(0.0)
    )

    # plot_reconfig_times_vs_delta_a(
    #     gom=gom,
    #     rom_a_km=R_EARTH + 504.59534097,
    #     epoch=epoch,
    #     target_lat_geod_deg=3.295,
    #     target_lon_deg=95.982,
    #     delta_min_km=-100,
    #     delta_max_km=200,
    #     delta_step_km=10,
    #     max_wrap=150,              # keep modest for speed during sweep
    #     drift_time_cap_days=15.0,  # cap drift to avoid very long solutions
    #     verbose=True
    # )

    # # Also produce the averaged plot across GOM altitudes (400–800 km)
    # Targets list: add as many (lat_deg, lon_deg) pairs as you like
    targets = [
        # sumatra 2004
        (3.295, 95.982),
        # los angeles
        (34.0522, -118.2437),
        # # Zero target (equator, prime meridian)
        (0.0, 0.0),
        # Rome
        (41.9028, 12.4964),
        # Hawaii
        (21.3069, -157.8583),
        # San Francisco
        (37.7749, -122.4194),
        # San Diego
        (32.7157, -117.1611),
        # Vancouver
        (49.2827, -123.1207),
        # New York City
        (40.7128, -74.0060),
        # London
        (51.5074, -0.1278),
        # Sydney
        (-33.8688, 151.2093),
        # Cape Town
        (-33.9249, 18.4241),
        # Buenos Aires
        (-34.6037, -58.3816),
        # Moscow
        (55.7558, 37.6173),
        # Tokyo
        (35.6895, 139.6917),
        # Singapore
        (1.3521, 103.8198),
        # Mumbai
        (19.0760, 72.8777),
        # Cairo
        (30.0444, 31.2357),
        # Nairobi
        (-1.2921, 36.8219),
        # Antarctica (McMurdo Station)
        (-77.8419, 166.6863),
        # Easter Island
        (-27.1127, -109.3497),
        # Tel Aviv
        (32.0853, 34.7818),
        # Miami
        (25.7617, -80.1918),
        # Athens
        (37.9838, 23.7275),
        # Bejing
        (39.9042, 116.4074),
        # Mexico City
        (19.4326, -99.1332),
        # Rio de Janeiro
        (-22.9068, -43.1729)
    ]

    for t_idx, (tlat, tlon) in enumerate(targets, start=1):
        print(f"\n[MAIN] Target {t_idx}/{len(targets)}: lat={tlat:+.3f} deg, lon={tlon:+.3f} deg", flush=True)

        # Single-scenario Δa sweep
        plot_reconfig_times_vs_delta_a(
            gom=gom,
            rom_a_km=R_EARTH + 504.59534097,
            epoch=epoch,
            target_lat_geod_deg=tlat,
            target_lon_deg=tlon,
            delta_min_km=-150,
            delta_max_km=200,
            delta_step_km=10,
            max_wrap=200,              # search breadth for (m,k)
            drift_time_cap_days=15.0,  # cap drift to avoid very long solutions
            verbose=True
        )

        # # Averaged plot across GOM altitudes (400–800 km) and random phases
        # plot_reconfig_times_vs_delta_a_averaged(
        #     epoch=epoch,
        #     target_lat_geod_deg=tlat,
        #     target_lon_deg=tlon,
        #     rom_a_km=R_EARTH + 504.59534097,
        #     delta_min_km=-200,
        #     delta_max_km=200,
        #     delta_step_km=10,
        #     gom_alt_min_km=400.0,
        #     gom_alt_max_km=800.0,
        #     n_alt_samples=9,
        #     n_phase_samples=16,
        #     phase_seed=42,
        #     i_deg=i_deg,              # keep inclination fixed; set inc_min/max to average over it
        #     max_wrap=80,
        #     drift_time_cap_days=15.0,
        #     verbose=True,
        # )

    #     # # Also print the other branch, for comparison
    #     # print("\n---- Ascending solution ----")
    #     # print(sol['ascending'])
    #     # print("\n---- Descending solution ----")
    #     # print(sol['descending'])

    #     simple_plotter_GOM_ROM(planner, sol, 3.295, 95.982)

    # else:
    #     print("\nNo solution from planner; cannot plot ROM ground track.")
    # planner = AnalyticRaanPhasingPlanner(
    #     gom=gom,
    #     rom_a_km=R_EARTH + 504.59534097,
    #     delta_a_km=50.0,
    #     epoch_gom=epoch,
    #     target_lat_geod_deg=3.295,   # geodetic latitude
    #     target_lon_deg=95.982,       # east-positive
    #     max_wrap=100                 # search breadth for (m,k)
    # )

    # sol = planner.solve_min_time()
    # print("\n==== Solver output (best branch) ====")
    # print(sol)

    # if sol.get('success', False):
    #     print("\nTimes (seconds):")
    #     print(f"  T_wait = {sol['T_wait']:.3f}")
    #     print(f"  T1     = {sol['T1']:.3f}")
    #     print(f"  T_d    = {sol['T_d']:.3f}")
    #     print(f"  T2     = {sol['T2']:.3f}")
    #     print(f"  Total  = {sol['total_time_s']:.3f}")
    #     print(f"Final epoch tf: {sol['tf'].isoformat()}")

    #     print("\nROM orbital parameters (radians + epoch):")
    #     print(sol['final_orbit'])
    #     print("\nROM orbital parameters (degrees, human-friendly):")
    #     print(sol['final_orbit_deg'])

    #     # Also print the other branch, for comparison
    #     print("\n---- Ascending solution ----")
    #     print(sol['ascending'])
    #     print("\n---- Descending solution ----")
    #     print(sol['descending'])

    #     # Optional quick stub to avoid NameError if you call the plotter.
    #     simple_plotter_GOM_ROM(planner, sol, 3.295, 95.982)

    #     if sol.get('success', False):
    #         # … your prints …
    #         plot_ground_tracks_simple(
    #             planner, sol,
    #             hours_gom=1.0, hours_rom=1.0, dt_s=10.0,
    #             target_lat_deg=planner.target_lat_deg,
    #             target_lon_deg=np.degrees(planner.target_lon_rad)
    # )
        # else:
        #     print("\nNo solution from planner; cannot plot ROM ground track.")
