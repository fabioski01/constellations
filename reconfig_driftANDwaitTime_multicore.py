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

Author: Fabio Meloni - TU Delft MSc Student - NASA JPL Visiting Researcher
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
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import sys
from reconfig_phasing_RGT import PhasingRGT

# Print guard so we only announce core usage once per process
_PRINTED_CORE_CONFIG = False
_CORE_BANNER_PID = os.getpid()  # track which process printed
_PRINTED_SOLVER_MODE_PIDS: set = set()  # print mode banner once per PID


def _env_truthy(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _print_core_banner(n_jobs: int, requested, env_jobs_val, mp_start: str) -> None:
    # Print to stderr to avoid buffering/capture issues
    try:
        try:
            aff = os.sched_getaffinity(0)
            aff_count = len(aff)
        except Exception:
            aff = None
            aff_count = os.cpu_count() or 1
        sys.stderr.write(
            f"[solver] n_jobs={n_jobs} cpu_max={os.cpu_count()} "
            f"affinity_count={aff_count} requested={requested} env={env_jobs_val} "
            f"mp_start={mp_start} pid={os.getpid()}\n"
        )
        if aff is not None:
            try:
                sys.stderr.write(
                    f"[solver] CPU affinity set: {sorted(list(aff))}\n")
            except Exception:
                pass
        sys.stderr.flush()
    except Exception:
        pass


# ---------------------------
# Constants and small helpers
# ---------------------------
target_planet = 'mars'  # 'earth' or 'mars'

if target_planet == 'earth':
    # Earth (WGS-84 / IERS-style)
    # km (WGS-84 mean equatorial radius)
    R_EARTH = constants.EARTH_RADIUS
    MU_EARTH = constants.GRAVITATIONAL_CONSTANT       # km^3/s^2
    J2 = constants.J2_EARTH                           # Earth's J2

    WGS84_f = 1.0 / 298.257223563
    WGS84_e2 = WGS84_f * (2.0 - WGS84_f)

    # "Planet-generic" ellipsoid parameters (used everywhere below)
    ELLIPSOID_F = WGS84_f
    ELLIPSOID_E2 = WGS84_e2

elif target_planet == 'mars':
    # Mars (IAU / MOLA reference ellipsoid)
    R_EARTH = constants.MARS_RADIUS                   # km (Mars mean radius)
    MU_EARTH = constants.MARS_GRAVITATIONAL_CONSTANT  # km^3/s^2
    J2 = constants.J2_MARS                            # Mars J2

    # IAU 2015 / USGS Mars reference ellipsoid
    MARS_A = 3396.19  # km, equatorial radius
    MARS_B = 3376.20  # km, polar radius
    MARS_ELLIPSOID_F = (MARS_A - MARS_B) / MARS_A
    MARS_ELLIPSOID_E2 = MARS_ELLIPSOID_F * (2.0 - MARS_ELLIPSOID_F)

    ELLIPSOID_F = MARS_ELLIPSOID_F
    ELLIPSOID_E2 = MARS_ELLIPSOID_E2

else:
    raise ValueError(f"Unsupported target_planet: {target_planet}")


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
    Planetary sidereal/prime-meridian angle in inertial space at UT1≈UTC.

    - Earth: IAU-82-style GMST (Greenwich sidereal angle).
      GMST [deg] = 280.46061837 + 360.98564736629*(JD-2451545.0)
                   + 0.000387933*T^2 - T^3/38710000
      T = (JD-2451545.0)/36525

    - Mars: IAU-2000-style prime meridian angle W for Mars:
      W [deg] = 176.630 + 350.89198226 * d
      d = JD - 2451545.0   (days since J2000.0; TDB≈UTC here)

    Returns wrapped radians in [0, 2π).
    """
    JD = datetime_to_jd(dt)
    D = JD - 2451545.0

    if target_planet == 'earth':
        T = D / 36525.0
        gmst_deg = 280.46061837 + 360.98564736629 * D \
            + 0.000387933 * T * T - (T**3) / 38710000.0
        return wrap_2pi(deg2rad(gmst_deg))

    elif target_planet == 'mars':
        # Mars prime meridian angle W(t) (Airy-0 / VL1 frame)
        W_deg = 176.630 + 350.89198226 * D
        return wrap_2pi(deg2rad(W_deg))

    else:
        raise ValueError(f"Unsupported target_planet: {target_planet}")


# ---------------------------
# Geodetic ↔ geocentric latitude
# ---------------------------


def geodetic_to_geocentric_latitude(phi_gd: float) -> float:
    """
    Convert geodetic latitude (rad) to geocentric latitude (rad) using the
    current planet's reference ellipsoid:

        tan φ_c = (1 - e^2) * tan φ_gd

    where e^2 = ELLIPSOID_E2 (WGS-84 for Earth, IAU/MOLA ellipsoid for Mars).
    """
    return atan((1.0 - ELLIPSOID_E2) * tan(phi_gd))


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
    domega = 0.75 * J2 * n * R_over_p2 * (5.0 * ci2 - 1.0)
    dM = n + 0.75 * J2 * n * R_over_a2 * \
        ((1.0 - e * e) ** -1.5) * (3.0 * ci2 - 1.0)
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
        drift_time_cap_days: float = 30.0,
        n_jobs: Optional[int] = None,
        executor: Optional[ProcessPoolExecutor] = None,
        mp_start: Optional[str] = None,
    ):
        self.gom = gom
        self.rom_a = rom_a_km
        self.delta_a = delta_a_km
        self.epoch = epoch_gom
        self.target_lat_deg = target_lat_geod_deg
        self.target_lon_rad = deg2rad(target_lon_deg)
        self.max_wrap = max_wrap
        self.drift_time_cap = drift_time_cap_days * 86400.0  # s
        # Parallelism for inner-branch solve (m-grid).
        cpu_max = os.cpu_count() or 1
        env_jobs = os.getenv("RECONFIG_N_JOBS")
        env_jobs_val: Optional[int] = None
        if env_jobs is not None:
            try:
                env_jobs_val = int(env_jobs)
            except Exception:
                env_jobs_val = None
        requested = n_jobs if n_jobs is not None else env_jobs_val
        if requested is None or requested <= 0:
            self.n_jobs = cpu_max
        else:
            self.n_jobs = int(max(1, min(requested, cpu_max)))
        # Info flag to avoid repeating prints during branch solves
        self._printed_cores: bool = False
        # Optional shared executor and start method
        self._executor: Optional[ProcessPoolExecutor] = executor
        self._mp_start = (
            mp_start or os.getenv("RECONFIG_MP_START") or (
                "spawn" if os.name == "nt" else "fork")
        )

        # Reduce risk of oversubscription inside workers
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        # Announce configured core usage once per process
        global _PRINTED_CORE_CONFIG, _CORE_BANNER_PID
        force_print = _env_truthy("RECONFIG_PRINT_CORES", False)
        if os.getpid() != _CORE_BANNER_PID:
            _CORE_BANNER_PID = os.getpid()
            _PRINTED_CORE_CONFIG = False
        if not _PRINTED_CORE_CONFIG or force_print:
            try:
                try:
                    aff = os.sched_getaffinity(0)
                    aff_count = len(aff)
                except Exception:
                    aff = None
                    aff_count = os.cpu_count() or 1
                print(
                    f"[solver] Configured n_jobs={self.n_jobs} "
                    f"(cpu_max={os.cpu_count()}, affinity_count={aff_count}, requested={requested}, env={env_jobs_val}, mp_start={self._mp_start})",
                    flush=True,
                )
                if aff is not None:
                    try:
                        print(
                            f"[solver] CPU affinity set: {sorted(list(aff))}", flush=True)
                    except Exception:
                        pass
            except Exception:
                pass
            _PRINTED_CORE_CONFIG = True

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
        self.dOm_T1, self.dom_T1, self.dM_T1 = j2_rates(
            self.aT1, self.eT1, self.i)
        self.dOm_T2, self.dom_T2, self.dM_T2 = j2_rates(
            self.aT2, self.eT2, self.i)

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

        asc_sol = self._solve_branch(
            'ascending', u_asc, lon_tol_schedule=lon_tol_schedule)
        desc_sol = self._solve_branch(
            'descending', u_desc, lon_tol_schedule=lon_tol_schedule)

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
        NSCAN_KDISC = 160         # discover k across Td interval 100-160
        # bracket across Td for (r==0) and band edges 120-160
        NSCAN_BRACKET = 160
        K_NEIGHBOR_PAD = 8  # 2-5
        LON_TOL_SCHEDULE = lon_tol_schedule or [0.25]  # deg (tune as you like)
        WAIT_CAP_HR = 200       # set None to disable wait cap. Usually is never more than 150 hr
        WAIT_CAP_S = None if WAIT_CAP_HR is None else WAIT_CAP_HR * 3600.0
        # -----------------------

        # Build m-range sequentially here; evaluation per-m can be parallelized
        all_candidates: List[Dict[str, Any]] = []

        # Branch-fixed constants and caps
        # Eq A uses const_u; WAIT cap converts to seconds if set
        const_u = -(self.du_T1 * self.T1 + self.du_T2 *
                    self.T2) + (u_req - self.u0)
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
        PAD = 2
        m_list = list(range(m_min - PAD, m_max + PAD + 1))

        # Build immutable state for workers
        state = _BranchSolveState(
            i=self.i,
            Omega0=self.Omega0,
            u0=self.u0,
            T1=self.T1,
            T2=self.T2,
            epoch=self.epoch,
            target_lon_rad=self.target_lon_rad,
            drift_time_cap=self.drift_time_cap,
            du_G=self.du_G,
            du_D=self.du_D,
            dOm_G=self.dOm_G,
            dOm_T1=self.dOm_T1,
            dOm_D=self.dOm_D,
            dOm_T2=self.dOm_T2,
            du_T1=self.du_T1,
            du_T2=self.du_T2,
            rom_a=self.rom_a,
            LON_TOL_SCHEDULE=LON_TOL_SCHEDULE,
            WAIT_CAP_S=WAIT_CAP_S,
            NSCAN_KDISC=NSCAN_KDISC,
            NSCAN_BRACKET=NSCAN_BRACKET,
            K_NEIGHBOR_PAD=K_NEIGHBOR_PAD,
        )

        # Decide whether to run parallel or sequential
        do_parallel = (self.n_jobs > 1) and (len(m_list) >= 4)

        global _PRINTED_SOLVER_MODE_PIDS
        if os.getpid() not in _PRINTED_SOLVER_MODE_PIDS:
            try:
                mode = "parallel" if do_parallel else "sequential"
                cpu_id = None
                try:
                    cpu_id = os.sched_getcpu()
                except Exception:
                    pass
                sys.stderr.write(
                    f"[solver] {mode}: n_jobs={self.n_jobs}, m_candidates={len(m_list)}, "
                    f"mp_start={self._mp_start}, pid={os.getpid()}"
                    + (f", cpu={cpu_id}" if cpu_id is not None else "") + "\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
            _PRINTED_SOLVER_MODE_PIDS.add(os.getpid())

        if do_parallel:
            # Fan out per-m solve to processes (reuse shared executor when possible)
            ex = self._executor
            created_local = False
            if ex is None:
                ex = ProcessPoolExecutor(
                    max_workers=self.n_jobs, mp_context=get_context(self._mp_start))
                created_local = True
            try:
                for chunk in ex.map(
                    _solve_branch_m_task,
                    [state]*len(m_list),
                    [u_req]*len(m_list),
                    [const_u]*len(m_list),
                    [branch_name]*len(m_list),
                    m_list,
                ):
                    if chunk:
                        all_candidates.extend(chunk)
            finally:
                if created_local:
                    ex.shutdown(wait=True)
        else:
            # Sequential evaluation using the same worker
            for m in m_list:
                chunk = _solve_branch_m_task(
                    state, u_req, const_u, branch_name, m)
                if chunk:
                    all_candidates.extend(chunk)

        # -------- selection: earliest within tolerance ladder --------
        if not all_candidates:
            return {'success': False, 'branch': branch_name,
                    'note': 'No solution within wait/drift caps.'}
        tol_used = None
        best = None
        for tol_deg in LON_TOL_SCHEDULE:
            feas = [c for c in all_candidates if abs(
                c['lon_err_deg']) <= tol_deg]
            if feas:
                best = min(feas, key=lambda s: s['total_time_s'])
                tol_used = tol_deg
                break
        if best is None:
            # No candidate inside the loosest tol: pick smallest error, then earliest
            best = min(all_candidates, key=lambda s: (
                abs(s['lon_err_deg']), s['total_time_s']))

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


# ------------- parallel worker plumbing -------------

@dataclass
class _BranchSolveState:
    # Orbital and timing constants
    i: float
    Omega0: float
    u0: float
    T1: float
    T2: float
    epoch: datetime
    target_lon_rad: float
    drift_time_cap: float

    # Rates
    du_G: float
    du_D: float
    dOm_G: float
    dOm_T1: float
    dOm_D: float
    dOm_T2: float
    du_T1: float
    du_T2: float

    # Other
    rom_a: float
    LON_TOL_SCHEDULE: List[float]
    WAIT_CAP_S: Optional[float]
    NSCAN_KDISC: int
    NSCAN_BRACKET: int
    K_NEIGHBOR_PAD: int


def _bisection_root(f, a: float, b: float) -> Optional[float]:
    fa = f(a)
    fb = f(b)
    if fa == 1e9 or fb == 1e9 or fa*fb > 0.0:
        return None
    for _ in range(80):
        m = 0.5*(a+b)
        fm = f(m)
        if fm == 1e9:
            a = m
            fa = f(a)
            continue
        if abs(fm) < 1e-12 or (b-a) < 1e-6:
            return m
        if fa*fm <= 0.0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    return 0.5*(a+b)


def _solve_branch_m_task(state: _BranchSolveState, u_req: float, const_u: float, branch_name: str, m: int) -> List[Dict[str, Any]]:
    """Compute all candidate solutions for a single m value. Designed to be picklable."""
    # Unpack
    i = state.i
    Omega0 = state.Omega0
    u0 = state.u0
    T1 = state.T1
    T2 = state.T2
    epoch = state.epoch
    target_lon = state.target_lon_rad
    du_G = state.du_G
    du_D = state.du_D
    dOm_G = state.dOm_G
    dOm_T1 = state.dOm_T1
    dOm_D = state.dOm_D
    dOm_T2 = state.dOm_T2
    du_T1 = state.du_T1
    du_T2 = state.du_T2
    rom_a = state.rom_a

    WAIT_CAP_S = state.WAIT_CAP_S
    NSCAN_KDISC = state.NSCAN_KDISC
    NSCAN_BRACKET = state.NSCAN_BRACKET
    K_NEIGHBOR_PAD = state.K_NEIGHBOR_PAD
    tol_rads = [deg2rad(t) for t in state.LON_TOL_SCHEDULE]

    # geometry offset
    psi = atan2(cos(i) * sin(u_req), cos(u_req))

    # time caps
    td_base_lo = 0.0
    td_base_hi = state.drift_time_cap
    Tw_cap = float('inf') if WAIT_CAP_S is None else WAIT_CAP_S

    candidates: List[Dict[str, Any]] = []

    A_m = const_u + 2.0*pi*m

    # Handle degenerate slopes
    if abs(du_D) < 1e-16 or abs(du_G) < 1e-16:
        return candidates

    # Tw=0 root
    t0 = A_m / du_D

    # Build feasible Td interval given wait cap
    if WAIT_CAP_S is None:
        s = -du_D / du_G  # dTw/dTd
        if s < 0.0:
            td_lo = td_base_lo
            td_hi = min(td_base_hi, t0)
        else:
            td_lo = max(td_base_lo, t0)
            td_hi = td_base_hi
    else:
        tc = (A_m - du_G * WAIT_CAP_S) / du_D
        td_lo = max(td_base_lo, min(t0, tc))
        td_hi = min(td_base_hi, max(t0, tc))

    if td_hi <= td_lo + 1e-6:
        return candidates

    def Tw_of_Td(Td: float, A_m_local: float) -> float:
        return (A_m_local - du_D * Td) / du_G

    def Omega_f_of_Td(Td: float, Tw: float) -> float:
        return (Omega0 + dOm_G * Tw + dOm_T1 * T1 + dOm_D * Td + dOm_T2 * T2)

    def total_time(Td: float, Tw: float) -> float:
        return Tw + T1 + Td + T2

    def phi_of(Td: float, A_m_local: float) -> Tuple[float, float, datetime]:
        Tw = Tw_of_Td(Td, A_m_local)
        tf = epoch + timedelta(seconds=total_time(Td, Tw))
        thetaG = gmst_rad(tf)
        phi = Omega_f_of_Td(Td, Tw) + psi - thetaG
        return phi, Tw, tf

    def package_candidate(Td_sol: float, Tw_sol: float, k_fix: int, tag: str) -> None:
        Ttot = total_time(Td_sol, Tw_sol)
        tf = epoch + timedelta(seconds=Ttot)
        Omega_f = wrap_2pi(Omega_f_of_Td(Td_sol, Tw_sol))
        u_f = wrap_2pi(u_req)
        M_f = u_f
        thetaG_tf = gmst_rad(tf)
        psi_final = atan2(cos(i) * sin(u_f), cos(u_f))
        lam_f = wrap_pm_pi(Omega_f + psi_final - thetaG_tf)
        lon_err = wrap_pm_pi(lam_f - target_lon)

        candidates.append({
            'success': True, 'branch': branch_name,
            'm_u_wrap': m, 'k_lon_wrap': k_fix,
            'T_wait': Tw_sol, 'T1': T1, 'T_d': Td_sol, 'T2': T2,
            'total_time_s': Ttot, 'tf': tf,
            'Omega_f_rad': Omega_f, 'u_f_rad': u_f,
            'alpha_f_rad': ra_of_position(i, u_f, Omega_f),
            'lambda_f_rad': lam_f, 'lambda_f_deg': rad2deg(lam_f),
            'lon_err_deg': rad2deg(lon_err), 'hit_type': tag,
            'final_orbit': {
                'a_km': rom_a, 'e': 0.0, 'i_rad': i,
                'Omega_rad': Omega_f, 'omega_rad': 0.0, 'M_rad': M_f, 'epoch': tf
            },
            'final_orbit_deg': {
                'a_km': rom_a, 'e': 0.0, 'i_deg': rad2deg(i),
                'RAAN_deg': rad2deg(Omega_f), 'arg_perigee_deg': 0.0,
                'M_deg': rad2deg(M_f), 'epoch': tf
            }
        })

    # Discover k across [td_lo, td_hi]
    kset: set[int] = set()
    for j in range(NSCAN_KDISC + 1):
        t_cur = td_lo + (td_hi - td_lo) * (j / NSCAN_KDISC)
        phi_cur, Tw_cur, _ = phi_of(t_cur, A_m)
        if Tw_cur < -1e-9 or (WAIT_CAP_S is not None and Tw_cur - WAIT_CAP_S > 1e-9):
            continue
        k_near = int(round((phi_cur - target_lon) / (2.0*pi)))
        kset.add(k_near)
    if not kset:
        return candidates
    kmin, kmax = min(kset), max(kset)
    for dk in range(1, K_NEIGHBOR_PAD + 1):
        kset.add(kmin - dk)
        kset.add(kmax + dk)

    # For each fixed k, bracket/solve
    for k_fix in sorted(kset):
        def r_of_Td(Td: float) -> Tuple[float, float]:
            phi, Tw, _ = phi_of(Td, A_m)
            if Tw < -1e-12 or (WAIT_CAP_S is not None and Tw - WAIT_CAP_S > 1e-12):
                return 1e9, Tw
            return phi - (target_lon + 2.0*pi*k_fix), Tw

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

        # exact roots r==0
        for (a, b, ra, rb) in segments:
            if (ra == 1e9) or (rb == 1e9):
                continue
            if ra * rb <= 0.0:
                Td0 = _bisection_root(lambda t: r_of_Td(t)[0], a, b)
                if Td0 is not None:
                    r0, Tw0 = r_of_Td(Td0)
                    if Tw0 >= 0.0 and (WAIT_CAP_S is None or Tw0 <= WAIT_CAP_S):
                        package_candidate(Td0, Tw0, k_fix, 'r==0')

        # band edges |r|=tol
        for tol_rad in tol_rads:
            for (a, b, ra, rb) in segments:
                if (ra == 1e9) or (rb == 1e9):
                    continue
                rma, rmb = ra - tol_rad, rb - tol_rad
                rpa, rpb = ra + tol_rad, rb + tol_rad
                if rma * rmb <= 0.0:
                    Td1 = _bisection_root(
                        lambda t, tr=tol_rad: r_of_Td(t)[0] - tr, a, b)
                    if Td1 is not None:
                        r1, Tw1 = r_of_Td(Td1)
                        if Tw1 >= 0.0 and (WAIT_CAP_S is None or Tw1 <= WAIT_CAP_S):
                            package_candidate(Td1, Tw1, k_fix, '|r|=tol')
                if rpa * rpb <= 0.0:
                    Td2 = _bisection_root(
                        lambda t, tr=tol_rad: r_of_Td(t)[0] + tr, a, b)
                    if Td2 is not None:
                        r2, Tw2 = r_of_Td(Td2)
                        if Tw2 >= 0.0 and (WAIT_CAP_S is None or Tw2 <= WAIT_CAP_S):
                            package_candidate(Td2, Tw2, k_fix, '|r|=tol')

        # near-miss fallback
        if best_sample[1] is not None and not any(c for c in candidates if c['m_u_wrap'] == m and c['k_lon_wrap'] == k_fix):
            Td_nm = float(best_sample[1])
            r_nm, Tw_nm = r_of_Td(Td_nm)
            if Tw_nm >= 0.0 and (WAIT_CAP_S is None or Tw_nm <= WAIT_CAP_S):
                package_candidate(Td_nm, Tw_nm, k_fix, 'near_miss')

    return candidates


# ---------------------------
# Helper to print details
# ---------------------------

def print_solution_summary(sol: dict) -> None:
    meta = sol.get('selection_meta', {})
    wait_hr = sol['T_wait'] / 3600.0
    drift_hr = sol['T_d'] / 3600.0
    tot_hr = sol['total_time_s'] / 3600.0
    tol = meta.get('tolerance_used_deg')
    tol_str = f"{tol:.2f}°" if tol is not None else "— (fallback)"
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
    T1 = float(sol['T1'])
    T_d = float(sol['T_d'])
    T2 = float(sol['T2'])
    Ttot = float(sol['total_time_s'])

    # Orbits (semi-major axes of segments)
    aG = planner.a_G
    aD = planner.a_D
    aT1 = planner.aT1
    eT1 = planner.eT1
    aT2 = planner.aT2
    eT2 = planner.eT2

    # Mean anomaly increments by segment (rad)
    dM_G = mean_motion(aG) * T_wait
    dM_T1 = mean_motion(aT1) * T1
    dM_D = mean_motion(aD) * T_d
    dM_T2 = mean_motion(aT2) * T2

    # Cumulative starts for each segment (deg), wrapped to [0,360)
    M_waiting_deg = (rad2deg(dM_G)) % 360.0
    M_T1_start_deg = (rad2deg(dM_G + dM_T1)) % 360.0
    M_D_start_deg = M_T1_start_deg
    M_T2_start_deg = (rad2deg(dM_G + dM_T1 + dM_D)) % 360.0
    M_ROM_start_deg = (rad2deg(dM_G + dM_T1 + dM_D + dM_T2)) % 360.0

    # Epochs per segment start
    epoch_T1 = epoch + timedelta(seconds=T_wait)
    epoch_D = epoch_T1 + timedelta(seconds=T1)
    epoch_T2 = epoch_D + timedelta(seconds=T_d)
    epoch_ROM = epoch_T2 + timedelta(seconds=T2)  # should match sol['tf']

    # Angles/RAANs
    RAAN0_deg = rad2deg(planner.Omega0)
    RAANf_deg = rad2deg(sol['Omega_f_rad'])
    i_deg = rad2deg(planner.i)

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
        'argument_of_periapsis': 0.0, \
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
    delta_max_km: float = 200.0,
    delta_step_km: float = 5.0,
    max_wrap: int = 300,
    drift_time_cap_days: float = 15.0,
    verbose: bool = False,
    n_jobs: Optional[int] = None,
    lon_tol_schedule: Optional[List[float]] = None,  # <-- added
):
    """
    Returns (delta_a_km, total_time_s, wait_time_s, drift_time_s), using NaN where no solution exists.
    """
    dAs = np.arange(delta_min_km, delta_max_km +
                    1e-9, delta_step_km, dtype=float)
    totals = np.full(dAs.shape, np.nan, dtype=float)
    waits = np.full(dAs.shape, np.nan, dtype=float)
    drifts = np.full(dAs.shape, np.nan, dtype=float)
    # if verbose:
    #     print(f"[Δa sweep] {len(dAs)} steps from {delta_min_km} to {delta_max_km} km (step {delta_step_km})...", flush=True)
    # Determine progress granularity (~10% of steps)
    progress_step = max(1, len(dAs)//10)

    # Resolve worker count and build a shared executor for the sweep
    cpu_max = os.cpu_count() or 1
    env_jobs = os.getenv("RECONFIG_N_JOBS")
    env_jobs_val: Optional[int] = None
    if env_jobs is not None:
        try:
            env_jobs_val = int(env_jobs)
        except Exception:
            env_jobs_val = None
    requested = n_jobs if n_jobs is not None else env_jobs_val
    use_workers = cpu_max if (requested is None or requested <= 0) else int(
        max(1, min(requested, cpu_max)))
    start_method = os.getenv("RECONFIG_MP_START") or (
        "spawn" if os.name == "nt" else "fork")
    shared_ex: Optional[ProcessPoolExecutor] = None
    if use_workers > 1:
        # if verbose:
        #     print(f"[Δa sweep] Creating shared pool: workers={use_workers} start={start_method}", flush=True)
        shared_ex = ProcessPoolExecutor(
            max_workers=use_workers, mp_context=get_context(start_method))

    try:
        for idx, dA in enumerate(dAs):
            # if verbose and ((idx % progress_step) == 0 or idx == 0):
            #     print(f"[Δa sweep] starting {idx+1}/{len(dAs)}: Δa={dA:g} km", flush=True)
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
                drift_time_cap_days=drift_time_cap_days,
                n_jobs=use_workers,
                executor=shared_ex,
                mp_start=start_method,
            )
            sol = planner.solve_min_time(
                lon_tol_schedule=lon_tol_schedule)  # <-- pass through
            if sol.get('success'):
                totals[idx] = sol['total_time_s']
                waits[idx] = sol['T_wait']
                drifts[idx] = sol['T_d']
            # if verbose:
            #     # Print every ~10% progress
            #     if (idx + 1) % progress_step == 0 or (idx + 1) == len(dAs):
            #         pct = 100.0 * (idx + 1) / len(dAs)
            #         print(f"[Δa sweep] {idx+1}/{len(dAs)} ({pct:.0f}%)", flush=True)
    finally:
        if shared_ex is not None:
            shared_ex.shutdown(wait=True)
    # if verbose:
        # print("[Δa sweep] done.", flush=True)
    return dAs, totals, waits, drifts


# ---------------------------
# Usage
# ---------------------------

if __name__ == "__main__":
    # Import the new plotter module
    from reconfig_plotter_multicore import run_reconfig_plots, PlotFlags

    # Example inputs (your sample)
    epoch = datetime(2025, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    # i_deg = 51.4476847948

    # gom = Orbit(
    #     a_km = R_EARTH + 500.0,
    #     i_rad= deg2rad(i_deg),
    #     Omega_rad = deg2rad(0.0),
    #     u_rad     = deg2rad(0.0)
    # )
    i_deg = 50.0  # or whatever inclination you actually want

    gom = Orbit(
        a_km=R_EARTH + 300.0,   # 300 km altitude around Mars
        i_rad=deg2rad(i_deg),
        Omega_rad=deg2rad(0.0),
        u_rad=deg2rad(0.0)
    )

    # ROM altitude (semi-major axis)
    # rom_a_km = R_EARTH + 504.59534097
    # RGT (Repeat Ground Track) ROM settings
    RGT_USE = True
    RGT_No = 15          # revolutions
    RGT_Nd = 1           # days
    RGT_e = 0.0         # eccentricity for RGT solution
    RGT_FALLBACK_ALT_KM = 250.0  # fallback if RGT solver fails

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
    FLAG_PARETO_TIME_VS_DV_AVG = False
    FLAG_ISO_TIME_CONTOURS_AVG = False

    # Diagnostics
    FLAG_PARETO_TIME_VS_DV = False  # averaged across grid
    FLAG_ISO_TIME_CONTOURS = False  # works, to be detailed and expand with avg
    FLAG_CHOSEN_VS_TOL = False  # works
    FLAG_TOL_SENSITIVITY = False  # doesnt work, type error
    FLAG_TARGET_JITTER = False  # doesnt work, type error
    FLAG_EPOCH_SENSITIVITY = False  # works, need to expand with avg and detail
    FLAG_RESIDUAL_CURVES = False  # doesnt work, name error

    # Common knobs
    max_wrap = 200
    drift_time_cap_days = 300.0/24.0  # ~200 hr
    verbose = False

    # Plot parameter sets
    delta_range = dict(delta_min_km=-200, delta_max_km=200, delta_step_km=10)
    tol_ladder = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
                  1.0, 2.0, 5.0, 10.0]  # used by time vs tolerance at given delta a
    # tol_ladder  = [0.001, 0.25]
    delta_a_fixed_km = 50.0

    # Diagnostics params
    iso_levels_hours = [12.0, 24.0, 48.0, 72.0, 96.0]
    tol_sens_series = (True, True, True)  # plot total/wait/drift
    jitter_radius_deg = 1.0
    jitter_samples = 200
    jitter_seed = 7
    jitter_tol_deg = 0.25
    epoch_window_hours = 24.0
    epoch_samples = 49
    epoch_tol_deg = 0.25
    # residuals: leave branch/m/k None to auto-pick best solution, or set explicitly
    residual_branch = None   # 'ascending' or 'descending' or None
    residual_m = None
    residual_k = None
    residual_nsamples = 600

    # -----------------------------------------------
    # Sweep GOM altitude and inclination combinations
    #  - altitudes: 500, 700, 1000 km
    #  - inclinations: 50°, 60°, and 98° only for 500 and 700 km
    # -----------------------------------------------
    ALT_LIST_KM = [500.0, 700.0, 1000.0]
    ALT_LIST_KM = [400, 300.0, 200.0]  # Mars test altitudes
    for alt_km in ALT_LIST_KM:
        incl_list = [50.0, 60.0] if alt_km == 1000.0 else [50.0, 60.0, 96.0]
        for i_deg in incl_list:
            print(f"\n=== Scenario: hG={alt_km:.0f} km, i={i_deg:.2f}° ===")

            # Build GOM and ROM (keep ~+5 km offset for ROM)
            gom = Orbit(
                a_km=R_EARTH + alt_km,
                i_rad=deg2rad(i_deg),
                Omega_rad=deg2rad(0.0),
                u_rad=deg2rad(0.0)
            )

            # Compute ROM from RGT solver for this inclination
            if RGT_USE:
                if target_planet == 'earth':
                    phasing = PhasingRGT()
                    a_solution, converged, iterations = phasing.solve_for_semi_major_axis(
                        i=i_deg, e=RGT_e, No=RGT_No, Nd=RGT_Nd
                    )
                elif target_planet == 'mars':
                    phasing = PhasingRGT(
                        J2=constants.J2_MARS,
                        Rearth=constants.MARS_RADIUS,
                        mu=constants.MARS_GRAVITATIONAL_CONSTANT,
                        rotation_rate_earth=2.0 * pi /
                        (24*3600 + 37*60 + 22.66),  # Mars sidereal day, rad/s
                    )
                    a_solution, converged, iterations = phasing.solve_for_semi_major_axis(
                        i=i_deg, e=RGT_e, No=RGT_No, Nd=RGT_Nd
                    )

                if converged and np.isfinite(a_solution):
                    rom_a_km = float(a_solution)
                    print(f"[RGT] ROM a={rom_a_km:.3f} km (h={rom_a_km - R_EARTH:.1f} km) "
                          f"from No/Nd={RGT_No}/{RGT_Nd}, e={RGT_e} (iters={iterations})")
                else:
                    rom_a_km = R_EARTH + RGT_FALLBACK_ALT_KM
                    print(f"[RGT] WARN: RGT solver failed (iters={iterations}). "
                          f"Falling back to h={RGT_FALLBACK_ALT_KM:.1f} km (a={rom_a_km:.3f} km).")
            else:
                rom_a_km = R_EARTH + RGT_FALLBACK_ALT_KM

            # ===============================
            # 1) Per-target case study plots
            # ===============================
            if any([FLAG_TIMES_VS_DELTA_A, FLAG_TIMES_VS_TOLERANCE,
                    FLAG_PARETO_TIME_VS_DV, FLAG_CHOSEN_VS_TOL,
                    FLAG_TOL_SENSITIVITY, FLAG_TARGET_JITTER,
                    FLAG_EPOCH_SENSITIVITY, FLAG_RESIDUAL_CURVES]):
                for (target_name, tlat, tlon) in targets:
                    print(
                        f"\n[CASE] {target_name} ({tlat:+.3f}°, {tlon:+.3f}°)")

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
            if FLAG_AVG_VS_DELTA_A or FLAG_AVG_VS_TOLERANCE or FLAG_PARETO_TIME_VS_DV_AVG or FLAG_ISO_TIME_CONTOURS_AVG:
                # Latitude band (deg): must be < inclination to ensure access
                lat_band_deg = i_deg - 1.0         # must be < inclination to ensure access

                # Default grid
                # number of latitude samples in [-band,+band]
                n_lat = 100  # 2
                # s3                  # number of longitudes in [-180,180)
                n_lon = 360

                # Adjust grid sizes by inclination
                if i_deg > 90.0:
                    n_lat = 18
                elif abs(i_deg - 60.0) < 1e-9:
                    n_lat = 13

                # Compute ROM from RGT solver for this inclination
                if RGT_USE:
                    if target_planet == 'earth':
                        phasing = PhasingRGT()
                        a_solution, converged, iterations = phasing.solve_for_semi_major_axis(
                            i=i_deg, e=RGT_e, No=RGT_No, Nd=RGT_Nd
                        )
                    elif target_planet == 'mars':
                        phasing = PhasingRGT(
                            J2=constants.J2_MARS,
                            Rearth=constants.MARS_RADIUS,
                            mu=constants.MARS_GRAVITATIONAL_CONSTANT,
                            rotation_rate_earth=2.0 * pi /
                            # Mars sidereal day, rad/s
                            (24*3600 + 37*60 + 22.66),
                        )
                        a_solution, converged, iterations = phasing.solve_for_semi_major_axis(
                            i=i_deg, e=RGT_e, No=RGT_No, Nd=RGT_Nd
                        )
                    if converged and np.isfinite(a_solution):
                        rom_a_km = float(a_solution)
                        print(f"[RGT] ROM a={rom_a_km:.3f} km (h={rom_a_km - R_EARTH:.1f} km) "
                              f"from No/Nd={RGT_No}/{RGT_Nd}, e={RGT_e} (iters={iterations})")
                    else:
                        rom_a_km = R_EARTH + RGT_FALLBACK_ALT_KM
                        print(f"[RGT] WARN: RGT solver failed (iters={iterations}). "
                              f"Falling back to h={RGT_FALLBACK_ALT_KM:.1f} km (a={rom_a_km:.3f} km).")
                else:
                    rom_a_km = R_EARTH + RGT_FALLBACK_ALT_KM

                print("\n[AVG] Generating averaged plots across target grid...")
                run_reconfig_plots(
                    gom=gom,
                    rom_a_km=rom_a_km,
                    epoch=epoch,
                    target_name="AVG",               # not used by averaged plots
                    target_lat_geod_deg=0.0,         # not used by averaged plots
                    target_lon_deg=0.0,              # not used by averaged plots
                    flags=PlotFlags(
                        times_vs_delta_a=False,
                        times_vs_tolerance=False,
                        avg_vs_delta_a=FLAG_AVG_VS_DELTA_A,
                        avg_vs_tolerance=FLAG_AVG_VS_TOLERANCE,
                        heatmap=False,
                        pareto_time_vs_dv_avg=FLAG_PARETO_TIME_VS_DV_AVG,     # added
                        iso_time_contours_avg=FLAG_ISO_TIME_CONTOURS_AVG       # added
                    ),
                    delta_min_km=delta_range["delta_min_km"],
                    delta_max_km=delta_range["delta_max_km"],
                    delta_step_km=delta_range["delta_step_km"],
                    # only for tol plot with multiple deltaA
                    delta_a_fixed_km=[50, 100, 200, 400],
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
                # Default grid
                n_lat_heatmap = 110  # 2  # 110          # across inclination band
                n_long_heatmap = 360  # 3  # 360         # around Earth

                # Adjust grid sizes by inclination
                if i_deg > 90.0:
                    n_lat_heatmap = 180
                elif abs(i_deg - 60.0) < 1e-9:
                    n_lat_heatmap = 130

                # Compute ROM from RGT solver for this inclination
                if RGT_USE:
                    if target_planet == 'earth':
                        phasing = PhasingRGT()
                        a_solution, converged, iterations = phasing.solve_for_semi_major_axis(
                            i=i_deg, e=RGT_e, No=RGT_No, Nd=RGT_Nd
                        )
                    elif target_planet == 'mars':
                        phasing = PhasingRGT(
                            J2=constants.J2_MARS,
                            Rearth=constants.MARS_RADIUS,
                            mu=constants.MARS_GRAVITATIONAL_CONSTANT,
                            rotation_rate_earth=2.0 * pi /
                            # Mars sidereal day, rad/s
                            (24*3600 + 37*60 + 22.66),
                        )
                        a_solution, converged, iterations = phasing.solve_for_semi_major_axis(
                            i=i_deg, e=RGT_e, No=RGT_No, Nd=RGT_Nd
                        )
                    if converged and np.isfinite(a_solution):
                        rom_a_km = float(a_solution)
                        print(f"[RGT] ROM a={rom_a_km:.3f} km (h={rom_a_km - R_EARTH:.1f} km) "
                              f"from No/Nd={RGT_No}/{RGT_Nd}, e={RGT_e} (iters={iterations})")
                    else:
                        rom_a_km = R_EARTH + RGT_FALLBACK_ALT_KM
                        print(f"[RGT] WARN: RGT solver failed (iters={iterations}). "
                              f"Falling back to h={RGT_FALLBACK_ALT_KM:.1f} km (a={rom_a_km:.3f} km).")
                else:
                    rom_a_km = R_EARTH + RGT_FALLBACK_ALT_KM

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
                    # a single tolerance for the heatmap
                    tolerances_deg=[0.25],
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
