"""Reconfiguration Plotting Utilities

Generates analytical reconfiguration performance figures using the J2 planner
(single or multicore). Central entry point: run_reconfig_plots(...).

Provided Plot Types (toggle via PlotFlags)
    times_vs_delta_a      : Case study Δa sweep (total / wait / drift vs Δa)
    times_vs_tolerance    : Case study tolerance sweep
    avg_vs_delta_a        : Same Δa sweep averaged over latitude/longitude grid
    avg_vs_tolerance      : Same tolerance sweep averaged over grid
    heatmap               : Global (band-limited) heatmaps for total / wait / drift at fixed (Δa, tol)
    iso_time_contours     : Iso-total-time contour overlay (optional with heatmap)

Core Function: run_reconfig_plots Arguments (quick reference)
    gom (Orbit)                 : Initial orbit (circular) object
    rom_a_km (float)            : Final ROM semi-major axis (km)
    epoch (datetime UTC)        : Scenario start epoch
    target_name (str)           : Label prefix for case-study outputs
    target_lat_geod_deg (float) : Case-study target latitude (deg)
    target_lon_deg (float)      : Case-study target longitude (deg)
    flags (PlotFlags)           : Enables selected plot families (see above)
    delta_min_km / delta_max_km / delta_step_km : Range for Δa sweep
    delta_a_fixed_km (float)    : Δa used when sweeping tolerance or making heatmaps
    tolerances_deg (list[float]): Tolerance ladder for sweeps / heatmaps
    inclination_deg (float)     : Constellation inclination (deg) (used for grid band cap)
    lat_band_deg (float)        : Half-width of latitude band for averaging / heatmaps (< inclination)
    n_lat (int)                 : Number of latitude samples for grid averaging/heatmap
    n_lon (int)                 : Number of longitude samples ([-180,180))
    max_wrap (int)              : Pass to planner (legacy constraint / safety bound)
    drift_time_cap_days (float) : Planner drift-time cap
    jitter_* (optional)         : Parameters for jitter / sensitivity diagnostics (if implemented)
    iso_levels_hours (list[float]): Contour levels (hours) for iso-time map
    verbose (bool)              : Print progress
    output_dir (str)            : Root directory for saving figures (default 'results')

Important Outputs
    Figures saved under results/reconfig/<target_name>/ or results/reconfig/avg/ etc.
    Filenames encode target, Δa range or tolerance, and flags.

Parallelism
    Uses multicore solver if import succeeds; otherwise falls back to single-core.
    External parallel control (RECONFIG_N_JOBS, etc.) handled by the solver module.

Author
Fabio Meloni - TU Delft MSc Student - NASA Jet Propulsion Laboratory Visiting Researcher
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Iterable
import constants
import inspect

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

# pull in your solver & helpers
# Prefer multicore solver if available, fall back to single-core
try:
    from reconfig_driftANDwaitTime_multicore import (
        AnalyticRaanPhasingPlanner,
        Orbit,
        deg2rad, rad2deg,
        R_EARTH,
        compute_reconfig_times_vs_delta_a,  # multicore-aware sweep
        gmst_rad,                           # for ground track
        j2_rates,
        u_required_branches,
    )
    _USING_MULTICORE_SOLVER = True
except Exception:
    from reconfig_driftANDwaitTime import (
        AnalyticRaanPhasingPlanner,
        Orbit,
        deg2rad, rad2deg,
        R_EARTH,
        compute_reconfig_times_vs_delta_a,  # single-core sweep (fallback)
        gmst_rad,                           # for ground track
        j2_rates,
        u_required_branches,
    )
    _USING_MULTICORE_SOLVER = False

MU_EARTH = constants.GRAVITATIONAL_CONSTANT  # km^3/s^2

# ---------------------------
# small utilities
# ---------------------------

def _call_with_supported_kwargs(func, *args, verbose=False, **kwargs):
    """
    Call `func` with only the kwargs it declares.
    Any extra kwargs are dropped (optionally warn if verbose=True).
    """
    sig = inspect.signature(func)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    if verbose:
        dropped = [k for k in kwargs if k not in allowed]
        if dropped:
            print(f"[warn] dropping unsupported kwargs for {func.__name__}: {dropped}")
    return func(*args, **filtered)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _hohmann_dv(a1_km: float, a2_km: float) -> tuple[float, float, float]:
    # returns (dv1, dv2, dv_total) in km/s
    mu = MU_EARTH  # km^3/s^2 (you already import this constant)
    r1 = float(a1_km)
    r2 = float(a2_km)
    v1   = (mu / r1) ** 0.5
    v2   = (mu / r2) ** 0.5
    v_tr1 = (mu * (2.0 / r1 - 1.0 / ((r1 + r2) * 0.5))) ** 0.5
    v_tr2 = (mu * (2.0 / r2 - 1.0 / ((r1 + r2) * 0.5))) ** 0.5
    dv1 = abs(v_tr1 - v1)
    dv2 = abs(v2 - v_tr2)
    return dv1, dv2, (dv1 + dv2)

def minutes(s: float) -> float:
    return s / 60.0

def hours(s: float) -> float:
    return s / 3600.0

def _fmt_target_tag(name: str, lat: float, lon: float) -> str:
    lat_tag = f"{lat:+.3f}".replace("+", "p").replace("-", "m")
    lon_tag = f"{lon:+.3f}".replace("+", "p").replace("-", "m")
    return f"{name}_lat{lat_tag}_lon{lon_tag}"

# --------------------------------------------------------------------------------------
# Core helpers
# --------------------------------------------------------------------------------------

def _safe_solve(
    gom: Orbit,
    rom_a_km: float,
    epoch,
    lat_deg: float,
    lon_deg: float,
    delta_a_km: float,
    max_wrap: int,
    drift_time_cap_days: float,
    tolerance_deg: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Run the analytic planner once. Returns (total_s, wait_s, drift_s) or (nan,nan,nan) if no solution.
    If tolerance_deg is provided, we pass a single-valued tolerance schedule so the
    solver picks the earliest solution within that tolerance band.
    """
    try:
        planner = AnalyticRaanPhasingPlanner(
            gom=gom,
            rom_a_km=rom_a_km,
            delta_a_km=delta_a_km,
            epoch_gom=epoch,
            target_lat_geod_deg=lat_deg,
            target_lon_deg=lon_deg,
            max_wrap=max_wrap,
            drift_time_cap_days=drift_time_cap_days,
        )
        lon_tol_sched = None if tolerance_deg is None else [float(tolerance_deg)]
        sol = planner.solve_min_time(lon_tol_schedule=lon_tol_sched)
        if sol.get('success'):
            return (float(sol['total_time_s']),
                    float(sol['T_wait']),
                    float(sol['T_d']))
        return (np.nan, np.nan, np.nan)
    except Exception:
        return (np.nan, np.nan, np.nan)

def compute_reconfig_times_vs_tolerance(
    gom: Orbit,
    rom_a_km: float,
    epoch,
    target_lat_geod_deg: float,
    target_lon_deg: float,
    delta_a_fixed_km: float,
    tolerances_deg: List[float],
    max_wrap: int = 200,
    drift_time_cap_days: float = 15.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For a single scenario, sweep tolerance and record the *earliest* total/wait/drift returned
    by the solver for that tolerance. Returns arrays (tols_deg, totals_s, waits_s, drifts_s).
    """
    tols = np.asarray(tolerances_deg, dtype=float)
    totals = np.full_like(tols, np.nan, dtype=float)
    waits  = np.full_like(tols, np.nan, dtype=float)
    drifts = np.full_like(tols, np.nan, dtype=float)

    for i, tol in enumerate(tols):
        if verbose:
            print(f"[TOL sweep] {i+1}/{len(tols)}: tol={tol:g}°")
        total, wait, drift = _safe_solve(
            gom, rom_a_km, epoch,
            target_lat_geod_deg, target_lon_deg,
            delta_a_fixed_km, max_wrap, drift_time_cap_days,
            tolerance_deg=tol,
        )
        totals[i], waits[i], drifts[i] = total, wait, drift

    return tols, totals, waits, drifts

def solve_once(
    gom: Orbit, rom_a_km: float, epoch: datetime,
    tgt_lat_deg: float, tgt_lon_deg: float,
    delta_a_km: float,
    drift_time_cap_days: float,
    max_wrap: int,
    tolerance_deg: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run the analytic planner once. If tolerance_deg is provided, we hand
    a single-rung tolerance schedule so selection is 'earliest within tol'.
    """
    planner = AnalyticRaanPhasingPlanner(
        gom=gom, rom_a_km=rom_a_km, delta_a_km=delta_a_km,
        epoch_gom=epoch, target_lat_geod_deg=tgt_lat_deg, target_lon_deg=tgt_lon_deg,
        max_wrap=max_wrap, drift_time_cap_days=drift_time_cap_days
    )
    lon_tol_schedule = None if tolerance_deg is None else [float(tolerance_deg)]
    return planner.solve_min_time(lon_tol_schedule=lon_tol_schedule)

def _pareto_frontier_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Given points (x,y) with x sorted ascending, return indices on the
    lower-left Pareto frontier (non-increasing y as x grows).
    """
    idx = []
    best = np.inf
    for i in range(len(x)):
        if y[i] < best - 1e-12:
            idx.append(i)
            best = y[i]
    return np.array(idx, dtype=int)

# --------------------------------------------------------------------------------------
# Target grids
# --------------------------------------------------------------------------------------

def build_target_grid(
    inclination_deg: float,
    lat_band_deg: float,
    n_lat: int,
    n_lon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce a regular lon/lat grid of target points.

    - Latitudes are linearly spaced across [-lat_band, +lat_band] (inclusive).
      Choose lat_band < inclination for visibility.
    - Longitudes are spaced across [-180, +180) with n_lon points.

    Returns (lats_deg array shape (n_lat,), lons_deg array shape (n_lon,))
    """
    if lat_band_deg >= abs(inclination_deg):
        raise ValueError("lat_band_deg must be smaller than the inclination to guarantee access.")
    lats = np.linspace(-lat_band_deg, +lat_band_deg, n_lat, dtype=float)
    lons = np.linspace(-180.0, 180.0, n_lon, endpoint=False, dtype=float)
    return lats, lons


# --------------------------------------------------------------------------------------
# Case-study plots
# --------------------------------------------------------------------------------------

def plot_times_vs_delta_a_case(
    gom: Orbit,
    rom_a_km: float,
    epoch,
    target_name: Optional[str],
    target_lat_geod_deg: float,
    target_lon_deg: float,
    delta_min_km: float,
    delta_max_km: float,
    delta_step_km: float,
    max_wrap: int,
    drift_time_cap_days: float,
    outdir: str = "results/reconfig/times_deltaA",
    verbose: bool = False,
) -> plt.Figure:
    
    # Get tolerance info by running one case first to extract default tolerance
    test_planner = AnalyticRaanPhasingPlanner(
        gom=gom, rom_a_km=rom_a_km, delta_a_km=0.0, epoch_gom=epoch,
        target_lat_geod_deg=target_lat_geod_deg, target_lon_deg=target_lon_deg,
        max_wrap=max_wrap, drift_time_cap_days=drift_time_cap_days
    )
    test_sol = test_planner.solve_min_time()
    tolerance_used = test_sol.get('selection_meta', {}).get('tolerance_used_deg', 0.25)
    if tolerance_used is None:
        tolerance_used = 0.25  # fallback default

    dAs, totals, waits, drifts = compute_reconfig_times_vs_delta_a(
        gom, rom_a_km, epoch, target_lat_geod_deg, target_lon_deg,
        delta_min_km, delta_max_km, delta_step_km, max_wrap, drift_time_cap_days, verbose
    )
    totals_h = totals / 3600.0
    waits_h  = waits  / 3600.0
    drifts_h = drifts / 3600.0

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 13))
    axs[0].plot(dAs, totals_h, lw=1.8, color='k');            axs[0].set_ylabel("Total [h]")
    axs[1].plot(dAs, waits_h,  lw=1.8, color='tab:blue');     axs[1].set_ylabel("Wait [h]")
    axs[2].plot(dAs, drifts_h, lw=1.8, color='tab:orange');   axs[2].set_ylabel("Drift [h]"); axs[2].set_xlabel("Δa [km]")
    for ax in axs: ax.grid(True, which='both', alpha=0.3); ax.axvline(0.0, color='0.7', ls='--', lw=1)
    
    # Handle case where target_name might be None
    target_display = target_name if target_name is not None else "Target"
    
    # Handle case where epoch might be None
    epoch_str = epoch.strftime('%Y-%m-%d') if epoch is not None else "Unknown"
    
    # Debug: check for None values before formatting
    print(f"[DEBUG] target_display: {target_display}")
    print(f"[DEBUG] gom.a_km: {gom.a_km}")
    print(f"[DEBUG] R_EARTH: {R_EARTH}")
    print(f"[DEBUG] gom.i_rad: {gom.i_rad}")
    print(f"[DEBUG] tolerance_used: {tolerance_used}")
    print(f"[DEBUG] epoch_str: {epoch_str}")
    
    fig.suptitle(f"Reconfig time vs Δa — {target_display} ({target_lat_geod_deg:+.3f}°, {target_lon_deg:+.3f}°)\n"
                 f"GOM h={(gom.a_km-R_EARTH):.1f} km, ROM h={(rom_a_km-R_EARTH):.1f} km, i={rad2deg(gom.i_rad):.3f}°\n"
                 f"tolerance={tolerance_used:.4f}° (auto), epoch {epoch_str}")
    os.makedirs(outdir, exist_ok=True)
    fname_target = target_name if target_name is not None else f"Target{target_lat_geod_deg:+.1f}_{target_lon_deg:+.1f}"
    fname = (f"TimesDeltaA_{fname_target}_GOM{gom.a_km:.0f}_ROM{rom_a_km:.0f}_"
             f"inc{rad2deg(gom.i_rad):.3f}_tolAuto_dA{delta_min_km}to{delta_max_km}_s{delta_step_km}.png")
    fig.savefig(os.path.join(outdir, fname), dpi=200, bbox_inches='tight')
    return fig


def plot_times_vs_tolerance_case(
    gom: Orbit,
    rom_a_km: float,
    epoch,
    target_name: Optional[str],
    target_lat_geod_deg: float,
    target_lon_deg: float,
    delta_a_fixed_km: float,
    tolerances_deg: List[float],
    max_wrap: int,
    drift_time_cap_days: float,
    outdir: str = "results/recconfig/times_tolerance",
    verbose: bool = False,
) -> plt.Figure:
    tols, totals, waits, drifts = compute_reconfig_times_vs_tolerance(
        gom, rom_a_km, epoch, target_lat_geod_deg, target_lon_deg,
        delta_a_fixed_km, tolerances_deg, max_wrap, drift_time_cap_days, verbose
    )
    totals_h = totals / 3600.0
    waits_h  = waits  / 3600.0
    drifts_h = drifts / 3600.0

    tol_min = np.nanmin(tols)
    tol_max = np.nanmax(tols)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 13))
    axs[0].plot(tols, totals_h, lw=1.8, marker='o', ms=3, color='k');            axs[0].set_ylabel("Total [h]")
    axs[1].plot(tols, waits_h,  lw=1.8, marker='o', ms=3, color='tab:blue');     axs[1].set_ylabel("Wait [h]")
    axs[2].plot(tols, drifts_h, lw=1.8, marker='o', ms=3, color='tab:orange');   axs[2].set_ylabel("Drift [h]"); axs[2].set_xlabel("Tolerance [deg]")
    for ax in axs:
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xscale('log')
        ax.set_xlim(tol_min, tol_max)
    
    # Handle case where target_name might be None
    target_display = target_name if target_name is not None else "Target"
    
    # Handle case where epoch might be None
    epoch_str = epoch.strftime('%Y-%m-%d') if epoch is not None else "Unknown"
    
    fig.suptitle(f"Reconfig time vs tolerance — {target_display} ({target_lat_geod_deg:+.3f}°, {target_lon_deg:+.3f}°)\n"
                 f"GOM a={gom.a_km:.1f} km, ROM a={rom_a_km:.1f} km, inc={rad2deg(gom.i_rad):.3f}°\n"
                 f"Δa={delta_a_fixed_km:+.1f} km, epoch {epoch_str}")
    os.makedirs(outdir, exist_ok=True)
    fname_target = target_name if target_name is not None else f"Target{target_lat_geod_deg:+.1f}_{target_lon_deg:+.1f}"
    fname = (f"timesTol_{fname_target}_GOM{gom.a_km:.0f}_ROM{rom_a_km:.0f}_"
             f"inc{rad2deg(gom.i_rad):.3f}_dA{delta_a_fixed_km:+.1f}_tol{tol_min:.4f}to{tol_max:.4f}.png")
    fig.savefig(os.path.join(outdir, fname), dpi=200, bbox_inches='tight')
    return fig


# --------------------------------------------------------------------------------------
# Averaged across a target grid
# --------------------------------------------------------------------------------------

def _avg_over_targets_delta_a(
    gom: Orbit, rom_a_km: float, epoch,
    lats_deg: np.ndarray, lons_deg: np.ndarray,
    delta_min_km: float, delta_max_km: float, delta_step_km: float,
    max_wrap: int, drift_time_cap_days: float, verbose=False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Average (per Δa) across the target set (lat x lon).
    """
    dAs = np.arange(delta_min_km, delta_max_km + 1e-9, delta_step_km, dtype=float)
    n = dAs.size
    sum_tot = np.zeros(n); sum_wait = np.zeros(n); sum_drift = np.zeros(n); counts = np.zeros(n, dtype=int)

    total_targets = len(lats_deg) * len(lons_deg)
    done = 0
    if verbose:
        print(f"[AVG Δa] Starting averaging across {total_targets} targets ({len(lats_deg)} lats × {len(lons_deg)} lons)")
        print(f"[AVG Δa] Δa range: {delta_min_km} to {delta_max_km} km (step {delta_step_km} km, {n} points)")
    for lat in lats_deg:
        for lon in lons_deg:
            if verbose:
                print(f"[AVG Δa] Target {done+1}/{total_targets}: ({lat:+.2f}°, {lon:+.3f}°) - computing {n} Δa points...")
            dA, tot, wai, dri = compute_reconfig_times_vs_delta_a(
                gom, rom_a_km, epoch, float(lat), float(lon),
                delta_min_km, delta_max_km, delta_step_km, max_wrap, drift_time_cap_days, verbose=verbose
            )
            mask = ~np.isnan(tot)
            sum_tot[mask]  += tot[mask]
            sum_wait[mask] += wai[mask]
            sum_drift[mask]+= dri[mask]
            counts[mask]   += 1
            done += 1
            if verbose:
                valid_solutions = int(np.sum(mask))
                pct = 100.0 * done / total_targets
                print(f"[AVG Δa] Completed target {done}/{total_targets} ({pct:.1f}%) - valid {valid_solutions}/{n}")

    with np.errstate(invalid='ignore', divide='ignore'):
        avg_tot   = np.where(counts>0, sum_tot / counts, np.nan)
        avg_wait  = np.where(counts>0, sum_wait / counts, np.nan)
        avg_drift = np.where(counts>0, sum_drift / counts, np.nan)
    
    if verbose:
        valid_points = np.sum(counts > 0)
        print(f"[AVG Δa] Completed! Valid solutions for {valid_points}/{n} Δa points")
    return dAs, avg_tot, avg_wait, avg_drift


def _avg_over_targets_tolerance(
    gom: Orbit, rom_a_km: float, epoch,
    lats_deg: np.ndarray, lons_deg: np.ndarray,
    delta_a_fixed_km: float, tolerances_deg: List[float],
    max_wrap: int, drift_time_cap_days: float, verbose=False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Average (per tolerance) across the target set (lat x lon).
    """
    tols = np.asarray(tolerances_deg, dtype=float)
    n = tols.size
    sum_tot = np.zeros(n); sum_wait = np.zeros(n); sum_drift = np.zeros(n); counts = np.zeros(n, dtype=int)

    total_targets = len(lats_deg) * len(lons_deg)
    done = 0
    if verbose:
        print(f"[AVG Tol] Starting averaging across {total_targets} targets ({len(lats_deg)} lats × {len(lons_deg)} lons)")
        print(f"[AVG Tol] Tolerance range: {min(tolerances_deg):.4f} to {max(tolerances_deg):.1f}° ({n} points)")
        print(f"[AVG Tol] Fixed Δa: {delta_a_fixed_km:+.1f} km")
    for lat in lats_deg:
        for lon in lons_deg:
            if verbose:
                print(f"[AVG Tol] Target {done+1}/{total_targets}: ({lat:+.2f}°, {lon:+.3f}°) - computing {n} tolerance points...")
            t, tot, wai, dri = compute_reconfig_times_vs_tolerance(
                gom, rom_a_km, epoch, float(lat), float(lon),
                delta_a_fixed_km, tolerances_deg, max_wrap, drift_time_cap_days, verbose=verbose
            )
            mask = ~np.isnan(tot)
            sum_tot[mask]  += tot[mask]
            sum_wait[mask] += wai[mask]
            sum_drift[mask]+= dri[mask]
            counts[mask]   += 1
            done += 1
            if verbose:
                valid_solutions = int(np.sum(mask))
                pct = 100.0 * done / total_targets
                print(f"[AVG Tol] Completed target {done}/{total_targets} ({pct:.1f}%) - valid {valid_solutions}/{n}")

    with np.errstate(invalid='ignore', divide='ignore'):
        avg_tot   = np.where(counts>0, sum_tot / counts, np.nan)
        avg_wait  = np.where(counts>0, sum_wait / counts, np.nan)
        avg_drift = np.where(counts>0, sum_drift / counts, np.nan)
    
    if verbose:
        valid_points = np.sum(~np.isnan(avg_tot))
        print(f"[AVG Tol] Completed averaging: {valid_points}/{len(tols)} tolerance points have valid data")
        print(f"[AVG Tol] Tolerance range: {tols[0]:.4f}°-{tols[-1]:.1f}°, Δa={delta_a_fixed_km:+.1f} km")
    
    return tols, avg_tot, avg_wait, avg_drift


def plot_times_vs_delta_a_avg_targets(
    gom: Orbit,
    rom_a_km: float,
    epoch,
    inclination_deg: float,
    lat_band_deg: float,
    n_lat: int,
    n_lon: int,
    delta_min_km: float,
    delta_max_km: float,
    delta_step_km: float,
    max_wrap: int,
    drift_time_cap_days: float,
    outdir: str = "results/reconfig/times_deltaA",
    verbose: bool = False,
) -> plt.Figure:
    lats, lons = build_target_grid(inclination_deg, lat_band_deg, n_lat, n_lon)
    dAs, avg_tot_s, avg_wait_s, avg_drift_s = _avg_over_targets_delta_a(
        gom, rom_a_km, epoch, lats, lons,
        delta_min_km, delta_max_km, delta_step_km,
        max_wrap, drift_time_cap_days, verbose
    )
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
    axs[0].plot(dAs, avg_tot_s/3600.0,  lw=1.8, color='k');            axs[0].set_ylabel("Total [h]")
    axs[1].plot(dAs, avg_wait_s/3600.0, lw=1.8, color='tab:blue');     axs[1].set_ylabel("Wait [h]")
    axs[2].plot(dAs, avg_drift_s/3600.0,lw=1.8, color='tab:orange');   axs[2].set_ylabel("Drift [h]"); axs[2].set_xlabel("Δa [km]")
    for ax in axs: ax.grid(True, which='both', alpha=0.3); ax.axvline(0.0, color='0.7', ls='--', lw=1)
    fig.suptitle(f"Reconfig time vs Δa — Averaged across {n_lat}×{n_lon} targets\n"
                 f"GOM h={(gom.a_km-R_EARTH):.1f} km, ROM h={(rom_a_km-R_EARTH):.1f} km, i={inclination_deg:.2f}°\n"
                 f"Latitude band ±{lat_band_deg:.1f}°, epoch {epoch.strftime('%Y-%m-%d')}")
    os.makedirs(outdir, exist_ok=True)
    fname = (f"TimesAvgDA_{n_lat}x{n_lon}_GOM{gom.a_km:.0f}_ROM{rom_a_km:.0f}_"
             f"inc{inclination_deg:.1f}_band{lat_band_deg:.1f}.png")
    fig.savefig(os.path.join(outdir, fname), dpi=200, bbox_inches='tight')
    return fig


def plot_times_vs_tolerance_avg_targets(
    gom: Orbit,
    rom_a_km: float,
    epoch,
    inclination_deg: float,
    lat_band_deg: float,
    n_lat: int,
    n_lon: int,
    delta_a_fixed_km: float,
    tolerances_deg: List[float],
    max_wrap: int,
    drift_time_cap_days: float,
    outdir: str = "results/reconfig/times_tolerance",
    verbose: bool = False,
) -> plt.Figure:
    lats, lons = build_target_grid(inclination_deg, lat_band_deg, n_lat, n_lon)
    tols, avg_tot_s, avg_wait_s, avg_drift_s = _avg_over_targets_tolerance(
        gom, rom_a_km, epoch, lats, lons,
        delta_a_fixed_km, tolerances_deg,
        max_wrap, drift_time_cap_days, verbose
    )
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
    axs[0].plot(tols, avg_tot_s/3600.0,  lw=1.8, marker='o', ms=3, color='k');            axs[0].set_ylabel("Total [h]")
    axs[1].plot(tols, avg_wait_s/3600.0, lw=1.8, marker='o', ms=3, color='tab:blue');     axs[1].set_ylabel("Wait [h]")
    axs[2].plot(tols, avg_drift_s/3600.0,lw=1.8, marker='o', ms=3, color='tab:orange');   axs[2].set_ylabel("Drift [h]"); axs[2].set_xlabel("Tolerance [deg]")
    for ax in axs: 
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xscale('log')
    tol_min = np.nanmin(tols)
    tol_max = np.nanmax(tols)
    fig.suptitle(f"Reconfig time vs tolerance — Averaged across {n_lat}×{n_lon} targets\n"
                 f"GOM h={(gom.a_km-R_EARTH):.1f} km, ROM h={(rom_a_km-R_EARTH):.1f} km, i={inclination_deg:.2f}°\n"
                 f"Δa={delta_a_fixed_km:+.1f} km, latitude band ±{lat_band_deg:.1f}°, epoch {epoch.strftime('%Y-%m-%d')}")
    os.makedirs(outdir, exist_ok=True)
    fname = (f"TimesAvgTol_{n_lat}x{n_lon}_GOM{gom.a_km:.0f}_ROM{rom_a_km:.0f}_"
             f"inc{inclination_deg:.1f}_dA{delta_a_fixed_km:+.1f}_band{lat_band_deg:.1f}.png")
    fig.savefig(os.path.join(outdir, fname), dpi=200, bbox_inches='tight')
    return fig


# --------------------------------------------------------------------------------------
# Heatmaps (total / wait / drift) + fading 1h GOM ground track
# --------------------------------------------------------------------------------------

def _eci_from_circular(i_rad, Omega_rad, u_rad, r=1.0):
    cO, sO = np.cos(Omega_rad), np.sin(Omega_rad)
    ci, si = np.cos(i_rad),     np.sin(i_rad)
    cu, su = np.cos(u_rad),     np.sin(u_rad)
    # r_pf = [cu, su, 0]; R3(Ω) R1(i) r_pf:
    x =  cO*cu - sO*ci*su
    y =  sO*cu + cO*ci*su
    z =  si*su
    return np.array([x,y,z])*r

def _groundtrack_GOM_first_hour(gom: Orbit, epoch, dt_s=10.0, hours=1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple secular J2 propagation of Ω(t), u(t) for the GOM; returns (lats_deg, lons_deg) arrays.
    """
    dOm, dom, dM = j2_rates(gom.a_km, 0.0, gom.i_rad)
    du = dom + dM
    N = max(2, int(round(hours*3600.0/dt_s))+1)
    lats = np.zeros(N); lons = np.zeros(N)
    for k, t in enumerate(np.linspace(0.0, hours*3600.0, N)):
        Om = gom.Omega_rad + dOm*t
        u  = gom.u_rad     + du*t
        r_eci = _eci_from_circular(gom.i_rad, Om, u, r=1.0)
        theta = gmst_rad(epoch + np.timedelta64(int(t), 's').astype(object))
        # ECI->ECEF rotation about Z by +theta:
        c, s = np.cos(theta), np.sin(theta)
        x =  c*r_eci[0] + s*r_eci[1]
        y = -s*r_eci[0] + c*r_eci[1]
        z =  r_eci[2]
        r = np.linalg.norm([x,y,z])
        lats[k] = np.degrees(np.arcsin(z/r))
        lon_raw = np.degrees(np.arctan2(y, x))
        lons[k] = ((lon_raw + 180.0) % 360.0) - 180.0
    return lats, lons

def _plot_lonwrap_segments(ax, lons_deg, lats_deg, **kw):
    lons = np.asarray(lons_deg); lats = np.asarray(lats_deg)
    jumps = np.where(np.abs(np.diff(lons)) > 180.0)[0]
    starts = np.r_[0, jumps+1]; ends = np.r_[jumps+1, len(lons)]
    for a,b in zip(starts, ends):
        if b-a >= 2:
            ax.plot(lons[a:b], lats[a:b], **kw)

def plot_heatmaps_for_case(
    gom: Orbit,
    rom_a_km: float,
    epoch,
    inclination_deg: float,
    lat_band_deg: float,
    n_lat: int,
    n_lon: int,
    delta_a_fixed_km: float,
    tolerance_deg: float,
    max_wrap: int,
    drift_time_cap_days: float,
    outdir: str = "results/reconfig/heatmaps",
    verbose: bool = False,
) -> plt.Figure:
    lats_vec, lons_vec = build_target_grid(inclination_deg, lat_band_deg, n_lat, n_lon)
    # grids for plotting
    LAT, LON = np.meshgrid(lats_vec, lons_vec, indexing='ij')  # (n_lat, n_lon)

    total_s = np.full(LAT.shape, np.nan, dtype=float)
    wait_s  = np.full(LAT.shape, np.nan, dtype=float)
    drift_s = np.full(LAT.shape, np.nan, dtype=float)

    total_targets = n_lat * n_lon
    done = 0
    for i in range(n_lat):
        for j in range(n_lon):
            tot, wai, dri = _safe_solve(
                gom, rom_a_km, epoch,
                float(LAT[i,j]), float(LON[i,j]),
                delta_a_fixed_km, max_wrap, drift_time_cap_days,
                tolerance_deg=tolerance_deg
            )
            total_s[i,j], wait_s[i,j], drift_s[i,j] = tot, wai, dri
            done += 1
            if verbose and (done % max(1, total_targets // 10) == 0):
                print(f"[HEATMAP] progress {done}/{total_targets}")

    # convert to hours for readability
    total_h = total_s / 3600.0
    wait_h  = wait_s  / 3600.0
    drift_h = drift_s / 3600.0

    # figure with Cartopy PlateCarree projection to show full Earth background
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             subplot_kw={"projection": ccrs.PlateCarree()})
    world_extent = (-180.0, 180.0, -90.0, 90.0)
    band_extent = (-180.0, 180.0, -lat_band_deg, +lat_band_deg)

    # choose vmin/vmax from finite percentiles to keep colors stable
    def finite_bounds(A):
        finite = np.isfinite(A)
        if not finite.any():
            return (0.0, 1.0)
        vmin = np.nanpercentile(A[finite], 5.0)
        vmax = np.nanpercentile(A[finite], 95.0)
        if vmin == vmax:
            vmax = vmin + 1e-3
        return vmin, vmax

    items = [("Total time [h]", total_h), ("Wait time [h]", wait_h), ("Drift time [h]", drift_h)]
    for ax, (title, data) in zip(axes, items):
        # Full Earth grayscale background
        ax.set_global()
        ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
        ax.add_feature(cfeature.OCEAN, color='white', zorder=0)
        ax.coastlines(color='black', linewidth=0.5, zorder=1)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          color="gray", alpha=0.5, linestyle="--",
                          x_inline=False, y_inline=False)
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
        gl.ylocator = mticker.FixedLocator(np.arange(-90,  91, 15))
        gl.right_labels = False
        gl.top_labels = False
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 8}

        # Semi-transparent heatmap overlay only over latitude band
        vmin, vmax = finite_bounds(data)
        im = ax.imshow(
            data, origin='lower', extent=band_extent, aspect='auto',
            vmin=vmin, vmax=vmax, cmap='viridis', alpha=0.65, zorder=2
        )
        # Ensure full globe extent remains after imshow
        ax.set_extent(world_extent, crs=ccrs.PlateCarree())

        ax.set_ylabel("Latitude [deg]")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.02)
        cbar.ax.set_ylabel(title)

    axes[-1].set_xlabel("Longitude [deg]")

    # overlay first-hour GOM ground track (fading alpha)
    gt_lat, gt_lon = _groundtrack_GOM_first_hour(gom, epoch, dt_s=20.0, hours=1.0)
    # draw with fading alpha by chunking
    N = len(gt_lon)
    chunks = 20
    idxs = np.linspace(0, N-1, chunks+1, dtype=int)
    for s, e in zip(idxs[:-1], idxs[1:]):
        alpha = 0.8 * (1.0 - s / (N-1)) + 0.1
        for ax in axes:
            _plot_lonwrap_segments(
                ax, gt_lon[s:e], gt_lat[s:e],
                color='k', lw=1.5, alpha=alpha, transform=ccrs.PlateCarree(), zorder=3
            )

    fig.suptitle(
        f"Reconfig time heatmaps — {n_lat}×{n_lon} targets\n"
        f"GOM h={(gom.a_km-R_EARTH):.1f} km, ROM h={(rom_a_km-R_EARTH):.1f} km, i={inclination_deg:.2f}°\n"
        f"Δa={delta_a_fixed_km:+.1f} km, tolerance={tolerance_deg:g}°, band ±{lat_band_deg:.1f}°, epoch {epoch.strftime('%Y-%m-%d')}"
    )
    os.makedirs(outdir, exist_ok=True)
    fname = (f"Heatmap_{n_lat}x{n_lon}_GOM{gom.a_km:.0f}_ROM{rom_a_km:.0f}_"
             f"inc{inclination_deg:.1f}_dA{delta_a_fixed_km:+.1f}_tol{tolerance_deg:g}.png")
    fig.savefig(os.path.join(outdir, fname), dpi=200, bbox_inches='tight')
    return fig

# ---------------------------
# Pareto: total time vs Δv
# ---------------------------

def plot_pareto_time_vs_dv(
    gom: Orbit, rom_a_km: float, epoch: datetime,
    target_name: str, tgt_lat_deg: float, tgt_lon_deg: float,
    delta_list_km: Iterable[float],
    drift_time_cap_days: float = 30.0,
    max_wrap: int = 120,
    tolerance_deg: float = 0.25,
    outdir: str = "results/reconfig/paretoDV",
    verbose: bool = False,
    include_wait_components: bool = True,   # <-- new
    mark_frontier: bool = True,             # optional highlight of Pareto front
):
    # --- safe fallbacks for helpers if not already defined in your module ---
    try:
        ensure_dir  # type: ignore[name-defined]
    except NameError:
        def ensure_dir(d): os.makedirs(d, exist_ok=True)

    try:
        _hohmann_dv  # type: ignore[name-defined]
        _hoh_dv = _hohmann_dv
    except NameError:
        # Minimal circular-to-circular Hohmann Δv (km/s)
        MU = constants.GRAVITATIONAL_CONSTANT  # km^3/s^2
        def _hoh_dv(a1_km: float, a2_km: float):
            v1 = np.sqrt(MU / a1_km)
            v2 = np.sqrt(MU / a2_km)
            a_tr = 0.5 * (a1_km + a2_km)
            v_peri_tr  = np.sqrt(MU * (2.0/a1_km - 1.0/a_tr))
            v_apo_tr   = np.sqrt(MU * (2.0/a2_km - 1.0/a_tr))
            dv1 = abs(v_peri_tr - v1)
            dv2 = abs(v2 - v_apo_tr)
            return dv1, dv2, dv1 + dv2

    try:
        hours  # type: ignore[name-defined]
    except NameError:
        def hours(x): return float(x) / 3600.0

    try:
        _fmt_target_tag  # type: ignore[name-defined]
    except NameError:
        def _fmt_target_tag(name, lat, lon):
            lat_tag = f"{lat:+.3f}".replace("+", "p").replace("-", "m")
            lon_tag = f"{lon:+.3f}".replace("+", "p").replace("-", "m")
            return f"{name}_lat{lat_tag}_lon{lon_tag}"

    try:
        solve_once  # type: ignore[name-defined]
    except NameError:
        # Fall back to your analytic planner if solve_once isn't available
        def solve_once(gom, rom_a_km, epoch, tlat, tlon, delta_a_km,
                       drift_time_cap_days, max_wrap, tolerance_deg=0.25):
            planner = AnalyticRaanPhasingPlanner(
                gom=gom, rom_a_km=rom_a_km, delta_a_km=delta_a_km,
                epoch_gom=epoch, target_lat_geod_deg=tlat, target_lon_deg=tlon,
                max_wrap=max_wrap, drift_time_cap_days=drift_time_cap_days
            )
            # pass a simple tolerance ladder so selection honors tolerance_deg
            sol = planner.solve_min_time(lon_tol_schedule=[tolerance_deg])
            return sol

    ensure_dir(outdir)
    dA = np.array(list(delta_list_km), dtype=float)

    # Δv per Δa (sum of two Hohmanns: GOM→Drift + Drift→ROM)
    dv = np.full_like(dA, np.nan, dtype=float)

    # times (hours)
    total_h = np.full_like(dA, np.nan, dtype=float)
    wait_h  = np.full_like(dA, np.nan, dtype=float)
    drift_h = np.full_like(dA, np.nan, dtype=float)

    for i, da in enumerate(dA):
        aG = gom.a_km
        aD = aG + da
        if aD <= 0:
            continue

        dv1, dv2, dv_g2d = _hoh_dv(aG, aD)
        dv3, dv4, dv_d2r = _hoh_dv(aD, rom_a_km)
        dv[i] = dv_g2d + dv_d2r

        sol = solve_once(
            gom, rom_a_km, epoch, tgt_lat_deg, tgt_lon_deg, da,
            drift_time_cap_days, max_wrap, tolerance_deg=tolerance_deg
        )
        if sol.get("success"):
            total_h[i] = hours(sol["total_time_s"])
            wait_h[i]  = hours(sol["T_wait"])
            drift_h[i] = hours(sol["T_d"])

        if verbose:
            print(f"[Pareto] Δa={da:+.1f} km  Δv={dv[i]:.3f} km/s  "
                  f"time={total_h[i]:.2f} h (wait={wait_h[i]:.2f}, drift={drift_h[i]:.2f})")

    # mask valid
    m = np.isfinite(dv) & np.isfinite(total_h)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.scatter(dv[m], total_h[m], s=28, label="Total time", zorder=3)

    if include_wait_components:
        mw = m & np.isfinite(wait_h)
        md = m & np.isfinite(drift_h)
        # vertical sticks (drift -> total), plus markers for wait & drift
        if np.any(md):
            ax.vlines(dv[md], drift_h[md], total_h[md], colors="0.75", lw=1, alpha=0.6, zorder=2)
            ax.scatter(dv[md], drift_h[md], s=18, marker="v", label="Drift time", zorder=3)
        if np.any(mw):
            ax.scatter(dv[mw], wait_h[mw], s=18, marker="^", label="Wait time", zorder=3)

    # Pareto front (time vs Δv) — non-dominated set (minimize both)
    if mark_frontier and np.any(m):
        idx = np.argsort(dv[m])
        dv_sorted = dv[m][idx]
        t_sorted  = total_h[m][idx]
        front_mask = np.zeros_like(dv_sorted, dtype=bool)
        best_t = np.inf
        for j in range(len(idx)):
            if t_sorted[j] < best_t - 1e-9:
                front_mask[j] = True
                best_t = t_sorted[j]
        ax.plot(dv_sorted[front_mask], t_sorted[front_mask],
                lw=2.4, color="tab:red", label="Pareto front", zorder=4)

    ax.set_xlabel("Δv (km/s)")
    ax.set_ylabel("Reconfiguration time (h)")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Pareto: time vs Δv  | tol={tolerance_deg:g}°  |  {target_name}")
    ax.legend(loc="best")

    tag = _fmt_target_tag(target_name, tgt_lat_deg, tgt_lon_deg)
    fig.savefig(os.path.join(outdir, f"pareto_time_vs_dv_{tag}.png"), dpi=200, bbox_inches="tight")
    return fig

# --------------- Averaged Pareto: time vs Δv ----------------
def plot_pareto_time_vs_dv_avg_targets(
    gom: Orbit,
    rom_a_km: float,
    epoch,
    *,
    # target grid
    inclination_deg: float,
    lat_band_deg: float,
    n_lat: int,
    n_lon: int,
    # Δa sweep
    delta_min_km: float,
    delta_max_km: float,
    delta_step_km: float,
    # solver knobs
    max_wrap: int,
    drift_time_cap_days: float,
    # plot options
    include_wait_components: bool = True,
    mark_frontier: bool = True,
    outdir: str = "results/reconfig/paretoDV",
    verbose: bool = False,
) -> plt.Figure:
    """
    Average reconfiguration times across an (n_lat × n_lon) target grid within
    ±lat_band_deg (|lat| ≤ band < inclination). Plot ⟨time⟩ vs Δv (km/s).

    Δv is computed per Δa as: Hohmann(GOM→DRIFT=a_G+Δa) + Hohmann(DRIFT→ROM=a_ROM).
    Times are the averaged totals/waits/drifts from your existing Δa-AVG routine.
    """
    ensure_dir(outdir)

    # Build targets
    lats_deg, lons_deg = build_target_grid(inclination_deg, lat_band_deg, n_lat, n_lon)

    # Average times vs Δa over the targets
    dA_km, avg_tot_s, avg_wait_s, avg_drift_s = _avg_over_targets_delta_a(
        gom, rom_a_km, epoch,
        lats_deg, lons_deg,
        delta_min_km, delta_max_km, delta_step_km,
        max_wrap, drift_time_cap_days,
        verbose=verbose
    )

    # Convert Δa → Δv for each point
    aG = gom.a_km
    dv_kms = np.empty_like(dA_km, dtype=float)
    for i, da in enumerate(dA_km):
        aD = aG + float(da)
        if aD <= 0.0:
            dv_kms[i] = np.nan
            continue
        _, _, dv_G2D = _hohmann_dv(aG, aD)
        _, _, dv_D2R = _hohmann_dv(aD, rom_a_km)
        dv_kms[i] = dv_G2D + dv_D2R

    # Drop NaNs and sort by Δv (clean curve & frontier)
    mask = np.isfinite(dv_kms) & np.isfinite(avg_tot_s)
    dv = dv_kms[mask]
    t_tot_h   = (avg_tot_s[mask]   / 3600.0)
    t_wait_h  = (avg_wait_s[mask]  / 3600.0)
    t_drift_h = (avg_drift_s[mask] / 3600.0)

    order = np.argsort(dv)
    dv = dv[order]
    t_tot_h = t_tot_h[order]
    t_wait_h = t_wait_h[order]
    t_drift_h = t_drift_h[order]

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(dv, t_tot_h, lw=2.0, label="⟨total⟩")

    if include_wait_components:
        ax.plot(dv, t_wait_h, lw=1.6, alpha=0.9, label="⟨wait⟩")
        ax.plot(dv, t_drift_h, lw=1.6, alpha=0.9, label="⟨drift⟩")

    if mark_frontier:
        fidx = _pareto_frontier_indices(dv, t_tot_h)
        ax.plot(dv[fidx], t_tot_h[fidx], lw=0, marker='o', ms=5,
                label="Pareto frontier", zorder=5)

    ax.set_xlabel("Δv (km/s)")
    ax.set_ylabel("Average reconfiguration time (h)")
    ax.grid(True, which='both', alpha=0.32)
    ax.legend(loc='best')

    # Title/subtitle in the same style as your AVG plots
    inc = inclination_deg
    alt_gom = gom.a_km - R_EARTH
    alt_rom = rom_a_km - R_EARTH
    npts = len(dv)
    title_main = "Pareto: ⟨time⟩ vs Δv — Averaged across target grid"
    title_line1 = (f"GOM h={alt_gom:.1f} km, ROM h={alt_rom:.1f} km, i={inc:.3f}°, "
                   f"lat band ±{lat_band_deg:.1f}°, grid {n_lat}×{n_lon}, "
                   f"epoch {epoch.strftime('%Y%m%dT%H%MZ')}")
    title_line2 = (f"Δa {delta_min_km:g}..{delta_max_km:g} km (step {delta_step_km:g}, N={npts}); "
                   f"max_wrap {max_wrap}, drift_cap {drift_time_cap_days:.1f} days")
    ax.set_title(f"{title_main}\n{title_line1}\n{title_line2}")

    # Save
    tag = _fmt_target_tag(n_lat, n_lon, lat_band_deg)
    fname = (
        f"pareto_time_vs_dv_AVG_{tag}_"
        f"GOM{gom.a_km:.0f}_ROM{rom_a_km:.0f}_"
        f"inc{inc:.2f}_"
        f"dA{delta_min_km:g}to{delta_max_km:g}_step{delta_step_km:g}_"
        f"cap{drift_time_cap_days:.1f}d.png"
    )
    import os
    ensure_dir(outdir)
    fig.savefig(os.path.join(outdir, fname), dpi=200, bbox_inches="tight")
    return fig

# ---------------------------
# Iso-time contours (Δa × tolerance)
# ---------------------------

def plot_iso_time_contours(
    gom: Orbit, rom_a_km: float, epoch: datetime,
    target_name: str, tgt_lat_deg: float, tgt_lon_deg: float,
    delta_min_km: float = -150, delta_max_km: float = 150, delta_step_km: float = 10,
    tol_list_deg: Iterable[float] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
    drift_time_cap_days: float = 30.0,
    max_wrap: int = 120,
    outdir: str = "results/iso_time",
    verbose: bool = False
):
    ensure_dir(outdir)
    dA = np.arange(delta_min_km, delta_max_km + 1e-9, delta_step_km, dtype=float)
    T = np.array(list(tol_list_deg), dtype=float)

    Z = np.full((len(T), len(dA)), np.nan, dtype=float)  # total time (h)

    for it, tol in enumerate(T):
        for ia, da in enumerate(dA):
            sol = solve_once(
                gom, rom_a_km, epoch, tgt_lat_deg, tgt_lon_deg, da,
                drift_time_cap_days, max_wrap, tolerance_deg=tol
            )
            if sol.get("success") and abs(sol["lon_err_deg"]) <= tol + 1e-6:
                Z[it, ia] = hours(sol["total_time_s"])
        if verbose:
            print(f"[IsoContour] tol={tol:g}° done")

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    cs = ax.contourf(dA, T, Z, levels=15)
    cbar = fig.colorbar(cs, ax=ax, label="Total time (h)")
    ax.set_xlabel("Δa (km)")
    ax.set_ylabel("Tolerance (deg)")
    ax.set_title(f"Iso-time contours | {target_name}")
    tag = _fmt_target_tag(target_name, tgt_lat_deg, tgt_lon_deg)
    fig.savefig(os.path.join(outdir, f"iso_time_contours_{tag}.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_iso_time_contours_avg_targets(
    gom: Orbit, rom_a_km: float, epoch: datetime,
    inclination_deg: float,
    lat_band_deg: float,
    n_lat: int, n_lon: int,
    delta_min_km: float = -150.0,
    delta_max_km: float =  150.0,
    delta_step_km: float = 5.0,
    tol_list_deg: Iterable[float] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10),
    drift_time_cap_days: float = 30.0,
    max_wrap: int = 200,
    outdir: str = "results/reconfig/iso_time_avg",
    verbose: bool = False,
) -> plt.Figure:
    """
    Iso-time contours of *average* total reconfiguration time vs (Δa, tolerance),
    averaged across an n_lat × n_lon target grid spanning ±lat_band_deg.
    """
    ensure_dir(outdir)

    # Build target grid
    lats_deg, lons_deg = build_target_grid(inclination_deg, lat_band_deg, n_lat, n_lon)

    # Axes
    dA = np.arange(delta_min_km, delta_max_km + 1e-9, delta_step_km, dtype=float)
    T  = np.array(list(tol_list_deg), dtype=float)
    na, nt = len(dA), len(T)

    # Accumulators for averaging
    sum_time_h = np.zeros((nt, na), dtype=float)
    counts     = np.zeros((nt, na), dtype=int)

    total_targets = len(lats_deg) * len(lons_deg)
    done_targets  = 0
    if verbose:
        print(f"[IsoAvg] Grid: {n_lat}×{n_lon} targets (±{lat_band_deg:.1f}°), "
              f"Δa {delta_min_km:g}..{delta_max_km:g} step {delta_step_km:g} (N={na}), "
              f"Tols {T.tolist()} (N={nt})")

    # Main loops
    for lat in lats_deg:
        for lon in lons_deg:
            for it, tol in enumerate(T):
                for ia, da in enumerate(dA):
                    sol = solve_once(
                        gom, rom_a_km, epoch,
                        float(lat), float(lon), float(da),
                        drift_time_cap_days, max_wrap,
                        tolerance_deg=float(tol)
                    )
                    if sol.get("success") and abs(sol["lon_err_deg"]) <= tol + 1e-6:
                        sum_time_h[it, ia] += hours(sol["total_time_s"])
                        counts[it, ia]     += 1
            done_targets += 1
            if verbose and (done_targets % max(1, total_targets // 10) == 0):
                print(f"[IsoAvg] Progress: {done_targets}/{total_targets} targets "
                      f"({100.0*done_targets/total_targets:.0f}%)")

    # Averages (NaN where no solutions)
    with np.errstate(invalid='ignore', divide='ignore'):
        Z_avg_h = np.where(counts > 0, sum_time_h / counts, np.nan)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6.2))
    finite_vals = Z_avg_h[np.isfinite(Z_avg_h)]
    if finite_vals.size:
        vmin, vmax = np.nanmin(Z_avg_h), np.nanmax(Z_avg_h)
        levels = np.linspace(vmin, vmax, 16)
        cs = ax.contourf(dA, T, Z_avg_h, levels=levels)
    else:
        cs = ax.contourf(dA, T, Z_avg_h, levels=15)

    fig.colorbar(cs, ax=ax, label="Average total reconfig time (h)")
    ax.set_xlabel("Δa (km)")
    ax.set_ylabel("Tolerance (deg)")

    # ----- Title (styled like your avg Δa plot, but richer) -----
    hG = gom.a_km - R_EARTH
    hR = rom_a_km - R_EARTH
    tol_min, tol_max = np.nanmin(T), np.nanmax(T)
    fig.suptitle(
        "Iso-time contours — Averaged across "
        f"{n_lat}×{n_lon} targets\n"
        f"GOM h={hG:.1f} km, ROM h={hR:.1f} km, i={inclination_deg:.2f}°\n"
        f"Latitude band ±{lat_band_deg:.1f}°, epoch {epoch.strftime('%Y-%m-%d')}\n"
        f"Δa {delta_min_km:+.0f}..{delta_max_km:+.0f} km (step {delta_step_km:g}), "
        f"tolerance {tol_min:g}..{tol_max:g}° (N={len(T)}), "
        f"drift cap {drift_time_cap_days:.1f} d, max_wrap {max_wrap}"
    )
    plt.tight_layout()

    # ----- Filename (compact, informative, filesystem-safe) -----
    import os
    fname = (
        f"IsoTimeAvg_{n_lat}x{n_lon}_"
        f"GOM{gom.a_km:.0f}_ROM{rom_a_km:.0f}_"
        f"inc{inclination_deg:.1f}_band{lat_band_deg:.1f}_"
        f"dA{delta_min_km:+.0f}to{delta_max_km:+.0f}_s{delta_step_km:g}_"
        f"tol{tol_min:g}to{tol_max:g}_N{len(T)}_"
        f"cap{drift_time_cap_days:.0f}d_"
        f"wrap{max_wrap}_"
        f"{epoch.strftime('%Y%m%d')}.png"
    )
    fig.savefig(os.path.join(outdir, fname), dpi=220, bbox_inches="tight")
    return fig


# ---------------------------
# Chosen solution vs tolerance
# ---------------------------

def plot_chosen_solution_vs_tolerance(
    gom: Orbit, rom_a_km: float, epoch: datetime,
    target_name: str, tgt_lat_deg: float, tgt_lon_deg: float,
    delta_a_fixed_km: float,
    tol_list_deg: Iterable[float] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2),
    drift_time_cap_days: float = 30.0,
    max_wrap: int = 120,
    outdir: str = "results/reconfig/sol_vs_tol",
    verbose: bool = False
):
    ensure_dir(outdir)
    T = np.array(list(tol_list_deg), dtype=float)
    total_h = np.full_like(T, np.nan, dtype=float)
    wait_h  = np.full_like(T, np.nan, dtype=float)
    drift_h = np.full_like(T, np.nan, dtype=float)
    m_sel   = np.full_like(T, np.nan, dtype=float)
    k_sel   = np.full_like(T, np.nan, dtype=float)
    branch  = []

    for i, tol in enumerate(T):
        sol = solve_once(
            gom, rom_a_km, epoch, tgt_lat_deg, tgt_lon_deg, delta_a_fixed_km,
            drift_time_cap_days, max_wrap, tolerance_deg=tol
        )
        if sol.get("success"):
            total_h[i] = hours(sol["total_time_s"])
            wait_h[i]  = hours(sol["T_wait"])
            drift_h[i] = hours(sol["T_d"])
            meta = sol.get("selection_meta", {})
            m_sel[i]  = meta.get("chosen_m", np.nan)
            k_sel[i]  = meta.get("chosen_k", np.nan)
            branch.append(sol.get("branch", ""))
        else:
            branch.append("")

        if verbose:
            print(f"[SolVsTol] tol={tol:g}°  total={total_h[i]:.2f}h  m={m_sel[i]} k={k_sel[i]} branch={branch[-1]}")

    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axs[0].plot(T, total_h, marker='o', label="total")
    axs[0].plot(T, wait_h,  marker='o', label="wait", alpha=0.8)
    axs[0].plot(T, drift_h, marker='o', label="drift", alpha=0.8)
    axs[0].set_ylabel("Time (h)")
    axs[0].grid(True, alpha=0.3); axs[0].legend()

    axs[1].step(T, m_sel, where='post'); axs[1].set_ylabel("m (u-wrap)")
    axs[1].grid(True, alpha=0.3)

    axs[2].step(T, k_sel, where='post'); axs[2].set_ylabel("k (lon-wrap)")
    axs[2].set_xlabel("Tolerance (deg)")
    axs[2].grid(True, alpha=0.3)

    fig.suptitle(f"Chosen solution vs tolerance | Δa={delta_a_fixed_km:+.1f} km | {target_name}")
    tag = _fmt_target_tag(target_name, tgt_lat_deg, tgt_lon_deg)
    fig.savefig(os.path.join(outdir, f"chosen_vs_tol_{tag}_dA{delta_a_fixed_km:+.1f}.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

# ---------------------------
# Tolerance sensitivity curves (several Δa)
# ---------------------------

def plot_tolerance_sensitivity_curves(
    gom: Orbit, rom_a_km: float, epoch: datetime,
    target_name: str, tgt_lat_deg: float, tgt_lon_deg: float,
    delta_list_km: Iterable[float],
    tol_list_deg: Iterable[float] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2),
    drift_time_cap_days: float = 30.0,
    max_wrap: int = 120,
    outdir: str = "results/reconfig/tol_sensitivity",
    verbose: bool = False
):
    ensure_dir(outdir)
    T = np.array(list(tol_list_deg), dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for da in delta_list_km:
        total_h = np.full_like(T, np.nan, dtype=float)
        for i, tol in enumerate(T):
            sol = solve_once(
                gom, rom_a_km, epoch, tgt_lat_deg, tgt_lon_deg, da,
                drift_time_cap_days, max_wrap, tolerance_deg=tol
            )
            if sol.get("success") and abs(sol["lon_err_deg"]) <= tol + 1e-6:
                total_h[i] = hours(sol["total_time_s"])
        ax.plot(T, total_h, marker='o', label=f"Δa={da:+.0f} km")
        if verbose:
            print(f"[TolSens] Δa={da:+.0f} km done")

    ax.set_xlabel("Tolerance (deg)")
    ax.set_ylabel("Total reconfig time (h)")
    ax.set_title(f"Tolerance sensitivity | {target_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    tag = _fmt_target_tag(target_name, tgt_lat_deg, tgt_lon_deg)
    fig.savefig(os.path.join(outdir, f"tol_sensitivity_{tag}.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

# ---------------------------
# Target jitter study (random offsets around target)
# ---------------------------

def plot_target_jitter_study(
    gom: Orbit, rom_a_km: float, epoch: datetime,
    target_name: str, tgt_lat_deg: float, tgt_lon_deg: float,
    delta_a_fixed_km: float,
    tol_fixed_deg: float = 0.25,
    n_samples: int = 200,
    jitter_lat_deg: float = 1.0,    # ±1° box by default
    jitter_lon_deg: float = 1.0,
    drift_time_cap_days: float = 30.0,
    max_wrap: int = 120,
    outdir: str = "results/reconfig/jitter",
    seed: int = 1,
    verbose: bool = False
):
    ensure_dir(outdir)
    rng = np.random.default_rng(seed)
    dlat = rng.uniform(-jitter_lat_deg, +jitter_lat_deg, size=n_samples)
    dlon = rng.uniform(-jitter_lon_deg, +jitter_lon_deg, size=n_samples)

    times_h = np.full(n_samples, np.nan)
    waits_h = np.full(n_samples, np.nan)
    drifts_h = np.full(n_samples, np.nan)

    for i in range(n_samples):
        sol = solve_once(
            gom, rom_a_km, epoch,
            tgt_lat_deg + dlat[i], tgt_lon_deg + dlon[i],
            delta_a_fixed_km, drift_time_cap_days, max_wrap, tolerance_deg=tol_fixed_deg
        )
        if sol.get("success"):
            times_h[i]  = hours(sol["total_time_s"])
            waits_h[i]  = hours(sol["T_wait"])
            drifts_h[i] = hours(sol["T_d"])

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    axs[0].hist(times_h[~np.isnan(times_h)], bins=24, alpha=0.85)
    axs[0].set_xlabel("Total time (h)"); axs[0].set_ylabel("count"); axs[0].grid(True, alpha=0.25)
    axs[0].set_title("Distribution of total times")

    sc = axs[1].scatter(dlon, dlat, c=times_h, s=20)
    axs[1].set_xlabel("lon offset (deg)"); axs[1].set_ylabel("lat offset (deg)")
    axs[1].grid(True, alpha=0.25); axs[1].set_title("Offsets colored by total time (h)")
    cbar = fig.colorbar(sc, ax=axs[1]); cbar.set_label("Total time (h)")

    tag = _fmt_target_tag(target_name, tgt_lat_deg, tgt_lon_deg)
    fig.suptitle(f"Target jitter study | Δa={delta_a_fixed_km:+.1f} km, tol={tol_fixed_deg}° | {target_name}")
    fig.savefig(os.path.join(outdir, f"jitter_{tag}_dA{delta_a_fixed_km:+.1f}_tol{tol_fixed_deg}.png"),
                dpi=210, bbox_inches="tight")
    plt.close(fig)

# ---------------------------
# Epoch sensitivity (sweep start epoch over 24h)
# ---------------------------

def plot_epoch_sensitivity(
    gom: Orbit, rom_a_km: float, base_epoch: datetime,
    target_name: str, tgt_lat_deg: float, tgt_lon_deg: float,
    delta_a_fixed_km: float,
    tol_fixed_deg: float = 0.25,
    hours_span: float = 24.0,
    step_minutes: float = 20.0,
    drift_time_cap_days: float = 30.0,
    max_wrap: int = 120,
    outdir: str = "results/reconfig/epoch_sensitivity",
    verbose: bool = False
):
    ensure_dir(outdir)
    N = int(np.floor(hours_span * 60.0 / step_minutes)) + 1
    dt_list = np.arange(N) * step_minutes
    times_h = np.full(N, np.nan)

    for i, m in enumerate(dt_list):
        epoch_i = base_epoch + timedelta(minutes=float(m))
        sol = solve_once(
            gom, rom_a_km, epoch_i, tgt_lat_deg, tgt_lon_deg,
            delta_a_fixed_km, drift_time_cap_days, max_wrap, tolerance_deg=tol_fixed_deg
        )
        if sol.get("success"):
            times_h[i] = hours(sol["total_time_s"])
        if verbose and (i % max(1, N//10) == 0):
            print(f"[Epoch] {i+1}/{N}  t0+{m:g} min  total={times_h[i]:.2f} h")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(dt_list/60.0, times_h, marker='o')
    ax.set_xlabel("Start epoch offset (hours)")
    ax.set_ylabel("Total reconfig time (h)")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Epoch sensitivity | Δa={delta_a_fixed_km:+.1f} km, tol={tol_fixed_deg}° | {target_name}")
    tag = _fmt_target_tag(target_name, tgt_lat_deg, tgt_lon_deg)
    fig.savefig(os.path.join(outdir, f"epoch_sensitivity_{tag}_dA{delta_a_fixed_km:+.1f}.png"),
                dpi=210, bbox_inches="tight")
    plt.close(fig)

# ---------------------------
# Residual curves for selected (m,k)
# ---------------------------

def _residual_functions(planner: AnalyticRaanPhasingPlanner, branch: str, m: int, k: int, u_req: float):
    """
    Recreate residual r(Td) = phi(Td) - (λ_t + 2πk) and give helper closures, using the same
    formulas as the planner.
    """
    # Eq A: du_G*Tw + du_D*Td = const_u + 2π m
    const_u = -(planner.du_T1 * planner.T1 + planner.du_T2 * planner.T2) + (u_req - planner.u0)
    A_m = const_u + 2.0*np.pi*m

    def Tw_of_Td(Td):
        return (A_m - planner.du_D * Td) / planner.du_G

    def Omega_f_of_Td(Td, Tw):
        return (planner.Omega0 +
                planner.dOm_G * Tw +
                planner.dOm_T1 * planner.T1 +
                planner.dOm_D * Td +
                planner.dOm_T2 * planner.T2)

    psi = np.arctan2(np.cos(planner.i) * np.sin(u_req), np.cos(u_req))

    def total_time(Td, Tw):
        return Tw + planner.T1 + Td + planner.T2

    def phi_of(Td):
        Tw = Tw_of_Td(Td)
        tf = planner.epoch + timedelta(seconds=float(total_time(Td, Tw)))
        thetaG = gmst_rad(tf)
        phi = Omega_f_of_Td(Td, Tw) + psi - thetaG  # unwrapped
        return phi, Tw

    lam_t = planner.target_lon_rad + 2.0*np.pi*k

    def residual(Td):
        phi, Tw = phi_of(Td)
        if Tw < -1e-9:
            return 1e9
        return phi - lam_t

    return residual, Tw_of_Td

def plot_residual_curves_for_selected_mk(
    gom: Orbit, rom_a_km: float, epoch: datetime,
    target_name: str, tgt_lat_deg: float, tgt_lon_deg: float,
    delta_a_fixed_km: float,
    tol_fixed_deg: float = 0.25,
    drift_time_cap_days: float = 30.0,
    wait_cap_hours: float = 150.0,       # match planner's internal default
    max_wrap: int = 120,
    outdir: str = "results/reconfig/residuals",
    verbose: bool = False
):
    ensure_dir(outdir)
    planner = AnalyticRaanPhasingPlanner(
        gom=gom, rom_a_km=rom_a_km, delta_a_km=delta_a_fixed_km,
        epoch_gom=epoch, target_lat_geod_deg=tgt_lat_deg, target_lon_deg=tgt_lon_deg,
        max_wrap=max_wrap, drift_time_cap_days=drift_time_cap_days
    )
    sol = planner.solve_min_time(lon_tol_schedule=[tol_fixed_deg])
    if not sol.get("success"):
        if verbose:
            print("[Residual] No solution — skipping plot.")
        return

    # pick branch and u_req
    branch = sol["branch"]
    u_asc, u_desc = u_required_branches(tgt_lat_deg, planner.i)
    u_req = u_asc if branch == "ascending" else u_desc

    # selected m,k
    meta = sol.get("selection_meta", {})
    m = int(meta.get("chosen_m", 0))
    k = int(meta.get("chosen_k", 0))

    residual, Tw_of_Td = _residual_functions(planner, branch, m, k, u_req)

    # feasible Td interval w.r.t wait cap (same logic as planner)
    Td_lo = 0.0
    Td_hi = planner.drift_time_cap
    if wait_cap_hours is None:
        # only Tw>=0
        # root for Tw=0: t0 = A/du_D (embedded in residual fn; approximate with sampling)
        pass
    # we mimic the full feasible window via dense sampling
    Ns = 800
    Td_grid = np.linspace(Td_lo, Td_hi, Ns)
    Tw = np.array([Tw_of_Td(t) for t in Td_grid])
    if wait_cap_hours is not None:
        Tw_cap = wait_cap_hours*3600.0
        mask = (Tw >= -1e-9) & (Tw <= Tw_cap + 1e-9)
    else:
        mask = (Tw >= -1e-9)

    Td_feas = Td_grid[mask]
    if Td_feas.size < 3:
        # fallback: plot nothing substantial
        return

    r = np.array([residual(t) for t in Td_feas])
    r_deg = np.degrees(np.where(r==1e9, np.nan, r))

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(Td_feas/3600.0, r_deg, lw=1.6)
    ax.axhline(+tol_fixed_deg, color='0.5', ls='--')
    ax.axhline(-tol_fixed_deg, color='0.5', ls='--')
    ax.axhline(0.0, color='k', lw=1)
    ax.set_xlabel("Drift time Td (h)")
    ax.set_ylabel("Residual r(Td) [deg]")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Residual curve (m={m}, k={k}, {branch}) | Δa={delta_a_fixed_km:+.1f} km, tol={tol_fixed_deg}° | {target_name}")

    tag = _fmt_target_tag(target_name, tgt_lat_deg, tgt_lon_deg)
    fig.savefig(os.path.join(outdir, f"residual_curve_{tag}_dA{delta_a_fixed_km:+.1f}.png"),
                dpi=220, bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------

@dataclass
class PlotFlags:
    # --- core case studies ---
    times_vs_delta_a: bool = False
    times_vs_tolerance: bool = False
    # --- averaged over target grid ---
    avg_vs_delta_a: bool = False
    avg_vs_tolerance: bool = False
    # --- heatmaps ---
    heatmap: bool = False
    # --- diagnostics (new) ---
    pareto_time_vs_dv: bool = False
    iso_time_contours: bool = False
    chosen_vs_tolerance: bool = False
    tolerance_sensitivity: bool = False
    target_jitter: bool = False
    epoch_sensitivity: bool = False
    residual_curves: bool = False


def run_reconfig_plots(
    *,
    # scenario
    gom: Orbit,
    rom_a_km: float,
    epoch: datetime,
    target_name: Optional[str],
    target_lat_geod_deg: float,
    target_lon_deg: float,

    # flags
    flags: PlotFlags = PlotFlags(),

    # -----------------------------
    # case: Δa sweep (core + Pareto)
    # -----------------------------
    delta_min_km: float = -200.0,
    delta_max_km: float =  200.0,
    delta_step_km: float = 10.0,

    # ------------------------------------------------
    # case: tolerance sweep (core + chosen_vs_tolerance)
    # ------------------------------------------------
    delta_a_fixed_km: float = 50.0,
    tolerances_deg: Optional[List[float]] = None,

    # -----------------------------------------
    # averaged target grid (core + iso-time/heatmap)
    # -----------------------------------------
    inclination_deg: Optional[float] = None,    # defaults to gom.i_rad
    lat_band_deg: Optional[float] = None,       # default derived from inc
    n_lat: int = 11,                            # targets across latitude band
    n_lon: int = 360,                            # targets across full longitude

    # -------------- diagnostics knobs --------------
    # Pareto time vs ΔV
    pareto_include_wait: bool = True,           # whether to show wait/drift/total in Pareto labels
    # Iso-time contours (hours)
    iso_levels_hours: Optional[List[float]] = None,  # e.g. [12, 24, 48, 72, 96]
    # Tolerance sensitivity curves (case study)
    tol_sens_include_series: Tuple[bool, bool, bool] = (True, True, True),  # (total, wait, drift)
    # Target jitter study
    jitter_radius_deg: float = 1.0,             # radial jitter (deg) around the target
    jitter_samples: int = 200,                  # number of random samples
    jitter_seed: int = 7,
    jitter_tolerance_deg: float = 0.25,         # fixed tol for jitter plot
    # Epoch sensitivity
    epoch_window_hours: float = 24.0,           # +/- window around epoch
    epoch_samples: int = 49,                    # odd number to include center
    epoch_tolerance_deg: float = 0.25,          # fixed tol for epoch plot
    # Residual curves
    residual_branch: Optional[str] = None,      # 'ascending'/'descending'/None(auto)
    residual_m: Optional[int] = None,           # specific m, or None for best
    residual_k: Optional[int] = None,           # specific k, or None for best
    residual_nsamples: int = 600,               # Td samples across feasible band

    # -------------- common solver knobs --------------
    max_wrap: int = 200,
    drift_time_cap_days: float = (200.0 / 24.0),

    # IO
    verbose: bool = False,
) -> Dict[str, plt.Figure]:
    """
    One-stop function to generate all requested plots; returns dict of figures
    keyed by short names. Keeps existing behavior and adds diagnostics.

    Core plots (kept):
      - times_vs_delta_a                    -> 'case_deltaA'
      - times_vs_tolerance                  -> 'case_tolerance'
      - avg_vs_delta_a                      -> 'avg_deltaA'
      - avg_vs_tolerance                    -> 'avg_tolerance'
      - heatmap                             -> 'heatmap'

    Diagnostics (new):
      - pareto_time_vs_dv                   -> 'pareto'
      - iso_time_contours                   -> 'iso_time'
      - chosen_vs_tolerance                 -> 'chosen_vs_tol'
      - tolerance_sensitivity               -> 'tol_sensitivity'
      - target_jitter                       -> 'jitter'
      - epoch_sensitivity                   -> 'epoch_sensitivity'
      - residual_curves                     -> 'residuals'
    """
    figs: Dict[str, plt.Figure] = {}

    # Default tolerance ladder (exponential-ish)
    if tolerances_deg is None:
        tolerances_deg = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
                          0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    # Defaults for iso-time levels
    if iso_levels_hours is None:
        iso_levels_hours = [12.0, 24.0, 48.0, 72.0, 96.0]

    # Derived inclination / latitude band for grid-based plots
    use_inc_deg = rad2deg(gom.i_rad) if inclination_deg is None else float(inclination_deg)
    use_band_deg = min(use_inc_deg - 1.0, 0.9 * use_inc_deg) if (lat_band_deg is None) else float(lat_band_deg)
    if use_band_deg <= 0.0 and (flags.avg_vs_delta_a or flags.avg_vs_tolerance or flags.heatmap or flags.iso_time_contours):
        raise ValueError("Latitude band must be positive and smaller than inclination for grid-based plots.")

    # --------------------------
    # Core: case-study plots
    # --------------------------
    if flags.times_vs_delta_a:
        figs['case_deltaA'] = plot_times_vs_delta_a_case(
            gom, rom_a_km, epoch,
            target_name, target_lat_geod_deg, target_lon_deg,
            delta_min_km, delta_max_km, delta_step_km,
            max_wrap, drift_time_cap_days,
            outdir="results/times_deltaA", verbose=verbose
        )

    if flags.times_vs_tolerance:
        figs['case_tolerance'] = plot_times_vs_tolerance_case(
            gom, rom_a_km, epoch,
            target_name, target_lat_geod_deg, target_lon_deg,
            delta_a_fixed_km, tolerances_deg,
            max_wrap, drift_time_cap_days,
            outdir="results/times_tolerance", verbose=verbose
        )

    # --------------------------
    # Core: averaged/grid plots
    # --------------------------
    if flags.avg_vs_delta_a:
        figs['avg_deltaA'] = plot_times_vs_delta_a_avg_targets(
            gom, rom_a_km, epoch,
            use_inc_deg, use_band_deg, n_lat, n_lon,
            delta_min_km, delta_max_km, delta_step_km,
            max_wrap, drift_time_cap_days,
            outdir="results/reconfig/times_deltaA", verbose=verbose
        )

    if flags.avg_vs_tolerance:
        figs['avg_tolerance'] = plot_times_vs_tolerance_avg_targets(
            gom, rom_a_km, epoch,
            use_inc_deg, use_band_deg, n_lat, n_lon,
            delta_a_fixed_km, tolerances_deg,
            max_wrap, drift_time_cap_days,
            outdir="results/reconfig/times_tolerance", verbose=verbose
        )

    if flags.heatmap:
        # choose a mid-ladder tolerance for the heatmap by default
        heat_tol = tolerances_deg[min(5, len(tolerances_deg) - 1)]  # ~0.1–0.25°
        figs['heatmap'] = plot_heatmaps_for_case(
            gom, rom_a_km, epoch,
            use_inc_deg, use_band_deg, n_lat, n_lon,
            delta_a_fixed_km, heat_tol,
            max_wrap, drift_time_cap_days,
            outdir="results/reconfig/heatmaps", verbose=verbose
        )

    # --------------------------
    # Diagnostics (new)
    # --------------------------
    if flags.pareto_time_vs_dv:
        figs["pareto"] = plot_pareto_time_vs_dv_avg_targets(
            gom=gom,
            rom_a_km=rom_a_km,
            epoch=epoch,
            inclination_deg=use_inc_deg,
            lat_band_deg=use_band_deg,
            n_lat=n_lat,
            n_lon=n_lon,
            delta_min_km=delta_min_km,
            delta_max_km=delta_max_km,
            delta_step_km=delta_step_km,
            max_wrap=max_wrap,
            drift_time_cap_days=drift_time_cap_days,
            include_wait_components=True,
            mark_frontier=True,
            outdir="results/reconfig/paretoDV",
            verbose=verbose,
        )

    if flags.iso_time_contours:
        figs['iso_time'] = _call_with_supported_kwargs(plot_iso_time_contours_avg_targets,
            gom, rom_a_km, epoch,
            use_inc_deg, use_band_deg, n_lat, n_lon,
            delta_a_fixed_km,
            levels_hours=iso_levels_hours,
            max_wrap=max_wrap, drift_time_cap_days=drift_time_cap_days,
            outdir="results/reconfig/iso_time", verbose=verbose
        )

    if flags.chosen_vs_tolerance:
        figs['chosen_vs_tol'] = plot_chosen_solution_vs_tolerance(
            gom, rom_a_km, epoch,
            target_name, target_lat_geod_deg, target_lon_deg,
            delta_a_fixed_km, tolerances_deg,
            max_wrap=max_wrap, drift_time_cap_days=drift_time_cap_days,
            outdir="results/reconfig/chosen_vs_tol", verbose=verbose
        )

    if flags.tolerance_sensitivity:
        figs['tol_sensitivity'] = _call_with_supported_kwargs(plot_tolerance_sensitivity_curves,
            gom, rom_a_km, epoch,
            target_name, target_lat_geod_deg, target_lon_deg,
            delta_a_fixed_km, tolerances_deg,
            include_series=tol_sens_include_series,  # (total, wait, drift)
            max_wrap=max_wrap, drift_time_cap_days=drift_time_cap_days,
            outdir="results/reconfig/tolerance_sensitivity", verbose=verbose
        )

    if flags.target_jitter:
        figs['jitter'] = _call_with_supported_kwargs(plot_target_jitter_study,
            gom, rom_a_km, epoch,
            center_name=target_name,
            center_lat_deg=target_lat_geod_deg,
            center_lon_deg=target_lon_deg,
            jitter_radius_deg=jitter_radius_deg,
            n_samples=jitter_samples,
            seed=jitter_seed,
            delta_a_fixed_km=delta_a_fixed_km,
            tolerance_deg=jitter_tolerance_deg,
            max_wrap=max_wrap, drift_time_cap_days=drift_time_cap_days,
            outdir="results/reconfig/jitter", verbose=verbose
        )

    if flags.epoch_sensitivity:
        figs['epoch_sensitivity'] = _call_with_supported_kwargs(plot_epoch_sensitivity,
            gom, rom_a_km, epoch,
            target_name, target_lat_geod_deg, target_lon_deg,
            delta_a_fixed_km=delta_a_fixed_km,
            tolerance_deg=epoch_tolerance_deg,
            window_hours=epoch_window_hours,
            n_samples=epoch_samples,
            max_wrap=max_wrap, drift_time_cap_days=drift_time_cap_days,
            outdir="results/reconfig/epoch_sensitivity", verbose=verbose
        )

    if flags.residual_curves:
        figs['residuals'] = _call_with_supported_kwargs(plot_residual_curves_for_selected_mk,
            gom, rom_a_km, epoch,
            target_name, target_lat_geod_deg, target_lon_deg,
            delta_a_fixed_km=delta_a_fixed_km,
            branch=residual_branch, m_sel=residual_m, k_sel=residual_k,
            nsamples=residual_nsamples,
            max_wrap=max_wrap, drift_time_cap_days=drift_time_cap_days,
            outdir="results/reconfig/residuals", verbose=verbose
        )

    return figs
