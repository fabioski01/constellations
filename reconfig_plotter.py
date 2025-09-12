# -*- coding: utf-8 -*-
"""
Plotting utilities for the analytic reconfiguration planner.

This module depends on your solver module `reconfig_driftANDwaitTime` and
now lives separately so you can delete plot code from the solver.

Provided plots (enable via flags in run_reconfig_plots):
  1) Reconfig time vs Δa (case study)
  2) Reconfig time vs tolerance (case study)
  3) Reconfig time vs Δa averaged across a target grid
  4) Reconfig time vs tolerance averaged across a target grid
  5) Heatmaps over the world (within a latitude band) for total / wait / drift
     at a fixed (Δa, tolerance), with the first hour of the GOM ground track overlaid.

Author: you :)
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# pull in your solver & helpers
from reconfig_driftANDwaitTime import (
    AnalyticRaanPhasingPlanner,
    Orbit,
    deg2rad, rad2deg,
    R_EARTH,
    compute_reconfig_times_vs_delta_a,  # reuse your existing sweep
    gmst_rad,                           # for ground track
    j2_rates,
)

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
    from reconfig_driftANDwaitTime import compute_reconfig_times_vs_delta_a
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
            if verbose and done == 0:
                print(f"[AVG Δa] Processing target ({lat:.2f}°, {lon:.2f}°) - first target...")
            dA, tot, wai, dri = compute_reconfig_times_vs_delta_a(
                gom, rom_a_km, epoch, float(lat), float(lon),
                delta_min_km, delta_max_km, delta_step_km, max_wrap, drift_time_cap_days, verbose=False
            )
            mask = ~np.isnan(tot)
            sum_tot[mask]  += tot[mask]
            sum_wait[mask] += wai[mask]
            sum_drift[mask]+= dri[mask]
            counts[mask]   += 1
            done += 1
            if verbose and (done % max(1, total_targets // 10) == 0 or done == total_targets):
                pct = 100.0 * done / total_targets
                print(f"[AVG Δa] Progress: {done}/{total_targets} ({pct:.1f}%) - current target ({lat:.2f}°, {lon:.2f}°)")

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
            if verbose and done == 0:
                print(f"[AVG Tol] Processing target ({lat:.2f}°, {lon:.2f}°) - first target...")
            t, tot, wai, dri = compute_reconfig_times_vs_tolerance(
                gom, rom_a_km, epoch, float(lat), float(lon),
                delta_a_fixed_km, tolerances_deg, max_wrap, drift_time_cap_days, verbose=False
            )
            mask = ~np.isnan(tot)
            sum_tot[mask]  += tot[mask]
            sum_wait[mask] += wai[mask]
            sum_drift[mask]+= dri[mask]
            counts[mask]   += 1
            done += 1
            if verbose and (done % max(1, total_targets // 10) == 0 or done == total_targets):
                pct = 100.0 * done / total_targets
                print(f"[AVG Tol] Progress: {done}/{total_targets} ({pct:.1f}%) - current target ({lat:.2f}°, {lon:.2f}°)")

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

    # figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    extent = (-180.0, 180.0, -lat_band_deg, +lat_band_deg)

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
        vmin, vmax = finite_bounds(data)
        im = ax.imshow(
            data, origin='lower', extent=extent, aspect='auto',
            vmin=vmin, vmax=vmax, cmap='viridis'
        )
        ax.set_ylabel("Latitude [deg]")
        ax.set_title(title)
        ax.grid(True, color='w', alpha=0.15, lw=0.8)
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
            _plot_lonwrap_segments(ax, gt_lon[s:e], gt_lat[s:e], color='k', lw=1.5, alpha=alpha)

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


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------

@dataclass
class PlotFlags:
    # case studies
    times_vs_delta_a: bool = False
    times_vs_tolerance: bool = False
    # averaged over target grid
    avg_vs_delta_a: bool = False
    avg_vs_tolerance: bool = False
    # heatmaps
    heatmap: bool = False


def run_reconfig_plots(
    *,
    # scenario
    gom: Orbit,
    rom_a_km: float,
    epoch,
    target_name: Optional[str],
    target_lat_geod_deg: float,
    target_lon_deg: float,

    # flags
    flags: PlotFlags = PlotFlags(),

    # case: Δa sweep
    delta_min_km: float = -200.0,
    delta_max_km: float = 200.0,
    delta_step_km: float = 10.0,

    # case: tolerance sweep
    delta_a_fixed_km: float = 50.0,
    tolerances_deg: Optional[List[float]] = None,

    # averaged grid
    inclination_deg: Optional[float] = None,
    lat_band_deg: Optional[float] = None,
    n_lat: int = 10,
    n_lon: int = 30,

    # common solver knobs
    max_wrap: int = 200,
    drift_time_cap_days: float = (200.0/24.0),

    # IO
    verbose: bool = False,
) -> Dict[str, plt.Figure]:
    """
    One-stop function to generate all requested plots; returns dict of figures.
    """
    figs: Dict[str, plt.Figure] = {}

    if tolerances_deg is None:
        # a sensible exponential ladder (similar to what you printed earlier)
        tolerances_deg = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

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

    # averaged across targets -> need an inclination & band
    use_inc_deg = rad2deg(gom.i_rad) if inclination_deg is None else float(inclination_deg)
    use_band_deg = min(use_inc_deg - 1.0, 0.9*use_inc_deg) if (lat_band_deg is None) else float(lat_band_deg)
    if use_band_deg <= 0.0:
        raise ValueError("Latitude band must be positive and smaller than inclination.")

    if flags.avg_vs_delta_a:
        figs['avg_deltaA'] = plot_times_vs_delta_a_avg_targets(
            gom, rom_a_km, epoch,
            use_inc_deg, use_band_deg, n_lat, n_lon,
            delta_min_km, delta_max_km, delta_step_km,
            max_wrap, drift_time_cap_days,
            outdir="results/plots_avg", verbose=verbose
        )

    if flags.avg_vs_tolerance:
        figs['avg_tolerance'] = plot_times_vs_tolerance_avg_targets(
            gom, rom_a_km, epoch,
            use_inc_deg, use_band_deg, n_lat, n_lon,
            delta_a_fixed_km, tolerances_deg,
            max_wrap, drift_time_cap_days,
            outdir="results/plots_avg", verbose=verbose
        )

    if flags.heatmap:
        # pick one tolerance from the ladder (e.g., the 0.25° you liked), but it's configurable
        heat_tol = tolerances_deg[min(5, len(tolerances_deg)-1)]  # default ~0.1–0.25 deg
        figs['heatmap'] = plot_heatmaps_for_case(
            gom, rom_a_km, epoch,
            use_inc_deg, use_band_deg, n_lat, n_lon,
            delta_a_fixed_km, heat_tol,
            max_wrap, drift_time_cap_days,
            outdir="results/plots_heatmaps", verbose=verbose
        )

    return figs


# --------------------------------------------------------------------------------------
# Optional: quick demo if run directly
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import datetime, timezone
    # demo scenario (same as your sample)
    epoch = datetime(2025, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    inc_deg = 51.4476847948
    gom = Orbit(a_km=R_EARTH+500.0, i_rad=deg2rad(inc_deg), Omega_rad=0.0, u_rad=0.0)

    # case target
    target_name = "LosAngeles"
    tgt_lat, tgt_lon = 34.0522, -118.2437

    figs = run_reconfig_plots(
        gom=gom,
        rom_a_km=R_EARTH + 504.59534097,
        epoch=epoch,
        target_name=target_name,
        target_lat_geod_deg=tgt_lat,
        target_lon_deg=tgt_lon,
        flags=PlotFlags(
            times_vs_delta_a=True,
            times_vs_tolerance=True,
            avg_vs_delta_a=True,
            avg_vs_tolerance=True,
            heatmap=True,
        ),
        delta_min_km=-100, delta_max_km=200, delta_step_km=20,
        delta_a_fixed_km=50.0,
        tolerances_deg=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        inclination_deg=inc_deg,
        lat_band_deg=40.0,
        n_lat=10, n_lon=30,
        max_wrap=200,
        drift_time_cap_days=200/24,
        verbose=True,
    )
    print(f"Generated {len(figs)} figure(s).")
