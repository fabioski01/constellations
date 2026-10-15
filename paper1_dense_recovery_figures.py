"""Make corrected-stage figures with the existing plotting framework."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper1_dense_figures import load_grid
from reconfig_plot_from_csv import plot_timesAvgDeltaA_from_csv
from reconfig_plotter_multicore import (Orbit, plot_saved_altitude_summary,
    plot_saved_tolerance_summary, plot_saved_heatmaps, plot_saved_geographic_gain)

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "dense-results/corrected"
OUTPUT = ROOT.parent / "dense-results/paper-results"


def main():
    completion = json.loads((DATA / "completion.json").read_text())
    audit = json.loads((DATA / "audit/independent_audit.json").read_text())
    if not completion["complete"] or audit["failures"]:
        raise ValueError("A complete audited corrected stage is required.")
    manifest = json.loads((DATA / "manifest.json").read_text())
    summary = pd.read_csv(DATA / "summary.csv")
    figures = OUTPUT / "figures"
    tables = OUTPUT / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(exist_ok=True)
    cases = list({e["case"]["case_id"]: e["case"] for e in manifest["experiments"]}.values())
    cases.sort(key=lambda c: c["case_id"])
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        for ext in ["png", "pdf"]:
            fig.savefig(figures / (name + "." + ext), dpi=230)
        plt.close(fig)

    common = summary[summary.profile.eq("common")]
    for case in cases:
        rows = common[common.case_id.eq(case["case_id"]) & np.isclose(common.tolerance_deg, .25)].sort_values("offset_km")
        data = pd.DataFrame(dict(delta_a_km=rows.offset_km,
            avg_total_s=rows.reported_mean_response_h * 3600,
            avg_wait_s=rows.mean_wait_h * 3600, avg_drift_s=rows.mean_drift_h * 3600,
            accepted_count=rows.reported_accepted_count, requested_count=rows.requested_count))
        file = tables / f'TimesAvgDeltaA_180x360_GOM{case["a_g_km"]:.0f}_ROM{case["a_r_km"]:.0f}_{case["case_id"]}_corrected.csv'
        data.to_csv(file, index=False)
        fig = plot_timesAvgDeltaA_from_csv(file, outdir=str(figures),
            gom_a_km=case["a_g_km"], rom_a_km=case["a_r_km"],
            figure_title=f'{case["case_id"]} | 64,800 targets | 0.25° tolerance')
        plt.close(fig)
    save(plot_saved_altitude_summary(common, cases[:3],
        title="Altitude trade | 64,800 common target coordinates"), "altitude_trade")
    save(plot_saved_altitude_summary(common, [cases[i] for i in [1, 3, 4]],
        title="Inclination configurations | common ±49° target domain"), "inclination_configurations")
    e2 = cases[1]
    epoch = datetime.fromisoformat(manifest["config"]["epoch_utc"])
    orbit = Orbit(e2["a_g_km"], np.deg2rad(e2["inclination_deg"]), 0., 0.)
    land = ROOT / "reference/maps/ne_110m_land.geojson"

    def select(case, offset):
        return next(e for e in manifest["experiments"] if e["profile"] == "common" and
            e["case"]["case_id"] == case and np.isclose(e["offset_km"], offset, rtol=0, atol=1e-8))

    grids = []
    for offset in [50., -200.]:
        grid = load_grid(DATA, select("E2", offset))
        grid["label"] = f"Δa={offset:+g} km"
        grids.append(grid)
    save(plot_saved_heatmaps(grids, gom=orbit, epoch=epoch, land_geojson=land,
        title="E2 response geography | 360 × 180 targets | 0.25° band"), "E2_response_geography")
    save(plot_saved_tolerance_summary(common, case_id="E2", offsets_km=[-200., 50., 200.],
        title="E2 tolerance sensitivity | 64,800 targets per curve"), "E2_tolerance")
    narrow = load_grid(DATA, select("E2", 50.), .001)
    wide = load_grid(DATA, select("E2", 50.), 5.)
    save(plot_saved_geographic_gain(narrow, narrow_s=narrow["response_s"], wide_s=wide["response_s"],
        land_geojson=land, title="E2 at +50 km | paired geographic effect of tolerance"), "E2_tolerance_location")
    paired = grids[0]["success"].astype(bool) & grids[1]["success"].astype(bool)
    delta = (grids[0]["response_s"] - grids[1]["response_s"])[paired] / 3600
    (OUTPUT / "location_paired_summary.json").write_text(json.dumps(dict(
        paired_count=int(paired.sum()), minus200_earlier_count=int(np.sum(delta > 1/3600)),
        plus50_earlier_count=int(np.sum(delta < -1/3600)), mean_reduction_h=float(delta.mean()),
        median_reduction_h=float(np.median(delta)), p10_reduction_h=float(np.quantile(delta, .1)),
        p90_reduction_h=float(np.quantile(delta, .9))), indent=2) + "\n")
    convergence = []
    for row in summary[summary.profile.isin(["coarse", "medium", "shifted"])].itertuples():
        dense = common[common.case_id.eq(row.case_id) & np.isclose(common.offset_km, row.offset_km) &
                       np.isclose(common.tolerance_deg, row.tolerance_deg)].iloc[0]
        convergence.append(dict(case_id=row.case_id, offset_km=row.offset_km,
            tolerance_deg=row.tolerance_deg, profile=row.profile, n_lon=row.n_lon, n_lat=row.n_lat,
            mean_response_h=row.reported_mean_response_h, dense_mean_response_h=dense.reported_mean_response_h,
            difference_from_dense_h=row.reported_mean_response_h - dense.reported_mean_response_h,
            accepted_fraction=row.reported_accepted_count / row.requested_count))
    pd.DataFrame(convergence).to_csv(tables / "grid_convergence.csv", index=False)
    pilot = pd.read_csv(ROOT / "results/paper1/grid-10x6-20261014/summary.csv")
    changes = []
    for row in common.itertuples():
        previous = pilot[pilot.case_id.eq(row.case_id) & np.isclose(pilot.offset_km, row.offset_km) &
                         np.isclose(pilot.tolerance_deg, row.tolerance_deg)]
        if previous.empty:
            continue
        old = float(previous.iloc[0].mean_response_s) / 3600
        changes.append(dict(case_id=row.case_id, offset_km=row.offset_km, tolerance_deg=row.tolerance_deg,
            pilot_mean_response_h=old, dense_mean_response_h=row.reported_mean_response_h,
            dense_minus_pilot_mean_h=row.reported_mean_response_h - old,
            scope="Corrected code in both; changed coordinate sampling."))
    pd.DataFrame(changes).to_csv(tables / "pilot_to_dense_sampling.csv", index=False)
    (figures / "figure_manifest.json").write_text(json.dumps(dict(
        data_manifest_sha256=hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest(),
        plotting_source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
            ["reconfig_plot_from_csv.py", "reconfig_plotter_multicore.py", Path(__file__).name]},
        scope="Existing plotting functions, saved corrected grids; no resimulation or smoothing."), indent=2) + "\n")
    print(json.dumps(dict(figures=len(list(figures.glob("*.png"))), output=str(OUTPUT))), flush=True)


if __name__ == "__main__":
    main()
