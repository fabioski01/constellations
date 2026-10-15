"""Finish the matched comparison from complete independently audited stages."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper1_dense_analysis import FIELDS
from reconfig_plotter_multicore import plot_saved_tolerance_summary

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "dense-results"
OUTPUT = DATA / "paper-results"


def main():
    primary = DATA / "corrected"
    paired = DATA / "paired"
    completion = json.loads((paired / "completion.json").read_text())
    audit = json.loads((paired / "audit/independent_audit.json").read_text())
    if not completion["complete"] or not audit["full_study_complete"] or audit["failures"]:
        raise ValueError("The complete audited matched stage is required.")
    manifests = [json.loads((p / "manifest.json").read_text()) for p in [primary, paired]]
    if manifests[0]["source_sha256"] != manifests[1]["source_sha256"]:
        raise ValueError("Numerical sources differ between stages.")
    configs = [{k:v for k,v in m["config"].items() if k not in ["scope", "recovery_stage"]} for m in manifests]
    if configs[0] != configs[1]:
        raise ValueError("Numerical configurations differ between stages.")
    compared = 0
    requests = 0
    for chunk in sorted((paired / "per_target").glob("*.npz")):
        with np.load(chunk) as second, np.load(primary / "per_target" / chunk.name) as first:
            for key in ["latitude_deg", "longitude_deg", "tolerances_deg", "latitude_status"] + ["corrected_" + k for k in FIELDS]:
                if not np.array_equal(first[key], second[key], equal_nan=True):
                    raise AssertionError("Repeated corrected arrays differ: " + chunk.name + " " + key)
            requests += int(first["corrected_success"].size)
            compared += 1
    (OUTPUT / "stage_repeatability.json").write_text(json.dumps(dict(
        compared_chunks=compared, compared_corrected_target_requests=requests,
        bitwise_equal=True, numerical_sources_and_configuration_equal=True,
        scope="Repeated corrected arrays in independently scheduled corrected and matched stages."), indent=2) + "\n")
    summary = pd.read_csv(paired / "summary.csv")
    physical = pd.read_csv(paired / "audit/continuous_clock_audit.csv")
    old = physical[physical.version.eq("original")].copy()
    old["continuous_rejected_fraction_of_reports"] = old.continuous_rejected_reported_count / old.reported_count
    old.to_csv(OUTPUT / "tables/original_code_comparison.csv", index=False)
    figures = OUTPUT / "figures"
    plt.rcParams.update({"font.family":"DejaVu Serif", "font.size":10,
        "axes.spines.top":False, "axes.spines.right":False})

    def save(fig, name):
        for extension in ["png", "pdf"]:
            target = figures / (name + "." + extension)
            temporary = target.with_name(target.name + ".tmp")
            fig.savefig(temporary, format=extension, dpi=230)
            temporary.replace(target)
        plt.close(fig)

    fig, axes = plt.subplots(2,1,figsize=(7.5,5.5),sharex=True)
    for case in sorted(old.case_id.unique()):
        r = old[old.case_id.eq(case) & np.isclose(old.offset_km,50.)].sort_values("tolerance_deg")
        axes[0].semilogx(r.tolerance_deg,r.continuous_rejected_fraction_of_reports,"o-",ms=3,label=case)
        r = summary[summary.version.eq("corrected") & summary.case_id.eq(case) & np.isclose(summary.offset_km,50.)].sort_values("tolerance_deg")
        axes[1].semilogx(r.tolerance_deg,r.reported_mean_response_h,"o-",ms=3,label=case)
    axes[0].set_ylim(-.02,1.02)
    axes[0].set_ylabel("Original reports failing\ncommon-clock acceptance")
    axes[1].set_ylabel("Corrected mean [h]")
    axes[1].set_xlabel("Longitude tolerance [deg]")
    axes[0].legend(ncol=5,fontsize=9)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle("Matched original-code audit | +50 km | 64,800 targets")
    fig.tight_layout(rect=(0,0,1,.95))
    save(fig,"original_common_clock_audit")
    save(plot_saved_tolerance_summary(summary,case_id="E2",offsets_km=[-200.,50.,200.],
        title="E2 | original reported and corrected component means"),"E2_original_corrected_tolerance")
    (figures / "comparison_figure_manifest.json").write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        paired_manifest_sha256=hashlib.sha256((paired / "manifest.json").read_bytes()).hexdigest(),
        scope="Complete saved matched data; original reported means and common-clock acceptance are separate metrics."),indent=2)+"\n")
    print(json.dumps(dict(compared_chunks=compared,bitwise_repeatable_requests=requests,
                         original_comparison_rows=len(old))),flush=True)


if __name__ == "__main__":
    main()
