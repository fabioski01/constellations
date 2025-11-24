import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable

def hohmann_delta_v(r1_km, r2_km, mu=398600.4418):
    r1 = r1_km
    r2 = r2_km
    v1 = np.sqrt(mu / r1)
    v2 = np.sqrt(mu / r2)
    a_trans = 0.5 * (r1 + r2)
    v_trans1 = np.sqrt(mu * (2/r1 - 1/a_trans))
    v_trans2 = np.sqrt(mu * (2/r2 - 1/a_trans))
    delta_v1 = v_trans1 - v1
    delta_v2 = v2 - v_trans2
    return delta_v1, delta_v2, abs(delta_v1) + abs(delta_v2)

def plot_timesAvgDeltaA_from_csv(csv_path, outdir="results/reconfig/times_deltaA", show=False):
    df = pd.read_csv(csv_path)
    fname = os.path.basename(csv_path)
    m = re.search(r"GOM(\d+)_ROM(\d+)", fname)
    if m:
        gom_a_km = float(m.group(1))
        rom_a_km = float(m.group(2))
    else:
        raise ValueError("Could not parse GOM/ROM altitudes from filename.")

    delta_a_km = df["delta_a_km"].values

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.subplots_adjust(bottom=0.15)

    axs[0].plot(delta_a_km, df["avg_total_s"]/3600.0, lw=1.8)
    axs[0].set_ylabel("Total [h]")
    axs[1].plot(delta_a_km, df["avg_wait_s"]/3600.0, lw=1.8)
    axs[1].set_ylabel("Wait [h]")
    axs[2].plot(delta_a_km, df["avg_drift_s"]/3600.0, lw=1.8)
    axs[2].set_ylabel("Drift [h]")

    delta_a_ticks = np.linspace(delta_a_km.min(), delta_a_km.max(), 7)

    # Hide x tick labels for upper subplots
    axs[0].tick_params(labelbottom=False)
    axs[1].tick_params(labelbottom=False)

    # Δa axis: only on bottom subplot
    axs[2].set_xticks(delta_a_ticks)
    axs[2].set_xticklabels([f"{da:.0f}" for da in delta_a_ticks], fontsize=12)
    axs[2].set_xlabel("Δa [km]", fontsize=12, labelpad=3)

    for ax in axs:
        ax.grid(True, which='both', alpha=0.3)
        ax.axvline(0.0, color='0.7', ls='--', lw=1)

    # Add a completely independent axis for ΔV below, aligned with Δa
    pos = axs[2].get_position()
    ax_dv = fig.add_axes([pos.x0, pos.y0 - 0.08, pos.width, 0.06])

    ax_dv.patch.set_visible(False)
    ax_dv.yaxis.set_visible(False)
    ax_dv.spines['top'].set_visible(False)
    ax_dv.spines['right'].set_visible(False)
    ax_dv.spines['left'].set_visible(False)
    ax_dv.spines['bottom'].set_visible(True)
    ax_dv.set_xlim(axs[2].get_xlim())

    def delta_a_to_dv(delta_a):
        r_mid = gom_a_km + delta_a
        _, _, dv1 = hohmann_delta_v(gom_a_km, r_mid)
        _, _, dv2 = hohmann_delta_v(r_mid, rom_a_km)
        return dv1 + dv2

    dv_tick_labels = [delta_a_to_dv(da) * 1000 for da in delta_a_ticks]  # Convert to m/s
    ax_dv.set_xticks(delta_a_ticks)
    ax_dv.set_xticklabels([f"{dv:.0f}" for dv in dv_tick_labels], fontsize=10)
    ax_dv.set_xlabel("ΔV [m/s]", fontsize=12, labelpad=3)

    fig.align_ylabels(axs)

    fig.suptitle(
        f"Average Reconfiguration Times vs Δa with GOM={gom_a_km:.0f} km, ROM={rom_a_km:.0f} km",
        fontsize=10, y=0.96
    )
    axs[0].set_title("Total, Wait, and Drift Times for Orbit Reconfiguration\n", fontsize=10)

    os.makedirs(outdir, exist_ok=True)
    fname = os.path.splitext(fname)[0] + "_fromcsv.png"
    outpath = os.path.join(outdir, fname)
    fig.savefig(outpath, dpi=200)
    print(f"Saved figure to {outpath}")
    if show:
        plt.show()

    return fig

if __name__ == "__main__":
    # Manually specify your list of CSV files here:
    csv_files = [
        "/scratch/nbruno/constellations/results/reconfig/times_deltaA/TimesAvgDeltaA_36x72_GOM3790_ROM3646_inc50.0_band49.0_dA-200to200_s10.csv",
        # "results/reconfig/times_deltaA/TimesAvgDeltaA_11x36_GOM700_ROM705_inc60.0_band59.0_dA-200to200_s10.csv",
        # Add more CSV paths as needed
    ]
    for csv_path in csv_files:
        plot_timesAvgDeltaA_from_csv(csv_path)