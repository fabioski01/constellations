import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
import math
import glob

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

    # Build "nice" Δa major ticks (avoid -217, 117, etc.)
    dmin = float(delta_a_km.min())
    dmax = float(delta_a_km.max())
    span = dmax - dmin

    def nice_step(span, target=6):
        if span <= 0:
            return 10.0
        raw = span / target
        exp = math.floor(math.log10(raw))
        frac = raw / (10 ** exp)
        if frac <= 1.5:
            base = 1
        elif frac <= 3:
            base = 2
        elif frac <= 7:
            base = 5
        else:
            base = 10
        return base * (10 ** exp)

    base = nice_step(span, target=6)   # e.g., for 500 km span → 100 km
    start = math.ceil(dmin / base) * base
    end   = math.floor(dmax / base) * base
    delta_a_ticks = np.arange(start, end + 0.5 * base, base)

    # Ensure 0 and ROM Δa are included if within range
    delta_a_rom = rom_a_km - gom_a_km  # e.g., 6874 - 7371 = -497 km
    rom_tick_in_range = (dmin <= delta_a_rom <= dmax)
    if dmin <= 0.0 <= dmax and not np.any(np.isclose(delta_a_ticks, 0.0, atol=1e-9)):
        delta_a_ticks = np.sort(np.append(delta_a_ticks, 0.0))
    if rom_tick_in_range and not np.any(np.isclose(delta_a_ticks, delta_a_rom, atol=1e-6)):
        delta_a_ticks = np.sort(np.append(delta_a_ticks, delta_a_rom))

    rng = float(dmax - dmin)
    min_sep = 0.06 * rng
    labels = [""] * len(delta_a_ticks)
    labeled_positions = []

    zero_idx = np.where(np.isclose(delta_a_ticks, 0.0, atol=1e-9))[0]
    rom_idx = np.where(np.isclose(delta_a_ticks, delta_a_rom, atol=1e-6))[0] if rom_tick_in_range else []

    if rom_tick_in_range and zero_idx.size > 0 and abs(delta_a_rom - 0.0) < 0.8 * min_sep:
        labels[int(rom_idx[0])] = "ROM"
        labeled_positions.append(delta_a_ticks[int(rom_idx[0])])
    else:
        if zero_idx.size > 0:
            labels[int(zero_idx[0])] = "0"
            labeled_positions.append(delta_a_ticks[int(zero_idx[0])])
        if rom_tick_in_range and len(rom_idx) > 0:
            labels[int(rom_idx[0])] = "ROM"
            labeled_positions.append(delta_a_ticks[int(rom_idx[0])])

    for i, t in enumerate(delta_a_ticks):
        if labels[i]:
            continue
        if all(abs(t - lp) >= min_sep for lp in labeled_positions):
            labels[i] = f"{t:.0f}"
            labeled_positions.append(t)

    axs[2].set_xticks(delta_a_ticks)
    axs[2].set_xticklabels(labels, fontsize=11, rotation=25, ha='right')
    axs[2].set_xlabel("Δa [km]", fontsize=12, labelpad=3)

    # Mark Δa at ROM with a vertical line (if in range) on all subplots
    if rom_tick_in_range:
        for ax in axs:
            ax.axvline(delta_a_rom, color='tab:red', ls='--', lw=1.2)
        # Color the ROM tick label in red
        for tick, pos in zip(axs[2].get_xticklabels(), delta_a_ticks):
            if np.isclose(pos, delta_a_rom, atol=1e-6):
                tick.set_color('tab:red')
                break


    for ax in axs:
        ax.grid(True, which='both', alpha=0.3)
        ax.grid(True, which='minor', color='0.8', linestyle=':', alpha=0.7)
        ax.axvline(0.0, color='0.7', ls='--', lw=1)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
        ax.tick_params(axis='both', which='minor', length=4, color='0.5')
        # Ensure zero is a y-tick and visible
        ymin, ymax = ax.get_ylim()
        if ymin > 0:
            ax.set_ylim(bottom=0)
        if ymax < 0:
            ax.set_ylim(top=0)
        yticks = ax.get_yticks()
        if 0.0 not in yticks:
            ax.set_yticks(np.sort(np.append(yticks, 0.0)))

    # Add ΔV axis aligned with Δa (mirror labeling of Δa)
    pos = axs[2].get_position()
    ax_dv = fig.add_axes([pos.x0, pos.y0 - 0.08, pos.width, 0.06])

    ax_dv.patch.set_visible(False)
    ax_dv.yaxis.set_visible(False)
    for spine in ("top", "right", "left"):
        ax_dv.spines[spine].set_visible(False)
    ax_dv.spines["bottom"].set_visible(True)
    ax_dv.set_xlim(axs[2].get_xlim())

    def delta_a_to_dv(delta_a):
        r_mid = gom_a_km + delta_a
        _, _, dv1 = hohmann_delta_v(gom_a_km, r_mid)
        _, _, dv2 = hohmann_delta_v(r_mid, rom_a_km)
        return dv1 + dv2  # km/s

    dv_vals = [delta_a_to_dv(da) * 1000.0 for da in delta_a_ticks]  # convert to m/s

    # Mirror Δa labels: show DV only where Δa label is non-empty
    dv_labels = []
    for lbl, dv in zip(labels, dv_vals):
        if lbl:           # Δa tick has a visible label
            dv_labels.append(f"{dv:.0f}")
        else:             # keep blank to avoid misalignment clutter
            dv_labels.append("")

    ax_dv.set_xticks(delta_a_ticks)
    ax_dv.set_xticklabels(dv_labels, fontsize=9, rotation=25, ha='right')
    ax_dv.set_xlabel("ΔV [m/s]", fontsize=12, labelpad=3)

    # Color ROM ΔV label red to match Δa ROM label
    if rom_tick_in_range:
        for tick, da, dv_lbl in zip(ax_dv.get_xticklabels(), delta_a_ticks, dv_labels):
            if np.isclose(da, delta_a_rom, atol=1e-6) and dv_lbl:
                tick.set_color('tab:red')
                break

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

    drift = df["avg_drift_s"] / 3600.0
    wait = df["avg_wait_s"] / 3600.0
    ratio = drift / wait

    # Quick plot of drift/wait ratio vs Δa and save
    fig_ratio, ax_ratio = plt.subplots(figsize=(8, 4))
    ax_ratio.plot(delta_a_km, ratio, color='purple', lw=1.8)
    ax_ratio.set_xlabel("Δa [km]", fontsize=12)
    ax_ratio.set_ylabel("Drift/Wait Ratio", fontsize=12)
    ax_ratio.grid(True, which='both', alpha=0.3)
    ax_ratio.axvline(0.0, color='0.7', ls='--', lw=1)
    ax_ratio.set_xticks(delta_a_ticks)
    ax_ratio.set_xticklabels([f"{da:.0f}" for da in delta_a_ticks], fontsize=10)
    ax_ratio.set_title("Drift/Wait Time Ratio vs Δa", fontsize=12)
    # Add minor ticks to ratio plot
    ax_ratio.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax_ratio.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax_ratio.tick_params(axis='both', which='minor', length=4, color='0.5')

    # Save the ratio plot
    ratio_outpath = os.path.join(outdir, os.path.splitext(fname)[0] + "_drift_wait_ratio.png")
    fig_ratio.savefig(ratio_outpath, dpi=200)
    print(f"Saved drift/wait ratio plot to {ratio_outpath}")

    plt.close(fig_ratio)  # Close to avoid display if running many files
    if show:
        plt.show()

    return fig

_SCENARIO_RE = re.compile(
    r"^TimesAvgDeltaA_(?P<grid>[^_]+)_GOM(?P<gom>\d+)_ROM(?P<rom>\d+)_inc(?P<inc>[0-9.]+)_band(?P<band>[0-9.]+)"
    r"_dA-(?P<min>\d+(?:\.\d+)?)to(?P<max>-?\d+(?:\.\d+)?)_s(?P<step>\d+)\.csv$"
)

def _parse_scenario(path: str):
    fname = os.path.basename(path)
    m = _SCENARIO_RE.match(fname)
    if not m:
        return None
    # Ignore grid in the grouping key
    gom = m.group("gom")
    rom = m.group("rom")
    inc = m.group("inc")
    band = m.group("band")
    base = f"TimesAvgDeltaA_GOM{gom}_ROM{rom}_inc{inc}_band{band}"
    da_min = -float(m.group("min"))           # dA-200to200 -> min = -200
    da_max = float(m.group("max"))            # supports negative max too
    step = float(m.group("step"))
    return dict(path=path, base=base, da_min=da_min, da_max=da_max, step=step)

def merge_scenario_files(csv_paths: list[str]) -> list[str]:
    """
    Group CSVs by (GOM, ROM, inc, band), ignoring grid size:
      - Prefer smaller step inside overlaps (smaller step wins)
      - Keep union of Δa ranges
      - Only create a merged file if there are >= 2 files in the group
    Returns list of merged CSV paths (only actual merges).
    """
    infos = [p for p in (_parse_scenario(p) for p in csv_paths) if p]
    if not infos:
        return []

    groups: dict[str, list[dict]] = {}
    for inf in infos:
        groups.setdefault(inf["base"], []).append(inf)

    out_paths = []
    for base, files in groups.items():
        # Skip groups with only one CSV (no actual merge)
        if len(files) < 2:
            print(f"Skip merge for {base}: only one CSV.")
            continue

        files_sorted = sorted(files, key=lambda x: x["step"])  # smaller step first

        frames = []
        for f in files_sorted:
            df = pd.read_csv(f["path"]).sort_values("delta_a_km")
            for c in ("delta_a_km", "avg_total_s", "avg_wait_s", "avg_drift_s"):
                if c not in df.columns:
                    raise ValueError(f"{f['path']} missing column {c}")
            frames.append(df)

        df_all = pd.concat(frames, ignore_index=True)
        # keep='first' → smaller step wins due to sorting
        df_all = df_all.drop_duplicates(subset=["delta_a_km"], keep="first").sort_values("delta_a_km")

        da_min = float(df_all["delta_a_km"].min())
        da_max = float(df_all["delta_a_km"].max())
        step_list = "+".join(str(int(s)) for s in sorted({int(x["step"]) for x in files}))

        # Save merged CSV under a 'merged' subfolder next to the inputs
        base_dir = os.path.dirname(files_sorted[0]["path"])
        merged_dir = os.path.join(base_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)

        out_name = f"{base}_dA{da_min:.0f}to{da_max:.0f}_s{step_list}_MERGED.csv"
        out_path = os.path.join(merged_dir, out_name)

        df_all.to_csv(out_path, index=False)
        print(f"Merged {len(files)} files → {out_path}")
        out_paths.append(out_path)

    return out_paths


if __name__ == "__main__":
    # Manually specify your list of CSV files here:
    csv_files = [
        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_72x240_GOM6871_ROM6874_inc50.0_band49.0_dA-200to200_s5.csv", # 500  50 high res, good step, low Da range
        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_72x240_GOM7071_ROM6874_inc50.0_band49.0_dA-200to200_s5.csv", # 700  50 high res, good step, low Da range
        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_72x240_GOM7371_ROM6874_inc50.0_band49.0_dA-200to200_s5.csv", # 1000 50 high res, good step, low Da range
        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_120x240_GOM6871_ROM6884_inc60.0_band59.0_dA-200to200_s5.csv", # 500  60 high res, good step, low Da range
        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_120x240_GOM7071_ROM6884_inc60.0_band59.0_dA-200to200_s5.csv", # 700  60 high res, good step, low Da range
        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_120x240_GOM7371_ROM6884_inc60.0_band59.0_dA-200to200_s5.csv", # 1000 60 high res, good step, low Da range
        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_80x240_GOM6871_ROM6940_inc98.0_band97.0_dA-200to200_s5.csv", # 500  98 deg cases still running (3 runs, one for each altitude, 6057557, 6057558, 6057559), still high res, good step, low Da range
        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_80x240_GOM7071_ROM6940_inc98.0_band97.0_dA-200to200_s5.csv", # 700  98 deg cases still running (3 runs, one for each altitude, 6057557, 6057558, 6057559), still high res, good step, low Da range
    
        # 98 deg cases still running (3 runs, one for each altitude, 6057557, 6057558, 6057559), still high res, good step, low Da range

        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_72x240_GOM6871_ROM6874_inc50.0_band49.0_dA-200to200_s20.csv", # 500  50 high res, bad step, wide Da range
        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_72x240_GOM7071_ROM6874_inc50.0_band49.0_dA-300to200_s20.csv", # 700  50 high res, bad step, wide Da range
        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_72x240_GOM7371_ROM6874_inc50.0_band49.0_dA-600to200_s20.csv", # 1000 50 high res, bad step, wide Da range
        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_80x240_GOM6871_ROM6884_inc60.0_band59.0_dA-200to200_s20.csv", # 500  60 high res, bad step, wide Da range
        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_80x240_GOM7071_ROM6884_inc60.0_band59.0_dA-300to200_s20.csv", # 700  60 high res, bad step, wide Da range
        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_80x240_GOM7371_ROM6884_inc60.0_band59.0_dA-600to200_s20.csv", # 1000 60 high res, bad step, wide Da range
        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_120x240_GOM6871_ROM6940_inc98.0_band97.0_dA-200to200_s20.csv", # 500  98 high res, bad step, wide Da range
        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_120x240_GOM7071_ROM6940_inc98.0_band97.0_dA-300to200_s20.csv", # 700  98 high res, bad step, wide Da range
        "hpc-downloads/results/reconfig/times_deltaA/TimesAvgDeltaA_120x240_GOM7371_ROM6940_inc98.0_band97.0_dA-600to200_s20.csv"  # 1000 98 high res, bad step, wide Da range

        # Add more CSV paths as needed
    ]
    for csv_path in csv_files:
        plot_timesAvgDeltaA_from_csv(csv_path)

    # Merge per scenario, then plot merged only (to a 'merged' plots folder)
    merged_paths = merge_scenario_files(csv_files)
    merged_outdir = "results/reconfig/times_deltaA/merged"
    os.makedirs(merged_outdir, exist_ok=True)
    for p in merged_paths:
        plot_timesAvgDeltaA_from_csv(p, outdir=merged_outdir)