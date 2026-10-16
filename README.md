# Satellite Constellation Modeling, Analysis & Reconfiguration Framework

High-level design, performance evaluation, cost/science multi-objective optimization, and reconfiguration analytics for constellations (Walker / Flower / variants) with J2 phasing.

Paper 1 dense study: [360×180 protocol, original/corrected audit, data and existing plotting entry points](docs/PAPER1_DENSE360X180.md).

Paper 1 maneuver execution: [connected Cartesian propagation, one worked-example figure, measured access errors, and numerical controls](docs/PAPER1_PROPAGATION_VALIDATION.md).

---

## Table of Contents
1. Purpose & Scope
2. End-to-End Workflow (Recommended Procedure)
3. Script Catalogue
4. Typical Outputs & Directory Structure
5. Installation & Quick Start
6. Minimal Usage Examples
7. Author & License

---

## 1. Purpose & Scope

This repository provides a modular toolkit to:
- Generate constellation layouts (Walker, Flower, custom patterns)
- Quantify coverage, visibility, cost, and science metrics
- Multi-objective optimization (Pareto front) of architectures
- Postprocessing and comparative analysis of design sweeps
- Spacecraft reconfiguration analysis (dependence on Δa, ground target coordinates, coordinate  tolerance, total time, ΔV)
- Visualization of ground tracks, coverage, and maneuver sequences

Supports both analytic and numerical approaches, with tools for batch optimization, postprocessing, deployment planning, and maneuver visualization.

---

## 2. End-to-End Workflow (Recommended Procedure)

**Step 1 – Architecture Optimization & Deployment Analysis:**  
Run `run_moea_sweep.py` to automate a series of multi-objective evolutionary algorithm (MOEA) runs, sweeping key parameters (altitude, inclination, mass, deployment ΔV). This script calls `multi_objective_optimizer.py`, which internally calls `launch_phasing_deployment.py` to analyze deployment phasing, time, and ΔV for multi-satellite launches. Results are saved as CSVs and plots.

**Step 2 – Postprocessing & Comparison:**  
Use `csv_postprocess.py` and `csv_compare.py` to analyze and visualize trade-offs, cost breakdowns, performance trends, and deployment constraints across sweeps and seeds.

**Step 3 – Detailed Coverage & Visibility:**  
Select candidate architectures and run `main.py` for ground tracks, 3D orbit views, coverage maps, visibility time series, and animations.

**Step 4 – Reconfiguration Analysis:**  
Run `reconfig_driftANDwaitTime_multicore.py` (or single-core variant) to perform parametric reconfiguration analysis (Δa/tolerance sweeps, timing, ΔV). This script calls `reconfig_plotter.py` and related plotting utilities.  
During this process, the RGT altitude solver (`reconfig_phasing_RGT.py`) is called internally to identify regional orbit parameters as needed for reconfiguration. Direct use of the RGT solver is only needed if you want to manually explore possible Nd/No combinations for RGT altitude and repeat ratios.  
Results are heavy and saved in CSV for further inspection.  
Optionally, use `reconfig_plot_from_csv.py` to visualize timing and ΔV breakdowns from saved CSVs after running the multicore analysis.

**Step 5 – Maneuver Sequence Visualization:**  
Use `reconfig_maneuver_visualizer.py` to render ground track evolution for specific reconfiguration scenarios.

Iterate as needed, adjusting weights, parameters, or operational strategies.

---

## 3. Script Catalogue

| File | Purpose | Typical Usage |
|------|---------|---------------|
| `multi_objective_optimizer.py` | Multi-objective optimizer (cost, coverage, science); NSGA-II, CSV/plot output; calls launch phasing deployment | Run for Pareto front generation |
| `run_moea_sweep.py` | Batch parameter sweep runner for optimizer; automates multi-run sweeps | Run for full sweep across parameters |
| `csv_postprocess.py` | Postprocessing & visualization of single Pareto CSV | Analyze one optimization result |
| `csv_compare.py` | Multi-CSV comparison, sweep analysis, cost/performance/deployment plots | Compare sweeps, seeds, or configs |
| `main.py` | Detailed analysis & visualization driver (ground tracks, coverage, visibility, animation) | Run for selected architecture |
| `utilities_constellations.py` | Constellation object builder (Walker, Flower, SGP4 propagation) | Imported by optimizer and analysis scripts |
| `utilities_coverage.py` | Coverage and visibility utilities, region/ground station lookup | Imported by optimizer and analysis scripts |
| `utilities_optimization.py` | Foundational optimization utilities (weights, scoring) | Internal support |
| `utilities.py` | Core coordinate/time conversions, orbital transforms, propagators, perturbations | Widely imported |
| `orbital_cost_function.py` | Launch cost scaling and OCF interpolation for Neutron DRL | Used in cost models |
| `reconfig_plotter.py` | Parametric reconfiguration timing, Δa/tolerance sweeps, heatmaps, iso-time plots | Called by multicore reconfig analysis |
| `reconfig_plotter_multicore.py` | Multicore variant of reconfiguration plotter | Used for large sweeps |
| `reconfig_plot_from_csv.py` | Plotting of reconfiguration timing from CSV (Δa, ΔV, time breakdown) | Visualize parametric results after heavy analysis |
| `reconfig_driftANDwaitTime.py` | Analytic J2 reconfiguration planner (single-core) | Imported by plotter/visualizer |
| `reconfig_driftANDwaitTime_multicore.py` | Multicore variant of analytic planner; main entry for reconfig analysis | Used for batch sweeps and main reconfig analysis |
| `reconfig_maneuver_visualizer.py` | Ground track visualizer for maneuver phases | Visualize reconfiguration sequences |
| `reconfig_phasing_RGT.py` | Repeat Ground Track (RGT) altitude solver under J2 | Called internally by reconfig analysis; optionally check Nd/No combinations manually |
| `launch_phasing_deployment.py` | RAAN phasing deployment planner (Δa, time, ΔV); called by optimizer | Optionally visualize deployment cost/time |
| `revisit.py` | Analytic revisit-time estimation | Optional, not precise or used in main workflow |
| `constants.py` | Physical/economic constants (μ, R⊕, J2, inflation, etc.) | Internal |

---

### Cross-Module Data Flow (Conceptual)

run_moea_sweep.py → multi_objective_optimizer.py (calls launch_phasing_deployment.py and other internal scripts) → (select candidate) → main.py (for detailed coverage or animations) → reconfig_driftANDwaitTime_multicore.py (calls reconfig_plotter.py, saves CSV; calls reconfig_phasing_RGT.py internally) → reconfig_maneuver_visualizer.py.

---

## 4. Typical Outputs & Directory Structure

```
results/                  # Aggregated analysis outputs
  MOEA/                   # Optimization results (Pareto CSVs, plots)
  walker-delta/           # Example constellation comparison outputs
  reconfig/               # Reconfiguration timing, heatmaps, parametric sweeps (CSV + plots)
  launch/                 # Launch phasing deployment analysis
plots/                    # General plotting products
  ground_tracks/          # Ground track figures
  coverage/               # Coverage maps
  animations/             # MP4 animations
postprocessing/           # CSV post-processing figures (cost/performance trade-offs)
hpc_downloads/            # (Optional) HPC-produced optimization results or plots
```

---

## 5. Installation & Quick Start

1. Clone repository.
2. Create Python >=3.10 environment (recommended).
3. Install core dependencies:
    ```bash
    pip install numpy scipy matplotlib cartopy sgp4 pandas seaborn tqdm
    ```
4. (Optional) For optimization extras:
    ```bash
    pip install pymoo
    ```
5. Run optimizer sweep:
    ```bash
    python run_moea_sweep.py
    ```
6. Post-process a generated CSV:
    ```bash
    python csv_postprocess.py path/to/pareto_solutions.csv
    ```
7. Compare multiple sweeps:
    ```bash
    python csv_compare.py "results/MOEA/walker-delta/p320_g120/sweep_alt"
    ```

Cartopy may require PROJ / GEOS system libraries (on Windows, wheels usually bundle them; on Linux use system package manager if needed).

---

## 6. Minimal Usage Examples

Refer to examples in each script. Most should be ready to run without changes. Some have flags to activate functionalities, such as plots.

For the corrected Paper 1 wait-and-drift workflow, see [corrections, verification, and remaining analysis](docs/PAPER1_CORRECTIONS.md). The new `paper1_analysis.py` runner saves every target's status and geographic errors, feasibility counts, paired departure-policy comparisons, and a reproducibility manifest. Start with `configs/paper1_smoke.json`; committed small-run results are in `results/paper1/verification-20261014/`.

---

## 7. Author & License

Author: Fabio Meloni – TU Delft MSc Student – NASA Jet Propulsion Laboratory Visiting Researcher

License: Not for public distribution.

