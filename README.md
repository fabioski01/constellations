# Satellite Constellation Modeling, Analysis & Reconfiguration Framework

High-level design, performance evaluation, cost / science multi‑objective optimization, and reconfiguration analytics for constellations (Walker / Flower / variants) with J2 phasing.

---
## Table of Contents
1. Purpose & Scope
2. End‑to‑End Workflow (Recommended Procedure)
3. Script Catalogue (14 Python files)
4. Typical Outputs & Directory Structure
5. Installation & Quick Start
6. Minimal Usage Examples
7. Author & License

---
## 1. Purpose & Scope
This repository (~13.5k lines, 14 Python modules) provides a modular toolkit to: (a) generate constellation layouts, (b) quantify coverage / visibility / cost / science metrics, (c) optimize architectures (Pareto front), and (d) analyze & visualize time, ΔV, and geometry for single-satellite reconfiguration to regional (RGT) or global orbits.

---
## 2. End‑to‑End Workflow (Recommended Procedure)
Step 1 – Architecture Exploration: Run `multi_objective_optimizer.py` to obtain a Pareto front (cost vs performance weights). Exported CSV contains orbital & pattern parameters of optimal candidates.

Step 2 – Post Processing: Use `csv_postprocess.py` on the produced CSV(s) to generate comparative plots (cost vs altitude, performance vs inclination, trade studies) and summary statistics.

Step 3 – Detailed Coverage & Visibility: Select one (or a few) candidate architectures from the Pareto front and run `main.py` to produce ground tracks, 3D orbit views, coverage maps (global/regional), visibility time series and animations.

Step 4 – RGT / Regional Orbit Tuning: If a repeat ground track (RGT) regional mode is desired, use `reconfig_phasing_RGT.py` to identify altitude / semi-major axis satisfying a specified repeat ratio for the given inclination.

Step 5 – Reconfiguration Analysis (Parametric): With global (GOM) and selected regional (ROM / RGT) orbits defined, use `reconfig_plotter.py` to sweep Δa and tolerance parameters and produce timing & minimum ΔV style figures (wait, drift, total times; averaged heatmaps; iso-time contours) to choose an operational Δa (e.g., 50 km) and tolerance strategy.

Step 6 – Maneuver Sequence Visualization: Use `reconfig_maneuver_visualizer.py` to render ground track evolution for a specific reconfiguration (GOM wait → Hohmann 1 → drift → Hohmann 2 → target pass) across multiple ground targets.

Iterate: Adjust weights, architectural parameters, or Δa / tolerance and repeat steps as needed.

---
## 3. Script Catalogue
Below each module is listed with a concise role and typical invocation context.

| File | Purpose (Short) | Typical User Action |
|------|-----------------|---------------------|
| `multi_objective_optimizer.py` | High-level multi-objective (cost / coverage / science) optimizer. Produces Pareto CSV + plots. | Edit weights & limits, run to generate new Pareto set. |
| `csv_postprocess.py` | Post-processing & visualization of Pareto CSV (cost vs performance, altitude/inclination trade-offs, correlation). | Provide CSV path, run to produce `postprocessing/` figures. |
| `main.py` | Comprehensive analysis & visualization driver (ground tracks, coverage, visibility, animations). | Set flags & constellation parameters; run for diagnostics & figures. |
| `utilities_constellations.py` | Constellation object: builds Walker / Flower patterns (SGP4-based propagation). | Imported by higher-level scripts. |
| `utilities_coverage.py` | Grid-based global/regional coverage & visibility utilities + region & ground station lookups. | Imported; rarely run standalone. |
| `utilities_optimization.py` | Foundational (earlier) optimization utilities (weights & scoring components). | Internal support / legacy reference. |
| `utilities.py` | Core coordinate & time conversions; orbital element transforms; basic propagators; perturbations. | Imported widely. |
| `reconfig_plotter.py` | Parametric reconfiguration timing & Δa/tolerance sweeps; heatmaps / iso-time plots; auto-selects multicore planner. | Provide GOM & ROM parameters; choose sweeps; run for `results/reconfig/` figures. |
| `reconfig_driftANDwaitTime_multicore.py` | J2 analytic reconfiguration planner (multicore); solves for wait & drift times across wrap branches. | Imported by plotter / visualizer; not typically executed directly. |
| `reconfig_driftANDwaitTime.py` | Single-core variant (fallback) of the analytic planner. | Automatic fallback if multicore import fails. |
| `reconfig_maneuver_visualizer.py` | Builds maneuver phase orbital segments & ground track PNGs for a specific reconfiguration scenario. | Edit scenario block or call programmatically; produces `plots/reconfig_groundtracks/`. |
| `reconfig_phasing_RGT.py` | Repeat Ground Track (RGT) altitude solver under J2 (semi-major axis for No/Nd ratio). | Use to determine ROM altitude satisfying repeat ratio. |
| `revisit.py` | Simplified analytic revisit-time estimation (coverage geometry approximations). | Run interactively for quick early analysis. |
| `constants.py` | Physical / economic constants (μ, R⊕, J2, inflation factors, etc.). | Internal. |

---
### Cross-Module Data Flow (Conceptual)
Optimizer → (select candidate) → Main (detailed coverage) → RGT (derive regional orbit) → Reconfig Plotter (Δa/tolerance trade) → Maneuver Visualizer (sequence depiction).

---
## 4. Typical Outputs & Directory Structure
```
results/                  # Aggregated analysis outputs
  walker-delta/           # Example constellation comparison outputs
  reconfig/               # Reconfiguration timing / heatmaps
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
4. (Optional) For optimization extras: `pip install pymoo` (if used in `multi_objective_optimizer.py`).
5. Run optimizer example:
    ```bash
    python multi_objective_optimizer.py
    ```
6. Post-process a generated CSV:
    ```bash
    python csv_postprocess.py path/to/pareto_solutions.csv
    ```

Cartopy may require PROJ / GEOS system libraries (on Windows, wheels usually bundle them; on Linux use system package manager if needed).

## 6. Minimal Usage Examples
Refer to examples in each script. Most should be ready to run without changes. Some have flags
to active functionalities, such as plots.

---

## 7. Author & License
Author: Fabio Meloni – TU Delft MSc Student – NASA Jet Propulsion Laboratory Visiting Researcher

License: Not for public distribution.