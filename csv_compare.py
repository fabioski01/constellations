"""
Multi-CSV Comparison Script for Satellite Constellation Optimization Results

This script is designed to load and compare multiple optimization result CSV files generated from parameter sweeps of satellite constellation designs. It provides a suite of comparative plots to analyze trade-offs between cost, performance, deployment constraints, and other mission metrics across different design choices (such as altitude, inclination, mass, and deployment delta-V).

Main Features:
 - Loads multiple CSVs from a directory or a manual list, extracting metadata and results for each sweep configuration.
 - Generates comparative plots including:
     * 3D scatter plots for visualizing relationships between cost, performance, and design parameters.
     * Pareto front comparisons (cost vs performance) with trend lines and multi-seed merging.
     * Cost component breakdowns and trends as a function of satellite number and sweep parameter.
     * Deployment constraint analysis (RAAN spacing, launch groups, phasing factors, launch cost).
     * Launch cost analysis and trade-off visualization.
     * Performance plots (coverage/science) vs satellite number and planes.
 - Supports merging solutions by sweep parameter, filtering non-dominated solutions, and visualizing only the relevant design variations (not per seed).
 - Interacts with other scripts in the workflow by consuming CSV outputs from optimization runs and producing summary plots for downstream analysis and reporting.

Main Functions:
 - MultiCSVComparator: Core class for loading, processing, and plotting results from multiple CSVs.
 - generate_all_plots: Orchestrates the creation of all comparative plots for a given sweep.
 - plot_3d_altitude_comparison, plot_3d_inclination_comparison: 3D visualizations for sweep analysis.
 - plot_pareto_comparison, plot_merged_pareto_multiseed: Pareto front and trade-off analysis.
 - plot_cost_components_comparison, plot_cost_components_per_satellite: Cost breakdowns and trends.
 - plot_deployment_comparison, plot_launch_cost_comparison: Deployment and launch cost analysis.
 - plot_performance_vs_sat_and_planes: Performance trends vs satellites and planes.

This script is typically used after running parameter sweeps with the optimization engine, and is compatible with outputs from scripts such as run_moea_sweep.py and multi_objective_optimizer.py.

Author: Fabio Meloni - TU Delft MSc Student - NASA JPL Visiting Researcher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
from pathlib import Path
import argparse
from scipy.stats import pearsonr, linregress
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MultiCSVComparator:
    """
    Loads and compares multiple optimization result CSVs.
    """
    
    def __init__(self, csv_dir, output_dir="comparison_plots"):
        """
        Initialize comparator.
        
        Parameters:
        -----------
        csv_dir : str or list
            Directory containing CSV files OR list of CSV file paths
        output_dir : str
            Output directory for comparison plots
        """
        self.csv_dir = csv_dir

        # Derive structured output directory: comparison_plots/<config>/<sweep>
        # Only auto-augment when caller uses the default base folder name.
        derived_output = output_dir
        try:
            import re as _re
            if isinstance(csv_dir, str) and os.path.isdir(csv_dir) and output_dir.strip().lower() == 'comparison_plots':
                p = Path(csv_dir)
                sweep_dir = p.name  # e.g., sweep_mass
                config_dir = p.parent.name  # e.g., p320_g120
                # Prefer a config folder like p###_g###; else still use whatever parent folder is
                if not _re.match(r"^p\d+_g\d+$", config_dir, flags=_re.IGNORECASE):
                    # Try one level higher
                    maybe_config = p.parent.name
                derived_output = os.path.join(output_dir, config_dir, sweep_dir)
            elif isinstance(csv_dir, list) and len(csv_dir) > 0 and output_dir.strip().lower() == 'comparison_plots':
                p = Path(csv_dir[0]).parent  # folder containing the first CSV
                sweep_dir = p.name
                config_dir = p.parent.name
                derived_output = os.path.join(output_dir, config_dir, sweep_dir)
        except Exception:
            # Fallback gracefully to provided output_dir
            derived_output = output_dir

        self.output_dir = derived_output
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.data_dict = {}  # {csv_name: dataframe}
        self.metadata = {}   # {csv_name: {target_alt, target_inc, mass, etc}}
        
        self.load_all_csvs()
        
    def load_all_csvs(self):
        """Load all CSV files from directory or list. Supports recursive search for multi-seed scenarios."""
        if isinstance(self.csv_dir, list):
            csv_files = self.csv_dir
        else:
            # Check if we need recursive search (multi-seed scenario)
            # Pattern: if csv_dir is like .../walker-delta/ (parent of p###_g###_sd### folders)
            # then search recursively for sweep folders
            csv_dir_path = Path(self.csv_dir)
            
            # Try direct search first (single seed folder like .../p320_g120_sd42/sweep_mass/)
            pop_candidates = glob(os.path.join(self.csv_dir, "population_solutions_*.csv"))
            pareto_candidates = glob(os.path.join(self.csv_dir, "pareto_solutions_*.csv"))
            
            if not pop_candidates and not pareto_candidates:
                # No CSVs found directly - try recursive search for multi-seed scenario
                # Search pattern: .../p*_g*_sd*/sweep_*/*.csv
                import re
                
                # Detect if we're at pattern level (e.g., walker-delta) or sweep level
                if re.match(r'^p\d+_g\d+_sd\d+$', csv_dir_path.name):
                    # We're at a specific config folder - don't recurse, user wants this seed only
                    pass
                else:
                    # Try recursive search for multi-seed CSVs
                    pop_candidates = glob(os.path.join(self.csv_dir, "**/population_solutions_*.csv"), recursive=True)
                    pareto_candidates = glob(os.path.join(self.csv_dir, "**/pareto_solutions_*.csv"), recursive=True)
            
            # Prefer population CSVs
            if pop_candidates:
                csv_files = sorted(pop_candidates)
            elif pareto_candidates:
                csv_files = sorted(pareto_candidates)
            else:
                csv_files = []
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.csv_dir}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_path in csv_files:
            csv_name = Path(csv_path).stem
            print(f"Loading: {csv_name}")
            
            df = pd.read_csv(csv_path)
            
            # Extract metadata from columns, filename, and parent folder
            metadata = self._extract_metadata(df, csv_path)
            
            # Debug: Print metadata for files with altitude 700 or inclination 50 or mass 190
            if metadata.get('target_altitude') == 700.0 or metadata.get('target_inclination') == 50.0 or metadata.get('sat_mass') == 190.0:
                print(f"  [DEBUG] {csv_name[:60]}...")
                print(f"    altitude={metadata.get('target_altitude')}, inc={metadata.get('target_inclination')}, mass={metadata.get('sat_mass')}, sweep_type={metadata.get('sweep_type')}")
            
            # Clean data: ensure essential columns exist; drop rows missing core fields only
            critical_cols = ['cost_musd', 'performance', 'num_satellites']
            before = len(df)
            missing = [c for c in critical_cols if c not in df.columns]
            if missing:
                print(f"[WARN] {csv_name} missing columns: {missing} — skipping file")
                continue
            df = df.dropna(subset=critical_cols)
            after = len(df)
            if after < before:
                print(f" - {csv_name}: {before} rows -> {after} after dropping NaNs in {critical_cols}")
            
            self.data_dict[csv_name] = df
            self.metadata[csv_name] = metadata
            
        print(f"Loaded {len(self.data_dict)} datasets")

        # Sort datasets by their sweep parameter for consistent legend ordering
        self._sort_datasets_by_parameter()
        
        # Add seed numbers to labels when duplicates exist
        self._disambiguate_labels()

    def _disambiguate_labels(self):
        """
        Add seed numbers to labels when multiple datasets have the same parameter value.
        For example, if two datasets both have "Alt 700 km", they become:
        "Alt 700 km (sd11)" and "Alt 700 km (sd42)"
        """
        # Group datasets by base_label
        label_groups = {}
        for name, meta in self.metadata.items():
            base_label = meta.get('base_label', meta.get('label', 'unknown'))
            label_groups.setdefault(base_label, []).append(name)
        
        # For groups with multiple datasets, append seed number
        for base_label, names in label_groups.items():
            if len(names) > 1:
                # Multiple datasets with same parameter value - add seed to distinguish
                for name in names:
                    seed = self.metadata[name].get('seed')
                    if seed is not None:
                        self.metadata[name]['label'] = f"{base_label} (sd{seed})"
                    else:
                        # Fallback: use csv name if seed not found
                        self.metadata[name]['label'] = f"{base_label} ({name})"

    def _nondominated_mask(self, df: pd.DataFrame) -> np.ndarray:
        """Return boolean mask for non-dominated points (minimize cost, maximize performance)."""
        costs = df['cost_musd'].to_numpy()
        perf = df['performance'].to_numpy()
        n = len(df)
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            # Eliminate points dominated by point i (strict in at least one objective)
            dominated = (costs[i] <= costs) & (perf[i] >= perf) & ((costs[i] < costs) | (perf[i] > perf))
            dominated[i] = False  # don't mark self as dominated
            keep[dominated] = False
        return keep

    @staticmethod
    def compute_global_pareto_from_arrays(costs: np.ndarray, performances: np.ndarray) -> np.ndarray:
        """
        Compute non-dominated front from cost and performance arrays.
        
        Parameters:
        -----------
        costs : np.ndarray
            Cost values (to minimize)
        performances : np.ndarray
            Performance values (to maximize)
            
        Returns:
        --------
        mask : np.ndarray
            Boolean mask indicating non-dominated solutions
        """
        n = len(costs)
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            # Eliminate points dominated by point i
            dominated = (costs[i] <= costs) & (performances[i] >= performances) & \
                       ((costs[i] < costs) | (performances[i] > performances))
            dominated[i] = False
            keep[dominated] = False
        return keep

    def _sort_datasets_by_parameter(self):
        """Sort datasets by primary parameter (alt/inc/mass/dv) for legends.
        Alt/Inc/Mass sorted descending, DV sorted ascending.
        """
        # Group by sweep type
        sweep_groups = {}
        for name, meta in self.metadata.items():
            st = meta.get('sweep_type', 'unknown')
            sweep_groups.setdefault(st, []).append((name, meta))

        # Sort each group by parameter value (descending)
        for st, items in sweep_groups.items():
            if st == 'altitude':
                items.sort(key=lambda x: x[1].get('target_altitude', -np.inf), reverse=True)
            elif st == 'inclination':
                items.sort(key=lambda x: x[1].get('target_inclination', -np.inf), reverse=True)
            elif st == 'mass':
                items.sort(key=lambda x: x[1].get('sat_mass', -np.inf), reverse=True)
            elif st == 'dv':
                # Smaller deployment DV is generally preferable; sort ascending
                items.sort(key=lambda x: x[1].get('dv_m_s', np.inf), reverse=False)
            else:
                # keep insertion order
                pass

        # Rebuild dicts in sorted order (preserves iteration order later)
        sorted_data = {}
        sorted_meta = {}
        for st, items in sweep_groups.items():
            for name, meta in items:
                sorted_data[name] = self.data_dict[name]
                sorted_meta[name] = meta

        self.data_dict = sorted_data
        self.metadata = sorted_meta
        
    def _extract_metadata(self, df, csv_path):
        """Extract metadata (alt/inc/mass/dv) and the sweep type from df, filename, and parent folder."""
        meta = {}

        # Resolve names and parents
        p = Path(csv_path)
        csv_name = p.stem
        parent_name = p.parent.name.lower()
        
        # Extract seed number from filename (e.g., _sd42, _sd11)
        import re
        seed_match = re.search(r'_sd(\d+)', csv_name)
        meta['seed'] = int(seed_match.group(1)) if seed_match else None

        # Try to get from dataframe columns first
        if 'optimization_params_target_altitude' in df.columns:
            # value may be per-row but constant for file
            try:
                meta['target_altitude'] = float(df['optimization_params_target_altitude'].iloc[0])
            except Exception:
                pass
        if 'optimization_params_sat_mass_kg' in df.columns:
            try:
                meta['sat_mass'] = float(df['optimization_params_sat_mass_kg'].iloc[0])
            except Exception:
                pass
        if 'launch_param_dv_deployment_m_s' in df.columns:
            try:
                meta['dv_m_s'] = float(df['launch_param_dv_deployment_m_s'].iloc[0])
            except Exception:
                pass

        # Parse from filename (e.g., ..._a700_i50_m190_dv200_...)
        # Filename values take precedence (more specific)
        alt_match = re.search(r'[_-]a(\d+)[km]?', csv_name)
        if alt_match:
            meta['target_altitude'] = float(alt_match.group(1))

        inc_match = re.search(r'[_-]i(\d+)[deg]?', csv_name)
        if inc_match:
            meta['target_inclination'] = float(inc_match.group(1))

        mass_match = re.search(r'[_-]m(\d+)[kg]?', csv_name)
        if mass_match:
            meta['sat_mass'] = float(mass_match.group(1))

        dv_match = re.search(r'[_-]dv(\d+)', csv_name)
        if dv_match:
            meta['dv_m_s'] = float(dv_match.group(1))

        # Determine sweep type from parent folder naming convention first
        if 'sweep_inc' in parent_name:
            sweep_type = 'inclination'
        elif 'sweep_mass' in parent_name:
            sweep_type = 'mass'
        elif 'sweep_vdep' in parent_name or 'sweep_dv' in parent_name:
            sweep_type = 'dv'
        elif 'sweep_alt' in parent_name or 'sweep_altitude' in parent_name:
            sweep_type = 'altitude'
        else:
            # Fallback heuristic: prefer a single distinguishing parameter
            sweep_type = 'unknown'

        # Build label based on sweep type to ensure legends reflect the swept parameter
        if sweep_type == 'inclination' and 'target_inclination' in meta:
            meta['label'] = f"Inc {meta['target_inclination']:.0f}°"
        elif sweep_type == 'mass' and 'sat_mass' in meta:
            meta['label'] = f"Mass {meta['sat_mass']:.0f} kg"
        elif sweep_type == 'dv' and 'dv_m_s' in meta:
            meta['label'] = f"ΔV {meta['dv_m_s']:.0f} m/s"
        elif sweep_type == 'altitude' and 'target_altitude' in meta:
            meta['label'] = f"Alt {meta['target_altitude']:.0f} km"
        else:
            # If still unknown, try to pick a sensible label priority
            if 'target_inclination' in meta:
                meta['label'] = f"Inc {meta['target_inclination']:.0f}°"
            elif 'sat_mass' in meta:
                meta['label'] = f"Mass {meta['sat_mass']:.0f} kg"
            elif 'dv_m_s' in meta:
                meta['label'] = f"ΔV {meta['dv_m_s']:.0f} m/s"
            elif 'target_altitude' in meta:
                meta['label'] = f"Alt {meta['target_altitude']:.0f} km"
            else:
                meta['label'] = csv_name

        meta['sweep_type'] = sweep_type
        meta['base_label'] = meta['label']  # Store base label for later modification
        return meta
    
    def generate_all_plots(self):
        """Generate all comparison plots."""
        print("\nGenerating comparison plots...")
        
        # Group datasets by sweep type
        sweep_groups = {}
        for name, meta in self.metadata.items():
            sweep_type = meta.get('sweep_type', 'unknown')
            if sweep_type not in sweep_groups:
                sweep_groups[sweep_type] = []
            sweep_groups[sweep_type].append(name)

        # Marker styles and high-contrast color palette
        self.marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        self.color_palette = [
            '#e41a1c',  # Red
            '#377eb8',  # Blue
            '#4daf4a',  # Green
            '#984ea3',  # Purple
            '#ff7f00',  # Orange
            '#ffff33',  # Yellow
            '#a65628',  # Brown
            '#f781bf',  # Pink
            '#999999',  # Gray
            '#66c2a5'   # Teal
        ]
        
        # 1. 3D scatter comparisons
        if 'altitude' in sweep_groups and len(sweep_groups['altitude']) > 1:
            self.plot_3d_altitude_comparison(sweep_groups['altitude'])
        
        if 'inclination' in sweep_groups and len(sweep_groups['inclination']) > 1:
            self.plot_3d_inclination_comparison(sweep_groups['inclination'])
        
        # 2. Pareto fronts with trend lines
        for sweep_type, names in sweep_groups.items():
            if len(names) > 1:
                # Only keep log-scale Pareto for mass sweep
                if sweep_type == 'mass':
                    self.plot_pareto_comparison_log_cost(names, sweep_type)
                # Keep merged multi-seed Pareto plots
                self.plot_merged_pareto_multiseed(names, sweep_type)
                if sweep_type == 'mass':
                    self.plot_merged_pareto_multiseed_logx(names, sweep_type)
        
        # 3. Cost components vs satellites
        for sweep_type, names in sweep_groups.items():
            if len(names) > 1:
                self.plot_cost_components_comparison(names, sweep_type)
        
        # # 4. Deployment constraints
        # for sweep_type, names in sweep_groups.items():
        #     if len(names) > 1:
        #         self.plot_deployment_comparison(names, sweep_type)
        
        # 5. Launch cost vs deployment DV
        for sweep_type, names in sweep_groups.items():
            if len(names) > 1:
                self.plot_launchdeploy_cost_comparison(names, sweep_type)

        # 6. Cost components per satellite vs satellites
        for sweep_type, names in sweep_groups.items():
            if len(names) > 1:
                self.plot_cost_components_per_satellite(names, sweep_type)

        # 7. Performance vs n of sats and n of planes
        for sweep_type, names in sweep_groups.items():
            if len(names) > 1:
                self.plot_performance_vs_sat_and_planes(names, sweep_type)
        
        # 8. Launch groups vs P and RAAN
        for sweep_type, names in sweep_groups.items():
            if len(names) > 1:
                self.plot_launch_groups_raan_comparison(names, sweep_type)

        # 9. Launch cost total and per sat vs n of planes
        for sweep_type, names in sweep_groups.items():
            if len(names) > 1:
                self.plot_launch_cost_planes_comparison(names, sweep_type)

        # 10. Cost ratios and absolute (NRE, RE, Launch)
        for sweep_type, names in sweep_groups.items():
            if len(names) > 1:
                self.plot_cost_ratios(names, sweep_type)

        print(f"\nAll comparison plots saved to: {self.output_dir}/")
    
    def plot_3d_altitude_comparison(self, csv_names):
        """3D scatter: Cost vs Performance vs Satellites, merged by altitude (not by seed/config)."""
        fig = plt.figure(figsize=(16, 6))
        # Group by altitude
        param_groups = {}
        for name in csv_names:
            meta = self.metadata[name]
            param_val = meta.get('target_altitude')
            if param_val is None:
                continue
            param_groups.setdefault(param_val, []).append(name)

        # Plot 1: Cost vs Performance vs Satellites
        ax1 = fig.add_subplot(131, projection='3d')
        for i, (alt, csv_list) in enumerate(sorted(param_groups.items())):
            color = self.color_palette[i % len(self.color_palette)]
            marker = self.marker_styles[i % len(self.marker_styles)]
            merged_df = pd.concat([self.data_dict[name] for name in csv_list], ignore_index=True)
            nd_mask = self._nondominated_mask(merged_df)
            merged_df = merged_df[nd_mask]
            ax1.scatter(merged_df['cost_musd'], merged_df['performance'], merged_df['num_satellites'],
                        c=color, s=60, alpha=0.7, marker=marker,
                        edgecolors='black', linewidth=0.5,
                        label=f"Alt {alt:.0f} km")
        ax1.set_xlabel('Cost (M$ FY2025)')
        ax1.set_ylabel('Performance')
        ax1.set_zlabel('Number of Satellites')
        ax1.set_title('Altitude Sweep Comparison\nCost vs Performance vs Satellites')
        ax1.legend()

        # Plot 2: Cost vs Satellites vs Altitude (colored by performance)
        ax2 = fig.add_subplot(132, projection='3d')
        for i, (alt, csv_list) in enumerate(sorted(param_groups.items())):
            marker = self.marker_styles[i % len(self.marker_styles)]
            merged_df = pd.concat([self.data_dict[name] for name in csv_list], ignore_index=True)
            nd_mask = self._nondominated_mask(merged_df)
            merged_df = merged_df[nd_mask]
            scatter = ax2.scatter(merged_df['cost_musd'], merged_df['num_satellites'],
                                [alt] * len(merged_df),
                                c=merged_df['performance'], s=60, alpha=0.7, marker=marker,
                                cmap='plasma', vmin=0, vmax=1,
                                edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Cost (M$ FY2025)')
        ax2.set_ylabel('Number of Satellites')
        ax2.set_zlabel('Target Altitude (km)')
        ax2.set_title('Cost vs Satellites vs Target Altitude\n(colored by performance)')
        plt.colorbar(scatter, ax=ax2, label='Performance', shrink=0.5)

        # Plot 3: Performance vs Altitude vs Satellites
        ax3 = fig.add_subplot(133, projection='3d')
        for i, (alt, csv_list) in enumerate(sorted(param_groups.items())):
            color = self.color_palette[i % len(self.color_palette)]
            marker = self.marker_styles[i % len(self.marker_styles)]
            merged_df = pd.concat([self.data_dict[name] for name in csv_list], ignore_index=True)
            nd_mask = self._nondominated_mask(merged_df)
            merged_df = merged_df[nd_mask]
            ax3.scatter([alt] * len(merged_df), merged_df['performance'], merged_df['num_satellites'],
                        c=color, s=60, alpha=0.7, marker=marker,
                        edgecolors='black', linewidth=0.5,
                        label=f"Alt {alt:.0f} km")
        ax3.set_xlabel('Target Altitude (km)')
        ax3.set_ylabel('Performance')
        ax3.set_zlabel('Number of Satellites')
        ax3.set_title('Altitude vs Performance vs Satellites')
        ax3.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/3d_altitude_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved 3d_altitude_comparison.png")
    
    def plot_3d_inclination_comparison(self, csv_names):
        """3D scatter: Cost vs Performance vs Satellites, merged by inclination (not by seed/config)."""
        fig = plt.figure(figsize=(16, 6))
        # Group by inclination
        param_groups = {}
        for name in csv_names:
            meta = self.metadata[name]
            param_val = meta.get('target_inclination')
            if param_val is None:
                continue
            param_groups.setdefault(param_val, []).append(name)

        # Plot 1: Cost vs Performance vs Satellites
        ax1 = fig.add_subplot(131, projection='3d')
        for i, (inc, csv_list) in enumerate(sorted(param_groups.items())):
            color = self.color_palette[i % len(self.color_palette)]
            marker = self.marker_styles[i % len(self.marker_styles)]
            merged_df = pd.concat([self.data_dict[name] for name in csv_list], ignore_index=True)
            nd_mask = self._nondominated_mask(merged_df)
            merged_df = merged_df[nd_mask]
            ax1.scatter(merged_df['cost_musd'], merged_df['performance'], merged_df['num_satellites'],
                        c=color, s=60, alpha=0.7, marker=marker,
                        edgecolors='black', linewidth=0.5,
                        label=f"Inc {inc:.0f}°")
        ax1.set_xlabel('Cost (M$ FY2025)')
        ax1.set_ylabel('Performance')
        ax1.set_zlabel('Number of Satellites')
        ax1.set_title('Inclination Sweep Comparison\nCost vs Performance vs Satellites')
        ax1.legend()

        # Plot 2: Cost vs Satellites vs Inclination (colored by performance)
        ax2 = fig.add_subplot(132, projection='3d')
        for i, (inc, csv_list) in enumerate(sorted(param_groups.items())):
            marker = self.marker_styles[i % len(self.marker_styles)]
            merged_df = pd.concat([self.data_dict[name] for name in csv_list], ignore_index=True)
            nd_mask = self._nondominated_mask(merged_df)
            merged_df = merged_df[nd_mask]
            scatter = ax2.scatter(merged_df['cost_musd'], merged_df['num_satellites'],
                                [inc] * len(merged_df),
                                c=merged_df['performance'], s=60, alpha=0.7, marker=marker,
                                cmap='coolwarm', vmin=0, vmax=1,
                                edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Cost (M$ FY2025)')
        ax2.set_ylabel('Number of Satellites')
        ax2.set_zlabel('Target Inclination (deg)')
        ax2.set_title('Cost vs Satellites vs Target Inclination\n(colored by performance)')
        plt.colorbar(scatter, ax=ax2, label='Performance', shrink=0.5)

        # Plot 3: Performance vs Inclination vs Satellites
        ax3 = fig.add_subplot(133, projection='3d')
        for i, (inc, csv_list) in enumerate(sorted(param_groups.items())):
            color = self.color_palette[i % len(self.color_palette)]
            marker = self.marker_styles[i % len(self.marker_styles)]
            merged_df = pd.concat([self.data_dict[name] for name in csv_list], ignore_index=True)
            nd_mask = self._nondominated_mask(merged_df)
            merged_df = merged_df[nd_mask]
            ax3.scatter([inc] * len(merged_df), merged_df['performance'], merged_df['num_satellites'],
                        c=color, s=60, alpha=0.7, marker=marker,
                        edgecolors='black', linewidth=0.5,
                        label=f"Inc {inc:.0f}°")
        ax3.set_xlabel('Target Inclination (deg)')
        ax3.set_ylabel('Performance')
        ax3.set_zlabel('Number of Satellites')
        ax3.set_title('Inclination vs Performance vs Satellites')
        ax3.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/3d_inclination_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved 3d_inclination_comparison.png")
    
    def plot_pareto_comparison(self, csv_names, sweep_type):
        """Show all solutions per CSV, and highlight the Pareto front (min cost, max performance)."""
        plt.figure(figsize=(10, 8))

        for i, name in enumerate(csv_names):
            df = self.data_dict[name].copy()
            meta = self.metadata[name]
            color = self.color_palette[i % len(self.color_palette)]
            marker = self.marker_styles[i % len(self.marker_styles)]

            # Identify non-dominated set
            nd_mask = self._nondominated_mask(df)
            df_front = df[nd_mask].sort_values('cost_musd')
            df_dominated = df[~nd_mask]

            # Plot dominated (all solutions below the front)
            if not df_dominated.empty:
                plt.scatter(df_dominated['cost_musd'], df_dominated['performance'],
                            c=color, s=28, alpha=0.25, marker=marker,
                            edgecolors='none', linewidth=0.0)

            # Plot Pareto front: line + highlighted markers
            if not df_front.empty:
                # Connecting line through front
                plt.plot(df_front['cost_musd'], df_front['performance'], '-', color=color, alpha=0.85)
                # Front points
                plt.scatter(df_front['cost_musd'], df_front['performance'],
                            c=color, s=70, alpha=0.95, marker=marker,
                            edgecolors='black', linewidth=0.6,
                            label=meta['label'])

        plt.xlabel('Total Cost (M$ FY2025)', fontsize=12)
        plt.ylabel('Performance Score', fontsize=12)
        plt.title(f'All Solutions + Pareto Front - {sweep_type.title()} Sweep', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pareto_comparison_{sweep_type}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved pareto_comparison_{sweep_type}.png")
    
    def plot_pareto_comparison_colored(self, csv_names, sweep_type):
        """Pareto fronts colored by number of satellites."""
        fig, axes = plt.subplots(1, len(csv_names), figsize=(6*len(csv_names), 5))
        
        if len(csv_names) == 1:
            axes = [axes]
        
        for i, name in enumerate(csv_names):
            df = self.data_dict[name]
            meta = self.metadata[name]
            marker = self.marker_styles[i % len(self.marker_styles)]

            scatter = axes[i].scatter(df['cost_musd'], df['performance'],
                                      c=df['num_satellites'], s=55, alpha=0.8,
                                      cmap='viridis', edgecolors='black', linewidth=0.5,
                                      marker=marker)
            
            axes[i].set_xlabel('Total Cost (M$ FY2025)')
            axes[i].set_ylabel('Performance Score')
            axes[i].set_title(f'{meta["label"]}\nPareto Front')
            axes[i].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[i], label='Number of Satellites')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pareto_colored_{sweep_type}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved pareto_colored_{sweep_type}.png")

    def plot_pareto_comparison_log_cost(self, csv_names, sweep_type):
        """Same as plot_pareto_comparison, but with cost in log scale (x-axis)."""
        plt.figure(figsize=(10, 8))

        for i, name in enumerate(csv_names):
            df = self.data_dict[name].copy()
            meta = self.metadata[name]
            color = self.color_palette[i % len(self.color_palette)]
            marker = self.marker_styles[i % len(self.marker_styles)]

            # Identify non-dominated set
            nd_mask = self._nondominated_mask(df)
            df_front = df[nd_mask].sort_values('cost_musd')
            df_dominated = df[~nd_mask]

            # Plot dominated (all solutions below the front)
            if not df_dominated.empty:
                plt.scatter(df_dominated['cost_musd'], df_dominated['performance'],
                            c=color, s=28, alpha=0.25, marker=marker,
                            edgecolors='none', linewidth=0.0)

            # Plot Pareto front: line + highlighted markers
            if not df_front.empty:
                # Connecting line through front
                plt.plot(df_front['cost_musd'], df_front['performance'], '-', color=color, alpha=0.85)
                # Front points
                plt.scatter(df_front['cost_musd'], df_front['performance'],
                            c=color, s=70, alpha=0.95, marker=marker,
                            edgecolors='black', linewidth=0.6,
                            label=meta['label'])

        # Use log scale with readable money ticks (M$ FY2025 / B$)
        from matplotlib.ticker import FuncFormatter, LogLocator
        ax = plt.gca()
        ax.set_xscale('log')
        # Major ticks at 1, 2, 5 within each decade
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))

        def _fmt_musd(x, pos):
            # x is in millions of dollars (M$ FY2025)
            import numpy as _np
            if x is None or not _np.isfinite(x) or x <= 0:
                return ""
            if x >= 1000:  # 1000 M$ FY2025 = 1 B$
                val_b = x / 1000.0
                # Show one decimal when needed
                return (f"{val_b:.0f} B$" if abs(val_b - round(val_b)) < 1e-6 else f"{val_b:.1f} B$")
            else:
                return f"{x:.0f} M$ FY2025"

        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_musd))
        plt.xlabel('Total Cost (M$ FY2025)')
        plt.ylabel('Performance Score')
        plt.title(f'All Solutions + Pareto Front (Log Cost) - {sweep_type.title()} Sweep')
        plt.legend()
        plt.grid(True, which='both', axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pareto_comparison_{sweep_type}_logx.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved pareto_comparison_{sweep_type}_logx.png")

    def plot_merged_pareto_multiseed(self, csv_names, sweep_type):
        """
        Merge solutions from multiple seeds and recompute global Pareto front.
        
        This function:
        1. Groups CSVs by their sweep parameter (e.g., same altitude, different seeds)
        2. Merges all solutions for each parameter value across seeds
        3. Recomputes the global non-dominated front from merged solutions
        4. Plots individual seed fronts (faded) + global merged front (highlighted)
        """
        # Group CSVs by sweep parameter (ignoring seed differences)
        import re
        
        param_groups = {}  # {param_value: [(csv_name, seed), ...]}
        
        for name in csv_names:
            meta = self.metadata[name]
            
            # Extract parameter value based on sweep type
            if sweep_type == 'altitude':
                param_val = meta.get('target_altitude')
            elif sweep_type == 'inclination':
                param_val = meta.get('target_inclination')
            elif sweep_type == 'mass':
                param_val = meta.get('sat_mass')
            elif sweep_type == 'dv':
                param_val = meta.get('dv_m_s')
            else:
                continue
            
            if param_val is None:
                continue
            
            # Extract seed from filename (e.g., sd42, sd11)
            seed_match = re.search(r'_sd(\d+)', name)
            seed = int(seed_match.group(1)) if seed_match else 0
            
            param_groups.setdefault(param_val, []).append((name, seed))
        
        # Check if we have any data at all
        if not param_groups:
            print(f"  [INFO] No data found for {sweep_type} sweep - skipping merged Pareto plot")
            return
        
        # Identify which groups have multiple seeds
        multi_seed_params = {k for k, v in param_groups.items() if len(v) > 1}
        single_seed_params = {k for k, v in param_groups.items() if len(v) == 1}
        
        if not multi_seed_params:
            print(f"  [INFO] No multi-seed data found for {sweep_type} sweep - skipping merged Pareto plot")
            print(f"        All {len(param_groups)} parameter values have only one seed each")
            return
        
        print(f"  Generating merged multi-seed Pareto plot for {sweep_type} sweep...")
        print(f"    Found {len(multi_seed_params)} parameter values with multiple seeds")
        if single_seed_params:
            print(f"    Found {len(single_seed_params)} parameter values with single seed (will still plot)")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Process each parameter group (including single-seed groups)
        for param_idx, (param_val, csv_seed_list) in enumerate(sorted(param_groups.items())):
            color = self.color_palette[param_idx % len(self.color_palette)]
            marker = self.marker_styles[param_idx % len(self.marker_styles)]
            
            # Get label from first CSV in group (use base_label to exclude seed suffix)
            first_csv_name = csv_seed_list[0][0]
            param_label = self.metadata[first_csv_name].get('base_label', 
                                                            self.metadata[first_csv_name]['label'])
            
            # Collect all solutions across seeds
            all_costs = []
            all_perfs = []
            
            is_multi_seed = len(csv_seed_list) > 1
            
            # Plot individual seed data
            for csv_name, seed in csv_seed_list:
                df = self.data_dict[csv_name].copy()
                costs = df['cost_musd'].to_numpy()
                perfs = df['performance'].to_numpy()
                
                # Store for merged computation
                all_costs.extend(costs)
                all_perfs.extend(perfs)
                
                # For multi-seed: plot faded individual seeds
                # For single-seed: skip (will plot final front directly)
                if is_multi_seed:
                    ax.scatter(costs, perfs, 
                              c=color, s=20, alpha=0.15, marker=marker,
                              edgecolors='none')
            
            # Compute global Pareto front from all solutions (merged if multi-seed, single if not)
            all_costs = np.array(all_costs)
            all_perfs = np.array(all_perfs)
            global_pareto_mask = self.compute_global_pareto_from_arrays(all_costs, all_perfs)
            
            merged_costs = all_costs[global_pareto_mask]
            merged_perfs = all_perfs[global_pareto_mask]
            
            # Sort for line plotting
            sorted_idx = np.argsort(merged_costs)
            merged_costs_sorted = merged_costs[sorted_idx]
            merged_perfs_sorted = merged_perfs[sorted_idx]
            
            # Plot Pareto front (highlighted)
            ax.plot(merged_costs_sorted, merged_perfs_sorted, '-', 
                   color=color, linewidth=2.5, alpha=0.9, zorder=10)
            
            label = f"{param_label}"
            
            ax.scatter(merged_costs_sorted, merged_perfs_sorted,
                      c=color, s=80, alpha=0.95, marker=marker,
                      edgecolors='black', linewidth=1.0, zorder=11,
                      label=label)
            
            if is_multi_seed:
                print(f"    {param_label}: merged {len(all_costs)} solutions → {len(merged_costs)} on global front")
            else:
                print(f"    {param_label}: single seed, {len(merged_costs)} solutions on Pareto front")
        
        ax.set_xlabel('Total Cost (M$ FY2025)', fontsize=12)
        ax.set_ylabel('Performance Score', fontsize=12)
        
        # Create title and subtitle
        ax.set_title(f'Pareto Front Comparison - {sweep_type.title()} Sweep', fontsize=13)
        subtitle = ""
        if multi_seed_params:
            subtitle += f"Parameters with multiple seeds: {len(multi_seed_params)} (merged)\n"
        if single_seed_params:
            subtitle += f"Parameters with single seed: {len(single_seed_params)}"
        if subtitle:
            ax.text(0.5, 1.02, subtitle.strip(), fontsize=11, ha='center', va='bottom', transform=ax.transAxes)
        
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pareto_merged_multiseed_{sweep_type}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved pareto_merged_multiseed_{sweep_type}.png")

    def plot_merged_pareto_multiseed_logx(self, csv_names, sweep_type):
        """
        Identical to plot_merged_pareto_multiseed, but with cost x-axis in log scale.
        Only used for mass sweep.
        """
        import re
        param_groups = {}
        for name in csv_names:
            meta = self.metadata[name]
            if sweep_type == 'mass':
                param_val = meta.get('sat_mass')
            elif sweep_type == 'altitude':
                param_val = meta.get('target_altitude')
            elif sweep_type == 'inclination':
                param_val = meta.get('target_inclination')
            elif sweep_type == 'dv':
                param_val = meta.get('dv_m_s')
            else:
                continue
            if param_val is None:
                continue
            seed_match = re.search(r'_sd(\d+)', name)
            seed = int(seed_match.group(1)) if seed_match else 0
            param_groups.setdefault(param_val, []).append((name, seed))

        if not param_groups:
            print(f"  [INFO] No data found for {sweep_type} sweep - skipping merged Pareto log plot")
            return

        multi_seed_params = {k for k, v in param_groups.items() if len(v) > 1}
        single_seed_params = {k for k, v in param_groups.items() if len(v) == 1}

        if not multi_seed_params:
            print(f"  [INFO] No multi-seed data found for {sweep_type} sweep - skipping merged Pareto log plot")
            print(f"        All {len(param_groups)} parameter values have only one seed each")
            return

        print(f"  Generating merged multi-seed Pareto LOG plot for {sweep_type} sweep...")
        print(f"    Found {len(multi_seed_params)} parameter values with multiple seeds")
        if single_seed_params:
            print(f"    Found {len(single_seed_params)} parameter values with single seed (will still plot)")

        fig, ax = plt.subplots(figsize=(12, 9))

        for param_idx, (param_val, csv_seed_list) in enumerate(sorted(param_groups.items())):
            color = self.color_palette[param_idx % len(self.color_palette)]
            marker = self.marker_styles[param_idx % len(self.marker_styles)]
            first_csv_name = csv_seed_list[0][0]
            param_label = self.metadata[first_csv_name].get('base_label', self.metadata[first_csv_name]['label'])
            all_costs = []
            all_perfs = []
            is_multi_seed = len(csv_seed_list) > 1
            for csv_name, seed in csv_seed_list:
                df = self.data_dict[csv_name].copy()
                costs = df['cost_musd'].to_numpy()
                perfs = df['performance'].to_numpy()
                all_costs.extend(costs)
                all_perfs.extend(perfs)
                if is_multi_seed:
                    ax.scatter(costs, perfs, c=color, s=20, alpha=0.15, marker=marker, edgecolors='none')
            all_costs = np.array(all_costs)
            all_perfs = np.array(all_perfs)
            global_pareto_mask = self.compute_global_pareto_from_arrays(all_costs, all_perfs)
            merged_costs = all_costs[global_pareto_mask]
            merged_perfs = all_perfs[global_pareto_mask]
            sorted_idx = np.argsort(merged_costs)
            merged_costs_sorted = merged_costs[sorted_idx]
            merged_perfs_sorted = merged_perfs[sorted_idx]
            ax.plot(merged_costs_sorted, merged_perfs_sorted, '-', color=color, linewidth=2.5, alpha=0.9, zorder=10)
            label = f"{param_label}"
            ax.scatter(merged_costs_sorted, merged_perfs_sorted, c=color, s=80, alpha=0.95, marker=marker, edgecolors='black', linewidth=1.0, zorder=11, label=label)
            if is_multi_seed:
                print(f"    {param_label}: merged {len(all_costs)} solutions → {len(merged_costs)} on global front")
            else:
                print(f"    {param_label}: single seed, {len(merged_costs)} solutions on Pareto front")

        ax.set_xlabel('Total Cost (M$ FY2025)', fontsize=12)
        ax.set_ylabel('Performance Score', fontsize=12)
        ax.set_xscale('log')
        from matplotlib.ticker import FuncFormatter, LogLocator
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        def _fmt_musd(x, pos):
            import numpy as _np
            if x is None or not _np.isfinite(x) or x <= 0:
                return ""
            if x >= 1000:
                val_b = x / 1000.0
                return (f"{val_b:.0f} B$" if abs(val_b - round(val_b)) < 1e-6 else f"{val_b:.1f} B$")
            else:
                return f"{x:.0f} M$ FY2025"
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_musd))
        ax.set_title(f'Pareto Front Comparison (Log Cost) - {sweep_type.title()} Sweep', fontsize=13)
        subtitle = ""
        if multi_seed_params:
            subtitle += f"Parameters with multiple seeds: {len(multi_seed_params)} (merged)\n"
        if single_seed_params:
            subtitle += f"Parameters with single seed: {len(single_seed_params)}"
        if subtitle:
            ax.text(0.5, 1.02, subtitle.strip(), fontsize=11, ha='center', va='bottom', transform=ax.transAxes)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pareto_merged_multiseed_{sweep_type}_logx.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved pareto_merged_multiseed_{sweep_type}_logx.png")
    
    def plot_cost_components_comparison(self, csv_names, sweep_type):
        """Cost components (RE, Launch) vs satellites, merged by sweep parameter."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        components = [
            ('RE_cost_musd', 'RE Cost', axes[0]),
            ('launch_cost_musd', 'Launch Cost', axes[1])
        ]

        param_groups = {}
        for name in csv_names:
            meta = self.metadata[name]
            if sweep_type == 'altitude':
                param_val = meta.get('target_altitude')
            elif sweep_type == 'inclination':
                param_val = meta.get('target_inclination')
            elif sweep_type == 'mass':
                param_val = meta.get('sat_mass')
            elif sweep_type == 'dv':
                param_val = meta.get('dv_m_s')
            else:
                continue
            if param_val is None:
                continue
            param_groups.setdefault(param_val, []).append(name)

        for param_idx, (param_val, csv_list) in enumerate(sorted(param_groups.items())):
            color = self.color_palette[param_idx % len(self.color_palette)]
            marker = self.marker_styles[param_idx % len(self.marker_styles)]
            merged_df = pd.concat([self.data_dict[name] for name in csv_list], ignore_index=True)
            nd_mask = self._nondominated_mask(merged_df)
            merged_df = merged_df[nd_mask]
            label_base = f"{sweep_type.title()} {param_val:.0f}" if param_val is not None else f"{sweep_type.title()}"

            # RE Cost subplot: scatter + linear trend line (dotted)
            r_re = None
            axes[0].scatter(merged_df['num_satellites'], merged_df['RE_cost_musd'],
                            c=color, s=45, alpha=0.75, marker=marker,
                            edgecolors='black', linewidth=0.4)
            if len(merged_df) > 1:
                x = merged_df['num_satellites']
                y = merged_df['RE_cost_musd']
                slope, intercept, r_re, _, _ = linregress(x, y)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = slope * x_fit + intercept
                axes[0].plot(x_fit, y_fit, color=color, alpha=0.7, linewidth=2, linestyle=':')
            legend_label_re = f"{label_base} (R={r_re:.2f})" if r_re is not None else label_base
            axes[0].plot([], [], color=color, marker=marker, linestyle='None', label=legend_label_re)
            
            # Launch Cost subplot: scatter + linear trend line (dotted)
            r = None
            axes[1].scatter(merged_df['num_satellites'], merged_df['launch_cost_musd'],
                            c=color, s=45, alpha=0.55, marker=marker,
                            edgecolors='black', linewidth=0.4)
            if len(merged_df) > 1:
                x = merged_df['num_satellites']
                y = merged_df['launch_cost_musd']
                slope, intercept, r, _, _ = linregress(x, y)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = slope * x_fit + intercept
                axes[1].plot(x_fit, y_fit, color=color, alpha=0.7, linewidth=2, linestyle=':')
            legend_label = f"{label_base} (R={r:.2f})" if r is not None else label_base
            axes[1].plot([], [], color=color, marker=marker, linestyle='None', label=legend_label)

        axes[0].set_xlabel('Number of Satellites')
        axes[0].set_ylabel('RE Cost (M$ FY2025)')
        axes[0].set_title(f'RE Cost vs Satellites - {sweep_type.title()} Sweep')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Number of Satellites')
        axes[1].set_ylabel('Launch Cost (M$ FY2025)')
        axes[1].set_title(f'Launch Cost vs Satellites - {sweep_type.title()} Sweep')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cost_components_{sweep_type}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved cost_components_{sweep_type}.png")

    def plot_launch_groups_raan_comparison(self, csv_names, sweep_type):
        """
        Plot only the top subplots from deployment comparison:
        - Launch groups vs number of planes
        - Required RAAN spacing vs launch groups
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        param_groups = {}
        for name in csv_names:
            meta = self.metadata[name]
            if sweep_type == 'altitude':
                param_val = meta.get('target_altitude')
            elif sweep_type == 'inclination':
                param_val = meta.get('target_inclination')
            elif sweep_type == 'mass':
                param_val = meta.get('sat_mass')
            elif sweep_type == 'dv':
                param_val = meta.get('dv_m_s')
            else:
                continue
            if param_val is None:
                continue
            param_groups.setdefault(param_val, []).append(name)

        for param_idx, (param_val, csv_list) in enumerate(sorted(param_groups.items())):
            color = self.color_palette[param_idx % len(self.color_palette)]
            marker = self.marker_styles[param_idx % len(self.marker_styles)]
            merged_df = pd.concat([self.data_dict[name] for name in csv_list], ignore_index=True)
            nd_mask = self._nondominated_mask(merged_df)
            merged_df = merged_df[nd_mask]
            label_base = f"{sweep_type.title()} {param_val:.0f}" if param_val is not None else f"{sweep_type.title()}"

            # 1. Launch groups vs number of planes
            r1 = None
            axes[0].scatter(merged_df['P_planes'], merged_df['launch_num_groups'],
                            c=color, s=55, alpha=0.7, marker=marker,
                            edgecolors='black', linewidth=0.4)
            if len(merged_df) > 1:
                x = merged_df['P_planes']
                y = merged_df['launch_num_groups']
                slope, intercept, r1, _, _ = linregress(x, y)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = slope * x_fit + intercept
                axes[0].plot(x_fit, y_fit, color=color, alpha=0.7, linewidth=2, linestyle=':')
            axes[0].plot([], [], color=color, marker=marker, linestyle='None',
                        label=f"{label_base} (R={r1:.2f})" if r1 is not None else label_base)

            # 2. Required RAAN spacing vs launch groups (hyperbolic fit, dotted)
            r2 = None
            axes[1].scatter(merged_df['launch_num_groups'], merged_df['launch_delta_raan_required_deg'],
                            c=color, s=55, alpha=0.7, marker=marker,
                            edgecolors='black', linewidth=0.4)
            if len(merged_df) > 1:
                x = merged_df['launch_num_groups']
                y = merged_df['launch_delta_raan_required_deg']
                from scipy.optimize import curve_fit
                def hyperbolic(x, a, b): return a / x + b
                try:
                    popt, _ = curve_fit(hyperbolic, x, y)
                    y_fit = hyperbolic(x, *popt)
                    r2 = np.corrcoef(y, y_fit)[0, 1]
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    axes[1].plot(x_fit, hyperbolic(x_fit, *popt), color=color, alpha=0.7, linewidth=2, linestyle=':')
                except Exception:
                    pass
            axes[1].plot([], [], color=color, marker=marker, linestyle='None',
                        label=f"{label_base} (R={r2:.2f})" if r2 is not None else label_base)

        axes[0].set_xlabel('Number of Planes (P)')
        axes[0].set_ylabel('Number of Launch Groups')
        axes[0].set_title('Launch Groups vs Planes')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Number of Launch Groups')
        axes[1].set_ylabel('Required RAAN Spacing (deg)')
        axes[1].set_title('RAAN Spacing vs Launch Groups')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'Launch Groups & RAAN Spacing - {sweep_type.title()} Sweep', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/launch_groupsP_RAAN_{sweep_type}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved launch_groupsP_RAAN_{sweep_type}.png")

    def plot_launch_cost_planes_comparison(self, csv_names, sweep_type):
        """
        Plot launch cost vs number of planes and launch cost per satellite vs number of planes.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        param_groups = {}
        for name in csv_names:
            meta = self.metadata[name]
            if sweep_type == 'altitude':
                param_val = meta.get('target_altitude')
            elif sweep_type == 'inclination':
                param_val = meta.get('target_inclination')
            elif sweep_type == 'mass':
                param_val = meta.get('sat_mass')
            elif sweep_type == 'dv':
                param_val = meta.get('dv_m_s')
            else:
                continue
            if param_val is None:
                continue
            param_groups.setdefault(param_val, []).append(name)

        for param_idx, (param_val, csv_list) in enumerate(sorted(param_groups.items())):
            color = self.color_palette[param_idx % len(self.color_palette)]
            marker = self.marker_styles[param_idx % len(self.marker_styles)]
            merged_df = pd.concat([self.data_dict[name] for name in csv_list], ignore_index=True)
            nd_mask = self._nondominated_mask(merged_df)
            merged_df = merged_df[nd_mask]
            label_base = f"{sweep_type.title()} {param_val:.0f}" if param_val is not None else f"{sweep_type.title()}"

            # 1. Launch cost vs number of planes
            r_cost = None
            axes[0].scatter(merged_df['P_planes'], merged_df['launch_cost_musd'],
                            c=color, s=55, alpha=0.7, marker=marker,
                            edgecolors='black', linewidth=0.4)
            if len(merged_df) > 1:
                x = merged_df['P_planes']
                y = merged_df['launch_cost_musd']
                slope, intercept, r_cost, _, _ = linregress(x, y)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = slope * x_fit + intercept
                axes[0].plot(x_fit, y_fit, color=color, alpha=0.7, linewidth=2, linestyle=':')
            axes[0].plot([], [], color=color, marker=marker, linestyle='None',
                        label=f"{label_base} (R={r_cost:.2f})" if r_cost is not None else label_base)

            # 2. Launch cost per satellite vs number of planes
            r_cost_per_sat = None
            merged_df = merged_df.copy()
            merged_df['launch_cost_per_sat'] = merged_df['launch_cost_musd'] / merged_df['num_satellites']
            axes[1].scatter(merged_df['P_planes'], merged_df['launch_cost_per_sat'],
                            c=color, s=55, alpha=0.7, marker=marker,
                            edgecolors='black', linewidth=0.4)
            if len(merged_df) > 1:
                x = merged_df['P_planes']
                y = merged_df['launch_cost_per_sat']
                slope, intercept, r_cost_per_sat, _, _ = linregress(x, y)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = slope * x_fit + intercept
                axes[1].plot(x_fit, y_fit, color=color, alpha=0.7, linewidth=2, linestyle=':')
            axes[1].plot([], [], color=color, marker=marker, linestyle='None',
                        label=f"{label_base} (R={r_cost_per_sat:.2f})" if r_cost_per_sat is not None else label_base)

        axes[0].set_xlabel('Number of Planes (P)')
        axes[0].set_ylabel('Launch Cost (M$ FY2025)')
        axes[0].set_title('Launch Cost vs Number of Planes')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Number of Planes (P)')
        axes[1].set_ylabel('Launch Cost per Satellite (M$ FY2025)')
        axes[1].set_title('Launch Cost per Satellite vs Number of Planes')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'Launch Cost Analysis - {sweep_type.title()} Sweep', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/launch_cost_planes_{sweep_type}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved launch_cost_planes_{sweep_type}.png")

    def plot_launchdeploy_cost_comparison(self, csv_names, sweep_type):
        """
        Launch deployment cost analysis:
        - Left: Launch cost vs deployment ΔV
        - Right: Launch cost per satellite vs deployment ΔV
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        param_groups = {}
        for name in csv_names:
            meta = self.metadata[name]
            if sweep_type == 'altitude':
                param_val = meta.get('target_altitude')
            elif sweep_type == 'inclination':
                param_val = meta.get('target_inclination')
            elif sweep_type == 'mass':
                param_val = meta.get('sat_mass')
            elif sweep_type == 'dv':
                param_val = meta.get('dv_m_s')
            else:
                continue
            if param_val is None:
                continue
            param_groups.setdefault(param_val, []).append(name)

        for param_idx, (param_val, csv_list) in enumerate(sorted(param_groups.items())):
            color = self.color_palette[param_idx % len(self.color_palette)]
            marker = self.marker_styles[param_idx % len(self.marker_styles)]
            merged_df = pd.concat([self.data_dict[name] for name in csv_list], ignore_index=True)
            nd_mask = self._nondominated_mask(merged_df)
            merged_df = merged_df[nd_mask]
            label_base = f"{sweep_type.title()} {param_val:.0f}" if param_val is not None else f"{sweep_type.title()}"

            # Left: Launch cost vs deployment ΔV
            r_lin = None
            axes[0].scatter(merged_df['launch_dv_used_m_s'], merged_df['launch_cost_musd'],
                            c=color, s=45, alpha=0.35, marker=marker,
                            edgecolors='black', linewidth=0.4)
            if len(merged_df) > 1:
                x = merged_df['launch_dv_used_m_s']
                y = merged_df['launch_cost_musd']
                slope, intercept, r_lin, _, _ = linregress(x, y)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit_lin = slope * x_fit + intercept
                axes[0].plot(x_fit, y_fit_lin, color=color, alpha=0.7, linewidth=2, linestyle=':')
            legend_label_left = f"{label_base} (R={r_lin:.2f})" if r_lin is not None else label_base
            axes[0].plot([], [], color=color, marker=marker, linestyle='None', label=legend_label_left)

            # Right: Launch cost per satellite vs deployment ΔV
            r_norm = None
            merged_df = merged_df.copy()
            merged_df['launch_cost_per_sat'] = merged_df['launch_cost_musd'] / merged_df['num_satellites']
            axes[1].scatter(merged_df['launch_dv_used_m_s'], merged_df['launch_cost_per_sat'],
                            c=color, s=45, alpha=0.35, marker=marker,
                            edgecolors='black', linewidth=0.4)
            if len(merged_df) > 1:
                x = merged_df['launch_dv_used_m_s']
                y = merged_df['launch_cost_per_sat']
                slope, intercept, r_norm, _, _ = linregress(x, y)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit_norm = slope * x_fit + intercept
                axes[1].plot(x_fit, y_fit_norm, color=color, alpha=0.7, linewidth=2, linestyle=':')
            legend_label_right = f"{label_base} (R={r_norm:.2f})" if r_norm is not None else label_base
            axes[1].plot([], [], color=color, marker=marker, linestyle='None', label=legend_label_right)

        axes[0].set_xlabel('Deployment ΔV Used (m/s)')
        axes[0].set_ylabel('Launch Cost (M$ FY2025)')
        axes[0].set_title('Launch Cost vs Deployment ΔV')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Deployment ΔV Used (m/s)')
        axes[1].set_ylabel('Launch Cost per Satellite (M$ FY2025)')
        axes[1].set_title('Launch Cost per Satellite vs Deployment ΔV')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'Launch Deployment Cost Analysis - {sweep_type.title()} Sweep', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/launchdeploy_cost_{sweep_type}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved launchdeploy_cost_{sweep_type}.png")

    def plot_cost_components_per_satellite(self, csv_names, sweep_type):
        """
        Plot cost per satellite components.
        Left: RE cost per sat + Total cost per sat (2 marker types)
        Right: Launch cost per sat
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        param_groups = {}
        for name in csv_names:
            meta = self.metadata[name]
            if sweep_type == 'altitude':
                param_val = meta.get('target_altitude')
            elif sweep_type == 'inclination':
                param_val = meta.get('target_inclination')
            elif sweep_type == 'mass':
                param_val = meta.get('sat_mass')
            elif sweep_type == 'dv':
                param_val = meta.get('dv_m_s')
            else:
                continue
            if param_val is None:
                continue
            param_groups.setdefault(param_val, []).append(name)

        # Legend handles
        sweep_handles = []
        marker_handles_left = [
            plt.Line2D([], [], color='black', marker='s', linestyle='None', markersize=7, label='RE'),
            plt.Line2D([], [], color='black', marker='o', linestyle='None', markersize=7, label='Total')
        ]

        for param_idx, (param_val, csv_list) in enumerate(sorted(param_groups.items())):
            color = self.color_palette[param_idx % len(self.color_palette)]
            marker = self.marker_styles[param_idx % len(self.marker_styles)]
            merged_df = pd.concat([self.data_dict[name] for name in csv_list], ignore_index=True)
            nd_mask = self._nondominated_mask(merged_df)
            merged_df = merged_df[nd_mask]
            label_base = f"{sweep_type.title()} {param_val:.0f}" if param_val is not None else f"{sweep_type.title()}"

            # Calculate cost per satellite
            merged_df = merged_df.copy()
            merged_df['RE_cost_per_sat'] = merged_df['RE_cost_musd'] / merged_df['num_satellites']
            merged_df['total_cost'] = merged_df['NRE_cost_musd'] + merged_df['RE_cost_musd'] + merged_df['launch_cost_musd']
            merged_df['total_cost_per_sat'] = merged_df['total_cost'] / merged_df['num_satellites']
            merged_df['launch_cost_per_sat'] = merged_df['launch_cost_musd'] / merged_df['num_satellites']

            # Add sweep color handle (once per color)
            sweep_handles.append(plt.Line2D([], [], color=color, marker='o', linestyle='None', markersize=7, label=label_base))

            # LEFT SUBPLOT: RE cost per satellite (square marker) + Total cost per satellite (circle marker)
            axes[0].scatter(merged_df['num_satellites'], merged_df['RE_cost_per_sat'],
                        c=color, s=45, alpha=0.5, marker='s',
                        edgecolors='black', linewidth=0.4)
            if len(merged_df) > 1:
                x = merged_df['num_satellites']
                y = merged_df['RE_cost_per_sat']
                from scipy.optimize import curve_fit
                def power_law(x, a, b):
                    return a * np.power(x, b)
                try:
                    popt, _ = curve_fit(power_law, x, y, maxfev=5000)
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    y_fit = power_law(x_fit, *popt)
                    axes[0].plot(x_fit, y_fit, color=color, alpha=0.6, linewidth=1.5, linestyle='--')
                except Exception:
                    pass

            axes[0].scatter(merged_df['num_satellites'], merged_df['total_cost_per_sat'],
                        c=color, s=45, alpha=0.5, marker='o',
                        edgecolors='black', linewidth=0.4)
            if len(merged_df) > 1:
                x = merged_df['num_satellites']
                y = merged_df['total_cost_per_sat']
                from scipy.optimize import curve_fit
                def power_law(x, a, b):
                    return a * np.power(x, b)
                try:
                    popt, _ = curve_fit(power_law, x, y, maxfev=5000)
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    y_fit = power_law(x_fit, *popt)
                    axes[0].plot(x_fit, y_fit, color=color, alpha=0.6, linewidth=1.5, linestyle=':')
                except Exception:
                    pass

            # RIGHT SUBPLOT: Launch cost per satellite (original)
            r_launch = None
            axes[1].scatter(merged_df['num_satellites'], merged_df['launch_cost_per_sat'],
                            c=color, s=45, alpha=0.55, marker=marker,
                            edgecolors='black', linewidth=0.4)
            if len(merged_df) > 1:
                x = merged_df['num_satellites']
                y = merged_df['launch_cost_per_sat']
                slope, intercept, r_launch, _, _ = linregress(x, y)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = slope * x_fit + intercept
                axes[1].plot(x_fit, y_fit, color=color, alpha=0.7, linewidth=2, linestyle=':')
            legend_label_launch = f"{label_base} (R={r_launch:.2f})" if r_launch is not None else label_base
            axes[1].plot([], [], color=color, marker=marker, linestyle='None', label=legend_label_launch)

        axes[0].set_xlabel('Number of Satellites')
        axes[0].set_ylabel('Cost per Satellite (M$ FY2025)')
        axes[0].set_title(f'RE & Total Cost per Satellite - {sweep_type.title()} Sweep')
        axes[0].legend(handles=sweep_handles + marker_handles_left, loc='best', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Number of Satellites')
        axes[1].set_ylabel('Launch Cost per Satellite (M$ FY2025)')
        axes[1].set_title(f'Launch Cost per Satellite vs Satellites - {sweep_type.title()} Sweep')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cost_components_per_satellite_{sweep_type}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved cost_components_per_satellite_{sweep_type}.png")

    def plot_performance_vs_sat_and_planes(self, csv_names, sweep_type):
        """
        Plot performance vs number of satellites (left) and vs number of planes (right),
        with trend lines and R in legend, colored by sweep parameter.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        param_groups = {}
        for name in csv_names:
            meta = self.metadata[name]
            if sweep_type == 'altitude':
                param_val = meta.get('target_altitude')
            elif sweep_type == 'inclination':
                param_val = meta.get('target_inclination')
            elif sweep_type == 'mass':
                param_val = meta.get('sat_mass')
            elif sweep_type == 'dv':
                param_val = meta.get('dv_m_s')
            else:
                continue
            if param_val is None:
                continue
            param_groups.setdefault(param_val, []).append(name)

        for param_idx, (param_val, csv_list) in enumerate(sorted(param_groups.items())):
            color = self.color_palette[param_idx % len(self.color_palette)]
            marker = self.marker_styles[param_idx % len(self.marker_styles)]
            merged_df = pd.concat([self.data_dict[name] for name in csv_list], ignore_index=True)
            nd_mask = self._nondominated_mask(merged_df)
            merged_df = merged_df[nd_mask]
            label_base = f"{sweep_type.title()} {param_val:.0f}" if param_val is not None else f"{sweep_type.title()}"

            # Left: Performance vs Number of Satellites
            r_sat = None
            axes[0].scatter(merged_df['num_satellites'], merged_df['performance'],
                            c=color, s=45, alpha=0.7, marker=marker,
                            edgecolors='black', linewidth=0.4)
            if len(merged_df) > 1:
                x = merged_df['num_satellites']
                y = merged_df['performance']
                from scipy.optimize import curve_fit
                def power_law(x, a, b):
                    return a * np.power(x, b)
                try:
                    popt, _ = curve_fit(power_law, x, y, maxfev=5000)
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    y_fit = power_law(x_fit, *popt)
                    y_fit_for_r = power_law(x, *popt)
                    r_sat = np.corrcoef(y, y_fit_for_r)[0, 1]
                    axes[0].plot(x_fit, y_fit, color=color, alpha=0.7, linewidth=2, linestyle=':')
                except Exception:
                    pass
            legend_label_sat = f"{label_base} (R={r_sat:.2f})" if r_sat is not None else label_base
            axes[0].plot([], [], color=color, marker=marker, linestyle='None', label=legend_label_sat)

            # Right: Performance vs Number of Planes
            r_planes = None
            axes[1].scatter(merged_df['P_planes'], merged_df['performance'],
                            c=color, s=45, alpha=0.7, marker=marker,
                            edgecolors='black', linewidth=0.4)
            if len(merged_df) > 1:
                x = merged_df['P_planes']
                y = merged_df['performance']
                from scipy.optimize import curve_fit
                def power_law(x, a, b):
                    return a * np.power(x, b)
                try:
                    popt, _ = curve_fit(power_law, x, y, maxfev=5000)
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    y_fit = power_law(x_fit, *popt)
                    y_fit_for_r = power_law(x, *popt)
                    r_planes = np.corrcoef(y, y_fit_for_r)[0, 1]
                    axes[1].plot(x_fit, y_fit, color=color, alpha=0.7, linewidth=2, linestyle=':')
                except Exception:
                    pass
            legend_label_planes = f"{label_base} (R={r_planes:.2f})" if r_planes is not None else label_base
            axes[1].plot([], [], color=color, marker=marker, linestyle='None', label=legend_label_planes)

        axes[0].set_xlabel('Number of Satellites')
        axes[0].set_ylabel('Performance Score')
        axes[0].set_title(f'Performance vs Satellites - {sweep_type.title()} Sweep')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Number of Planes (P)')
        axes[1].set_ylabel('Performance Score')
        axes[1].set_title(f'Performance vs Planes - {sweep_type.title()} Sweep')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/performance_vs_sat_and_planes_{sweep_type}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved performance_vs_sat_and_planes_{sweep_type}.png")

    def plot_cost_ratios(self, csv_names, sweep_type):
        """
        Plot cost ratios and absolute costs vs number of satellites.
        Left: Percentage breakdown of NRE, RE, Launch cost over total cost.
        Right: Absolute NRE, RE, Launch cost vs number of satellites.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        param_groups = {}
        for name in csv_names:
            meta = self.metadata[name]
            if sweep_type == 'altitude':
                param_val = meta.get('target_altitude')
            elif sweep_type == 'inclination':
                param_val = meta.get('target_inclination')
            elif sweep_type == 'mass':
                param_val = meta.get('sat_mass')
            elif sweep_type == 'dv':
                param_val = meta.get('dv_m_s')
            else:
                continue
            if param_val is None:
                continue
            param_groups.setdefault(param_val, []).append(name)

        cost_categories = [
            ('NRE_cost_musd', 'NRE', 'o', ':'),
            ('RE_cost_musd', 'RE', 's', '--'),
            ('launch_cost_musd', 'Launch', '^', '-.')
        ]

        # For legend handles
        sweep_handles = []
        cost_handles = []

        # Add cost type handles (black markers) only once
        for idx, (col, cat, marker, lstyle) in enumerate(cost_categories):
            handle = plt.Line2D([], [], color='black', marker=marker, linestyle='None', markersize=7, label=cat)
            cost_handles.append(handle)
        handle_total = plt.Line2D([], [], color='black', marker='X', linestyle='None', markersize=7, label='Total')
        cost_handles.append(handle_total)

        for param_idx, (param_val, csv_list) in enumerate(sorted(param_groups.items())):
            color = self.color_palette[param_idx % len(self.color_palette)]
            label_base = f"{sweep_type.title()} {param_val:.0f}" if param_val is not None else f"{sweep_type.title()}"
            merged_df = pd.concat([self.data_dict[name] for name in csv_list], ignore_index=True)
            nd_mask = self._nondominated_mask(merged_df)
            merged_df = merged_df[nd_mask]

            # Calculate percentages
            merged_df = merged_df.copy()
            total_cost = merged_df['NRE_cost_musd'] + merged_df['RE_cost_musd'] + merged_df['launch_cost_musd']
            merged_df['NRE_pct'] = 100 * merged_df['NRE_cost_musd'] / total_cost
            merged_df['RE_pct'] = 100 * merged_df['RE_cost_musd'] / total_cost
            merged_df['Launch_pct'] = 100 * merged_df['launch_cost_musd'] / total_cost

            # Add sweep type legend handle (color only, marker 'o')
            handle = plt.Line2D([], [], color=color, marker='o', linestyle='None', markersize=7, label=label_base)
            sweep_handles.append(handle)

            for col, cat, marker, lstyle in cost_categories:
                pct_col = f"{cat}_pct"
                axes[0].scatter(merged_df['num_satellites'], merged_df[pct_col],
                                c=color, s=45, alpha=0.35, marker=marker,
                                edgecolors='black', linewidth=0.4)
                # Trend line
                if len(merged_df) > 1:
                    x = merged_df['num_satellites']
                    y = merged_df[pct_col]
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    from scipy.optimize import curve_fit

                    if cat in ['NRE', 'RE']:
                        def power_law(x, a, b):
                            return a * np.power(x, b)
                        try:
                            popt, _ = curve_fit(power_law, x, y, maxfev=5000)
                            y_fit = power_law(x_fit, *popt)
                            axes[0].plot(x_fit, y_fit, color=color, linestyle=lstyle, linewidth=2, alpha=0.7)
                        except Exception:
                            pass
                    else:
                        # Linear fit for Launch cost percentage
                        slope, intercept, r, _, _ = linregress(x, y)
                        y_fit = slope * x_fit + intercept
                        axes[0].plot(x_fit, y_fit, color=color, linestyle=lstyle, linewidth=2, alpha=0.7)

            # Right subplot: absolute costs
            for col, cat, marker, lstyle in cost_categories:
                axes[1].scatter(merged_df['num_satellites'], merged_df[col],
                                c=color, s=45, alpha=0.35, marker=marker,
                                edgecolors='black', linewidth=0.4)
                # Trend line
                if len(merged_df) > 1:
                    x = merged_df['num_satellites']
                    y = merged_df[col]
                    slope, intercept, r, _, _ = linregress(x, y)
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    y_fit = slope * x_fit + intercept
                    axes[1].plot(x_fit, y_fit, color=color, linestyle=lstyle, linewidth=2, alpha=0.7)

            # Absolute total cost (marker 'X'), only once per sweep value
            merged_df['total_cost'] = (
                merged_df['NRE_cost_musd'] + merged_df['RE_cost_musd'] + merged_df['launch_cost_musd']
            )
            axes[1].scatter(
                merged_df['num_satellites'], merged_df['total_cost'],
                c=color, s=55, alpha=0.35, marker='X', edgecolors='black', linewidth=0.4
            )
            if len(merged_df) > 1:
                x = merged_df['num_satellites']
                y = merged_df['total_cost']
                slope, intercept, r, _, _ = linregress(x, y)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = slope * x_fit + intercept
                axes[1].plot(x_fit, y_fit, color=color, linestyle='-', linewidth=2, alpha=0.7)

        axes[0].set_xlabel('Number of Satellites')
        axes[0].set_ylabel('Cost Percentage (%)')
        axes[0].set_ylim(0, 100)  # Limit y-axis to 0–100%
        axes[0].set_yticks([0, 2.5, 5, 7.5, 10, 15, 20, 25, 50, 75, 100])  # Example: set ticks at 0, 20, ..., 100
        axes[0].set_title('NRE / RE / Launch Cost Percentage vs Number of Satellites')
        axes[0].grid(True, alpha=0.3)
        # Legend: sweep type (color) + cost type (black marker)
        axes[0].legend(handles=sweep_handles + cost_handles[:-1], loc='best', fontsize=10)

        axes[1].set_xlabel('Number of Satellites')
        axes[1].set_ylabel('Cost (M$ FY2025)')
        axes[1].set_yscale('log')
        custom_ticks = [0.3, 0.5, 1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 5000, 10000]
        axes[1].set_yticks(custom_ticks)
        axes[1].set_yticklabels([str(tick) for tick in custom_ticks])
        # axes[1].yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        axes[1].yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(1, 10), numticks=12))
        # axes[1].yaxis.set_major_formatter(mticker.ScalarFormatter())
        axes[1].set_title('Absolute NRE / RE / Launch Cost vs Number of Satellites')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(handles=sweep_handles + cost_handles, loc='best', fontsize=10)

        plt.suptitle(f'Cost Ratios and Breakdown - {sweep_type.title()} Sweep', fontsize=15)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cost_ratios_{sweep_type}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved cost_ratios_{sweep_type}.png")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Compare multiple optimization CSV results')
    parser.add_argument('csv_dir', help='Directory containing CSV files OR comma-separated list of CSV paths')
    parser.add_argument('--output_dir', default='comparison_plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Check if input is a directory or list of files
    if ',' in args.csv_dir:
        csv_input = [path.strip() for path in args.csv_dir.split(',')]
    else:
        csv_input = args.csv_dir
    
    # Create comparator and generate plots
    print(f"Processing CSVs from: {csv_input}")
    comparator = MultiCSVComparator(csv_input, args.output_dir)
    comparator.generate_all_plots()
    
    print("\nComparison complete!")

if __name__ == "__main__":
    # =====================================================================
    # MANUAL CONTROL EXAMPLES
    # =====================================================================
    # This section shows different ways to run the comparator with manual control.
    # Comment/uncomment the examples you want to run.
    
    # ---------------------------------------------------------------------
    # Option 1: Command line interface
    # ---------------------------------------------------------------------
    # Uncomment to use command-line arguments:
    # python csv_compare.py "results/MOEA/walker-delta/p320_g120/sweep_alt"
    # python csv_compare.py "path/to/file1.csv,path/to/file2.csv,path/to/file3.csv"
    # main()
    
    # ---------------------------------------------------------------------
    # Option 2: Single sweep from specific seed/config
    # ---------------------------------------------------------------------
    # # Process altitude sweep from a single configuration/seed
    # csv_dir_altitude = "results/MOEA/walker-delta/p320_g120_sd11/sweep_alt"
    # comparator = MultiCSVComparator(csv_dir_altitude, "comparison_plots")
    # comparator.generate_all_plots()

    # ---------------------------------------------------------------------
    # Option 3: Other single-seed sweeps
    # ---------------------------------------------------------------------
    # # Inclination sweep
    # csv_dir_inclination = "results/MOEA/walker-delta/p320_g120_sd11/sweep_inc"
    # comparator = MultiCSVComparator(csv_dir_inclination, "comparison_plots")
    # comparator.generate_all_plots()

    # # Mass sweep (will use log-scale Pareto)
    # csv_dir_mass = "results/MOEA/walker-delta/p320_g120_sd11/sweep_mass"
    # comparator = MultiCSVComparator(csv_dir_mass, "comparison_plots")
    # comparator.generate_all_plots()

    # # Deployment ΔV sweep
    # csv_dir_vdep = "results/MOEA/walker-delta/p320_g120_sd11/sweep_vdep"
    # comparator = MultiCSVComparator(csv_dir_vdep, "comparison_plots")
    # comparator.generate_all_plots()

    # ---------------------------------------------------------------------
    # Option 4: Multi-seed analysis (recursive loading)
    # ---------------------------------------------------------------------
    # Point to parent folder containing multiple seed folders.
    # Will automatically find and merge CSVs from all seeds:
    # e.g., p320_g120_sd11/sweep_mass/, p320_g120_sd42/sweep_mass/, etc.
    
    # # Multi-seed mass sweep (merges across all seeds)
    # csv_dir_multiseed = "results/MOEA/walker-delta/p320_g120/sweep_mass"
    # comparator = MultiCSVComparator(csv_dir_multiseed, "comparison_plots")
    # comparator.generate_all_plots()  # Includes individual + merged Pareto
    
    # # Or point to pattern level to get all configs/seeds
    # csv_dir_all = "results/MOEA/walker-delta/"
    # comparator = MultiCSVComparator(csv_dir_all, "comparison_plots")
    # comparator.generate_all_plots()


    # ---------------------------------------------------------------------
    # Option 5: Manual list of specific CSV files
    # ---------------------------------------------------------------------
    # Provide exact file paths as a list
    # List all CSVs for altitude sweep (3 folders × 3 altitudes = 9 files)
    csv_files_altitude = [
        # p480_g50_sd11
        "results/MOEA/walker-delta/p480_g50_sd11/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1500_sd11_m190_a500_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p480_g50_sd11/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1500_sd11_m190_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p480_g50_sd11/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1500_sd11_m190_a1000_i50_dv200_T180d.csv",
        # p320_g120_sd42
        "results/MOEA/walker-delta/p320_g120_sd42/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p320_g120_sp1500_sd42_m190_a500_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p320_g120_sd42/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p320_g120_sp1500_sd42_m190_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p320_g120_sd42/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p320_g120_sp1500_sd42_m190_a1000_i50_dv200_T180d.csv",
        # p160_g81_sd42
        "results/MOEA/walker-delta/p160_g81_sd42/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p160_g81_sp1500_sd42_m190_a500_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p160_g81_sd42/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p160_g81_sp1500_sd42_m190_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p160_g81_sd42/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p160_g81_sp1500_sd42_m190_a1000_i50_dv200_T180d.csv",
        # p480_g50_sd12 step 1000 
        "results/MOEA/walker-delta/p480_g50_sd12/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1000_sd12_m190_a500_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p480_g50_sd12/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1000_sd12_m190_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p480_g50_sd12/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1000_sd12_m190_a1000_i50_dv200_T180d.csv",
        # p480_g50_sd12 step 500
        "results/MOEA/walker-delta/p480_g50_sd12/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp500_sd12_m190_a500_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p480_g50_sd12/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp500_sd12_m190_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p480_g50_sd12/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp500_sd12_m190_a1000_i50_dv200_T180d.csv",
        # p500_g50_sd12 step 500
        "results/MOEA/walker-delta/p500_g50_sd12/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p500_g50_sp500_sd12_m190_a500_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p500_g50_sd12/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p500_g50_sp500_sd12_m190_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p500_g50_sd12/sweep_alt/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p500_g50_sp500_sd12_m190_a1000_i50_dv200_T180d.csv",
    ]
    comparator = MultiCSVComparator(csv_files_altitude, "comparison_plots/manual_altitude")
    comparator.generate_all_plots()
        
    csv_files_inclination = [
        # p480_g50_sd11
        "results/MOEA/walker-delta/p480_g50_sd11/sweep_inc/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1500_sd11_m190_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p480_g50_sd11/sweep_inc/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1500_sd11_m190_a700_i60_dv200_T180d.csv",
        "results/MOEA/walker-delta/p480_g50_sd11/sweep_inc/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1500_sd11_m190_a700_i98_dv200_T180d.csv",
        # p320_g120_sd42
        "results/MOEA/walker-delta/p320_g120_sd42/sweep_inc/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p320_g120_sp1500_sd42_m190_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p320_g120_sd42/sweep_inc/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p320_g120_sp1500_sd42_m190_a700_i60_dv200_T180d.csv",
        "results/MOEA/walker-delta/p320_g120_sd42/sweep_inc/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p320_g120_sp1500_sd42_m190_a700_i98_dv200_T180d.csv",
        # p160_g81_sd42
        "results/MOEA/walker-delta/p160_g81_sd42/sweep_inc/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p160_g81_sp1500_sd42_m190_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p160_g81_sd42/sweep_inc/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p160_g81_sp1500_sd42_m190_a700_i60_dv200_T180d.csv",
        "results/MOEA/walker-delta/p160_g81_sd42/sweep_inc/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p160_g81_sp1500_sd42_m190_a700_i98_dv200_T180d.csv",
        # p480_g50_sd12
        "results/MOEA/walker-delta/p480_g50_sd12/sweep_inc/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1000_sd12_m190_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p480_g50_sd12/sweep_inc/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1000_sd12_m190_a700_i60_dv200_T180d.csv",
        "results/MOEA/walker-delta/p480_g50_sd12/sweep_inc/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1000_sd12_m190_a700_i98_dv200_T180d.csv",
        # p500_g50_sd12 step 500
        "results/MOEA/walker-delta/p500_g50_sd12/sweep_inc/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p500_g50_sp500_sd12_m190_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p500_g50_sd12/sweep_inc/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p500_g50_sp500_sd12_m190_a700_i60_dv200_T180d.csv",
        "results/MOEA/walker-delta/p500_g50_sd12/sweep_inc/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p500_g50_sp500_sd12_m190_a700_i98_dv200_T180d.csv",
    ]
    comparator = MultiCSVComparator(csv_files_inclination, "comparison_plots/manual_inclination")
    comparator.generate_all_plots()

    csv_files_mass = [
        # p480_g50_sd11
        "results/MOEA/walker-delta/p480_g50_sd11/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1500_sd11_m190_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p480_g50_sd11/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1500_sd11_m370_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p480_g50_sd11/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1500_sd11_m760_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p480_g50_sd11/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1500_sd11_m95_a700_i50_dv200_T180d.csv",
        # p320_g120_sd42
        "results/MOEA/walker-delta/p320_g120_sd42/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p320_g120_sp1500_sd42_m190_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p320_g120_sd42/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p320_g120_sp1500_sd42_m370_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p320_g120_sd42/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p320_g120_sp1500_sd42_m760_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p320_g120_sd42/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p320_g120_sp1500_sd42_m95_a700_i50_dv200_T180d.csv",
        # p160_g81_sd42
        "results/MOEA/walker-delta/p160_g81_sd42/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p160_g81_sp1500_sd42_m190_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p160_g81_sd42/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p160_g81_sp1500_sd42_m370_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p160_g81_sd42/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p160_g81_sp1500_sd42_m760_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p160_g81_sd42/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p160_g81_sp1500_sd42_m95_a700_i50_dv200_T180d.csv",
        # p480_g50_sd12
        "results/MOEA/walker-delta/p480_g50_sd12/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1000_sd12_m190_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p480_g50_sd12/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1000_sd12_m370_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p480_g50_sd12/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1000_sd12_m760_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p480_g50_sd12/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p480_g50_sp1000_sd12_m95_a700_i50_dv200_T180d.csv",
        # p500_g50_sd12 step 500
        "results/MOEA/walker-delta/p500_g50_sd12/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p500_g50_sp500_sd12_m190_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p500_g50_sd12/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p500_g50_sp500_sd12_m370_a700_i50_dv200_T180d.csv",
        # "results/MOEA/walker-delta/p500_g50_sd12/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p500_g50_sp500_sd12_m760_a700_i50_dv200_T180d.csv",
        "results/MOEA/walker-delta/p500_g50_sd12/sweep_mass/pareto_solutions_walker-delta_w0.0-0.0-0.9-0.7_p500_g50_sp500_sd12_m95_a700_i50_dv200_T180d.csv",
    ]
    comparator = MultiCSVComparator(csv_files_mass, "comparison_plots/manual_mass")
    comparator.generate_all_plots()

    # ---------------------------------------------------------------------
    # Option 6: Selective plotting
    # ---------------------------------------------------------------------
    # # Run only specific plots instead of generate_all_plots()
    # csv_dir = "results/MOEA/walker-delta/p320_g120_sd11/sweep_inc"
    # comparator = MultiCSVComparator(csv_dir, "comparison_plots")
    # comparator.plot_3d_comparison()
    # comparator.plot_pareto_comparison()
    # # comparator.plot_merged_pareto_multiseed()  # Only if multi-seed
    # comparator.plot_cost_components_comparison()
    
    print("\nComparison complete!")