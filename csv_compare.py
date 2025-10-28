"""
Multi-CSV Comparison Script for Constellation Optimization Results

This script loads multiple optimization CSVs (from parameter sweeps) and generates
comparative plots for trade-off analysis.

Useful plots:
- 3D scatter comparing altitude/inclination sweeps
- Cost vs performance (Pareto) with trend lines across sweeps
- Cost components vs satellites with trends for different sweeps
- Deployment constraints (RAAN spacing, launch groups vs planes)
- Launch cost analysis

Author: Fabio Meloni - TU Delft MSc Student - NASA JPL Visiting Researcher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        """Load all CSV files from directory or list."""
        if isinstance(self.csv_dir, list):
            csv_files = self.csv_dir
        else:
            # Prefer population-level CSVs if available, otherwise fall back to Pareto-only
            pop_candidates = glob(os.path.join(self.csv_dir, "population_solutions_*.csv"))
            if pop_candidates:
                csv_files = sorted(pop_candidates)
            else:
                pareto_candidates = glob(os.path.join(self.csv_dir, "pareto_solutions_*.csv"))
                csv_files = sorted(pareto_candidates)
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.csv_dir}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_path in csv_files:
            csv_name = Path(csv_path).stem
            print(f"Loading: {csv_name}")
            
            df = pd.read_csv(csv_path)
            
            # Extract metadata from columns, filename, and parent folder
            metadata = self._extract_metadata(df, csv_path)
            
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
        import re
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
                self.plot_pareto_comparison(names, sweep_type)
                self.plot_pareto_comparison_colored(names, sweep_type)
                # Extra: for mass sweep, also provide a log-scale cost variant
                if sweep_type == 'mass':
                    self.plot_pareto_comparison_log_cost(names, sweep_type)
        
        # 3. Cost components vs satellites
        for sweep_type, names in sweep_groups.items():
            if len(names) > 1:
                self.plot_cost_components_comparison(names, sweep_type)
        
        # 4. Deployment constraints
        for sweep_type, names in sweep_groups.items():
            if len(names) > 1:
                self.plot_deployment_comparison(names, sweep_type)
        
        # 5. Launch cost analysis
        for sweep_type, names in sweep_groups.items():
            if len(names) > 1:
                self.plot_launch_cost_comparison(names, sweep_type)
        
        print(f"\nAll comparison plots saved to: {self.output_dir}/")
    
    def plot_3d_altitude_comparison(self, csv_names):
        """3D scatter: Cost vs Performance vs Satellites, colored by target altitude."""
        fig = plt.figure(figsize=(16, 6))
        
        # Plot 1: Cost vs Performance vs Satellites (colored by target altitude)
        ax1 = fig.add_subplot(131, projection='3d')
        
        for i, name in enumerate(csv_names):
            df = self.data_dict[name]
            meta = self.metadata[name]
            color = self.color_palette[i % len(self.color_palette)]
            marker = self.marker_styles[i % len(self.marker_styles)]

            ax1.scatter(df['cost_musd'], df['performance'], df['num_satellites'],
                        c=color, s=60, alpha=0.7, marker=marker,
                        edgecolors='black', linewidth=0.5,
                        label=meta['label'])
        
        ax1.set_xlabel('Cost (M$)')
        ax1.set_ylabel('Performance')
        ax1.set_zlabel('Number of Satellites')
        ax1.set_title('Altitude Sweep Comparison\nCost vs Performance vs Satellites')
        ax1.legend()
        
        # Plot 2: Cost vs Satellites vs Altitude (colored by performance)
        ax2 = fig.add_subplot(132, projection='3d')
        
        for i, name in enumerate(csv_names):
            df = self.data_dict[name]
            meta = self.metadata[name]
            target_alt = meta['target_altitude']
            marker = self.marker_styles[i % len(self.marker_styles)]

            scatter = ax2.scatter(df['cost_musd'], df['num_satellites'],
                                  [target_alt] * len(df),
                                  c=df['performance'], s=60, alpha=0.7, marker=marker,
                                  cmap='plasma', vmin=0, vmax=1,
                                  edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('Cost (M$)')
        ax2.set_ylabel('Number of Satellites')
        ax2.set_zlabel('Target Altitude (km)')
        ax2.set_title('Cost vs Satellites vs Target Altitude\n(colored by performance)')
        plt.colorbar(scatter, ax=ax2, label='Performance', shrink=0.5)
        
        # Plot 3: Performance vs Altitude vs Satellites
        ax3 = fig.add_subplot(133, projection='3d')
        
        for i, name in enumerate(csv_names):
            df = self.data_dict[name]
            meta = self.metadata[name]
            target_alt = meta['target_altitude']
            color = self.color_palette[i % len(self.color_palette)]
            marker = self.marker_styles[i % len(self.marker_styles)]

            ax3.scatter([target_alt] * len(df), df['performance'], df['num_satellites'],
                        c=color, s=60, alpha=0.7, marker=marker,
                        edgecolors='black', linewidth=0.5,
                        label=meta['label'])
        
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
        """3D scatter: Cost vs Performance vs Satellites, colored by target inclination."""
        fig = plt.figure(figsize=(16, 6))
        
        # Plot 1: Cost vs Performance vs Satellites (colored by target inclination)
        ax1 = fig.add_subplot(131, projection='3d')
        
        for i, name in enumerate(csv_names):
            df = self.data_dict[name]
            meta = self.metadata[name]
            color = self.color_palette[i % len(self.color_palette)]
            marker = self.marker_styles[i % len(self.marker_styles)]

            ax1.scatter(df['cost_musd'], df['performance'], df['num_satellites'],
                        c=color, s=60, alpha=0.7, marker=marker,
                        edgecolors='black', linewidth=0.5,
                        label=meta['label'])
        
        ax1.set_xlabel('Cost (M$)')
        ax1.set_ylabel('Performance')
        ax1.set_zlabel('Number of Satellites')
        ax1.set_title('Inclination Sweep Comparison\nCost vs Performance vs Satellites')
        ax1.legend()
        
        # Plot 2: Cost vs Satellites vs Inclination (colored by performance)
        ax2 = fig.add_subplot(132, projection='3d')
        
        for i, name in enumerate(csv_names):
            df = self.data_dict[name]
            meta = self.metadata[name]
            target_inc = meta['target_inclination']
            marker = self.marker_styles[i % len(self.marker_styles)]

            scatter = ax2.scatter(df['cost_musd'], df['num_satellites'],
                                  [target_inc] * len(df),
                                  c=df['performance'], s=60, alpha=0.7, marker=marker,
                                  cmap='coolwarm', vmin=0, vmax=1,
                                  edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('Cost (M$)')
        ax2.set_ylabel('Number of Satellites')
        ax2.set_zlabel('Target Inclination (deg)')
        ax2.set_title('Cost vs Satellites vs Target Inclination\n(colored by performance)')
        plt.colorbar(scatter, ax=ax2, label='Performance', shrink=0.5)
        
        # Plot 3: Performance vs Inclination vs Satellites
        ax3 = fig.add_subplot(133, projection='3d')
        
        for i, name in enumerate(csv_names):
            df = self.data_dict[name]
            meta = self.metadata[name]
            target_inc = meta['target_inclination']
            color = self.color_palette[i % len(self.color_palette)]
            marker = self.marker_styles[i % len(self.marker_styles)]

            ax3.scatter([target_inc] * len(df), df['performance'], df['num_satellites'],
                        c=color, s=60, alpha=0.7, marker=marker,
                        edgecolors='black', linewidth=0.5,
                        label=meta['label'])
        
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

        plt.xlabel('Total Cost (M$)', fontsize=12)
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
            
            axes[i].set_xlabel('Total Cost (M$)')
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

        # Use log scale with readable money ticks (M$ / B$)
        from matplotlib.ticker import FuncFormatter, LogLocator
        ax = plt.gca()
        ax.set_xscale('log')
        # Major ticks at 1, 2, 5 within each decade
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))

        def _fmt_musd(x, pos):
            # x is in millions of dollars (M$)
            import numpy as _np
            if x is None or not _np.isfinite(x) or x <= 0:
                return ""
            if x >= 1000:  # 1000 M$ = 1 B$
                val_b = x / 1000.0
                # Show one decimal when needed
                return (f"{val_b:.0f} B$" if abs(val_b - round(val_b)) < 1e-6 else f"{val_b:.1f} B$")
            else:
                return f"{x:.0f} M$"

        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_musd))
        plt.xlabel('Total Cost (M$)')
        plt.ylabel('Performance Score')
        plt.title(f'All Solutions + Pareto Front (Log Cost) - {sweep_type.title()} Sweep')
        plt.legend()
        plt.grid(True, which='both', axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pareto_comparison_{sweep_type}_logx.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved pareto_comparison_{sweep_type}_logx.png")
    
    def plot_cost_components_comparison(self, csv_names, sweep_type):
        """Cost components (RE, Launch) vs satellites with trends."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        components = [
            ('RE_cost_musd', 'RE Cost', axes[0]),
            ('launch_cost_musd', 'Launch Cost', axes[1])
        ]
        
        for comp_col, comp_title, ax in components:
            for i, name in enumerate(csv_names):
                df = self.data_dict[name]
                meta = self.metadata[name]
                color = self.color_palette[i % len(self.color_palette)]
                marker = self.marker_styles[i % len(self.marker_styles)]

                # Scatter
                ax.scatter(df['num_satellites'], df[comp_col],
                           c=color, s=45, alpha=0.75, marker=marker,
                           edgecolors='black', linewidth=0.4,
                           label=meta['label'])
                
                # Trend line
                if len(df) > 1:
                    slope, intercept, r_value, _, _ = linregress(df['num_satellites'], df[comp_col])
                    x_fit = np.array([df['num_satellites'].min(), df['num_satellites'].max()])
                    y_fit = slope * x_fit + intercept
                    ax.plot(x_fit, y_fit, '--', color=color, linewidth=2, alpha=0.8)
                    
                    # Add r-value as text
                    mid_x = (df['num_satellites'].min() + df['num_satellites'].max()) / 2
                    mid_y = slope * mid_x + intercept
                    ax.text(mid_x, mid_y, f'r={r_value:.2f}', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor=color, alpha=0.25))
            
            ax.set_xlabel('Number of Satellites')
            ax.set_ylabel(f'{comp_title} (M$)')
            ax.set_title(f'{comp_title} vs Satellites - {sweep_type.title()} Sweep')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cost_components_{sweep_type}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved cost_components_{sweep_type}.png")
    
    def plot_deployment_comparison(self, csv_names, sweep_type):
        """Deployment constraints: RAAN spacing, launch groups vs planes."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for i, name in enumerate(csv_names):
            df = self.data_dict[name]
            meta = self.metadata[name]
            color = self.color_palette[i % len(self.color_palette)]
            marker = self.marker_styles[i % len(self.marker_styles)]
            
            # 1. Launch groups vs number of planes
            axes[0, 0].scatter(df['P_planes'], df['launch_num_groups'],
                               c=color, s=55, alpha=0.7, marker=marker,
                               edgecolors='black', linewidth=0.4,
                               label=meta['label'])
            
            # 2. Required RAAN spacing vs launch groups
            axes[0, 1].scatter(df['launch_num_groups'], df['launch_delta_raan_required_deg'],
                               c=color, s=55, alpha=0.7, marker=marker,
                               edgecolors='black', linewidth=0.4,
                               label=meta['label'])
            
            # 3. Phasing factor F vs number of launches
            axes[1, 0].scatter(df['F_phasing'], df['launch_num_launches'],
                               c=color, s=55, alpha=0.7, marker=marker,
                               edgecolors='black', linewidth=0.4,
                               label=meta['label'])
            
            # 4. Launch cost vs number of planes (new)
            axes[1, 1].scatter(df['P_planes'], df['launch_cost_musd'],
                               c=color, s=55, alpha=0.7, marker=marker,
                               edgecolors='black', linewidth=0.4,
                               label=meta['label'])
        
        # Configure axes
        axes[0, 0].set_xlabel('Number of Planes (P)')
        axes[0, 0].set_ylabel('Number of Launch Groups')
        axes[0, 0].set_title('Launch Groups vs Planes')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Number of Launch Groups')
        axes[0, 1].set_ylabel('Required RAAN Spacing (deg)')
        axes[0, 1].set_title('RAAN Spacing vs Launch Groups')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Phasing Factor (F)')
        axes[1, 0].set_ylabel('Number of Launches')
        axes[1, 0].set_title('Launches vs Phasing Factor')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Number of Planes (P)')
        axes[1, 1].set_ylabel('Launch Cost (M$)')
        axes[1, 1].set_title('Launch Cost vs Number of Planes')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Deployment Constraints - {sweep_type.title()} Sweep', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/deployment_{sweep_type}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved deployment_{sweep_type}.png")
    
    def plot_launch_cost_comparison(self, csv_names, sweep_type):
        """Launch cost analysis: cost vs ΔV, launches vs planes vs satellites."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for i, name in enumerate(csv_names):
            df = self.data_dict[name]
            meta = self.metadata[name]
            color = self.color_palette[i % len(self.color_palette)]
            marker = self.marker_styles[i % len(self.marker_styles)]
            
            # 1. Launch cost vs deployment ΔV
            axes[0].scatter(df['launch_dv_used_m_s'], df['launch_cost_musd'],
                            c=color, s=55, alpha=0.7, marker=marker,
                            edgecolors='black', linewidth=0.4,
                            label=meta['label'])
            
            # Add correlation
            if len(df) > 1:
                corr, _ = pearsonr(df['launch_dv_used_m_s'], df['launch_cost_musd'])
                # Add text in legend area
                label_text = f"{meta['label']} (r={corr:.2f})"
                axes[0].plot([], [], ' ', label=label_text)
            
            # 2. Launches vs planes (sized by satellites)
            scatter = axes[1].scatter(df['P_planes'], df['launch_num_launches'],
                                      s=df['num_satellites']*2, alpha=0.55,
                                      c=color, marker=marker,
                                      edgecolors='black', linewidth=0.5,
                                      label=meta['label'])
        
        axes[0].set_xlabel('Deployment ΔV Used (m/s)')
        axes[0].set_ylabel('Launch Cost (M$)')
        axes[0].set_title('Launch Cost vs Deployment ΔV')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Number of Planes (P)')
        axes[1].set_ylabel('Number of Launches')
        axes[1].set_title('Launches vs Planes\n(marker size = number of satellites)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Launch Cost Analysis - {sweep_type.title()} Sweep', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/launch_cost_{sweep_type}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved launch_cost_{sweep_type}.png")


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
    # Option 1: Use command line (uncomment)
    # main()
    
    # Option 2: Hardcode paths
    
    # Process altitude sweep
    csv_dir_altitude = "results/MOEA/walker-delta/p320_g120/sweep_alt"
    comparator = MultiCSVComparator(csv_dir_altitude, "comparison_plots")
    comparator.generate_all_plots()

    # Example: Inclination sweep
    csv_dir_inclination = "results/MOEA/walker-delta/p320_g120/sweep_inc"
    comparator = MultiCSVComparator(csv_dir_inclination, "comparison_plots")
    comparator.generate_all_plots()

    # Example: Mass sweep
    csv_dir_mass = "results/MOEA/walker-delta/p320_g120/sweep_mass"
    comparator = MultiCSVComparator(csv_dir_mass, "comparison_plots")
    comparator.generate_all_plots()

    # Vdep sweep
    csv_dir_vdep = "results/MOEA/walker-delta/p320_g120/sweep_vdep"  # or ".../sweep_dv"
    comparator = MultiCSVComparator(csv_dir_vdep, "comparison_plots")
    comparator.generate_all_plots()

    
    print("\nComparison complete!")