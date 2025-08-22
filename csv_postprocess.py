"""
CSV Postprocessing Script for Multi-Objective Optimization Results

This script loads optimization results from CSV files and generates various plots
analyzing relationships between design parameters, costs, and performance metrics.

Author: Fabio Meloni - TU Delft MSc Student - NASA JPL Visiting Researcher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import argparse
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CSVPostProcessor:
    """
    Post-processor for multi-objective optimization results.
    
    Loads CSV data and generates comprehensive analysis plots.
    """
    
    def __init__(self, csv_filepath, output_dir="postprocessing"):
        """
        Initialize the post-processor.
        
        Parameters:
        -----------
        csv_filepath : str
            Path to the CSV file containing optimization results
        output_dir : str
            Directory to save generated plots
        """
        self.csv_filepath = csv_filepath
        self.output_dir = output_dir
        self.data = None
        self.pattern_type = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and process data
        self.load_data()
        self.detect_pattern_type()
        
    def load_data(self):
        """Load CSV data and perform basic preprocessing."""
        print(f"Loading data from: {self.csv_filepath}")
        self.data = pd.read_csv(self.csv_filepath)
        
        print(f"Loaded {len(self.data)} solutions")
        print(f"Columns: {list(self.data.columns)}")
        
        # Basic data validation
        if self.data.empty:
            raise ValueError("CSV file is empty")
            
        # Convert any string 'inf' or 'nan' to numeric
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
        # Remove any rows with NaN in critical columns
        critical_cols = ['cost_musd', 'performance', 'num_satellites']
        self.data = self.data.dropna(subset=critical_cols)
        
        print(f"After cleaning: {len(self.data)} solutions")
        
        # Print altitude distribution for diagnostics
        if 'perigee_height_km' in self.data.columns:
            print(f"\nPERIGEE ALTITUDE (actual orbital altitude) distribution:")
            print(f"  Min perigee altitude: {self.data['perigee_height_km'].min():.1f} km")
            print(f"  Max perigee altitude: {self.data['perigee_height_km'].max():.1f} km")
            print(f"  Mean perigee altitude: {self.data['perigee_height_km'].mean():.1f} km")
            if 'optimization_params_target_altitude' in self.data.columns:
                print(f"  Target altitude: {self.data['optimization_params_target_altitude'].iloc[0]} km")
            
            # Count solutions by altitude ranges
            alt_ranges = pd.cut(self.data['perigee_height_km'], bins=[0, 300, 400, 500, 600, 1000], 
                              labels=['<300', '300-400', '400-500', '500-600', '>600'])
            print(f"  Solutions by perigee altitude range:")
            for range_name, count in alt_ranges.value_counts().items():
                print(f"    {range_name} km: {count} solutions")
        
        if 'altitude_km' in self.data.columns:
            print(f"\nALTITUDE_KM (semi-major axis - Earth radius, NOT actual altitude!):")
            print(f"  Min altitude_km: {self.data['altitude_km'].min():.1f} km")
            print(f"  Max altitude_km: {self.data['altitude_km'].max():.1f} km")
            print(f"  Mean altitude_km: {self.data['altitude_km'].mean():.1f} km")
            print(f"  NOTE: This is NOT the actual orbital altitude for elliptical orbits!")
                
        # Optional: Filter to solutions near target altitude (uncomment to use)
        # target_alt = self.data['optimization_params_target_altitude'].iloc[0]
        # tolerance = 50  # km
        # mask = abs(self.data['altitude_km'] - target_alt) <= tolerance
        # print(f"\nFiltering to solutions within {tolerance} km of target altitude...")
        # print(f"  Before filter: {len(self.data)} solutions")
        # self.data = self.data[mask]
        # print(f"  After filter: {len(self.data)} solutions")
        
    def detect_pattern_type(self):
        """Detect constellation pattern type from column names."""
        if 'P_planes' in self.data.columns and 'F_phasing' in self.data.columns:
            self.pattern_type = "Walker-Delta"
        elif 'Np' in self.data.columns and 'Nd' in self.data.columns:
            self.pattern_type = "Flower"
        else:
            self.pattern_type = "Unknown"
        
        print(f"Detected pattern type: {self.pattern_type}")
    
    def generate_all_plots(self):
        """Generate all analysis plots."""
        print("\nGenerating plots...")
        
        # 1. Basic relationship plots
        self.plot_cost_vs_performance()
        self.plot_cost_vs_satellites()
        self.plot_performance_vs_satellites()
        
        # 2. Design parameter analysis
        self.plot_performance_vs_altitude()
        self.plot_altitude_trade_offs()  # New detailed altitude analysis
        self.plot_performance_vs_inclination()
        self.plot_cost_vs_altitude()
        self.plot_cost_vs_inclination()
        
        # 3. Cost breakdown analysis
        self.plot_cost_breakdown()
        self.plot_cost_components_vs_satellites()
        
        # 4. Performance component analysis
        self.plot_performance_breakdown()
        self.plot_coverage_analysis()
        
        # 5. Pattern-specific analysis
        if self.pattern_type == "Walker-Delta":
            self.plot_walker_delta_analysis()
        elif self.pattern_type == "Flower":
            self.plot_flower_analysis()
            
        # 6. Advanced analysis
        self.plot_correlation_matrix()
        self.plot_pareto_front_colored()
        self.plot_3d_analysis()
        
        # 7. Summary statistics
        self.generate_summary_statistics()
        
        print(f"\nAll plots saved to: {self.output_dir}/")
        
    def plot_cost_vs_performance(self):
        """Plot the Pareto front: cost vs performance."""
        plt.figure(figsize=(10, 8))
        
        # Color by number of satellites
        scatter = plt.scatter(self.data['cost_musd'], self.data['performance'], 
                            c=self.data['num_satellites'], s=50, alpha=0.7, 
                            cmap='viridis', edgecolors='black', linewidth=0.5)
        
        plt.colorbar(scatter, label='Number of Satellites')
        plt.xlabel('Total Cost (M$)', fontsize=12)
        plt.ylabel('Performance Score', fontsize=12)
        plt.title('Pareto Front: Cost vs Performance\n(colored by number of satellites)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr, p_value = pearsonr(self.data['cost_musd'], self.data['performance'])
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cost_vs_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_cost_vs_satellites(self):
        """Plot cost vs number of satellites."""
        plt.figure(figsize=(12, 8))
        
        # Main scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(self.data['num_satellites'], self.data['cost_musd'], 
                   alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        plt.xlabel('Number of Satellites')
        plt.ylabel('Total Cost (M$)')
        plt.title('Total Cost vs Number of Satellites')
        plt.grid(True, alpha=0.3)
        
        # Cost per satellite
        plt.subplot(2, 2, 2)
        cost_per_sat = self.data['cost_musd'] / self.data['num_satellites']
        plt.scatter(self.data['num_satellites'], cost_per_sat, 
                   alpha=0.6, s=30, color='orange', edgecolors='black', linewidth=0.3)
        plt.xlabel('Number of Satellites')
        plt.ylabel('Cost per Satellite (M$)')
        plt.title('Cost per Satellite vs Number of Satellites')
        plt.grid(True, alpha=0.3)
        
        # Histogram of satellite numbers
        plt.subplot(2, 2, 3)
        plt.hist(self.data['num_satellites'], bins=20, alpha=0.7, color='skyblue', 
                edgecolor='black')
        plt.xlabel('Number of Satellites')
        plt.ylabel('Frequency')
        plt.title('Distribution of Satellite Numbers')
        plt.grid(True, alpha=0.3)
        
        # Cost components vs satellites
        plt.subplot(2, 2, 4)
        plt.scatter(self.data['num_satellites'], self.data['NRE_cost_musd'], 
                   alpha=0.6, s=20, label='NRE Cost', color='red')
        plt.scatter(self.data['num_satellites'], self.data['RE_cost_musd'], 
                   alpha=0.6, s=20, label='RE Cost', color='blue')
        plt.scatter(self.data['num_satellites'], self.data['launch_cost_musd'], 
                   alpha=0.6, s=20, label='Launch Cost', color='green')
        plt.xlabel('Number of Satellites')
        plt.ylabel('Cost Component (M$)')
        plt.title('Cost Components vs Number of Satellites')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cost_vs_satellites.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_performance_vs_satellites(self):
        """Plot performance vs number of satellites."""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(self.data['num_satellites'], self.data['performance'], 
                   alpha=0.6, s=30, color='purple', edgecolors='black', linewidth=0.3)
        plt.xlabel('Number of Satellites')
        plt.ylabel('Performance Score')
        plt.title('Performance vs Number of Satellites')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.data['num_satellites'], self.data['performance'], 1)
        p = np.poly1d(z)
        plt.plot(sorted(self.data['num_satellites']), p(sorted(self.data['num_satellites'])), 
                "r--", alpha=0.8, linewidth=2)
        
        plt.subplot(1, 2, 2)
        plt.hist(self.data['performance'], bins=20, alpha=0.7, color='lightcoral', 
                edgecolor='black')
        plt.xlabel('Performance Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Performance Scores')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/performance_vs_satellites.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_performance_vs_altitude(self):
        """Plot performance vs altitude."""
        plt.figure(figsize=(12, 8))
        
        # Use perigee height (actual orbital altitude) instead of altitude_km
        altitude_col = 'perigee_height_km' if 'perigee_height_km' in self.data.columns else 'altitude_km'
        altitude_label = 'Perigee Altitude (km)' if altitude_col == 'perigee_height_km' else 'Altitude (km) [semi-major axis]'
        
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(self.data[altitude_col], self.data['performance'], 
                            c=self.data['num_satellites'], s=40, alpha=0.7, 
                            cmap='plasma', edgecolors='black', linewidth=0.3)
        plt.colorbar(scatter, label='Number of Satellites')
        plt.xlabel(altitude_label)
        plt.ylabel('Performance Score')
        plt.title('Performance vs Altitude (colored by satellites)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.scatter(self.data[altitude_col], self.data['global_coverage_score'], 
                   alpha=0.6, s=30, color='green', label='Global Coverage')
        plt.scatter(self.data[altitude_col], self.data['regional_coverage_score'], 
                   alpha=0.6, s=30, color='blue', label='Regional Coverage')
        plt.scatter(self.data[altitude_col], self.data['science_score'], 
                   alpha=0.6, s=30, color='red', label='Science Score')
        plt.xlabel(altitude_label)
        plt.ylabel('Component Score')
        plt.title('Performance Components vs Altitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.hist(self.data[altitude_col], bins=20, alpha=0.7, color='lightgreen', 
                edgecolor='black')
        plt.xlabel(altitude_label)
        plt.ylabel('Frequency')
        plt.title('Distribution of Altitudes')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        # Performance vs altitude binned by altitude ranges
        alt_bins = pd.cut(self.data[altitude_col], bins=5)
        alt_performance = self.data.groupby(alt_bins)['performance'].mean()
        alt_performance.plot(kind='bar', color='orange', alpha=0.7)
        plt.xlabel('Altitude Range (km)')
        plt.ylabel('Average Performance')
        plt.title('Average Performance by Altitude Range')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/performance_vs_altitude.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_altitude_trade_offs(self):
        """Plot detailed altitude trade-off analysis."""
        plt.figure(figsize=(15, 10))
        
        # Use perigee height (actual orbital altitude) instead of altitude_km
        altitude_col = 'perigee_height_km' if 'perigee_height_km' in self.data.columns else 'altitude_km'
        altitude_label = 'Perigee Altitude (km)' if altitude_col == 'perigee_height_km' else 'Altitude (km) [semi-major axis]'
        
        # Altitude distribution
        plt.subplot(2, 3, 1)
        plt.hist(self.data[altitude_col], bins=30, alpha=0.7, color='lightblue', 
                edgecolor='black')
        if 'optimization_params_target_altitude' in self.data.columns:
            plt.axvline(self.data['optimization_params_target_altitude'].iloc[0], 
                       color='red', linestyle='--', linewidth=2, label='Target Altitude')
            plt.legend()
        plt.xlabel(altitude_label)
        plt.ylabel('Number of Solutions')
        plt.title('Distribution of Solution Altitudes')
        plt.grid(True, alpha=0.3)
        
        # Cost vs Altitude (detailed)
        plt.subplot(2, 3, 2)
        scatter = plt.scatter(self.data[altitude_col], self.data['cost_musd'], 
                            c=self.data['performance'], s=40, alpha=0.7, 
                            cmap='viridis', edgecolors='black', linewidth=0.3)
        plt.colorbar(scatter, label='Performance')
        plt.xlabel(altitude_label)
        plt.ylabel('Total Cost (M$)')
        plt.title('Cost vs Altitude\n(colored by performance)')
        plt.grid(True, alpha=0.3)
        
        # Performance vs Altitude (detailed)
        plt.subplot(2, 3, 3)
        scatter = plt.scatter(self.data[altitude_col], self.data['performance'], 
                            c=self.data['cost_musd'], s=40, alpha=0.7, 
                            cmap='plasma', edgecolors='black', linewidth=0.3)
        plt.colorbar(scatter, label='Cost (M$)')
        plt.xlabel(altitude_label)
        plt.ylabel('Performance')
        plt.title('Performance vs Altitude\n(colored by cost)')
        plt.grid(True, alpha=0.3)
        
        # Launch cost vs altitude
        plt.subplot(2, 3, 4)
        plt.scatter(self.data[altitude_col], self.data['launch_cost_musd'], 
                   alpha=0.6, s=30, color='red', edgecolors='black', linewidth=0.3)
        plt.xlabel(altitude_label)
        plt.ylabel('Launch Cost (M$)')
        plt.title('Launch Cost vs Altitude')
        plt.grid(True, alpha=0.3)
        
        # Science score vs altitude (should peak near target)
        plt.subplot(2, 3, 5)
        plt.scatter(self.data[altitude_col], self.data['science_score'], 
                   alpha=0.6, s=30, color='green', edgecolors='black', linewidth=0.3)
        if 'optimization_params_target_altitude' in self.data.columns:
            plt.axvline(self.data['optimization_params_target_altitude'].iloc[0], 
                       color='red', linestyle='--', linewidth=2, label='Target Altitude')
            plt.legend()
        plt.xlabel(altitude_label)
        plt.ylabel('Science Score')
        plt.title('Science Score vs Altitude')
        plt.grid(True, alpha=0.3)
        
        # Altitude ranges analysis
        plt.subplot(2, 3, 6)
        alt_ranges = pd.cut(self.data[altitude_col], bins=5)
        range_stats = self.data.groupby(alt_ranges).agg({
            'cost_musd': 'mean',
            'performance': 'mean',
            'num_satellites': 'mean'
        })
        
        x_pos = range(len(range_stats))
        width = 0.25
        
        plt.bar([x - width for x in x_pos], range_stats['cost_musd']/100, 
               width, label='Avg Cost (×100 M$)', alpha=0.7, color='red')
        plt.bar(x_pos, range_stats['performance'], 
               width, label='Avg Performance', alpha=0.7, color='blue')
        plt.bar([x + width for x in x_pos], range_stats['num_satellites']/10, 
               width, label='Avg Satellites (×10)', alpha=0.7, color='green')
        
        plt.xlabel('Altitude Range')
        plt.ylabel('Normalized Values')
        plt.title('Average Metrics by Altitude Range')
        plt.xticks(x_pos, [f'{interval.left:.0f}-{interval.right:.0f}' for interval in range_stats.index])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/altitude_trade_offs.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_performance_vs_inclination(self):
        """Plot performance vs inclination."""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(self.data['inclination_deg'], self.data['performance'], 
                            c=self.data['altitude_km'], s=40, alpha=0.7, 
                            cmap='coolwarm', edgecolors='black', linewidth=0.3)
        plt.colorbar(scatter, label='Altitude (km)')
        plt.xlabel('Inclination (degrees)')
        plt.ylabel('Performance Score')
        plt.title('Performance vs Inclination (colored by altitude)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(self.data['inclination_deg'], bins=20, alpha=0.7, color='lightblue', 
                edgecolor='black')
        plt.xlabel('Inclination (degrees)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Inclinations')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/performance_vs_inclination.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_cost_vs_altitude(self):
        """Plot cost vs altitude."""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(self.data['altitude_km'], self.data['cost_musd'], 
                            c=self.data['num_satellites'], s=40, alpha=0.7, 
                            cmap='viridis', edgecolors='black', linewidth=0.3)
        plt.colorbar(scatter, label='Number of Satellites')
        plt.xlabel('Altitude (km)')
        plt.ylabel('Total Cost (M$)')
        plt.title('Cost vs Altitude (colored by satellites)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Cost breakdown vs altitude
        plt.scatter(self.data['altitude_km'], self.data['launch_cost_musd'], 
                   alpha=0.6, s=30, label='Launch Cost', color='red')
        plt.scatter(self.data['altitude_km'], self.data['RE_cost_musd'], 
                   alpha=0.6, s=30, label='RE Cost', color='blue')
        plt.xlabel('Altitude (km)')
        plt.ylabel('Cost Component (M$)')
        plt.title('Cost Components vs Altitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cost_vs_altitude.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_cost_vs_inclination(self):
        """Plot cost vs inclination."""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(self.data['inclination_deg'], self.data['cost_musd'], 
                            c=self.data['altitude_km'], s=40, alpha=0.7, 
                            cmap='plasma', edgecolors='black', linewidth=0.3)
        plt.colorbar(scatter, label='Altitude (km)')
        plt.xlabel('Inclination (degrees)')
        plt.ylabel('Total Cost (M$)')
        plt.title('Cost vs Inclination (colored by altitude)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(self.data['inclination_deg'], self.data['launch_cost_musd'], 
                   alpha=0.6, s=30, color='red', label='Launch Cost')
        plt.xlabel('Inclination (degrees)')
        plt.ylabel('Launch Cost (M$)')
        plt.title('Launch Cost vs Inclination')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cost_vs_inclination.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_cost_breakdown(self):
        """Plot cost breakdown analysis."""
        plt.figure(figsize=(14, 10))
        
        # Stacked bar chart of cost components
        plt.subplot(2, 3, 1)
        cost_components = ['NRE_cost_musd', 'RE_cost_musd', 'launch_cost_musd']
        bottom = np.zeros(len(self.data))
        colors = ['red', 'blue', 'green']
        
        for i, component in enumerate(cost_components):
            plt.bar(range(len(self.data)), self.data[component], bottom=bottom, 
                   color=colors[i], alpha=0.7, label=component.replace('_musd', '').replace('_', ' ').title())
            bottom += self.data[component]
        
        plt.xlabel('Solution Index')
        plt.ylabel('Cost (M$)')
        plt.title('Cost Component Breakdown')
        plt.legend()
        
        # Pie chart of average cost breakdown
        plt.subplot(2, 3, 2)
        avg_costs = [self.data[comp].mean() for comp in cost_components]
        labels = ['NRE Cost', 'RE Cost', 'Launch Cost']
        plt.pie(avg_costs, labels=labels, autopct='%1.1f%%', colors=colors)
        plt.title('Average Cost Breakdown')
        
        # Cost ratios
        plt.subplot(2, 3, 3)
        total_cost = self.data['cost_musd']
        nre_ratio = self.data['NRE_cost_musd'] / total_cost
        re_ratio = self.data['RE_cost_musd'] / total_cost
        launch_ratio = self.data['launch_cost_musd'] / total_cost
        
        plt.hist([nre_ratio, re_ratio, launch_ratio], bins=15, alpha=0.7, 
                label=['NRE Ratio', 'RE Ratio', 'Launch Ratio'], color=colors)
        plt.xlabel('Cost Ratio')
        plt.ylabel('Frequency')
        plt.title('Distribution of Cost Ratios')
        plt.legend()
        
        # Cost per satellite breakdown
        plt.subplot(2, 3, 4)
        cost_per_sat = self.data['cost_musd'] / self.data['num_satellites']
        nre_per_sat = self.data['NRE_cost_musd'] / self.data['num_satellites']
        re_per_sat = self.data['RE_cost_musd'] / self.data['num_satellites']
        launch_per_sat = self.data['launch_cost_musd'] / self.data['num_satellites']
        
        plt.scatter(self.data['num_satellites'], nre_per_sat, alpha=0.6, s=20, 
                   color='red', label='NRE per Sat')
        plt.scatter(self.data['num_satellites'], re_per_sat, alpha=0.6, s=20, 
                   color='blue', label='RE per Sat')
        plt.scatter(self.data['num_satellites'], launch_per_sat, alpha=0.6, s=20, 
                   color='green', label='Launch per Sat')
        plt.xlabel('Number of Satellites')
        plt.ylabel('Cost per Satellite (M$)')
        plt.title('Cost per Satellite Components')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Payload and spacecraft costs
        plt.subplot(2, 3, 5)
        plt.scatter(self.data['num_satellites'], self.data['payload_tfu_musd'], 
                   alpha=0.6, s=30, color='purple', label='Payload TFU')
        plt.scatter(self.data['num_satellites'], self.data['sc_tfu_musd'], 
                   alpha=0.6, s=30, color='orange', label='Spacecraft TFU')
        plt.xlabel('Number of Satellites')
        plt.ylabel('TFU Cost (M$)')
        plt.title('TFU Costs vs Number of Satellites')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning curve analysis
        plt.subplot(2, 3, 6)
        plt.scatter(self.data['num_satellites'], self.data['learning_rate'], 
                   alpha=0.6, s=30, color='brown')
        plt.xlabel('Number of Satellites')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate vs Number of Satellites')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cost_breakdown.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_cost_components_vs_satellites(self):
        """Plot detailed cost components vs satellites."""
        plt.figure(figsize=(12, 8))
        
        # Individual subplots for each cost component
        cost_components = [
            ('NRE_cost_musd', 'NRE Cost', 'red'),
            ('RE_cost_musd', 'RE Cost', 'blue'), 
            ('launch_cost_musd', 'Launch Cost', 'green'),
            ('payload_tfu_musd', 'Payload TFU', 'purple')
        ]
        
        for i, (component, title, color) in enumerate(cost_components):
            plt.subplot(2, 2, i+1)
            plt.scatter(self.data['num_satellites'], self.data[component], 
                       alpha=0.6, s=30, color=color, edgecolors='black', linewidth=0.3)
            
            # Add trend line
            if len(self.data) > 1:
                z = np.polyfit(self.data['num_satellites'], self.data[component], 1)
                p = np.poly1d(z)
                x_trend = sorted(self.data['num_satellites'])
                plt.plot(x_trend, p(x_trend), "k--", alpha=0.8, linewidth=2)
                
            plt.xlabel('Number of Satellites')
            plt.ylabel(f'{title} (M$)')
            plt.title(f'{title} vs Number of Satellites')
            plt.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr, _ = pearsonr(self.data['num_satellites'], self.data[component])
            plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cost_components_vs_satellites.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_performance_breakdown(self):
        """Plot performance component breakdown."""
        plt.figure(figsize=(14, 10))
        
        # Performance components
        perf_components = ['local_coverage_score', 'regional_coverage_score', 
                          'global_coverage_score', 'science_score']
        
        # Individual component distributions
        for i, component in enumerate(perf_components):
            plt.subplot(2, 3, i+1)
            plt.hist(self.data[component], bins=15, alpha=0.7, color=f'C{i}', 
                    edgecolor='black')
            plt.xlabel(component.replace('_', ' ').title())
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {component.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
        
        # Component correlation heatmap
        plt.subplot(2, 3, 5)
        perf_data = self.data[perf_components + ['performance']]
        corr_matrix = perf_data.corr()
        
        im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.xticks(range(len(corr_matrix.columns)), 
                  [col.replace('_', '\n') for col in corr_matrix.columns], rotation=45)
        plt.yticks(range(len(corr_matrix.columns)), 
                  [col.replace('_', '\n') for col in corr_matrix.columns])
        plt.title('Performance Component Correlations')
        
        # Add correlation values to heatmap
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', fontsize=8)
        
        # Weighted components analysis
        plt.subplot(2, 3, 6)
        weighted_components = ['weighted_local', 'weighted_regional', 
                              'weighted_global', 'weighted_science']
        
        if all(col in self.data.columns for col in weighted_components):
            avg_weights = [self.data[comp].mean() for comp in weighted_components]
            labels = ['Local', 'Regional', 'Global', 'Science']
            plt.pie(avg_weights, labels=labels, autopct='%1.1f%%')
            plt.title('Average Weighted Component Contributions')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/performance_breakdown.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_coverage_analysis(self):
        """Plot detailed coverage analysis."""
        plt.figure(figsize=(12, 8))
        
        coverage_components = ['local_coverage_score', 'regional_coverage_score', 
                              'global_coverage_score']
        
        # Coverage vs altitude
        plt.subplot(2, 2, 1)
        for i, component in enumerate(coverage_components):
            plt.scatter(self.data['altitude_km'], self.data[component], 
                       alpha=0.6, s=20, label=component.replace('_', ' ').title(), 
                       color=f'C{i}')
        plt.xlabel('Altitude (km)')
        plt.ylabel('Coverage Score')
        plt.title('Coverage Components vs Altitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Coverage vs inclination
        plt.subplot(2, 2, 2)
        for i, component in enumerate(coverage_components):
            plt.scatter(self.data['inclination_deg'], self.data[component], 
                       alpha=0.6, s=20, label=component.replace('_', ' ').title(), 
                       color=f'C{i}')
        plt.xlabel('Inclination (degrees)')
        plt.ylabel('Coverage Score')
        plt.title('Coverage Components vs Inclination')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Coverage vs number of satellites
        plt.subplot(2, 2, 3)
        for i, component in enumerate(coverage_components):
            plt.scatter(self.data['num_satellites'], self.data[component], 
                       alpha=0.6, s=20, label=component.replace('_', ' ').title(), 
                       color=f'C{i}')
        plt.xlabel('Number of Satellites')
        plt.ylabel('Coverage Score')
        plt.title('Coverage Components vs Number of Satellites')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Overlap penalty analysis
        plt.subplot(2, 2, 4)
        if 'overlap_penalty' in self.data.columns:
            plt.scatter(self.data['num_satellites'], self.data['overlap_penalty'], 
                       alpha=0.6, s=30, color='red')
            plt.xlabel('Number of Satellites')
            plt.ylabel('Overlap Penalty')
            plt.title('Overlap Penalty vs Number of Satellites')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/coverage_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_walker_delta_analysis(self):
        """Plot Walker-Delta specific analysis."""
        if self.pattern_type != "Walker-Delta":
            return
            
        plt.figure(figsize=(14, 10))
        
        # P planes analysis
        plt.subplot(2, 3, 1)
        plt.scatter(self.data['P_planes'], self.data['performance'], 
                   alpha=0.6, s=40, color='blue', edgecolors='black', linewidth=0.3)
        plt.xlabel('Number of Planes (P)')
        plt.ylabel('Performance Score')
        plt.title('Performance vs Number of Planes')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.scatter(self.data['P_planes'], self.data['cost_musd'], 
                   alpha=0.6, s=40, color='red', edgecolors='black', linewidth=0.3)
        plt.xlabel('Number of Planes (P)')
        plt.ylabel('Total Cost (M$)')
        plt.title('Cost vs Number of Planes')
        plt.grid(True, alpha=0.3)
        
        # F phasing analysis
        plt.subplot(2, 3, 3)
        plt.scatter(self.data['F_phasing'], self.data['performance'], 
                   alpha=0.6, s=40, color='green', edgecolors='black', linewidth=0.3)
        plt.xlabel('Phasing Factor (F)')
        plt.ylabel('Performance Score')
        plt.title('Performance vs Phasing Factor')
        plt.grid(True, alpha=0.3)
        
        # Satellites per plane
        sats_per_plane = self.data['num_satellites'] / self.data['P_planes']
        plt.subplot(2, 3, 4)
        plt.scatter(sats_per_plane, self.data['performance'], 
                   alpha=0.6, s=40, color='purple', edgecolors='black', linewidth=0.3)
        plt.xlabel('Satellites per Plane')
        plt.ylabel('Performance Score')
        plt.title('Performance vs Satellites per Plane')
        plt.grid(True, alpha=0.3)
        
        # P vs F relationship
        plt.subplot(2, 3, 5)
        scatter = plt.scatter(self.data['P_planes'], self.data['F_phasing'], 
                            c=self.data['performance'], s=60, alpha=0.7, 
                            cmap='viridis', edgecolors='black', linewidth=0.3)
        plt.colorbar(scatter, label='Performance')
        plt.xlabel('Number of Planes (P)')
        plt.ylabel('Phasing Factor (F)')
        plt.title('P vs F (colored by performance)')
        plt.grid(True, alpha=0.3)
        
        # Distribution of P and F
        plt.subplot(2, 3, 6)
        plt.hist(self.data['P_planes'], bins=range(int(self.data['P_planes'].min()), 
                                                  int(self.data['P_planes'].max())+2), 
                alpha=0.7, color='blue', label='P planes', edgecolor='black')
        plt.hist(self.data['F_phasing'], bins=range(int(self.data['F_phasing'].min()), 
                                                   int(self.data['F_phasing'].max())+2), 
                alpha=0.7, color='red', label='F phasing', edgecolor='black')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of P and F Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/walker_delta_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_flower_analysis(self):
        """Plot Flower constellation specific analysis."""
        if self.pattern_type != "Flower":
            return
            
        plt.figure(figsize=(14, 10))
        
        flower_params = ['Np', 'Nd', 'Fn', 'Fd', 'Fh']
        
        for i, param in enumerate(flower_params[:5]):
            if param in self.data.columns:
                plt.subplot(2, 3, i+1)
                plt.scatter(self.data[param], self.data['performance'], 
                           alpha=0.6, s=40, color=f'C{i}', edgecolors='black', linewidth=0.3)
                plt.xlabel(f'{param} Parameter')
                plt.ylabel('Performance Score')
                plt.title(f'Performance vs {param}')
                plt.grid(True, alpha=0.3)
        
        # Flower parameter relationships
        if len([p for p in flower_params if p in self.data.columns]) >= 2:
            plt.subplot(2, 3, 6)
            if 'Np' in self.data.columns and 'Nd' in self.data.columns:
                scatter = plt.scatter(self.data['Np'], self.data['Nd'], 
                                    c=self.data['performance'], s=60, alpha=0.7, 
                                    cmap='viridis', edgecolors='black', linewidth=0.3)
                plt.colorbar(scatter, label='Performance')
                plt.xlabel('Np (Petals)')
                plt.ylabel('Nd (Days)')
                plt.title('Np vs Nd (colored by performance)')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/flower_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_correlation_matrix(self):
        """Plot correlation matrix of key parameters."""
        # Select numeric columns of interest
        key_columns = [
            'cost_musd', 'performance', 'num_satellites', 'altitude_km', 
            'inclination_deg', 'eccentricity', 'science_score',
            'global_coverage_score', 'regional_coverage_score', 'local_coverage_score'
        ]
        
        # Add pattern-specific columns
        if self.pattern_type == "Walker-Delta":
            key_columns.extend(['P_planes', 'F_phasing'])
        elif self.pattern_type == "Flower":
            flower_cols = ['Np', 'Nd', 'Fn', 'Fd', 'Fh']
            key_columns.extend([col for col in flower_cols if col in self.data.columns])
        
        # Filter to existing columns
        available_columns = [col for col in key_columns if col in self.data.columns]
        
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_data = self.data[available_columns]
        corr_matrix = corr_data.corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
        
        plt.title('Correlation Matrix of Key Parameters', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_pareto_front_colored(self):
        """Plot Pareto front with multiple coloring schemes."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Color by altitude
        scatter1 = axes[0, 0].scatter(self.data['cost_musd'], self.data['performance'], 
                                     c=self.data['altitude_km'], s=50, alpha=0.7, 
                                     cmap='viridis', edgecolors='black', linewidth=0.5)
        axes[0, 0].set_xlabel('Total Cost (M$)')
        axes[0, 0].set_ylabel('Performance Score')
        axes[0, 0].set_title('Pareto Front (colored by altitude)')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0, 0], label='Altitude (km)')
        
        # Color by inclination
        scatter2 = axes[0, 1].scatter(self.data['cost_musd'], self.data['performance'], 
                                     c=self.data['inclination_deg'], s=50, alpha=0.7, 
                                     cmap='plasma', edgecolors='black', linewidth=0.5)
        axes[0, 1].set_xlabel('Total Cost (M$)')
        axes[0, 1].set_ylabel('Performance Score')
        axes[0, 1].set_title('Pareto Front (colored by inclination)')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[0, 1], label='Inclination (deg)')
        
        # Color by global coverage
        scatter3 = axes[1, 0].scatter(self.data['cost_musd'], self.data['performance'], 
                                     c=self.data['global_coverage_score'], s=50, alpha=0.7, 
                                     cmap='coolwarm', edgecolors='black', linewidth=0.5)
        axes[1, 0].set_xlabel('Total Cost (M$)')
        axes[1, 0].set_ylabel('Performance Score')
        axes[1, 0].set_title('Pareto Front (colored by global coverage)')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=axes[1, 0], label='Global Coverage Score')
        
        # Color by cost per satellite
        cost_per_sat = self.data['cost_musd'] / self.data['num_satellites']
        scatter4 = axes[1, 1].scatter(self.data['cost_musd'], self.data['performance'], 
                                     c=cost_per_sat, s=50, alpha=0.7, 
                                     cmap='magma', edgecolors='black', linewidth=0.5)
        axes[1, 1].set_xlabel('Total Cost (M$)')
        axes[1, 1].set_ylabel('Performance Score')
        axes[1, 1].set_title('Pareto Front (colored by cost per satellite)')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=axes[1, 1], label='Cost per Satellite (M$)')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pareto_front_colored.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_3d_analysis(self):
        """Plot 3D analysis plots."""
        fig = plt.figure(figsize=(16, 6))
        
        # 3D plot: Cost vs Performance vs Satellites
        ax1 = fig.add_subplot(131, projection='3d')
        scatter1 = ax1.scatter(self.data['cost_musd'], self.data['performance'], 
                              self.data['num_satellites'], c=self.data['altitude_km'], 
                              s=40, alpha=0.6, cmap='viridis')
        ax1.set_xlabel('Cost (M$)')
        ax1.set_ylabel('Performance')
        ax1.set_zlabel('Number of Satellites')
        ax1.set_title('3D: Cost vs Performance vs Satellites\n(colored by altitude)')
        
        # 3D plot: Cost vs Performance vs Altitude
        ax2 = fig.add_subplot(132, projection='3d')
        scatter2 = ax2.scatter(self.data['cost_musd'], self.data['performance'], 
                              self.data['altitude_km'], c=self.data['num_satellites'], 
                              s=40, alpha=0.6, cmap='plasma')
        ax2.set_xlabel('Cost (M$)')
        ax2.set_ylabel('Performance')
        ax2.set_zlabel('Altitude (km)')
        ax2.set_title('3D: Cost vs Performance vs Altitude\n(colored by satellites)')
        
        # 3D plot: Satellites vs Altitude vs Inclination
        ax3 = fig.add_subplot(133, projection='3d')
        scatter3 = ax3.scatter(self.data['num_satellites'], self.data['altitude_km'], 
                              self.data['inclination_deg'], c=self.data['performance'], 
                              s=40, alpha=0.6, cmap='coolwarm')
        ax3.set_xlabel('Number of Satellites')
        ax3.set_ylabel('Altitude (km)')
        ax3.set_zlabel('Inclination (deg)')
        ax3.set_title('3D: Satellites vs Altitude vs Inclination\n(colored by performance)')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/3d_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_summary_statistics(self):
        """Generate and save summary statistics."""
        summary_file = f"{self.output_dir}/summary_statistics.txt"
        
        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MULTI-OBJECTIVE OPTIMIZATION RESULTS SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"CSV File: {self.csv_filepath}\n")
            f.write(f"Pattern Type: {self.pattern_type}\n")
            f.write(f"Total Solutions: {len(self.data)}\n\n")
            
            # Basic statistics
            f.write("BASIC STATISTICS\n")
            f.write("-"*20 + "\n")
            
            key_metrics = ['cost_musd', 'performance', 'num_satellites', 'altitude_km', 'inclination_deg']
            for metric in key_metrics:
                if metric in self.data.columns:
                    f.write(f"{metric}:\n")
                    f.write(f"  Mean: {self.data[metric].mean():.3f}\n")
                    f.write(f"  Std:  {self.data[metric].std():.3f}\n")
                    f.write(f"  Min:  {self.data[metric].min():.3f}\n")
                    f.write(f"  Max:  {self.data[metric].max():.3f}\n\n")
            
            # Pareto front extremes
            f.write("PARETO FRONT EXTREMES\n")
            f.write("-"*25 + "\n")
            
            min_cost_idx = self.data['cost_musd'].idxmin()
            max_perf_idx = self.data['performance'].idxmax()
            
            f.write("Minimum Cost Solution:\n")
            f.write(f"  Cost: {self.data.loc[min_cost_idx, 'cost_musd']:.2f} M$\n")
            f.write(f"  Performance: {self.data.loc[min_cost_idx, 'performance']:.3f}\n")
            f.write(f"  Satellites: {self.data.loc[min_cost_idx, 'num_satellites']:.0f}\n")
            f.write(f"  Altitude: {self.data.loc[min_cost_idx, 'altitude_km']:.1f} km\n\n")
            
            f.write("Maximum Performance Solution:\n")
            f.write(f"  Cost: {self.data.loc[max_perf_idx, 'cost_musd']:.2f} M$\n")
            f.write(f"  Performance: {self.data.loc[max_perf_idx, 'performance']:.3f}\n")
            f.write(f"  Satellites: {self.data.loc[max_perf_idx, 'num_satellites']:.0f}\n")
            f.write(f"  Altitude: {self.data.loc[max_perf_idx, 'altitude_km']:.1f} km\n\n")
            
            # Cost breakdown
            f.write("AVERAGE COST BREAKDOWN\n")
            f.write("-"*25 + "\n")
            cost_components = ['NRE_cost_musd', 'RE_cost_musd', 'launch_cost_musd']
            total_avg_cost = sum(self.data[comp].mean() for comp in cost_components)
            
            for comp in cost_components:
                avg_cost = self.data[comp].mean()
                percentage = (avg_cost / total_avg_cost) * 100
                f.write(f"  {comp.replace('_musd', '').replace('_', ' ').title()}: "
                       f"{avg_cost:.2f} M$ ({percentage:.1f}%)\n")
            
            f.write(f"\nTotal Average Cost: {total_avg_cost:.2f} M$\n\n")
            
            # Pattern-specific statistics
            if self.pattern_type == "Walker-Delta":
                f.write("WALKER-DELTA PATTERN STATISTICS\n")
                f.write("-"*35 + "\n")
                f.write(f"  Number of Planes (P) - Range: {self.data['P_planes'].min():.0f} to {self.data['P_planes'].max():.0f}\n")
                f.write(f"  Phasing Factor (F) - Range: {self.data['F_phasing'].min():.0f} to {self.data['F_phasing'].max():.0f}\n")
                f.write(f"  Average Satellites per Plane: {(self.data['num_satellites']/self.data['P_planes']).mean():.1f}\n\n")
            
            elif self.pattern_type == "Flower":
                f.write("FLOWER PATTERN STATISTICS\n")
                f.write("-"*25 + "\n")
                flower_params = ['Np', 'Nd', 'Fn', 'Fd', 'Fh']
                for param in flower_params:
                    if param in self.data.columns:
                        f.write(f"  {param} - Range: {self.data[param].min():.0f} to {self.data[param].max():.0f}\n")
                f.write("\n")
            
            # Correlations
            f.write("KEY CORRELATIONS\n")
            f.write("-"*15 + "\n")
            
            corr_pairs = [
                ('cost_musd', 'performance'),
                ('cost_musd', 'num_satellites'),
                ('performance', 'num_satellites'),
                ('performance', 'altitude_km'),
                ('cost_musd', 'altitude_km')
            ]
            
            for var1, var2 in corr_pairs:
                if var1 in self.data.columns and var2 in self.data.columns:
                    corr, p_value = pearsonr(self.data[var1], self.data[var2])
                    f.write(f"  {var1} vs {var2}: r = {corr:.3f} (p = {p_value:.3f})\n")
        
        print(f"Summary statistics saved to: {summary_file}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Post-process multi-objective optimization CSV results')
    parser.add_argument('csv_file', help='Path to CSV file containing optimization results')
    parser.add_argument('--output_dir', default='postprocessing', help='Output directory for plots')
    parser.add_argument('--pattern', choices=['Walker-Delta', 'Flower', 'auto'], default='auto',
                       help='Constellation pattern type (auto-detect by default)')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found.")
        return
    
    # Create post-processor and generate plots
    print(f"Processing CSV file: {args.csv_file}")
    processor = CSVPostProcessor(args.csv_file, args.output_dir)
    processor.generate_all_plots()
    
    print("\nPost-processing complete!")

if __name__ == "__main__":
    # Option 1: Use command line arguments (uncomment next line)
    # main()
    
    # Option 2: Hardcode the file path (current approach)
    csv_file = "hpc_downloads/results_multi_seed/walker-delta/AFB_Andersen/L0.1_R0.1_G0.9_Sci0.9/pareto_solutions_walker-delta_gsAFB_Andersen_w0.1-0.1-0.9-0.9_pop640_gen480_step100_seed69_mass130kg_alt500km.csv"
    
    # Option 3: Interactive input (uncomment next 2 lines)
    # csv_file = input("Enter path to CSV file: ")
    # output_dir = input("Enter output directory (default: postprocessing): ") or "postprocessing"
    
    # Create post-processor and generate plots
    print(f"Processing CSV file: {csv_file}")
    processor = CSVPostProcessor(csv_file, "postprocessing")
    processor.generate_all_plots()
    print("\nPost-processing complete!")
