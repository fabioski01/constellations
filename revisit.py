"""
Constellation Revisit Time Calculator

This script calculates maximum revisit times for satellite constellations
based on orbital mechanics and coverage geometry.

Author: GitHub Copilot
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional

class RevisitTimeCalculator:
    """
    Calculate revisit times for satellite constellations.
    
    Supports Walker-Delta, single plane, and custom constellation patterns.
    """
    
    def __init__(self):
        self.EARTH_RADIUS = 6371.0  # km
        self.MU_EARTH = 3.986004418e5  # km³/s², Earth's gravitational parameter
        self.EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s
        
    def orbital_period_hours(self, semi_major_axis: float) -> float:
        """
        Calculate orbital period in hours.
        
        Parameters
        ----------
        semi_major_axis : float
            Semi-major axis in km
            
        Returns
        -------
        float
            Orbital period in hours
        """
        period_seconds = 2 * np.pi * np.sqrt(semi_major_axis**3 / self.MU_EARTH)
        return period_seconds / 3600.0
    
    def ground_track_separation(self, latitude: float, inclination: float, 
                               num_planes: int) -> Dict[str, float]:
        """
        Calculate ground track separation at given latitude.
        
        Parameters
        ----------
        latitude : float
            Target latitude in degrees10
        inclination : float
            Orbital inclination in degrees
        num_planes : int
            Number of orbital planes
            
        Returns
        -------
        Dict[str, float]
            Dictionary with separation in degrees and km
        """
        # Track separation in degrees longitude
        sep_deg = 360 / num_planes
        
        # Adjust for latitude and inclination effects
        if abs(latitude) <= abs(inclination):
            effective_sep_deg = sep_deg * np.cos(np.radians(latitude))
        else:
            # Outside coverage zone
            effective_sep_deg = float('inf')
        
        # Convert to km
        sep_km = effective_sep_deg * np.pi/180 * self.EARTH_RADIUS * np.cos(np.radians(latitude))
        
        return {
            'separation_deg': effective_sep_deg,
            'separation_km': sep_km
        }
    
    def effective_swath_width(self, altitude: float, fov_half_angle: float, 
                             min_elevation: float = 0.0) -> float:
        """
        Calculate effective swath width considering elevation constraints.
        
        Parameters
        ----------
        altitude : float
            Satellite altitude in km
        fov_half_angle : float
            Half-angle of field of view in degrees
        min_elevation : float, optional
            Minimum elevation angle in degrees (default: 0)
            
        Returns
        -------
        float
            Effective swath width in km
        """
        # Basic swath width from FOV
        swath_fov = 2 * altitude * np.tan(np.radians(fov_half_angle))
        
        # Constraint from minimum elevation angle
        if min_elevation > 0:
            max_nadir_angle = 90 - min_elevation
            swath_elevation = 2 * altitude * np.tan(np.radians(max_nadir_angle))
            swath_width = min(swath_fov, swath_elevation)
        else:
            swath_width = swath_fov
            
        return swath_width
    
    def coverage_gap_factor(self, swath_width: float, track_separation: float) -> float:
        """
        Calculate coverage gap factor.
        
        Parameters
        ----------
        swath_width : float
            Effective sensor swath width in km
        track_separation : float
            Ground track separation in km
            
        Returns
        -------
        float
            Gap factor (1.0 = perfect coverage, >1.0 = gaps exist)
        """
        if track_separation == 0 or np.isinf(track_separation):
            return float('inf')
        
        coverage_ratio = swath_width / track_separation
        
        if coverage_ratio >= 1.0:
            # Complete coverage, account for overlap
            gap_factor = 1.0
        else:
            # Coverage gaps exist
            gap_factor = 1.0 / coverage_ratio
            
        return gap_factor
    
    def walker_delta_revisit(self, semi_major_axis: float, total_satellites: int, 
                           num_planes: int, latitude: float, inclination: float,
                           fov_half_angle: float = 5.0, min_elevation: float = 0.0) -> Dict:
        """
        Calculate revisit time for Walker-Delta constellation.
        
        Parameters
        ----------
        semi_major_axis : float
            Semi-major axis in km
        total_satellites : int
            Total number of satellites
        num_planes : int
            Number of orbital planes
        latitude : float
            Target latitude in degrees
        inclination : float
            Orbital inclination in degrees
        fov_half_angle : float, optional
            Half-angle of field of view in degrees (default: 5)
        min_elevation : float, optional
            Minimum elevation angle in degrees (default: 0)
            
        Returns
        -------
        Dict
            Comprehensive revisit time analysis
        """
        # Basic orbital parameters
        altitude = semi_major_axis - self.EARTH_RADIUS
        period_hours = self.orbital_period_hours(semi_major_axis)
        sats_per_plane = total_satellites // num_planes
        
        # Check if latitude is within coverage zone
        if abs(latitude) > abs(inclination):
            return {
                'max_revisit_hours': float('inf'),
                'average_revisit_hours': float('inf'),
                'coverage_available': False,
                'reason': f'Latitude {latitude}° outside inclination coverage ±{inclination}°'
            }
        
        # Ground track geometry
        track_info = self.ground_track_separation(latitude, inclination, num_planes)
        track_sep_km = track_info['separation_km']
        
        # Swath coverage
        swath_width = self.effective_swath_width(altitude, fov_half_angle, min_elevation)
        
        # Gap factor
        gap_factor = self.coverage_gap_factor(swath_width, track_sep_km)
        
        # Time between plane crossings
        ground_speed_kmh = 2 * np.pi * self.EARTH_RADIUS * np.cos(np.radians(latitude)) / period_hours
        time_between_planes = track_sep_km / ground_speed_kmh if ground_speed_kmh > 0 else float('inf')
        
        # Time between satellites in same plane
        time_between_sats = period_hours / sats_per_plane
        
        # Effective revisit time
        base_revisit = min(time_between_planes, time_between_sats)
        max_revisit = base_revisit * gap_factor
        average_revisit = base_revisit * 0.5  # Statistical average
        
        return {
            'max_revisit_hours': max_revisit,
            'average_revisit_hours': average_revisit,
            'orbital_period_hours': period_hours,
            'altitude_km': altitude,
            'sats_per_plane': sats_per_plane,
            'track_separation_km': track_sep_km,
            'swath_width_km': swath_width,
            'gap_factor': gap_factor,
            'ground_speed_kmh': ground_speed_kmh,
            'coverage_available': True,
            'coverage_efficiency': min(1.0, swath_width / track_sep_km) if track_sep_km > 0 else 0.0
        }
    
    def latitude_scan_analysis(self, semi_major_axis: float, total_satellites: int,
                              num_planes: int, inclination: float, 
                              fov_half_angle: float = 5.0, min_elevation: float = 0.0,
                              lat_range: Tuple[float, float] = (-90, 90),
                              lat_step: float = 5.0) -> pd.DataFrame:
        """
        Analyze revisit times across latitude range.
        
        Parameters
        ----------
        semi_major_axis : float
            Semi-major axis in km
        total_satellites : int
            Total number of satellites
        num_planes : int
            Number of orbital planes
        inclination : float
            Orbital inclination in degrees
        fov_half_angle : float, optional
            Half-angle of field of view in degrees (default: 5)
        min_elevation : float, optional
            Minimum elevation angle in degrees (default: 0)
        lat_range : Tuple[float, float], optional
            Latitude range to analyze (default: (-90, 90))
        lat_step : float, optional
            Latitude step size in degrees (default: 5)
            
        Returns
        -------
        pd.DataFrame
            Latitude scan results
        """
        latitudes = np.arange(lat_range[0], lat_range[1] + lat_step, lat_step)
        results = []
        
        for lat in latitudes:
            analysis = self.walker_delta_revisit(
                semi_major_axis, total_satellites, num_planes, lat, inclination,
                fov_half_angle, min_elevation
            )
            
            results.append({
                'latitude': lat,
                'max_revisit_hours': analysis['max_revisit_hours'],
                'average_revisit_hours': analysis['average_revisit_hours'],
                'coverage_available': analysis['coverage_available'],
                'gap_factor': analysis.get('gap_factor', float('inf')),
                'coverage_efficiency': analysis.get('coverage_efficiency', 0.0)
            })
        
        return pd.DataFrame(results)
    
    def plot_latitude_analysis(self, df: pd.DataFrame, constellation_name: str = "Constellation",
                              save_path: Optional[str] = None):
        """
        Plot latitude analysis results.
        
        Parameters
        ----------
        df : pd.DataFrame
            Results from latitude_scan_analysis
        constellation_name : str, optional
            Name for plot title (default: "Constellation")
        save_path : str, optional
            Path to save plot (default: None, shows plot)
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Filter valid coverage points
        valid_df = df[df['coverage_available'] == True].copy()
        
        if len(valid_df) == 0:
            print("No valid coverage points to plot.")
            return
        
        # Replace infinite values for plotting
        valid_df.loc[valid_df['max_revisit_hours'] == float('inf'), 'max_revisit_hours'] = np.nan
        valid_df.loc[valid_df['average_revisit_hours'] == float('inf'), 'average_revisit_hours'] = np.nan
        
        # Plot 1: Revisit times
        ax1.plot(valid_df['latitude'], valid_df['max_revisit_hours'], 'r-', 
                linewidth=2, label='Maximum Revisit Time')
        ax1.plot(valid_df['latitude'], valid_df['average_revisit_hours'], 'b-', 
                linewidth=2, label='Average Revisit Time')
        ax1.set_xlabel('Latitude (degrees)')
        ax1.set_ylabel('Revisit Time (hours)')
        ax1.set_title(f'{constellation_name} - Revisit Time vs Latitude')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Gap factor
        ax2.plot(valid_df['latitude'], valid_df['gap_factor'], 'g-', linewidth=2)
        ax2.set_xlabel('Latitude (degrees)')
        ax2.set_ylabel('Gap Factor')
        ax2.set_title('Coverage Gap Factor vs Latitude')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Coverage efficiency
        ax3.plot(valid_df['latitude'], valid_df['coverage_efficiency'], 'purple', linewidth=2)
        ax3.set_xlabel('Latitude (degrees)')
        ax3.set_ylabel('Coverage Efficiency')
        ax3.set_title('Coverage Efficiency vs Latitude')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

def main():
    """
    Interactive main function for constellation analysis.
    """
    print("🛰️  Constellation Revisit Time Calculator")
    print("=" * 50)
    
    # Initialize calculator
    calc = RevisitTimeCalculator()
    
    # Get user inputs
    print("\n📋 Enter Constellation Parameters:")
    
    # Orbital parameters
    altitude = float(input("Satellite altitude (km) [500]: ") or "500")
    semi_major_axis = altitude + calc.EARTH_RADIUS
    
    inclination = float(input("Inclination (degrees) [51.4]: ") or "51.4")
    
    # Constellation parameters
    total_satellites = int(input("Total number of satellites [24]: ") or "24")
    num_planes = int(input("Number of orbital planes [4]: ") or "4")
    
    # Sensor parameters
    fov_half_angle = float(input("FOV half-angle (degrees) [5]: ") or "5")
    min_elevation = float(input("Minimum elevation angle (degrees) [0]: ") or "0")
    
    # Analysis parameters
    target_latitude = float(input("Target latitude for analysis (degrees) [0]: ") or "0")
    
    print(f"\n🔍 Analyzing constellation:")
    print(f"   • {total_satellites} satellites in {num_planes} planes")
    print(f"   • Altitude: {altitude} km")
    print(f"   • Inclination: {inclination}°")
    print(f"   • FOV: ±{fov_half_angle}°")
    print(f"   • Min elevation: {min_elevation}°")
    
    # Single point analysis
    print(f"\n📍 Analysis for latitude {target_latitude}°:")
    result = calc.walker_delta_revisit(
        semi_major_axis, total_satellites, num_planes, 
        target_latitude, inclination, fov_half_angle, min_elevation
    )
    
    if result['coverage_available']:
        print(f"   • Maximum revisit time: {result['max_revisit_hours']:.2f} hours")
        print(f"   • Average revisit time: {result['average_revisit_hours']:.2f} hours")
        print(f"   • Coverage efficiency: {result['coverage_efficiency']:.2f}")
        print(f"   • Gap factor: {result['gap_factor']:.2f}")
        print(f"   • Ground track separation: {result['track_separation_km']:.1f} km")
        print(f"   • Sensor swath width: {result['swath_width_km']:.1f} km")
    else:
        print(f"   • {result['reason']}")
    
    # Latitude scan analysis
    do_scan = input(f"\n🌍 Perform latitude scan analysis? (y/n) [y]: ").lower()
    if do_scan != 'n':
        print("Performing latitude scan...")
        
        # Coverage zone limits
        lat_min = max(-abs(inclination), -85)
        lat_max = min(abs(inclination), 85)
        
        df = calc.latitude_scan_analysis(
            semi_major_axis, total_satellites, num_planes, inclination,
            fov_half_angle, min_elevation, (lat_min, lat_max), 2.0
        )
        
        # Find worst coverage location
        valid_df = df[df['coverage_available'] == True]
        if len(valid_df) > 0:
            worst_idx = valid_df['max_revisit_hours'].idxmax()
            worst_lat = valid_df.loc[worst_idx, 'latitude']
            worst_time = valid_df.loc[worst_idx, 'max_revisit_hours']
            
            print(f"\n📊 Latitude Scan Results:")
            print(f"   • Coverage zone: {lat_min:.1f}° to {lat_max:.1f}°")
            print(f"   • Worst revisit time: {worst_time:.2f} hours at {worst_lat:.1f}°")
            
            best_idx = valid_df['max_revisit_hours'].idxmin()
            best_lat = valid_df.loc[best_idx, 'latitude']
            best_time = valid_df.loc[best_idx, 'max_revisit_hours']
            print(f"   • Best revisit time: {best_time:.2f} hours at {best_lat:.1f}°")
        
        # Plot results
        constellation_name = f"{total_satellites}-sat Walker-Delta ({num_planes} planes)"
        save_plot = input(f"\n📈 Save analysis plot? (y/n) [y]: ").lower()
        if save_plot != 'n':
            filename = f"revisit_analysis_{total_satellites}sat_{num_planes}planes_{altitude}km.png"
            calc.plot_latitude_analysis(df, constellation_name, filename)
        else:
            calc.plot_latitude_analysis(df, constellation_name)
        
        # Save CSV
        save_csv = input(f"\n💾 Save results to CSV? (y/n) [n]: ").lower()
        if save_csv == 'y':
            csv_filename = f"revisit_analysis_{total_satellites}sat_{num_planes}planes_{altitude}km.csv"
            df.to_csv(csv_filename, index=False)
            print(f"Results saved to: {csv_filename}")

if __name__ == "__main__":
    main()