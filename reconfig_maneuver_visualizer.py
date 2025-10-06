"""Reconfiguration Maneuver / Ground Track Visualizer

Short Description
    Generate ground track PNGs for analytic reconfiguration phases
    (GOM wait, transfers, drift, final ROM) for multiple target sites.

Core Scenario Inputs (edit in main or adapt programmatically)
    base_epoch : UTC start datetime
    inc_deg    : inclination (deg)
    gom_a_km   : initial (GOM) semi-major axis (km)
    rom_a_km   : target (ROM) semi-major axis (km)
    delta_a_km : drift Δa used by planner (km, can be ±)
    targets    : list of (name, lat_deg, lon_deg)

Derived From Planner (build_maneuver_sequence_for_visualizer)
    times   : dict with T_wait_s, T1_s, T_d_s, T2_s
    epochs  : phase epoch datetimes
    orbits  : dict of orbital element dicts (gom, transfer1, drift, transfer2, rom, rom_test)

Visualizer add_orbit(...) Key Args
    orbital_params (dict) : semi_major_axis, eccentricity, inclination, RAAN,
                                                    argument_of_periapsis, true_anomaly (+ optional bstar/ndot/nddot)
    epoch (datetime)      : orbit epoch
    label (str)           : legend label
    propagation_time_hours: span to propagate
    plot_segments         : optional list of (start_h, end_h) for partial + fade

Plot Controls (plot_ground_tracks)
    save_path, time_step_seconds, target_lat_deg/target_lon_deg, tolerance_deg, lon_err_deg

Output
    PNG: plots/reconfig_groundtracks/<auto_named>.png

Quick Example
    python reconfig_maneuver_visualizer.py  (generates multiple target figures)

Author
Fabio Meloni - TU Delft MSc Student - NASA Jet Propulsion Laboratory Visiting Researcher
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timezone, timedelta
from sgp4.api import Satrec, WGS84
import os
from utilities import TimeConversion, CoordinateConversion
import constants
import matplotlib.ticker as mticker
from reconfig_driftANDwaitTime import (
    Orbit as PlannerOrbit,
    deg2rad,
    build_maneuver_sequence_for_visualizer,
)

class OrbitGroundTrackVisualizer:
    """
    Class for visualizing ground tracks of multiple orbits on the same plot.
    Each orbit can have its own epoch.
    """
    
    def __init__(self):
        """
        Initialize the visualizer.
        """
        self.orbits = []
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        self.linestyles = ['--', '-.', ':', '-', (0, (3, 1, 1, 1)), (0, (5, 10)), (0, (3, 10, 1, 10))]
    
    def add_orbit(self, orbital_params, epoch, label, propagation_time_hours=24, color=None, linestyle='--'):
        """
        Add an orbit to be visualized.
        
        Parameters:
        -----------
        orbital_params : dict
            Dictionary containing orbital parameters:
            - semi_major_axis: km
            - eccentricity: unitless
            - inclination: degrees
            - RAAN: degrees  
            - argument_of_periapsis: degrees
            - true_anomaly: degrees
            - ndot: rev/day² (optional, default: 1e-8)
            - nddot: rev/day³ (optional, default: 1e-12)
            - bstar: 1/er (optional, default: 2e-5)
        epoch : datetime
            Epoch for this specific orbit
        label : str
            Label for this orbit in the legend
        propagation_time_hours : float, optional
            Propagation time in hours (default: 24 hours)
        color : str, optional
            Color for this orbit (auto-assigned if None)
        linestyle : str, optional
            Line style for this orbit (default: '--')
        """
        # Set defaults for optional parameters
        params = orbital_params.copy()
        params.setdefault('ndot', 1e-8)
        params.setdefault('nddot', 1e-12)
        params.setdefault('bstar', 2e-5)
        
        # Auto-assign color if not provided
        if color is None:
            color = self.colors[len(self.orbits) % len(self.colors)]
        
        # Auto-assign linestyle cycling through available styles
        if linestyle == 'auto':
            linestyle = self.linestyles[len(self.orbits) % len(self.linestyles)]
        
        orbit = {
            'params': params,
            'epoch': epoch,
            'label': label,
            'propagation_time_hours': propagation_time_hours,
            'color': color,
            'linestyle': linestyle
        }
        
        self.orbits.append(orbit)
    
    def create_satellite(self, orbital_params, epoch):
        """
        Create a Satrec from classical elements at the given epoch.
        Angles in degrees in, mean motion computed from a (km).
        """
        from sgp4.api import Satrec, WGS84
        import numpy as np

        MU_EARTH = 398600.4418  # km^3/s^2

        # Inputs
        a_km = float(orbital_params['semi_major_axis'])
        e    = float(orbital_params['eccentricity'])
        i    = np.radians(orbital_params['inclination'])
        raan = np.radians(orbital_params['RAAN'])
        argp = np.radians(orbital_params['argument_of_periapsis'])
        # Allow either mean_anomaly or true_anomaly in degrees
        if 'mean_anomaly' in orbital_params:
            M = np.radians(float(orbital_params['mean_anomaly']))
            nu = None
        else:
            nu = np.radians(orbital_params['true_anomaly'])
            M = None

        # ν → M (for e>0; for circular it’s the same)
        if M is None:
            if e < 1e-12:
                M = nu
            else:
                beta = np.sqrt((1.0 - e) / (1.0 + e))
                E = 2.0 * np.arctan(beta * np.tan(nu / 2.0))
                E = (E + 2.0*np.pi) % (2.0*np.pi)
                M = (E - e*np.sin(E)) % (2.0*np.pi)

        # Mean motion for sgp4init (rad/min)
        n_rad_s  = np.sqrt(MU_EARTH / (a_km**3))
        no_kozai = n_rad_s * 60.0

        # Epoch as “days since 1949-12-31 00:00 UT” (Vallado convention)
        jd, fr = TimeConversion.datetime_to_sgp4time(epoch)
        epoch_days = (jd + fr) - 2433281.5

        # Keep perturbation/drag terms zero for 1-hour checks
        bstar = float(orbital_params.get('bstar', 0.0))
        ndot  = 0.0   # rev/day^2
        nddot = 0.0   # rev/day^3

        sat = Satrec()
        sat.sgp4init(
            WGS84, 'i', 0,
            epoch_days,         # <-- this was missing; REQUIRED by this API
            bstar, ndot, nddot,
            e, argp, i, M,
            no_kozai, raan
        )
        return sat

    def add_orbit(self, orbital_params, epoch, label, propagation_time_hours=24, color=None, linestyle='--', plot_segments=None):
        """
        Add an orbit to be visualized.
        
        Parameters:
        -----------
        orbital_params : dict
            Dictionary containing orbital parameters:
            - semi_major_axis: km
            - eccentricity: unitless
            - inclination: degrees
            - RAAN: degrees  
            - argument_of_periapsis: degrees
            - true_anomaly: degrees
            - ndot: rev/day² (optional, default: 1e-8)
            - nddot: rev/day³ (optional, default: 1e-12)
            - bstar: 1/er (optional, default: 2e-5)
        epoch : datetime
            Epoch for this specific orbit
        label : str
            Label for this orbit in the legend
        propagation_time_hours : float, optional
            Propagation time in hours (default: 24 hours)
        color : str, optional
            Color for this orbit (auto-assigned if None)
        linestyle : str, optional
            Line style for this orbit (default: '--')
        plot_segments : list of tuples, optional
            List of (start_hour, end_hour) tuples to plot only specific segments.
            Example: [(0, 1), (192, 193)] to plot first and last hours of a long orbit
        """
        # Set defaults for optional parameters
        params = orbital_params.copy()
        params.setdefault('ndot', 1e-8)
        params.setdefault('nddot', 1e-12)
        params.setdefault('bstar', 2e-5)
        
        # Auto-assign color if not provided
        if color is None:
            color = self.colors[len(self.orbits) % len(self.colors)]
        
        # Auto-assign linestyle cycling through available styles
        if linestyle == 'auto':
            linestyle = self.linestyles[len(self.orbits) % len(self.linestyles)]
        
        orbit = {
            'params': params,
            'epoch': epoch,
            'label': label,
            'propagation_time_hours': propagation_time_hours,
            'color': color,
            'linestyle': linestyle,
            'plot_segments': plot_segments
        }
        
        self.orbits.append(orbit)

    def propagate_orbit(self, orbital_params, epoch, propagation_time_hours, time_step_seconds=10):
        """
        Propagate an orbit and return ECI positions.
        
        Parameters:
        -----------
        orbital_params : dict
            Orbital parameters
        epoch : datetime
            Epoch for this orbit
        propagation_time_hours : float
            Propagation time in hours
        time_step_seconds : float, optional
            Time step in seconds (default: 60s)
            
        Returns:
        --------
        tuple
            (times, positions, epoch) where times is in seconds from epoch and positions in km
        """
        # Create satellite
        sat = self.create_satellite(orbital_params, epoch)
        
        # Create time array
        total_seconds = propagation_time_hours * 3600
        times = np.arange(0, total_seconds + time_step_seconds, time_step_seconds)
        
        # Propagate
        positions = np.zeros((len(times), 3))
        
        for i, t in enumerate(times):
            current_time = epoch + timedelta(seconds=float(t))
            jd, fr = TimeConversion.datetime_to_sgp4time(current_time)
            
            err, r, v = sat.sgp4(jd, fr)
            if err != 0:
                print(f"SGP4 error {err} at time {t}s for orbit '{orbital_params.get('label', 'unknown')}'")
                positions[i] = [np.nan, np.nan, np.nan]
            else:
                positions[i] = r
        
        return times, positions, epoch
    
    # Removed duplicate plot_ground_tracks definition – keep the richer one below
    
    def _plot_orbit_segments(self, ax, lons, lats, color, label, linestyle):
        """
        Plot orbit segments handling longitude discontinuities.
        """
        # Convert to numpy arrays
        lons = np.array(lons)
        lats = np.array(lats)
        
        # Find discontinuities (longitude jumps > 180 degrees)
        lon_diff = np.diff(lons)
        breaks = np.where(np.abs(lon_diff) > 180)[0]
        
        # Add start and end indices
        segment_starts = np.concatenate(([0], breaks + 1))
        segment_ends = np.concatenate((breaks + 1, [len(lons)]))
        
        # Plot each continuous segment
        for i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
            if start < end:
                segment_lons = lons[start:end]
                segment_lats = lats[start:end]
                
                # Only show label for first segment
                segment_label = label if i == 0 else None
                
                ax.plot(segment_lons, segment_lats, 
                       color=color, linestyle=linestyle, linewidth=2,
                       label=segment_label, transform=ccrs.PlateCarree())

    def _plot_orbit_segments_with_fade(self, ax, lons, lats, color, label, linestyle, is_first_segment=False, is_last_segment=False):
        """
        Plot orbit segments with fade effects at the ends for partial orbit visualization.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.collections import LineCollection
        
        # Convert to numpy arrays
        lons = np.array(lons)
        lats = np.array(lats)
        
        # Find discontinuities (longitude jumps > 180 degrees)
        lon_diff = np.diff(lons)
        breaks = np.where(np.abs(lon_diff) > 180)[0]
        
        # Add start and end indices
        segment_starts = np.concatenate(([0], breaks + 1))
        segment_ends = np.concatenate((breaks + 1, [len(lons)]))
        
        # Plot each continuous segment
        for i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
            if start < end:
                segment_lons = lons[start:end]
                segment_lats = lats[start:end]
                
                # Only show label for first segment
                segment_label = label if i == 0 else None
                
                # Create fade effect if this is a first or last segment
                if (is_first_segment or is_last_segment) and len(segment_lons) > 10:
                    # Create points for LineCollection
                    points = np.array([segment_lons, segment_lats]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    
                    # Create alpha values for fade effect
                    n_points = len(segment_lons)
                    alphas = np.ones(n_points - 1)
                    
                    fade_length = min(20, n_points // 4)  # Fade over 20 points or 1/4 of segment
                    
                    if is_first_segment:
                        # Fade at the end of first segment
                        for j in range(fade_length):
                            alphas[-(j+1)] = (j + 1) / fade_length * 0.3  # Fade to 30% opacity
                    
                    if is_last_segment:
                        # Fade at the beginning of last segment (smoother fade)
                        for j in range(fade_length):
                            alphas[j] = (j + 1) / fade_length * 0.7 + 0.3  # Fade from 30% to full opacity more gradually
                    
                    # Create LineCollection with varying alpha
                    lc = LineCollection(segments, colors=[color]*len(segments), 
                                      linestyle=linestyle, linewidths=2, alpha=1.0,
                                      transform=ccrs.PlateCarree())
                    
                    # Set individual alpha values using proper color handling
                    lc.set_alpha(None)  # Remove global alpha
                    base_rgba = mcolors.to_rgba(color)
                    colors_with_alpha = [(base_rgba[0], base_rgba[1], base_rgba[2], alpha) 
                                       for alpha in alphas]
                    lc.set_color(colors_with_alpha)
                    
                    ax.add_collection(lc)
                    
                    # Add invisible line for legend
                    if segment_label:
                        ax.plot([], [], color=color, linestyle=linestyle, linewidth=2, 
                               label=segment_label, transform=ccrs.PlateCarree())
                else:
                    # Regular plot without fade
                    ax.plot(segment_lons, segment_lats, 
                           color=color, linestyle=linestyle, linewidth=2,
                           label=segment_label, transform=ccrs.PlateCarree())
    
    def print_orbit_summary(self):
        """
        Print a summary of all added orbits.
        """
        if not self.orbits:
            print("No orbits added yet.")
            return
        
        print(f"\nOrbit Summary ({len(self.orbits)} orbits):")
        print("-" * 80)
        
        for i, orbit in enumerate(self.orbits):
            params = orbit['params']
            print(f"{i+1:2d}. {orbit['label']}")
            print(f"    Epoch: {orbit['epoch'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"    SMA: {params['semi_major_axis']:.1f} km, "
                  f"Inc: {params['inclination']:.1f}°, "
                  f"RAAN: {params['RAAN']:.1f}°")
            print(f"    Propagation: {orbit['propagation_time_hours']:.1f}h, "
                  f"Color: {orbit['color']}, "
                  f"Style: {orbit['linestyle']}")
            print()

    def plot_ground_tracks(self, save_path=None, time_step_seconds=10, target_lat_deg=None, target_lon_deg=None, tolerance_deg=None, lon_err_deg=None):
        """
        Plot ground tracks for all added orbits.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        time_step_seconds : float, optional
            Time step for propagation in seconds (default: 60s)
        """
        if not self.orbits:
            print("No orbits added. Please add orbits using add_orbit() before plotting.")
            return
        
        # Create figure
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Set global extent to show the entire Earth
        ax.set_global()
        
        # Use grayscale map instead of colored stock image
        ax.add_feature(cfeature.LAND, color='lightgray')
        ax.add_feature(cfeature.OCEAN, color='white')
        ax.coastlines(color='black', linewidth=0.5)
        # ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  color="gray", alpha=0.5, linestyle="--",
                  x_inline=False, y_inline=False)
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 10))  # denser vertical lines (meridians)
        gl.ylocator = mticker.FixedLocator(np.arange(-90,  90, 10))   # parallels
        gl.right_labels = False
        gl.top_labels = False
        gl.xlabel_style = {'size': 7}
        gl.ylabel_style = {'size': 8}
        # Collect epoch info for title
        epochs = [orbit['epoch'] for orbit in self.orbits]
        min_epoch = min(epochs)
        max_epoch = max(epochs)
        
    # Plot each orbit
        for i, orbit in enumerate(self.orbits):
            print(f"Plotting orbit {i+1}/{len(self.orbits)}: {orbit['label']}")
            print(f"  Epoch: {orbit['epoch'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"  Propagation time: {orbit['propagation_time_hours']:.1f} hours")
            
            if orbit.get('plot_segments'):
                print(f"  Plot segments: {orbit['plot_segments']}")
            
            # Check if we need to plot segments or full orbit
            if orbit.get('plot_segments'):
                # Propagate specific segments
                times_list, positions_list, epoch = self.propagate_orbit_segments(
                    orbit['params'], 
                    orbit['epoch'],
                    orbit['propagation_time_hours'],
                    orbit['plot_segments'],
                    time_step_seconds
                )
                
                # Plot each segment
                for j, (times, positions) in enumerate(zip(times_list, positions_list)):
                    # Convert ECI to geographic coordinates
                    ECEF_cartesian_orbit = CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(
                        positions * 1000,  # Convert to meters
                        times.astype(float),
                        reference_time=epoch
                    )
                    
                    ECEF_geodetic_orbit = [CoordinateConversion.ECEF_cartesian_to_geodetic(x, y, z) 
                                        for x, y, z in ECEF_cartesian_orbit]
                    
                    # Extract lat/lon
                    lats = [coord[0] for coord in ECEF_geodetic_orbit]  # latitude
                    lons = [coord[1] for coord in ECEF_geodetic_orbit]  # longitude
                    
                    # Handle longitude wrapping
                    unwrapped_lons = np.unwrap(np.radians(lons))
                    normalized_lons = np.degrees(unwrapped_lons)
                    
                    # Create segment label
                    start_hour, end_hour = orbit['plot_segments'][j]
                    
                    # Check if this is GOM or Drift orbit for fade effect
                    is_gom_or_drift = 'GOM' in orbit['label'] or 'Drift' in orbit['label']
                    
                    # For GOM and Drift orbits, remove time notation from legend
                    if j == 0:  # Only create label for first segment
                        if is_gom_or_drift:
                            # Remove the time part from the label for GOM/Drift orbits
                            base_label = orbit['label'].split(' (t=')[0]  # Get part before time notation
                            segment_label = base_label
                        else:
                            segment_label = f"{orbit['label']} (t={start_hour:.1f}-{end_hour:.1f}h)"
                    else:
                        segment_label = None
                    
                    is_first_segment = j == 0  # First segment (first hour)
                    is_last_segment = j == len(orbit['plot_segments']) - 1  # Last segment (last hour)
                    
                    # Plot segments with fade effect for GOM/Drift orbits
                    if is_gom_or_drift and len(orbit['plot_segments']) > 1:
                        self._plot_orbit_segments_with_fade(ax, normalized_lons, lats, orbit['color'], 
                                                          segment_label, orbit['linestyle'],
                                                          is_first_segment, is_last_segment)
                    else:
                        # Regular plotting for other orbits or single segments
                        self._plot_orbit_segments(ax, normalized_lons, lats, orbit['color'], 
                                                segment_label, orbit['linestyle'])
            else:
                # Propagate full orbit (existing code)
                times, positions, epoch = self.propagate_orbit(
                    orbit['params'], 
                    orbit['epoch'],
                    orbit['propagation_time_hours'],
                    time_step_seconds
                )
                
                # Convert ECI to geographic coordinates
                ECEF_cartesian_orbit = CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(
                    positions * 1000,  # Convert to meters
                    times.astype(float),
                    reference_time=epoch
                )
                
                ECEF_geodetic_orbit = [CoordinateConversion.ECEF_cartesian_to_geodetic(x, y, z) 
                                    for x, y, z in ECEF_cartesian_orbit]
                
                # Extract lat/lon
                lats = [coord[0] for coord in ECEF_geodetic_orbit]  # latitude
                lons = [coord[1] for coord in ECEF_geodetic_orbit]  # longitude
                
                # Handle longitude wrapping
                unwrapped_lons = np.unwrap(np.radians(lons))
                normalized_lons = np.degrees(unwrapped_lons)
                
                # Plot segments to avoid longitude wrapping issues
                self._plot_orbit_segments(ax, normalized_lons, lats, orbit['color'], 
                                        orbit['label'], orbit['linestyle'])
        
        # Target marker (more visible) if provided
        if target_lat_deg is not None and target_lon_deg is not None:
            lon_wrapped = ((float(target_lon_deg) + 180.0) % 360.0) - 180.0
            ax.scatter(
                [lon_wrapped], [float(target_lat_deg)],
                s=100, marker='*', color='gold', alpha=0.9,
                edgecolors='black', linewidths=1.2,
                transform=ccrs.PlateCarree(), label='Target'
            )
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Add title
        if min_epoch == max_epoch:
            title = f'Reconfiguration Visualization\nEpoch: {min_epoch.strftime("%Y-%m-%d %H:%M:%S UTC")}'
        else:
            title = f'Reconfiguration Visualization\nEpochs: {min_epoch.strftime("%Y-%m-%d %H:%M")} to {max_epoch.strftime("%Y-%m-%d %H:%M")} UTC'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add info box
        info_text = f'Number of orbits: {len(self.orbits)}\n'
        info_text += f'Time step: {time_step_seconds}s\n'
        info_text += f'Max propagation: {max([o["propagation_time_hours"] for o in self.orbits]):.1f}h\n'
        
        # Add longitude error if provided
        if lon_err_deg is not None:
            info_text += f'Longitude error: {lon_err_deg:.3f}°\n'
        
        # Add tolerance info if provided
        if tolerance_deg is not None:
            info_text += f'ROM tolerance: {tolerance_deg:.3f}°\n'
        
        # Add epoch info if different epochs
        if min_epoch != max_epoch:
            time_span = max_epoch - min_epoch
            info_text += f'Epoch span: {time_span.total_seconds()/3600:.1f}h'
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        # plt.show()

    def propagate_orbit_segments(self, orbital_params, epoch, propagation_time_hours, plot_segments=None, time_step_seconds=10):
        """
        Propagate an orbit and return ECI positions for specific segments only.
        
        Parameters:
        -----------
        orbital_params : dict
            Orbital parameters
        epoch : datetime
            Epoch for this orbit
        propagation_time_hours : float
            Total propagation time in hours
        plot_segments : list of tuples, optional
            List of (start_hour, end_hour) tuples to plot only specific segments
        time_step_seconds : float, optional
            Time step in seconds (default: 60s)
            
        Returns:
        --------
        tuple
            (times_list, positions_list, epoch) where each list contains arrays for each segment
        """
        if plot_segments is None:
            # If no segments specified, plot the entire orbit
            return self.propagate_orbit(orbital_params, epoch, propagation_time_hours, time_step_seconds)
        
        # Create satellite
        sat = self.create_satellite(orbital_params, epoch)
        
        times_list = []
        positions_list = []
        
        for start_hour, end_hour in plot_segments:
            # Validate segment bounds
            start_hour = max(0, start_hour)
            end_hour = min(propagation_time_hours, end_hour)
            
            if start_hour >= end_hour:
                continue
                
            # Create time array for this segment
            start_seconds = start_hour * 3600
            end_seconds = end_hour * 3600
            times = np.arange(start_seconds, end_seconds + time_step_seconds, time_step_seconds)
            
            # Propagate this segment
            positions = np.zeros((len(times), 3))
            
            for i, t in enumerate(times):
                current_time = epoch + timedelta(seconds=float(t))
                jd, fr = TimeConversion.datetime_to_sgp4time(current_time)
                
                err, r, v = sat.sgp4(jd, fr)
                if err != 0:
                    print(f"SGP4 error {err} at time {t}s for orbit '{orbital_params.get('label', 'unknown')}'")
                    positions[i] = [np.nan, np.nan, np.nan]
                else:
                    positions[i] = r
            
            times_list.append(times)
            positions_list.append(positions)
        
        return times_list, positions_list, epoch

def main():
    """
    Use the analytic planner to compute maneuver times and segment orbits,
    then visualize ground tracks without duplicating inputs.
    """
    # 1) Single source of truth: scenario inputs
    base_epoch = datetime(2025, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    inc_deg    = 51.447684794794995
    raan0_deg  = 0.0
    u0_deg     = 0.0  # for circular, u=M when ω=0

    gom_a_km   = constants.EARTH_RADIUS + 500.0
    rom_a_km   = 6875.59534097
    delta_a_km = 50.0  # example drift Δa; adjust as needed

    # 2) Build planner Orbit and call helper
    gom = PlannerOrbit(
        a_km=gom_a_km,
        i_rad=deg2rad(inc_deg),
        Omega_rad=deg2rad(raan0_deg),
        u_rad=deg2rad(u0_deg),
        e=0.0,
    )

    # Helper to run one target and save a plot
    def run_one(name: str, target_lat_deg: float, target_lon_deg: float):
        pkg = build_maneuver_sequence_for_visualizer(
            gom=gom,
            rom_a_km=rom_a_km,
            delta_a_km=delta_a_km,
            epoch=base_epoch,
            target_lat_geod_deg=target_lat_deg,
            target_lon_deg=target_lon_deg,
            max_wrap=500,
            drift_time_cap_days=(1000.0/24),
        )
        if not pkg.get('success'):
            print(f"[{name}] Planner failed:", pkg.get('note'))
            return

        from reconfig_driftANDwaitTime import print_solution_summary
        sol = pkg['solution']
        print_solution_summary(sol)

        times  = pkg['times']
        epochs = pkg['epochs']
        orbits = pkg['orbits']

        # Extract tolerance information from solution metadata
        solution = pkg['solution']
        selection_meta = solution.get('selection_meta', {})
        tolerance_deg = selection_meta.get('tolerance_used_deg')
        
        # Extract longitude error from solution
        lon_err_deg = solution.get('lon_err_deg')
        
        # Also print what we found for debugging
        print(f"[TOLERANCE DEBUG] tolerance_deg = {tolerance_deg}")
        print(f"[TOLERANCE DEBUG] lon_err_deg = {lon_err_deg}")
        print(f"[TOLERANCE DEBUG] selection_meta keys = {list(selection_meta.keys())}")
        
        tol_str = f"{tolerance_deg:.3f}" if tolerance_deg is not None else "fallback"
        print(f"[TOLERANCE DEBUG] final tol_str = '{tol_str}'")

        visualizer = OrbitGroundTrackVisualizer()

        t_wait_h = times['T_wait_s'] / 3600.0
        t1_h     = times['T1_s'] / 3600.0
        td_h     = times['T_d_s'] / 3600.0
        t2_h     = times['T2_s'] / 3600.0

        # For GOM orbit, if wait time > 2h, plot only first and last hour
        gom_segments = None
        if t_wait_h > 2.0:
            gom_segments = [(0, 1), (max(0.0, t_wait_h - 1.0), t_wait_h)]
        
        visualizer.add_orbit(
            orbits['gom'], epochs['gom_epoch'], f'GOM Orbit (t_wait={t_wait_h:.3f} h)',
            propagation_time_hours=t_wait_h, color='blue', linestyle='--', plot_segments=gom_segments
        )
        visualizer.add_orbit(
            orbits['transfer1'], epochs['transfer1_epoch'], f'Transfer 1 (t1={t1_h:.3f} h)',
            propagation_time_hours=t1_h, color='cyan', linestyle='-.'
        )
        drift_segments = None
        if td_h > 2.0:
            drift_segments = [(0, 1), (max(0.0, td_h - 1.0), td_h)]
        visualizer.add_orbit(
            orbits['drift'], epochs['drift_epoch'], f'Drift Orbit (td={td_h:.3f} h)',
            propagation_time_hours=td_h, color='green', linestyle=':', plot_segments=drift_segments
        )
        visualizer.add_orbit(
            orbits['transfer2'], epochs['transfer2_epoch'], f'Transfer 2 (t2={t2_h:.3f} h)',
            propagation_time_hours=t2_h, color='orange', linestyle='-'
        )
        visualizer.add_orbit(
            orbits['rom'], epochs['rom_epoch'], 'ROM Theoretical Orbit',
            propagation_time_hours=1.0, color='red', linestyle='--'
        )
        visualizer.add_orbit(
            orbits['rom_test'], epochs['rom_epoch'], 'ROM Propagated Orbit',
            propagation_time_hours=1.0, color='red', linestyle=':'
        )

        tgt_lat_fmt = f"{target_lat_deg:+.6f}".replace("+", "p").replace("-", "m")
        tgt_lon_fmt = f"{target_lon_deg:+.6f}".replace("+", "p").replace("-", "m")
        fname = (
            f"{name}_GOM{gom_a_km:.1f}km_ROM{rom_a_km:.1f}km_"
            f"dA{delta_a_km:+.1f}km_inc{inc_deg:.3f}deg_"
            f"tgtLat{tgt_lat_fmt}_tgtLon{tgt_lon_fmt}_"
            f"tol{tol_str}deg.png"
        )
        save_path = os.path.join("plots", "reconfig_groundtracks", fname)
        visualizer.plot_ground_tracks(
            save_path=save_path,
            time_step_seconds=30,
            target_lat_deg=target_lat_deg,
            target_lon_deg=target_lon_deg,
            tolerance_deg=tolerance_deg,
            lon_err_deg=lon_err_deg,
        )
        print(f"[{name}] Plot saved: {save_path}")
    
    # 1) Pre-existing targets
    targets = [
        ("LosAngeles", 34.0522, -118.2437),
        ("Sumatra_2004", 3.295, 95.982),
        ("Zero_0_0", 0.0, 0.0),
        ("Rome", 41.9028, 12.4964),
        ("Hawaii", 21.3069, -157.8583),
        ("SanFrancisco", 37.7749, -122.4194),
        ("SanDiego", 32.7157, -117.1611),
        ("Vancouver", 49.2827, -123.1207),
        ("Sydney", -33.8688, 151.2093),
        ("Beijing", 39.9042, 116.4074),
        ("Tokyo", 35.6895, 139.6917),
    ]
    for name, lat, lon in targets:
        run_one(f"case_{name}", lat, lon)

    # # 2) Los Angeles variations (separate loop): Δ from −1.000° to +1.000° in 0.001° steps
    # la_lat0 = 34.0522
    # la_lon0 = -118.2437
    # step_deg = 0.5
    # max_delta = 10.0

    # # lat+Δ, lon+Δ
    # delta = -max_delta
    # while delta <= max_delta + 1e-12:
    #     lat = la_lat0 + delta
    #     lon = la_lon0 + delta
    #     run_one(f"LA_var_add_d{delta:+.3f}".replace("+", "p").replace("-", "m"), lat, lon)
    #     delta = round(delta + step_deg, 6)  # avoid FP drift

    # # lat+Δ, lon−Δ
    # delta = -max_delta
    # while delta <= max_delta + 1e-12:
    #     lat = la_lat0 + delta
    #     lon = la_lon0 - delta
    #     run_one(f"LA_var_sub_d{delta:+.3f}".replace("+", "p").replace("-", "m"), lat, lon)
    #     delta = round(delta + step_deg, 6)

def compute_anomaly_change(dt, a, e):
        """
        Compute mean anomaly change for any orbit (circular or eccentric).
        
        Parameters:
        -----------
        dt : float
            Time interval in seconds
        a : float
            Semi-major axis in km
        e : float
            Eccentricity
            
        Returns:
        --------
        float
            Mean anomaly change in radians
        """
        from reconfig_phasing_RGT import compute_meanmotion
        # Mean motion depends only on semi-major axis (Kepler's 3rd law)
        n = compute_meanmotion(a)  # rad/s
        return n * dt  # radians

if __name__ == "__main__":
    main()
    

    