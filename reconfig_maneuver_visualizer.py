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
    
    # def create_satellite(self, orbital_params, epoch):
    #     """
    #     Create a Satrec object from orbital parameters and epoch.
        
    #     Parameters:
    #     -----------
    #     orbital_params : dict
    #         Orbital parameters dictionary
    #     epoch : datetime
    #         Epoch for this satellite
            
    #     Returns:
    #     --------
    #     Satrec
    #         SGP4 satellite object
    #     """
    #     # Convert datetime to SGP4 epoch
    #     jd, fr = TimeConversion.datetime_to_sgp4time(epoch)
    #     epoch_days = (jd + fr) - 2433281.5
        
    #     # Convert angles to radians
    #     inclination = np.radians(orbital_params['inclination'])
    #     raan = np.radians(orbital_params['RAAN'])
    #     arg_perigee = np.radians(orbital_params['argument_of_periapsis'])
        
    #     # Convert true anomaly to mean anomaly (assuming circular orbit for simplicity)
    #     # For more accurate conversion, implement eccentric anomaly calculation
    #     mean_anomaly = np.radians(orbital_params['true_anomaly'])  # Simplified
        
    #     # Compute mean motion from semi-major axis
    #     n_rad_s = np.sqrt(constants.GRAVITATIONAL_CONSTANT / orbital_params['semi_major_axis']**3)
    #     no_kozai = n_rad_s * 60.0  # Convert to rad/min for SGP4
        
    #     # Create Satrec object
    #     sat = Satrec()
    #     sat.sgp4init(
    #         WGS84,                                      # gravity model
    #         'i',                                        # opsmode
    #         0,                                          # satnum
    #         epoch_days,                                 # epoch
    #         orbital_params['bstar'],                    # bstar
    #         orbital_params['ndot'] / (24*60*60),        # ndot (convert from rev/day² to rev/min²)
    #         orbital_params['nddot'] / (24*60*60)**2,    # nddot (convert from rev/day³ to rev/min³)
    #         orbital_params['eccentricity'],             # eccentricity
    #         arg_perigee,                                # argument of periapsis (radians)
    #         inclination,                                # inclination (radians)
    #         mean_anomaly,                               # mean anomaly (radians)
    #         no_kozai,                                   # mean motion (rad/min)
    #         raan                                        # RAAN (radians)
    #     )
        
    #     return sat
    
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
        nu   = np.radians(orbital_params['true_anomaly'])

        # ν → M (for e>0; for circular it’s the same)
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
    
    def plot_ground_tracks(self, save_path=None, time_step_seconds=10):
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
        ax.stock_img()
        ax.coastlines()
        ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")
        
        # Collect epoch info for title
        epochs = [orbit['epoch'] for orbit in self.orbits]
        min_epoch = min(epochs)
        max_epoch = max(epochs)
        
        # Plot each orbit
        for i, orbit in enumerate(self.orbits):
            print(f"Plotting orbit {i+1}/{len(self.orbits)}: {orbit['label']}")
            print(f"  Epoch: {orbit['epoch'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"  Propagation time: {orbit['propagation_time_hours']:.1f} hours")
            
            # Propagate orbit
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
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Add title
        if min_epoch == max_epoch:
            title = f'Ground Tracks Comparison\nEpoch: {min_epoch.strftime("%Y-%m-%d %H:%M:%S UTC")}'
        else:
            title = f'Ground Tracks Comparison\nEpochs: {min_epoch.strftime("%Y-%m-%d %H:%M")} to {max_epoch.strftime("%Y-%m-%d %H:%M")} UTC'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add info box
        info_text = f'Number of orbits: {len(self.orbits)}\n'
        info_text += f'Time step: {time_step_seconds}s\n'
        info_text += f'Max propagation: {max([o["propagation_time_hours"] for o in self.orbits]):.1f}h\n'
        
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

    def plot_ground_tracks(self, save_path=None, time_step_seconds=10):
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
        ax.stock_img()
        ax.coastlines()
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
                    segment_label = f"{orbit['label']} (t={start_hour:.1f}-{end_hour:.1f}h)" if j == 0 else None
                    
                    # Plot segments to avoid longitude wrapping issues
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
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Add title
        if min_epoch == max_epoch:
            title = f'Ground Tracks Comparison\nEpoch: {min_epoch.strftime("%Y-%m-%d %H:%M:%S UTC")}'
        else:
            title = f'Ground Tracks Comparison\nEpochs: {min_epoch.strftime("%Y-%m-%d %H:%M")} to {max_epoch.strftime("%Y-%m-%d %H:%M")} UTC'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add info box
        info_text = f'Number of orbits: {len(self.orbits)}\n'
        info_text += f'Time step: {time_step_seconds}s\n'
        info_text += f'Max propagation: {max([o["propagation_time_hours"] for o in self.orbits]):.1f}h\n'
        
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
    Example usage of the OrbitGroundTrackVisualizer.
    """

    t_wait_s = 35984.435
    t_transfer1_s = 2849.55382
    t_drift_s = 52300.033
    t_transfer2_s = 2850.97810
    
    M_waiting_GOM_rad = compute_anomaly_change(t_wait_s, 6871, 0)
    M_transfer1_rad = compute_anomaly_change(t_transfer1_s, 6896, 0.00362529)
    M_drift_rad = compute_anomaly_change(t_drift_s, 6921, 0.0)
    M_transfer2_rad = compute_anomaly_change(t_transfer2_s, ((6921+6875.59534135)/2), ((6921-6875.59534135)/(6921+6875.59534135)))
    M_waiting_GOM_deg = np.degrees(M_waiting_GOM_rad)
    M_transfer1_deg = np.degrees(M_transfer1_rad)
    M_drift_deg = np.degrees(M_drift_rad)   
    M_transfer2_deg = np.degrees(M_transfer2_rad)

    # print('M waiting GOM', compute_anomaly_change(t_wait_s, 6871, 0), np.degrees(compute_anomaly_change(t_wait_s, 6871, 0)))
    # print('M T1', compute_anomaly_change(t_transfer1_s, 6896, 0.00362529), np.degrees(compute_anomaly_change(t_transfer1_s, 6896, 0.00362529)))
    # print('M Drift', compute_anomaly_change(t_drift_s, 6921, 0.0), np.degrees(compute_anomaly_change(t_drift_s, 6921, 0.0)))
    # print('M T2', compute_anomaly_change(t_transfer2_s, ((6921+6875.59534135)/2), ((6921-6875.59534135)/(6921+6875.59534135))), np.degrees(compute_anomaly_change(t_transfer2_s, ((6921+6875.59534135)/2), ((6921-6875.59534135)/(6921+6875.59534135)))))

    visualizer = OrbitGroundTrackVisualizer()
    
    # Define different epochs for different orbit phases
    base_epoch = datetime(2025, 5, 5, 12, 0, 0, tzinfo=timezone.utc)

    # transfer1_epoch = base_epoch + timedelta(hours=(0.92823)) # waiting time 0.92823 hr or 3341.63527 s
    transfer1_epoch = base_epoch + timedelta(hours=(t_wait_s/3600)) # waiting time x hr or 35984.435 s

    # drift_epoch = base_epoch + timedelta(hours=(0.92823+0.7915555)) # transfer 1 time 2849.55382 s or 0.79154 hr
    drift_epoch = base_epoch + timedelta(hours=((t_wait_s/3600)+(t_transfer1_s/3600))) # transfer 1 time 2849.55382 s or 0.79154 hr

    # transfer2_epoch = base_epoch + timedelta(hours=(0.92823+0.7915555+193.23699)) # drifting 193.23699 hours or 695653.16822 s
    transfer2_epoch = base_epoch + timedelta(hours=((t_wait_s/3600)+(t_transfer1_s/3600)+(t_drift_s/3600))) # drifting x hours or 52300.033 s

    # transfer2_fix_epoch = base_epoch + timedelta(hours=(-0.79194)) # drifting 193.23699 hours or 695653.16822 s
    # rom_epoch = base_epoch # transfer 2 time 2850.97810 s or 0.79194 hr
    rom_epoch = datetime(2025, 5, 6, 14, 6, 25, 00, tzinfo=timezone.utc)

    # rom_orbit_test_epoch = base_epoch + timedelta(hours=(0.92823+0.7915555+193.23699+0.79194))
    rom_orbit_test_epoch = base_epoch + timedelta(hours=((t_wait_s/3600)+(t_transfer1_s/3600)+(t_drift_s/3600)+(t_transfer2_s/3600)))


    # Example orbit 1: GOM orbit (initial)
    gom_orbit = {
        'semi_major_axis': 6871.0,    # km (500 km altitude)
        'eccentricity': 0.0,
        'inclination': 51.447684794794995,          # degrees
        'RAAN': 0.0,                  # degrees
        'argument_of_periapsis': 0.0, # degrees
        'true_anomaly': 0.0,          # degrees
    }
    
    # Example orbit 2: Transfer orbit 1 (Hohmann to drift)
    # transfer1_orbit = {
    #     'semi_major_axis': 6896.0,    # km (elliptical)
    #     'eccentricity': 0.00362529,        # elliptical
    #     'inclination': 51.447684794794995,          # degrees
    #     'RAAN': 0.0,                  # degrees 
    #     'argument_of_periapsis': 0.0, # degrees
    #     'true_anomaly': 212.23453773800986,          # degrees, mean anomaly by sat in gom for waiting time, including perturbations 
    # }
    
    transfer1_orbit = {
        'semi_major_axis': 6896.0,    # km (elliptical)
        'eccentricity': 0.00362529,        # elliptical
        'inclination': 51.447684794794995,          # degrees
        'RAAN': 0.0,                  # degrees 
        'argument_of_periapsis': 0.0, # degrees
        'true_anomaly': np.mod(M_waiting_GOM_deg, 360),          # degrees, mean anomaly by sat in gom for waiting time, including perturbations
    }

    # Example orbit 3: Drift orbit (higher altitude)
    # drift_orbit = {
    #     'semi_major_axis': 6921.0,    # km (550 km altitude)
    #     'eccentricity': 0.0,
    #     'inclination': 51.447684794794995,          # degrees
    #     'RAAN': 0,                 # degrees 
    #     'argument_of_periapsis': 0.0, # degrees
    #     'true_anomaly': np.mod(180.00291705692214+212.23453773800986, 360),         # degrees, mean anomaly by sat in gom for waiting time PLUS sat in t1 for transfer time, including perturbations
    # }
    drift_orbit = {
        'semi_major_axis': 6921.0,    # km (550 km altitude)
        'eccentricity': 0.0,
        'inclination': 51.447684794794995,          # degrees
        'RAAN': 0,                 # degrees 
        'argument_of_periapsis': 0.0, # degrees
        'true_anomaly': np.mod((M_waiting_GOM_deg + M_transfer1_deg), 360),         # degrees, mean anomaly by sat in gom for waiting time PLUS sat in t1 for transfer time, including perturbations
    }

    # Example orbit 4: Transfer orbit 2 (Hohmann to ROM)
    # transfer2_orbit = {
    #     'semi_major_axis': 6898.3,    # km (elliptical)
    #     'eccentricity': ((6921-6875.59534135)/(6921+6875.59534135)),        # elliptical
    #     'inclination': 51.447684794794995,          # degrees
    #     'RAAN': 322.207, # degrees 
    #     'argument_of_periapsis': 0, # degrees
    #     'true_anomaly': np.mod(212.23453773800986 + 180.00291705692214 + 43704.99084848064, 360),          # degrees, mean anomaly by sat in gom for waiting time PLUS sat in t1 for transfer time PLUS sat in drift orbit for drift time, including perturbations
    #     # 'true_anomaly': np.mod(175.813-179.93824653951972, 360),          # degrees, mean anomaly by sat in gom for waiting time PLUS sat in t1 for transfer time PLUS sat in drift orbit for drift time, including perturbations
    # }
    transfer2_orbit = {
        'semi_major_axis': 6898.3,    # km (elliptical)
        'eccentricity': ((6921-6875.59534135)/(6921+6875.59534135)),        # elliptical
        'inclination': 51.447684794794995,          # degrees
        'RAAN': 354.882448, # degrees 
        'argument_of_periapsis': 0, # degrees
        'true_anomaly': np.mod((M_waiting_GOM_deg + M_transfer1_deg + M_drift_deg), 360),          # degrees, mean anomaly by sat in gom for waiting time PLUS sat in t1 for transfer time PLUS sat in drift orbit for drift time, including perturbations
        # 'true_anomaly': np.mod(175.813-179.93824653951972, 360),          # degrees, mean anomaly by sat in gom for waiting time PLUS sat in t1 for transfer time PLUS sat in drift orbit for drift time, including perturbations
    }
    
    # Example orbit 4: Transfer orbit 2 (Hohmann to ROM)
    # transfer2_orbit_fix = {
    #     'semi_major_axis': 6898.3,    # km (elliptical)
    #     'eccentricity': ((6921-6875.59534135)/(6921+6875.59534135)),        # elliptical
    #     'inclination': 51.447684794794995,          # degrees
    #     'RAAN': 322.207, # degrees 
    #     'argument_of_periapsis': 0, # degrees
    #     # 'true_anomaly': np.mod(212.23453773800986 + 180.00291705692214 + 43704.99084848064, 360),          # degrees, mean anomaly by sat in gom for waiting time PLUS sat in t1 for transfer time PLUS sat in drift orbit for drift time, including perturbations
    #     'true_anomaly': np.mod(175.813-179.93824653951972, 360),          # degrees, mean anomaly by sat in gom for waiting time PLUS sat in t1 for transfer time PLUS sat in drift orbit for drift time, including perturbations
    # }
    
    # Example orbit 5: ROM orbit (final RGT)
    # rom_orbit = {
    #     'semi_major_axis': 6875.59534135,  # km (specific RGT altitude)
    #     'eccentricity': 0.0,
    #     'inclination': 51.447684794794995,               # degrees
    #     'RAAN': 322.207,                      # degrees (target RAAN)
    #     'argument_of_periapsis': 0.0,      # degrees
    #     'true_anomaly': 175.813,              # degrees (target phase)
    # }
    rom_orbit = {
        'semi_major_axis': 6875.59534097,  # km (specific RGT altitude)
        'eccentricity': 0.0,
        'inclination': 51.447684794794995,               # degrees
        'RAAN': 354.882448,                      # degrees (target RAAN)
        'argument_of_periapsis': 0.0,      # degrees
        'true_anomaly': 175.813353,              # degrees (target phase)
    }

    
    # rom_orbit_test = {
    #     'semi_major_axis': 6875.59534135,  # km (specific RGT altitude)
    #     'eccentricity': 0.0,
    #     'inclination': 51.447684794794995,               # degrees
    #     'RAAN': 322.207,                      # degrees (target RAAN)
    #     'argument_of_periapsis': 0.0,      # degrees
    #     'true_anomaly': np.mod(212.23453773800986 + 180.00291705692214 + 43704.99084848064 + 179.93824653951972, 360), # degrees, mean anomaly by sat in gom for waiting time PLUS sat in t1 for transfer time PLUS sat in drift orbit for drift time PLUS transfer2, including perturbations
    # }

    rom_orbit_test = {
        'semi_major_axis': 6875.59534135,  # km (specific RGT altitude)
        'eccentricity': 0.0,
        'inclination': 51.447684794794995,               # degrees
        'RAAN': 354.882448,                      # degrees (target RAAN)
        'argument_of_periapsis': 0.0,      # degrees
        'true_anomaly': np.mod(M_waiting_GOM_deg + M_transfer1_deg + M_drift_deg + M_transfer2_deg, 360), # degrees, mean anomaly by sat in gom for waiting time PLUS sat in t1 for transfer time PLUS sat in drift orbit for drift time PLUS transfer2, including perturbations
    }
    
    # Add orbits to visualizer with different epochs
    # visualizer.add_orbit(gom_orbit, base_epoch, 'GOM Orbit (t=0h)', 
    #                     propagation_time_hours=0.928, color='blue', linestyle='--')
    visualizer.add_orbit(gom_orbit, base_epoch, f'GOM Orbit (t_wait={t_wait_s/3600:.3f} h)', 
                        propagation_time_hours=(t_wait_s/3600), color='blue', linestyle='--') # 35984.435/3600 = 9.996 hr something of waiting time

    visualizer.add_orbit(transfer1_orbit, transfer1_epoch, f'Transfer 1 (t1={t_transfer1_s/3600.:3f} h)', 
                        propagation_time_hours=(t_transfer1_s/3600), color='cyan', linestyle='-.')
    
    # # Add drift orbit with only first and last hours plotted
    # visualizer.add_orbit(drift_orbit, drift_epoch, 'Drift Orbit (first & last hour)', 
    #                     propagation_time_hours=193.2370, color='green', linestyle=':',
    #                     plot_segments=[(0, 1), (192.2370, 193.2370)])  # First and last hours
    visualizer.add_orbit(drift_orbit, drift_epoch, f'Drift Orbit (first last hour only), (td={t_drift_s/3600:.3f} h)', 
                        propagation_time_hours=(t_drift_s/3600), color='green', linestyle=':',
                        plot_segments=[(0, 1), (((t_drift_s/3600) -1), (t_drift_s/3600))])  # First and last hours new td 52300.033
    
    # visualizer.add_orbit(transfer2_orbit_fix, transfer2_fix_epoch, 'Transfer 2 FIX (t=0.79h)', 
    #                     propagation_time_hours=0.791444, color='orange', linestyle=':')
    
    visualizer.add_orbit(transfer2_orbit, transfer2_epoch, f'Transfer 2 (t2={t_transfer2_s/3600.:3f} h)', 
                        propagation_time_hours=(t_transfer2_s/3600), color='orange', linestyle='-')
    
    visualizer.add_orbit(rom_orbit, rom_epoch, f'ROM RGT Orbit', 
                        propagation_time_hours=1, color='red', linestyle='--')
    
    visualizer.add_orbit(rom_orbit_test, rom_orbit_test_epoch, f'ROM TEST RGT Orbit', 
                        propagation_time_hours=1, color='red', linestyle=':')
    
    # Print summary
    visualizer.print_orbit_summary()
    
    # Create plot
    save_path = "plots/orbit_comparison/maneuver_sequence_ground_tracks.png"
    visualizer.plot_ground_tracks(save_path=save_path, time_step_seconds=30)
    
    print("Maneuver sequence ground track visualization complete!")

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
    

    