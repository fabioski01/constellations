from utilities import CoordinateConversion, orbiter, TimeConversion
from utilities_constellations import Constellation, create_label_from_satrec
from utilities_coverage import Coverage, get_ground_station_coordinates
import constants

from datetime import datetime, timezone
import numpy as np
from sgp4.api import Satrec, jday, WGS84

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
import cartopy.crs as ccrs
from matplotlib.colors import hsv_to_rgb
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

import math

import os # to create directories if they do not exist

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")

### INPUTS ###
### SELECT TOTAL NUMBER OF SATELLITES ###
# num_satellites_list = [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,88,100] # compatible for P=4
# num_satellites_list = [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,66,72,78,84,90,96,99] # compatible for P=3
# num_satellites_list = [4]
num_satellites_list = list(range(1, 101))

### SELECT ORBITAL PARAMETERS ###
orbital_parameters = {
    'semi_major_axis':     6800.0,    # km
    'eccentricity':        0.00,    # unitless
    'inclination':         50.0,     # deg
    'RAAN':                0,     # deg
    'argument_of_periapsis': 250.0,  # deg     
    'true_anomaly':        0.0,      # deg
    'ndot':                1e-8,     # rev/day² # Includes derivatives
    'nddot':               1e-12,    # rev/day³
    'bstar':               2e-5      # 1/er
}
### SELECT EPOCH ###
epoch = datetime(2025,5,5,0,0,0)

### SELECT A PATTERN ###
pattern_list = [("Walker-Delta", {"P": 2, "F": 1}), ("Walker-Delta", {"P": 3, "F": 1}), ("Walker-Delta", {"P": 4, "F": 1}), ("Walker-Delta", {"P": 5, "F": 1}), ("Walker-Delta", {"P": 6, "F": 1}), ("Walker-Delta", {"P": 7, "F": 1}), ("Walker-Delta", {"P": 8, "F": 1}), ("Walker-Delta", {"P": 9, "F": 1}), ("Walker-Delta", {"P": 10, "F": 1}),
    "Walker-Delta-SinglePlane"
]
# pattern_list = [ "Walker-Delta", "Walker-Delta-SingleSatPerPlane", "Walker-Delta-SinglePlane"]
# pattern_list = ["Walker-Delta-SingleSatPerPlane"]
# pattern_list = ["Walker-Delta-SinglePlane"]
# pattern_list = [("Walker-Delta", {"P": 5, "F": 1})]
pattern_params = {} # saving parameters for each pattern for plotting outside the pattern loop

### INPUT for visibility plots ###
gs = "Fucino"  # Ground station location

### INPUTS for coverage plots ###
# masking_angles = [5.0,10.0,15.0]  # degrees, the larger the smaller the coverage area. Same as elevation angle threshold. Usually 10-15 deg min
masking_angles = [20.0]  # degrees, the larger the smaller the coverage area. Same as elevation angle threshold
# fovs = [20,70,120,180] # degrees, field of view of the sensor. If too low is limiting, if it's large then the masking angle is limiting
fovs = [170] # degrees, field of view of the sensor

# Define propagation window & step. Make sure is at least the number of periods to be plotted
start_time_datetime = datetime(2025, 5, 5, 12, 00, 00, 00, tzinfo=timezone.utc) # my input
end_time_datetime =   datetime(2025, 5, 15, 12, 00, 00, 00, tzinfo=timezone.utc) # my input
step =  102 # Time step input

# Switches for plotting
PLOT_GROUND_TRACKS = False
PLOT_INERTIAL_GROUND_TRACKS = False # for debugging walker spaced by mean anomaly
PLOT_INERTIAL_3D = False
PLOT_COVERAGE = False
COMPUTE_VISIBILITY = True
PLOT_VISIBILITY = False # needs COMPUTE_VISIBILITY to be True
PLOT_DATA_COMPARISON = False # needs COMPUTE_VISIBILITY to be True
PLOT_PATTERN_VISIBILITY_COMPARISON = True # needs COMPUTE_VISIBILITY to be True

# Estimate one period index to stop ground track after one orbit
a = orbital_parameters['semi_major_axis']  # km from Earth's center
orbital_period = 2 * np.pi * np.sqrt(a**3 / constants.GRAVITATIONAL_CONSTANT)

orbital_period_hours = int(orbital_period // 3600)
orbital_period_minutes = int((orbital_period % 3600) // 60)
orbital_period_seconds = int(orbital_period % 60)
print(f"Orbital period: {orbital_period_hours} hours, {orbital_period_minutes} minutes, {orbital_period_seconds} seconds ({orbital_period/3600:.4f} hr)")

if __name__ == "__main__":   
    # Get ground station coordinates based on the selected name
    gs_lat, gs_lon, gs_alt = get_ground_station_coordinates(gs)
    gs_geodetic = (gs_lat, gs_lon, gs_alt)
    
    # Store data for pattern comparison to avoid recalculating later
    pattern_visibility_data = {}  # Store visibility and overlap data for each pattern
    pattern_num_satellites = {}   # Store corresponding num_satellites for each pattern
    # New: store all visibilities, elevations, and times for plotting
    if 'pattern_visibility_plotdata' not in locals():
        pattern_visibility_plotdata = {}

    for pattern_item in pattern_list:
        # Check if pattern is a tuple with parameters or just a string
        if isinstance(pattern_item, tuple):
            pattern = pattern_item[0]
            pattern_params[pattern] = pattern_item[1]
            # Create unique pattern key for Walker-Delta with parameters
            if pattern == "Walker-Delta":
                P = pattern_item[1].get('P', 1)
                F = pattern_item[1].get('F', 1)
                pattern_key = f"{pattern}_{P}_{F}"  # Create unique key with P and F values
            else:
                pattern_key = pattern
        else:
            pattern = pattern_item
            pattern_key = pattern
            pattern_params[pattern] = {}
        
        # Extract P and F for Walker-Delta pattern if they exist
        P = pattern_params[pattern].get('P', 1)  # Default to 1 if not specified
        F = pattern_params[pattern].get('F', 1)  # Default to 1 if not specified
        
        if pattern == "Walker-Delta":
            print(f"========== Processing pattern: {pattern} {orbital_parameters['inclination']}:N/{P}/{F} ==========")
        else:
            print(f'========== Processing pattern: {pattern_key} ==========')

        # visibility data analysis
        total_time_with_any_visible_list = []
        overlap_percentage_list = []
        num_satellites_record = []
        # Initialize pattern data storage for this specific pattern
        pattern_visibility_data[pattern_key] = {
            'total_time': [],     # For storing total_time_with_any_visible values
            'overlap': []         # For storing overlap_percentage values
        }
        pattern_num_satellites[pattern_key] = []  # For storing num_satellites values
        # New: store all visibilities, elevations, and times for plotting
        pattern_visibility_plotdata[pattern_key] = {}

        # loop through the number of satellites for this pattern
        for num_satellites in num_satellites_list:
            print(f"Processing constellation with {num_satellites} satellites...")
            # Build the constellation
            if pattern == "Walker-Delta":
                if num_satellites < P:
                    print(f"WARNING: Skipping Walker-Delta pattern with {num_satellites} satellites (requires at least {P} satellites).")
                    continue  # Skip to the next iteration in the num_satellites loop
                try:
                    const = Constellation(num_satellites, orbital_parameters, epoch, pattern, P=P, F=F)
                except ValueError as e:
                    print(f"WARNING: Skipping Walker-Delta pattern with {num_satellites} satellites: {str(e)}")
                    continue  # Skip to the next iteration in the num_satellites loop
            else:
                const = Constellation(num_satellites, orbital_parameters, epoch, pattern)

            ### Propagation by constellation class function
            result = const.generate_positions_with_time(start_time_datetime, end_time_datetime, step) # ECI Cartesian

            positions = result["positions"] # A 3D numpy array of shape (num_satellites, num_time_steps, 3) with satellite positions in km
            velocities = result["velocities"] # A 3D numpy array of shape (num_satellites, num_time_steps, 3) with satellite velocities in km/s
            time_stamps = result["time_stamps"] # A 1D numpy array of shape (num_time_steps,) with time stamps in seconds since epoch.

            # #####################################
            # ### PLOTTING ########################
            # #####################################

            ## SELECT COLOR OF LEGEND
            # colors = cm.tab10(np.linspace(0, 1, num_satellites))  # Unique color for each satellite
            # colors = cm.plasma(np.linspace(0, 1, num_satellites))  # Use the 'plasma' colormap
            colors = [hsv_to_rgb((i / num_satellites, 0.8, 0.9)) for i in range(num_satellites)]    
            # colors = cm.tab20(np.linspace(0, 1, num_satellites))

            if PLOT_GROUND_TRACKS:
                #############################################
                # Plot ground track for this satellite (in loop)
                print(f"Plotting ground tracks of {num_satellites} satellites...")
                    # Add this debug code right after the ECI propagation in your main.py:
                
                n_periods = 1  # Number of periods to plot

                idx_oneperiod = np.searchsorted(time_stamps, orbital_period) # Index of one orbital period
                idx_max = n_periods * idx_oneperiod  # Index of total number of time steps to plot

                fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
                ax.stock_img()
                ax.coastlines()
                ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")

                PLOT_GROUND_STATION_MARKER = True  # Show ground station location on ground track plots

                if PLOT_GROUND_STATION_MARKER:
                    # Plot the ground station position
                    ax.plot(
                        gs_lon,
                        gs_lat,
                        marker="*",  # Star marker for ground station
                        markersize=12,
                        markerfacecolor='yellow',
                        markeredgecolor='black',
                        linestyle='',
                        transform=ccrs.PlateCarree(),
                        label=f"GS: {gs}"
                    )
                    # Add a text label next to the marker
                    ax.text(
                        gs_lon + 2,  # Offset to the right of the marker
                        gs_lat,
                        f"{gs}",
                        fontsize=9,
                        color='black',
                        transform=ccrs.PlateCarree()
                    )


                for satellite in range(num_satellites):
                    satellite_considered = satellite
                    # geographic_orbit = CoordinateConversion.ECI_cartesian_to_geographic(positions[satellite_considered, :idx_max, :], time_stamps[:idx_max]) # A list of tuples (time, latitude, longitude, radial_distance)

                    ECEF_cartesian_orbit = CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(positions[satellite_considered, :idx_max, :]*1000, time_stamps[:idx_max]) # A list of tuples (time, x, y, z), km to m
                    ECEF_geodetic_orbit = [CoordinateConversion.ECEF_cartesian_to_geodetic(x, y, z) for x, y, z in ECEF_cartesian_orbit] # A list of tuples (latitude, longitude, altitude)

                    # # Extract data
                    # times = [entry[0] for entry in geographic_orbit]
                    # latitudes = [entry[1] for entry in geographic_orbit]
                    # longitudes = [entry[2] for entry in geographic_orbit]
                    latitudes, longitudes, altitudes = zip(*ECEF_geodetic_orbit)  # Unzip the tuples
                    
                    # Handle longitude discontinuities
                    unwrapped_longitudes = np.unwrap(np.radians(longitudes))  # Unwrap in radians
                    normalized_longitudes = np.degrees(unwrapped_longitudes)  # Convert back to degrees
        
                    label = create_label_from_satrec(satellite, const.satrecs[satellite], pattern, num_satellites, P if pattern == 'Walker-Delta' else None)                
                    
                    ax.plot(
                        normalized_longitudes,
                        latitudes,
                        marker="o",
                        markersize=2,
                        linestyle="-.",
                        color=colors[satellite],  # Assign unique color
                        label=label,
                        transform=ccrs.PlateCarree(),
                    )

                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.13),  # Centered below the plot
                    fontsize="x-small",
                    ncol=5,                       # Adjust columns as needed
                    borderaxespad=0
                )
                
                if num_satellites > 90:
                    plt.tight_layout(rect=[0, -0.06, 1, 0.95]) # Leave more space at the bottom for bigger legend
                elif num_satellites > 49:
                    plt.tight_layout(rect=[0, -0.03, 1, 0.95])  # Leave less space at the bottom for the legend
                else:
                    plt.tight_layout(rect=[0, 0.00, 1, 0.95])  # Does not leave space at the bottom for the legend

                # Dynamically include orbital parameters in the title and filename
                orbital_params_str = (
                    f"a={orbital_parameters['semi_major_axis']}km / h={orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
                    f"e={orbital_parameters['eccentricity']}, "
                    f"i={orbital_parameters['inclination']}°, "
                    f"RAAN varies by satellite, "
                    f"ω={orbital_parameters['argument_of_periapsis']}°"
                )

                if pattern == 'Walker-Delta':
                    ax.set_title(f"Ground tracks of a {pattern} constellation {orbital_parameters['inclination']}:{num_satellites}/{P}/{F}, during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/ground_tracks/{pattern}/{orbital_parameters['inclination']}-{P}-{F}/{n_periods}T/constellation_ground_tracks_step{step}_N{num_satellites}-{P}-{F}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                elif pattern == 'Walker-Delta-SinglePlane':
                    # Dynamically include orbital parameters in the title and filename
                    orbital_params_str = (
                        f"a={orbital_parameters['semi_major_axis']}km / h={orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
                        f"e={orbital_parameters['eccentricity']}, "
                        f"i={orbital_parameters['inclination']}°, "
                        f"RAAN={orbital_parameters['RAAN']}°, "
                        f"ω={orbital_parameters['argument_of_periapsis']}°"
                    )     
                    ax.set_title(f"Ground tracks of a {pattern} constellation with {num_satellites} sat(s) during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/ground_tracks/{pattern}/{n_periods}T/constellation_ground_tracks_step{step}_N{num_satellites}_RAAN{orbital_parameters['RAAN']}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                else:
                    ax.set_title(f"Ground tracks of a {pattern} constellation with {num_satellites} sat(s) during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/ground_tracks/{pattern}/{n_periods}T/constellation_ground_tracks_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename)

                pass

            if PLOT_INERTIAL_GROUND_TRACKS:
                #############################################
                # Plot inertial ground track (without Earth rotation)
                print(f"Plotting inertial ground tracks of {num_satellites} satellites...")
                
                n_periods = 1  # Number of periods to plot

                idx_oneperiod = np.searchsorted(time_stamps, orbital_period)
                idx_max = n_periods * idx_oneperiod

                fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
                ax.stock_img()
                ax.coastlines()
                ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")

                for satellite in range(num_satellites):
                    satellite_positions = positions[satellite, :idx_max, :]  # ECI positions in km
                    
                    # Convert ECI to inertial geodetic coordinates using the new function
                    eci_positions_m = satellite_positions * 1000  # Convert km to m
                    inertial_geodetic_orbit = CoordinateConversion.ECI_cartesian_to_ECI_geodetic(eci_positions_m)
                    
                    # Extract coordinates
                    latitudes, longitudes, altitudes = zip(*inertial_geodetic_orbit)
                    
                    # Handle longitude discontinuities
                    unwrapped_longitudes = np.unwrap(np.radians(longitudes))
                    normalized_longitudes = np.degrees(unwrapped_longitudes)

                    label = create_label_from_satrec(satellite, const.satrecs[satellite], pattern, num_satellites, P if pattern == 'Walker-Delta' else None)                
                    
                    # Plot the ground track
                    ax.plot(
                        normalized_longitudes,
                        latitudes,
                        marker="o",
                        markersize=2,
                        linestyle="--",
                        color=colors[satellite],
                        label=label,
                        transform=ccrs.PlateCarree(),
                    )

                # Add legend and formatting
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.13),
                    fontsize="x-small",
                    ncol=5,
                    borderaxespad=0
                )
                
                if num_satellites > 90:
                    plt.tight_layout(rect=[0, -0.06, 1, 0.95])
                elif num_satellites > 49:
                    plt.tight_layout(rect=[0, -0.03, 1, 0.95])
                else:
                    plt.tight_layout(rect=[0, 0.00, 1, 0.95])

                # Create title and filename
                orbital_params_str = (
                    f"a={orbital_parameters['semi_major_axis']}km / h={orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
                    f"e={orbital_parameters['eccentricity']}, "
                    f"i={orbital_parameters['inclination']}°, "
                    f"RAAN={orbital_parameters['RAAN']}°, "
                    f"ω={orbital_parameters['argument_of_periapsis']}°"
                )
                
                ax.set_title(f"Inertial ground tracks of {pattern} constellation with {num_satellites} sat(s) during {n_periods} orbital period(s)\n({orbital_params_str})")
                filename = f"plots/inertial_ground_tracks/{pattern}/{n_periods}T/constellation_inertial_ground_tracks_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename)

            if PLOT_INERTIAL_3D:
                #############################################
                # Plot inertial orbit of satellite in 3D (in loop)
                print(f"Plotting inertial 3D orbits of {num_satellites} satellites...")
                
                n_periods = 1  # Number of periods to plot

                idx_oneperiod = np.searchsorted(time_stamps, orbital_period) # Index of one orbital period
                idx_max = n_periods * idx_oneperiod  # Index of total number of time steps to plot
                # 3D Plot of Satellite Orbits
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')

                # Plot the Earth
                earth_radius = constants.EARTH_RADIUS  # km
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x = earth_radius * np.outer(np.cos(u), np.sin(v))
                y = earth_radius * np.outer(np.sin(u), np.sin(v))
                z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, color='b', alpha=0.05, edgecolor='k')  # Earth as a sphere

                # Plot satellite positions as scatter points
                for satellite in range(num_satellites):
                    # positions[satellite_considered, :idx_max, :]*1000, time_stamps[:idx_max]
                    satellite_positions = positions[satellite, :idx_max, :]  # Extract positions for this satellite in km
                    
                    label = create_label_from_satrec(satellite, const.satrecs[satellite], pattern, num_satellites, P if pattern == 'Walker-Delta' else None)                

                    # Plot the scatter points
                    ax.scatter(
                        satellite_positions[:, 0],  # X-coordinates
                        satellite_positions[:, 1],  # Y-coordinates
                        satellite_positions[:, 2],  # Z-coordinates
                        label=label,
                        color=colors[satellite],
                        s=5,  # Size of scatter points
                        alpha=0.8  # Transparency for better visualization
                    )

                    # Plot the line connecting the scatter points as a dashed line
                    ax.plot(
                        satellite_positions[:, 0],  # X-coordinates
                        satellite_positions[:, 1],  # Y-coordinates
                        satellite_positions[:, 2],  # Z-coordinates
                        color=colors[satellite],
                        linewidth=1,  # Line width
                        alpha=0.6,    # Transparency for better visualization
                        linestyle='--'  # Dashed line
                    )

                # Set plot limits for better visualization
                max_range = np.max(np.abs(positions))  # Maximum range of satellite positions
                ax.set_xlim([-max_range, max_range])
                ax.set_ylim([-max_range, max_range])
                ax.set_zlim([-max_range, max_range])

                # Add labels and legend
                ax.set_xlabel("X [km]")
                ax.set_ylabel("Y [km]")
                ax.set_zlabel("Z [km]")
                # ax.legend(loc="upper left", fontsize="small")
                
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.13),  # Centered below the plot
                    fontsize="x-small",
                    ncol=4,                       # Adjust columns as needed
                    borderaxespad=0
                )
                
                if num_satellites > 90:
                    plt.tight_layout(rect=[0, -0.06, 1, 0.95]) # Leave more space at the bottom for bigger legend
                elif num_satellites > 49:
                    plt.tight_layout(rect=[0, -0.03, 1, 0.95])  # Leave less space at the bottom for the legend
                else:
                    plt.tight_layout(rect=[0, 0.00, 1, 0.95])  # Does not leave space at the bottom for the legend

                # Dynamically include orbital parameters in the title and filename
                orbital_params_str = (
                    f"a={orbital_parameters['semi_major_axis']}km / h={orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
                    f"e={orbital_parameters['eccentricity']}, "
                    f"i={orbital_parameters['inclination']}°, "
                    f"RAAN varies, "
                    f"ω={orbital_parameters['argument_of_periapsis']}°"
                )

                if pattern == 'Walker-Delta':
                    ax.set_title(f"3D inertial visualization of {pattern} constellation {orbital_parameters['inclination']}:{num_satellites}/{P}/{F}, during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/inertial_3D/{pattern}/{orbital_parameters['inclination']}-{P}-{F}/{n_periods}T/3D_ECI_Orbit_step{step}_N{num_satellites}-{P}-{F}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                elif pattern == 'Walker-Delta-SinglePlane':
                    # Dynamically include orbital parameters in the title and filename
                    orbital_params_str = (
                        f"a={orbital_parameters['semi_major_axis']}km / h={orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
                        f"e={orbital_parameters['eccentricity']}, "
                        f"i={orbital_parameters['inclination']}°, "
                        f"RAAN={orbital_parameters['RAAN']}°, "
                        f"ω={orbital_parameters['argument_of_periapsis']}°"
                    )     
                    ax.set_title(f"3D inertial visualization of {pattern} constellation with {num_satellites} sat(s) during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/inertial_3D/{pattern}/{n_periods}T/3D_ECI_Orbit_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                else:            
                    ax.set_title(f"3D inertial visualization of {pattern} constellation with {num_satellites} sat(s) during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/inertial_3D/{pattern}/{n_periods}T/3D_ECI_Orbit_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename)

                pass

            if PLOT_COVERAGE:
                #############################################
                ### Plot coverage for multiple satellites
                print(f"Plotting coverage of {num_satellites} satellites...")

                n_periods = 1  # Number of periods to plot

                idx_oneperiod = np.searchsorted(time_stamps, orbital_period) # Index of one orbital period
                idx_max = n_periods * idx_oneperiod  # Index of total number of time steps to plot


                # Extract orbital parameters for the subtitle
                orbital_params_str = (
                    f"a={orbital_parameters['semi_major_axis']}km / h={orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
                    f"e={orbital_parameters['eccentricity']}, "
                    f"i={orbital_parameters['inclination']}°, "
                    f"RAAN varies, "
                    f"ω={orbital_parameters['argument_of_periapsis']}°"
                )

                for masking_angle in masking_angles:
                    for fov in fovs:
                        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
                        ax.stock_img()
                        ax.coastlines()
                        ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")

                    PLOT_GROUND_STATION_MARKER = True
                        
                    if PLOT_GROUND_STATION_MARKER:
                        # Plot the ground station position
                        ax.plot(
                            gs_lon,
                            gs_lat,
                            marker="*",  # Star marker for ground station
                            markersize=12,
                            markerfacecolor='yellow',
                            markeredgecolor='black',
                            linestyle='',
                            transform=ccrs.PlateCarree(),
                            label=f"GS: {gs}"
                        )
                        # Add a text label next to the marker
                        ax.text(
                            gs_lon + 2,  # Offset to the right of the marker
                            gs_lat,
                            f"{gs}",
                            fontsize=9,
                            color='black',
                            transform=ccrs.PlateCarree()
                    )

                        # Loop through all satellites
                        for satellite_index in range(num_satellites):
                            for index in range(0, idx_max): # up until idx_max for one orbital period
                                state = {
                                    "position": positions[satellite_index, index, :],  # Position of the satellite in km
                                    "velocity": velocities[satellite_index, index, :],  # Velocity of the satellite in km/s
                                    "time": time_stamps[index],  # Time stamp
                                }

                                # Instantiate the Coverage class
                                coverage = Coverage(orbital_parameters, state)

                                # Compute sample coverage
                                coverage_points = coverage.compute_sample_coverage(masking_angle=masking_angle, fov=fov) # The function will decide which parameter is limiting the cap angle

                                coverage_points, fraction_covered, cap_angle = coverage.compute_sample_coverage(masking_angle=masking_angle, fov=fov)
                                
                                if coverage_points:  # Ensure there are valid coverage points
                                    lats, lons = zip(*coverage_points)
                                    ax.scatter(
                                        lons, lats, s=1, color=colors[satellite_index], alpha=0.99, transform=ccrs.PlateCarree()
                                    )

                        # Create legend labels
                        legend_labels = [
                            create_label_from_satrec(i, const.satrecs[i], pattern, num_satellites, P if pattern == 'Walker-Delta' else None)
                            for i in range(num_satellites)
                        ]

                        # Create legend handles (proxy artists)
                        legend_handles = [
                            Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=8)
                            for i in range(num_satellites)
                        ]

                        if pattern == 'Walker-Delta':
                            ax.set_title(f"Coverage areas of a {pattern} constellation {orbital_parameters['inclination']}:{num_satellites}/{P}/{F}, during {n_periods} orbital period(s)\n({orbital_params_str})\n Masking Angle: {masking_angle}°, FOV: {fov}°, Cap Angle: {cap_angle:.2f}°Instant. Fraction Covered: {100*fraction_covered:.4f}%")
                            filename = f"plots/coverage/{pattern}/{orbital_parameters['inclination']}-{P}-{F}/{n_periods}T/constellation_coverage_step{step}_N{num_satellites}-{P}-{F}_mask{masking_angle}_fov{fov}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                        elif pattern == 'Walker-Delta-SinglePlane':
                            # Dynamically include orbital parameters in the title and filename
                            orbital_params_str = (
                                f"a={orbital_parameters['semi_major_axis']}km / h={orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
                                f"e={orbital_parameters['eccentricity']}, "
                                f"i={orbital_parameters['inclination']}°, "
                                f"RAAN={orbital_parameters['RAAN']}°, "
                                f"ω={orbital_parameters['argument_of_periapsis']}°"
                            )     
                            ax.set_title(f"Coverage areas of a {pattern} constellation with {num_satellites} sat(s) during {n_periods} orbital period(s)\n({orbital_params_str})\n Masking Angle: {masking_angle}°, FOV: {fov}°, Cap Angle: {cap_angle:.2f}°Instant. Fraction Covered: {100*fraction_covered:.4f}%")
                            filename = f"plots/coverage/{pattern}/{n_periods}T/constellation_coverage_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                        else:
                            ax.set_title(f"Coverage areas of a {pattern} constellation with {num_satellites} sat(s) during {n_periods} orbital period(s)\n({orbital_params_str})\n Masking Angle: {masking_angle}°, FOV: {fov}°, Cap Angle: {cap_angle:.2f}°Instant. Fraction Covered: {100*fraction_covered:.4f}%")
                            filename = f"plots/coverage/{pattern}/{n_periods}T/constellation_coverage_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"

                        ax.legend(handles=legend_handles, labels=legend_labels,
                            loc="upper center",
                            bbox_to_anchor=(0.5, -0.13),  # Centered below the plot
                            fontsize="x-small",
                            ncol=5,                       # Adjust columns as needed
                            borderaxespad=0
                        )
                
                        if num_satellites > 90:
                            plt.tight_layout(rect=[-0.03, -0.06, 1, 1]) # Leave more space at the bottom for bigger legend
                        elif num_satellites > 49:
                            plt.tight_layout(rect=[-0.03, -0.03, 1, 1])  # Leave less space at the bottom for the legend
                        else:
                            plt.tight_layout(rect=[-0.03, 0.00, 1, 1])  # Does not leave space at the bottom for the legend

                        # Save the plot
                        # filename = f"plots/coverage/{pattern}/{n_periods}T/constellation_coverage_step{step}_N{num_satellites}-{P}-{F}_mask{masking_angle}_fov{fov}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        plt.savefig(filename)
                        
                        pass

            if COMPUTE_VISIBILITY:
                #############################################
                ### Compute and store visibility/overlap data for all satellites (NO plotting)
                print(f"Computing visibility and overlap for {num_satellites} satellites in {pattern}...")
                n_periods = 1
                idx_oneperiod = np.searchsorted(time_stamps, orbital_period) # Index of one orbital period
                idx_max = n_periods * idx_oneperiod  # Index of total number of time steps to plot
                elevation_threshold = 10.0  # degrees
                # Extract orbital parameters for the subtitle
                orbital_params_str = (
                    f"a={orbital_parameters['semi_major_axis']}km / h={orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
                    f"e={orbital_parameters['eccentricity']}, "
                    f"i={orbital_parameters['inclination']}°, "
                    f"RAAN varies, "
                    f"ω={orbital_parameters['argument_of_periapsis']}°"
                )
                all_visibilities = []
                all_elevations = []
                all_times = []
                for satellite_index in range(num_satellites): # saving visibilities, elevation, times, for each N satellites. Same pattern.
                    visibility = []
                    elevations = []
                    times = []
                    for idx in range(idx_max):
                        sat_position = positions[satellite_index, idx, :]  # km 
                        sat_time = time_stamps[idx]
                        ECEF_cartesian_orbit = CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(sat_position*1000, sat_time)
                        vis, elev = Coverage.is_visible_cartesian_ECEF(ECEF_cartesian_orbit, gs_geodetic, elevation_threshold)
                        visibility.append(int(np.asarray(vis).item()))
                        elevations.append(elev)
                        times.append(sat_time / 3600.0)
                    all_visibilities.append(visibility)
                    all_elevations.append(elevations)
                    all_times.append(times)
                # Store for plotting
                pattern_visibility_plotdata[pattern_key][num_satellites] = {
                    'visibilities': all_visibilities,
                    'elevations': all_elevations,
                    'times': all_times
                }
                # Store results for later plotting
                total_visible_seconds_all = np.sum(all_visibilities) * step
                any_visible_per_timestep = np.any(all_visibilities, axis=0)
                total_time_with_any_visible = np.sum(any_visible_per_timestep) * step
                if total_time_with_any_visible > 0: # this is to avoid division by zero when there is no visibility at all
                    overlap_percentage = ((total_visible_seconds_all - total_time_with_any_visible) / total_time_with_any_visible) * 100
                else:
                    overlap_percentage = 0
                # Store results for later plotting
                total_time_with_any_visible_list.append(total_time_with_any_visible / 3600)
                overlap_percentage_list.append(overlap_percentage)
                pattern_visibility_data[pattern_key]['total_time'].append(total_time_with_any_visible / 3600) # this saves the total visibility time for the pattern comparison
                pattern_visibility_data[pattern_key]['overlap'].append(overlap_percentage) # this saves the overlap percentage for the pattern comparison
                num_satellites_record.append(num_satellites) 
                pattern_num_satellites[pattern_key].append(num_satellites) # this saves the number of satellites for the pattern comparison
                # Precompute and store percentage of time with at least one visible satellite
                percentage_visible = total_time_with_any_visible * 100 / (len(times) * step) if len(times) > 0 else 0
                if 'percentage_visible_list' not in locals():
                    percentage_visible_list = []
                percentage_visible_list.append(percentage_visible)
                # Prepare overlap_text for the title
                overlap_text = f"overlap: {overlap_percentage:.2f}%" if total_time_with_any_visible > 0 else "no visibility"
                pass

            if PLOT_VISIBILITY:
                #############################################
                ### Only plot using precomputed data (NO computation)
                print(f"Plotting visibility and elevation angle of {num_satellites} satellites in {pattern}...")
                plotdata = pattern_visibility_plotdata[pattern_key][num_satellites]
                all_visibilities = np.array(plotdata['visibilities'])
                all_elevations = plotdata['elevations']
                all_times = plotdata['times']
                legend_labels = []
                fig = plt.figure(figsize=(12, 7))
                gs_plot = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.05)
                ax1 = fig.add_subplot(gs_plot[0])
                ax2 = fig.add_subplot(gs_plot[1], sharex=ax1)
                for satellite_index in range(num_satellites):
                    label = create_label_from_satrec(
                        satellite_index, const.satrecs[satellite_index], pattern, num_satellites, P if pattern == 'Walker-Delta' else None
                    )
                    legend_labels.append(label)
                    ax1.plot(all_times[satellite_index], all_visibilities[satellite_index], drawstyle='steps-post', color=colors[satellite_index], linestyle='--', label=label)
                    ax2.plot(all_times[satellite_index], all_elevations[satellite_index], color=colors[satellite_index], linestyle='-')
                ax1.set_ylabel(f"Visible (elev > {elevation_threshold}°)")
                ax1.set_yticks([0, 1])
                ax1.set_yticklabels(["No", "Yes"])
                ax1.grid(True)
                plt.setp(ax1.get_xticklabels(), visible=False)

                ax2.set_xlabel("Time [hours]")
                ax2.set_ylabel("Elevation [deg]")
                ax2.grid(True)
                # Add dashed horizontal line for elevation threshold
                ax2.axhline(elevation_threshold, color='gray', linestyle='--', linewidth=1, label=f"Threshold ({elevation_threshold}°)")
                
                handles, labels = ax1.get_legend_handles_labels()
                fig.legend(
                    handles, labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.01),
                    fontsize="x-small",  # or "small"
                    ncol=5,              # Adjust as needed
                    borderaxespad=0
                )

                # Only show legend for threshold line if not already present
                handles, labels = ax2.get_legend_handles_labels()
                if handles:
                    ax2.legend(loc="upper right", fontsize="small")

                plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space at the bottom for the legend
                
                # Save the plot
                if pattern == 'Walker-Delta':
                    ax1.set_title(f"Satellite Visibility from {gs} Ground Station a {pattern} constellation\n{orbital_parameters['inclination']}:{num_satellites}/{P}/{F}, during {n_periods} orbital period(s)\n({orbital_params_str})\nTotal time with at least one visible sat: {total_time_with_any_visible:.2f} s ({total_time_with_any_visible/3600:.2f} hr), {percentage_visible:.2f}%; {overlap_text}", fontsize=10)
                    filename = f"plots/visibility/{pattern}/{orbital_parameters['inclination']}-{P}-{F}/{n_periods}T/visibility_multiple_satellites_{gs}_step{step}_N{num_satellites}-{P}-{F}_el{elevation_threshold}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                elif pattern == 'Walker-Delta-SinglePlane':
                    # Extract orbital parameters for the subtitle
                    orbital_params_str = (
                        f"a={orbital_parameters['semi_major_axis']}km / h={orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
                        f"e={orbital_parameters['eccentricity']}, "
                        f"i={orbital_parameters['inclination']}°, "
                        f"RAAN={orbital_parameters['RAAN']}°, "
                        f"ω={orbital_parameters['argument_of_periapsis']}°"
                    )
                    ax1.set_title(f"Satellite Visibility from {gs} Ground Station a {pattern} constellation\nwith N{num_satellites} sat(s) at RAAN {orbital_parameters['RAAN']}°, during {n_periods} orbital period(s)\n({orbital_params_str})\nTotal time with at least one visible sat: {total_time_with_any_visible:.2f} s ({total_time_with_any_visible/3600:.2f} hr), {percentage_visible:.2f}%; {overlap_text}", fontsize=10)
                    filename = f"plots/visibility/{pattern}/{n_periods}T/visibility_multiple_satellites_{gs}_step{step}_N{num_satellites}_RAAN{orbital_parameters['RAAN']}_el{elevation_threshold}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                else:
                    ax1.set_title(f"Satellite Visibility from {gs} Ground Station a {pattern} constellation\nwith N{num_satellites} sat(s), during {n_periods} orbital period(s)\n({orbital_params_str})\nTotal time with at least one visible sat: {total_time_with_any_visible:.2f} s ({total_time_with_any_visible/3600:.2f} hr), {percentage_visible:.2f}%; {overlap_text}", fontsize=10)
                    filename = f"plots/visibility/{pattern}/{n_periods}T/visibility_multiple_satellites_{gs}_step{step}_N{num_satellites}_el{elevation_threshold}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename, bbox_inches='tight')

                pass
            
            # END OF NUM_SATELLITES LOOP
            #############################################

        ### STILL IN THE PATTERN LOOP
        if PLOT_DATA_COMPARISON:
            print(f"Plotting comparison of visibility and elevation angle wrt to number of satellites in {pattern_key}...")
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 7), gridspec_kw={'height_ratios': [2, 1]})

            # Save the plot
            if pattern == 'Walker-Delta':
                ax1.set_title(f"Visibility and Overlap vs. N of Satellites\n from {gs} Ground Station over {n_periods} orbital period(s)\n({orbital_params_str})\nPattern: {pattern} {orbital_parameters['inclination']}:/{P}/{F}")
                filename = f"plots/visibility/{pattern}/{orbital_parameters['inclination']}-{P}-{F}/{n_periods}T/DATA_comparison_{gs}_step{step}_{orbital_parameters['inclination']}-{P}-{F}-el{elevation_threshold}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
            elif pattern == 'Walker-Delta-SinglePlane':
                # orbital_params_str already passed before 

                ax1.set_title(f"Visibility and Overlap vs. N of Satellites\n from {gs} Ground Station over {n_periods} orbital period(s)\n({orbital_params_str})\nPattern: {pattern} RAAN={orbital_parameters['RAAN']}")
                filename = f"plots/visibility/{pattern}/{n_periods}T/DATA_comparison_{gs}_step{step}_el{elevation_threshold}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}_RAAN{orbital_parameters['RAAN']}.png"
            else:
                ax1.set_title(f"Visibility and Overlap vs. N of Satellites\n from {gs} Ground Station over {n_periods} orbital period(s)\n({orbital_params_str})\nPattern: {pattern}")
                filename = f"plots/visibility/{pattern}/{n_periods}T/DATA_comparison_{gs}_step{step}_el{elevation_threshold}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"

            # Top plot: total time with at least one satellite visible
            ax1.plot(num_satellites_record, total_time_with_any_visible_list, marker='o')
            ax1.set_ylabel("Total time with\nat least one visible sat [hr]")
            ax1.grid(True)
            # ax1.set_title(f"Visibility and Overlap vs. N of Satellites\n from {gs} Ground Station over {n_periods} orbital period(s)\n({orbital_params_str})")

            # Bottom plot: overlap percentage
            ax2.plot(num_satellites_record, overlap_percentage_list, marker='o', color='orange')
            ax2.set_xlabel("Number of satellites in constellation")
            ax2.set_ylabel("Overlap percentage [%]")
            # ax2.set_yscale('log') # There are important 0 values, so log scale is not appropriate
            ax2.grid(True)

            # Add a secondary y-axis for percentage of simulation time
            total_simulation_time_hr = n_periods * (orbital_period / 3600)  # in hours
            ax1b = ax1.twinx()
            # Set the secondary y-axis limits to match the left axis, but as percentage
            left_min, left_max = ax1.get_ylim()
            ax1b.set_ylim(left_min / total_simulation_time_hr * 100, left_max / total_simulation_time_hr * 100)
            ax1b.set_ylabel("Total time with\nat least one visible sat [% of propagation time]")
            ax1b.grid(False)  # Don't show grid for secondary axis

            plt.tight_layout()

            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
        
            pass

    ### Pattern Comparison Plot ###
    if PLOT_PATTERN_VISIBILITY_COMPARISON:
        print(f"Plotting comparison of visibility and elevation angle wrt to number of satellites for different patterns: {pattern_list}...")
        
        # Define a richer set of colors using colormap
        num_patterns = len([p for p in pattern_list if isinstance(p, tuple) and p[0] == "Walker-Delta"]) + 2  # Walker-Delta patterns + 2 special patterns
        pattern_colors = []
        
        # Add colors from a colormap - using tab20 for up to 20 distinct colors
        if num_patterns <= 20:
            cmap = plt.colormaps['tab20']
            pattern_colors = [cmap(i/num_patterns) for i in range(num_patterns)]
        else:
            # For more than 20 patterns, we'll use hsv which can generate more distinct colors
            cmap = plt.colormaps['hsv']
            pattern_colors = [cmap(i/num_patterns) for i in range(num_patterns)]
        
        # Enhanced line styles
        pattern_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10)), (0, (3, 10, 1, 10))]
        
        # Build pattern_labels dynamically from the patterns we've already processed
        patterns_to_plot = []
        pattern_labels = []

        # Skip initializing the dictionaries here since they were already created in the main processing loop
        
        # Only use patterns that have data
        for pattern_item in pattern_list:
            # Extract pattern key properly
            if isinstance(pattern_item, tuple):
                base_pattern = pattern_item[0]
                if base_pattern == "Walker-Delta":
                    P = pattern_item[1].get('P', 1)
                    F = pattern_item[1].get('F', 1)
                    pattern_key = f"{base_pattern}_{P}_{F}"
                else:
                    pattern_key = base_pattern
            else:
                pattern_key = pattern_item
                base_pattern = pattern_item
            
            if pattern_key in pattern_visibility_data and len(pattern_visibility_data[pattern_key]['total_time']) > 0:
                patterns_to_plot.append(pattern_key)
                
                # Generate appropriate label based on pattern type
                if pattern_key.startswith("Walker-Delta_"):
                    # Extract P and F from pattern_key
                    parts = pattern_key.split('_')
                    P = int(parts[1])
                    F = int(parts[2])
                    label = f"Walker-Delta i={orbital_parameters['inclination']}:N/{P}/{F}"
                elif pattern_key == "Walker-Delta-SingleSatPerPlane":
                    label = f"Walker-Delta-SingleSatPerPlane i={orbital_parameters['inclination']}°"
                elif pattern_key == "Walker-Delta-SinglePlane":
                    label = f"Walker-Delta-SinglePlane i={orbital_parameters['inclination']}° RAAN={orbital_parameters['RAAN']}°"
                else:
                    label = pattern_key
                
                pattern_labels.append(label)
        
        # Group patterns by type for better color organization
        walker_indices = []
        special_indices = []
        for i, pat in enumerate(patterns_to_plot):
            if pat.startswith("Walker-Delta_"):
                walker_indices.append(i)
            else:
                special_indices.append(i)
        
        # Plot the comparison using the pre-calculated data
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Define only dashed/dotted line styles (no solid lines)
        dashed_styles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (1, 1))]
        
        # Plot each pattern's data with organized colors
        for idx, i in enumerate(walker_indices + special_indices):
            pat = patterns_to_plot[i]
            # Use the pre-computed data
            time_data = pattern_visibility_data[pat]['total_time']
            overlap_data = pattern_visibility_data[pat]['overlap']
            satellites = pattern_num_satellites[pat]
            
            # Determine style based on pattern type
            style_idx = idx % len(dashed_styles)
            
            # Special styling for non-Walker-Delta patterns
            if not pat.startswith("Walker-Delta_"):
                # Use distinctive styles for special patterns
                if pat == "Walker-Delta-SinglePlane":
                    color = 'red'
                    linestyle = '--'  # Changed to dashed
                    marker = 's'  # square marker
                    markersize = 4  # Reduced marker size
                    zorder = 10  # bring to front
                elif pat == "Walker-Delta-SingleSatPerPlane":
                    color = 'blue'
                    linestyle = '-.'  # Changed to dash-dot
                    marker = 'd'  # diamond marker
                    markersize = 4  # Reduced marker size
                    zorder = 10  # bring to front
                else:
                    color = pattern_colors[idx]
                    linestyle = dashed_styles[style_idx]
                    marker = 'o'
                    markersize = 3  # Reduced marker size
                    zorder = 5
            else:
                # Walker-Delta patterns - get P value for color grading
                parts = pat.split('_')
                P = int(parts[1])
                
                # Use color based on P value
                color = pattern_colors[idx]
                linestyle = dashed_styles[style_idx]
                marker = 'o'
                markersize = 3  # Reduced marker size
                zorder = 5
            
            # Plot visibility time (top plot)
            ax1.plot(
                satellites, 
                time_data, 
                linestyle=linestyle, 
                color=color, 
                marker=marker, 
                label=pattern_labels[i],
                markersize=markersize,
                linewidth=2,
                zorder=zorder
            )
            
            # Plot overlap percentage (bottom plot)
            ax2.plot(
                satellites, 
                overlap_data, 
                linestyle=linestyle, 
                color=color, 
                marker=marker, 
                label=pattern_labels[i],
                markersize=markersize,
                linewidth=2,
                zorder=zorder
            )
        
        # Add labels and styling for top plot (visibility time)
        ax1.set_ylabel("Total time with\nat least one visible sat [hr]", fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(
            f"Pattern Comparison: Visibility and Overlap vs. N of Satellites\n"
            f"from {gs} Ground Station over {n_periods} orbital period(s)\n"
            f"(a={orbital_parameters['semi_major_axis']}km, e={orbital_parameters['eccentricity']}, i={orbital_parameters['inclination']}°)",
            fontsize=12
        )
        
        # Group legends by type
        walker_delta_lines = []
        walker_delta_labels = []
        special_lines = []
        special_labels = []
        
        for i, pat in enumerate(patterns_to_plot):
            if pat.startswith("Walker-Delta_"):
                walker_delta_lines.append(ax1.get_lines()[i])
                walker_delta_labels.append(pattern_labels[i])
            else:
                special_lines.append(ax1.get_lines()[i])
                special_labels.append(pattern_labels[i])
        
        # Create legends outside the plot at the bottom
        if walker_delta_lines:
            # Remove legend from inside the plot and place it at the bottom
            first_legend = fig.legend(
                handles=walker_delta_lines,
                labels=walker_delta_labels,
                title="Walker-Delta Patterns",
                fontsize=9,
                loc='lower center',
                bbox_to_anchor=(0.5, 0),
                ncol=3
            )
        
        if special_lines:
            # Position the special patterns legend below the Walker-Delta legend
            legend_y_pos = -0.05 if not walker_delta_lines else -0.12
            second_legend = fig.legend(
                handles=special_lines,
                labels=special_labels,
                title="Special Patterns",
                fontsize=9,
                loc='lower center',
                bbox_to_anchor=(0.5, legend_y_pos),
                ncol=2
            )
        
        # Add labels and styling for bottom plot (overlap percentage)
        ax2.set_xlabel("Number of satellites in constellation", fontsize=11)
        ax2.set_ylabel("Overlap percentage [%]", fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Removed annotations for highest values to avoid covering data points
        
        # Add a secondary y-axis for percentage of simulation time
        total_simulation_time_hr = n_periods * (orbital_period / 3600)  # in hours
        ax1b = ax1.twinx()
        # Set the secondary y-axis limits to match the left axis, but as percentage
        left_min, left_max = ax1.get_ylim()
        ax1b.set_ylim(left_min / total_simulation_time_hr * 100, left_max / total_simulation_time_hr * 100)
        ax1b.set_ylabel("Total time with\nat least one visible sat [% of propagation time]", fontsize=11)
        ax1b.grid(False)  # Don't show grid for secondary axis

        # Add pattern count information to the plot
        ax1.text(
            0.02, 0.95, 
            f"Total patterns: {len(patterns_to_plot)}", 
            transform=ax1.transAxes,
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
        )
        
        # Improve tick label size
        for ax in [ax1, ax2, ax1b]:
            ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Calculate extra space needed for legends based on number of patterns
        legend_space = 0.02 if len(patterns_to_plot) <= 5 else 0.5
        if special_lines and walker_delta_lines:
            legend_space += 0.01  # Add more space for two legend groups
            
        # Adjust spacing between subplots - leave space at bottom for legends
        plt.tight_layout(rect=[0, legend_space, 1, 0.98])
        
        # Save the comparison plot with secondary axis
        filename = f"plots/visibility/PatternComparison/{n_periods}T/PatternComparison_{gs}_step{step}_el{elevation_threshold}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')