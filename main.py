from utilities import CoordinateConversion, orbiter, TimeConversion
from utilities_constellations import Constellation, create_label_from_satrec
from utilities_coverage import Coverage, get_ground_station_coordinates, get_available_regions
import constants

from datetime import datetime, timezone, timedelta
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
# Switches for plotting
PLOT_GROUND_TRACKS = True
PLOT_INERTIAL_GROUND_TRACKS = False
n_periods_ground_tracks = 1
PLOT_INERTIAL_3D = False
n_periods_3d = 1
PLOT_ECEF_3D = False
n_periods_ecef3d = n_periods_3d
ANIMATE_CONSTELLATION = False # needs PLOT_ECEF_3D to be True
COMPUTE_GLOBAL_COVERAGE = False
masking_angles = [10.0]  # degrees, the larger the smaller the coverage area. Same as elevation angle threshold. Usually 10-15 deg min
fovs = [160] # degrees, field of view of the sensor. If too low is limiting, if it's large then the masking angle is limiting
n_periods_global_coverage = 2
COMPUTE_REGIONAL_COVERAGE = False
# regions_to_analyze = ["Western_US", "Europe", "Japan"]
regions_to_analyze = ["Western_US", "Eastern_US", "Continental_US", "Europe", "East_Asia", "South_America", "Africa", "Australia", "Arctic", "Antarctica", "Japan", "Italy", "United_Kingdom", "India", "Brazil", "Turkey", "Central_Mexico", "Alaska", "Hawaii", "Iran", "Nepal", "Philippines", "New_Zealand", "Chile", "Indonesia", "Greece", "Central_China", "Kamchatka", "Caribbean", "Eastern_Europe", "Central_Asia_Eastern_China", "Western_South_America_Coast", "West_Coast_US", "California", "Mediterranean", "Iberian_Peninsula", "Southeast_Asia", "Middle_East", "North_Africa", "Great_Lakes"]
n_periods_regional_coverage = n_periods_global_coverage
PLOT_COVERAGE = False
n_periods_plot_coverage = n_periods_global_coverage
COMPUTE_VISIBILITY = False
gs = "Sumatra_EQ"  # Ground station location INPUT
n_periods_compute_visibility = 1
PLOT_VISIBILITY = False # needs COMPUTE_VISIBILITY to be True
PLOT_DATA_COMPARISON = False # needs COMPUTE_VISIBILITY to be True
PLOT_PATTERN_VISIBILITY_COMPARISON = False # needs COMPUTE_VISIBILITY to be True

# num_satellites_list = [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,88,100] # compatible for walker P=4
# num_satellites_list = [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,66,72,78,84,90,96,99] # compatible for walker P=3
# num_satellites_list = [77] # Flower
# num_satellites_list = list(range(1, 101))
num_satellites_list = [1]

## SELECT ORBITAL PARAMETERS ###
orbital_parameters = {
    'semi_major_axis':     6882.732340969999,    # km
    'eccentricity':        0.00,    # unitless
    'inclination':         51.447684794794995,     # deg
    'RAAN':                354.870847,     # deg
    'argument_of_periapsis': 0,  # deg     
    'true_anomaly':        175.813353,      # deg
    'ndot':                1e-8,     # rev/day² # Includes derivatives
    'nddot':               1e-12,    # rev/day³
    'bstar':               2e-5      # 1/er
}
### FLOWERS ###
FLOWER_CONSTELLATIONS = {
    "basic_flower": { # need 2-4 sat
        "pattern_params": {"Np": 3, "Nd": 1, "Fn": 1, "Fd": 4, "Fh": 0}, # pattern_list.append(("Flower", {"Np": 3, "Nd": 1, "Fn": 1, "Fd": 4, "Fh": 0}))      # Basic flower (full)
                                                                        # pattern_list.append(("Flower", {"Np": 3, "Nd": 1, "Fn": 1, "Fd": 2, "Fh": 0}))      # Basic flower (not enough slots)
                                                                        # pattern_list.append(("Flower", {"Np": 3, "Nd": 2, "Fn": 1, "Fd": 2, "Fh": 0}))      # Basic flower (full)
                                                                        # pattern_list.append(("Flower", {"Np": 3, "Nd": 2, "Fn": 1, "Fd": 4, "Fh": 0}))      # Basic flower (empty slots)
        "orbital_params": {
            'semi_major_axis': 20270.418,    # km
            'eccentricity': 0.6557478,       # unitless
            'inclination': 63.4,             # deg
            'RAAN': 0,                       # deg
            'argument_of_periapsis': 270.0,  # deg     
            'true_anomaly': 0.0,             # deg
            'ndot': 1e-8,                    # rev/day²
            'nddot': 1e-12,                  # rev/day³
            'bstar': 2e-5,                   # 1/er
            'description': "Basic flower (full)"
        }
    },
    "tre_lacci": { # need 30 sat
        "pattern_params": {"Np": 31, "Nd": 11, "Fn": 7, "Fd": 10, "Fh": 0},
        "orbital_params": {
            'semi_major_axis': 21133.149,    # km
            'eccentricity': 0.2723216,       # unitless
            'inclination': 180.0,            # deg
            'RAAN': 0,                       # deg
            'argument_of_periapsis': 270.0,  # deg     
            'true_anomaly': 0.0,             # deg
            'ndot': 1e-8,                    # rev/day²
            'nddot': 1e-12,                  # rev/day³
            'bstar': 2e-5,                   # 1/er
            'description': "Planar (Tre Lacci): 31-11-N-7-10"
        }
    },
    "rings": { # need 90 sat
        "pattern_params": {"Np": 8, "Nd": 1, "Fn": 1, "Fd": 90, "Fh": 0},
        "orbital_params": {
            'semi_major_axis': 21133.149,    # km
            'eccentricity': 0.5565734193233578,       # unitless
            'inclination': 165.0,            # deg
            'RAAN': 0,                       # deg
            'argument_of_periapsis': 270.0,  # deg     
            'true_anomaly': 0.0,             # deg
            'ndot': 1e-8,                    # rev/day²
            'nddot': 1e-12,                  # rev/day³
            'bstar': 2e-5,                   # 1/er
            'description': "Rings: 8-1-N-1-90" 
        }
    },
    "straight_line": { # needs 5 sat
        "pattern_params": {"Np": 4, "Nd": 1, "Fn": 1, "Fd": 5, "Fh": 0},
        "orbital_params": {
            'semi_major_axis': 16732.862,    # km
            'eccentricity': 0.0000434,       # unitless
            'inclination': 174.8,            # deg
            'RAAN': 0,                       # deg
            'argument_of_periapsis': 270.0,  # deg     
            'true_anomaly': 0.0,             # deg
            'ndot': 1e-8,                    # rev/day²
            'nddot': 1e-12,                  # rev/day³
            'bstar': 2e-5,                   # 1/er
            'description': "Straight line: 4-1-N-1-5"
        }
    },
    "helix": { # needs 49 sat
        "pattern_params": {"Np": 15, "Nd": 7, "Fn": 23, "Fd": 49, "Fh": 0},
        "orbital_params": {
            'semi_major_axis': 25367.718,    # km
            'eccentricity': 0.3937911,       # unitless
            'inclination': 63.4,             # deg
            'RAAN': 0,                       # deg
            'argument_of_periapsis': 270.0,  # deg     
            'true_anomaly': 0.0,             # deg
            'ndot': 1e-8,                    # rev/day²
            'nddot': 1e-12,                  # rev/day³
            'bstar': 2e-5,                   # 1/er
            'description': "Helix: 15-7-N-23-49"
        }
    },
    "3_figure_8": { # needs 57 sat
        "pattern_params": {"Np": 37, "Nd": 18, "Fn": 57, "Fd": 6, "Fh": 19},
        "orbital_params": {
            'semi_major_axis': 26080.990,    # km
            'eccentricity': 0.0000327,       # unitless
            'inclination': 63.4,             # deg
            'RAAN': 0,                       # deg
            'argument_of_periapsis': 270.0,  # deg     
            'true_anomaly': 0.0,             # deg
            'ndot': 1e-8,                    # rev/day²
            'nddot': 1e-12,                  # rev/day³
            'bstar': 2e-5,                   # 1/er
            'description': "3 figure 8: 37-18-57-6-19"
        }
    },
    "basic_figure_8": { # needs 13 sat
        "pattern_params": {"Np": 12, "Nd": 1, "Fn": 1, "Fd": 13, "Fh": 0},
        "orbital_params": {
            'semi_major_axis': 8044.321,    # km
            'eccentricity': 0.0000,       # unitless
            'inclination': 116.6,             # deg
            'RAAN': 0,                       # deg
            'argument_of_periapsis': 270.0,  # deg     
            'true_anomaly': 0.0,             # deg
            'ndot': 1e-8,                    # rev/day²
            'nddot': 1e-12,                  # rev/day³
            'bstar': 2e-5,                   # 1/er
            'description': "Basic figure 8: 12-1-13-1-13"
        }
    },
    "lone_star": { # needs 77 sat
        "pattern_params": {"Np": 38, "Nd": 23, "Fn": 23, "Fd": 77, "Fh": 0},
        "orbital_params": {
            'semi_major_axis': 20000,    # km. needs h_p=1300 km
            'eccentricity': 0.615,       # unitless
            'inclination': 0,             # deg
            'RAAN': 0,                       # deg
            'argument_of_periapsis': 270.0,  # deg     
            'true_anomaly': 0.0,             # deg
            'ndot': 1e-8,                    # rev/day²
            'nddot': 1e-12,                  # rev/day³
            'bstar': 2e-5,                   # 1/er
            'description': "Basic figure 8: 12-1-13-1-13"
        }
    },
    
}
# selected_flower = "lone_star"  # Change this to try different configurations
# orbital_parameters = FLOWER_CONSTELLATIONS[selected_flower]["orbital_params"].copy()
# pattern_params_flower = FLOWER_CONSTELLATIONS[selected_flower]["pattern_params"]

# h_perigee = 1300  # km
# print(f'Target e given hp of {h_perigee} km and a of {orbital_parameters["semi_major_axis"]} km', {1-((constants.EARTH_RADIUS + h_perigee) / orbital_parameters["semi_major_axis"])})

### SELECT EPOCH ###
# epoch = datetime(2025,5,5,12,0,0, tzinfo=timezone.utc)
# epoch =datetime(2025, 5, 10, 0, 48, 8, 679437, tzinfo=timezone.utc) # v1
epoch = datetime(2025, 5, 6, 14, 6, 23, tzinfo=timezone.utc) #  v2

# ### SELECT A PATTERN ###
# pattern_list = []
# # pattern_list.append(("Flower", pattern_params_flower))

# pattern_list.append(("Walker-Delta", {"P": 1, "F": 1}))
# if COMPUTE_GLOBAL_COVERAGE or COMPUTE_REGIONAL_COVERAGE:
#     pattern_list = [
#         ("Walker-Delta", {"P": 1, "F": 1}),
#         ("Walker-Delta", {"P": 2, "F": 1}),
#         ("Walker-Delta", {"P": 4, "F": 1}),
#         ("Walker-Delta", {"P": 8, "F": 1})
#     ]
# else:
#     # Generate Walker-Delta patterns with P values and F values from 1 to P-1
#     pattern_list = []
#     for P in range(1, 10):  # P values from 1 to 10
#         for F in range(1, P):  # F values from 1 to P-1 (F must be less than P)
#             pattern_list.append(("Walker-Delta", n{"P": P, "F": F}))

#     # For P=1, there's no valid F value, so add this manually
#     pattern_list.append(("Walker-Delta", {"P": 1, "F": 1}))

#     pattern_list = [ "Walker-Delta", "Walker-Delta-SingleSatPerPlane", "Walker-Delta-SinglePlane"]
pattern_list = [("Walker-Delta", {"P": 1, "F": 1}),]

pattern_params = {} # saving parameters for each pattern for plotting outside the pattern loop

# Define propagation window & step. Make sure is at least the number of periods to be plotted
# start_time_datetime = datetime(2025, 5, 5, 12, 00, 00, 00, tzinfo=timezone.utc) # my input
# end_time_datetime =   datetime(2025, 5, 15, 12, 00, 00, 00, tzinfo=timezone.utc) # my input
start_time_datetime = epoch # my input
end_time_datetime =  epoch + timedelta(days=10) # my input
step =  100 # Time step input

# Estimate one period index to stop ground track after one orbit
a = orbital_parameters['semi_major_axis']  # km from Earth's center
orbital_period = 2 * np.pi * np.sqrt(a**3 / constants.GRAVITATIONAL_CONSTANT)

orbital_period_hours = int(orbital_period // 3600)
orbital_period_minutes = int((orbital_period % 3600) // 60)
orbital_period_seconds = int(orbital_period % 60)
print(f"Orbital period: {orbital_period_hours} hours, {orbital_period_minutes} minutes, {orbital_period_seconds} seconds ({orbital_period/3600:.4f} hr)\nPerigee: {((1-orbital_parameters['eccentricity'])*orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS):.0f} km")

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
            elif pattern == "Flower":
                Np = pattern_item[1].get('Np')
                Nd = pattern_item[1].get('Nd')
                Fn = pattern_item[1].get('Fn')
                Fd = pattern_item[1].get('Fd')
                Fh = pattern_item[1].get('Fh')
                pattern_key = f"{pattern}_Np{Np}_Nd{Nd}_Fn{Fn}_Fd{Fd}_Fh{Fh}"  # Unique key for Flower            
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
        elif pattern == "Flower":
            print(f"========== Processing pattern: {pattern} with Np={Np}, Nd={Nd}, Fn={Fn}, Fd={Fd}, Fh={Fh} ==========")
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

        ################################
        ### NUMBER OF SATELLITE LOOP ###
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
            
            elif pattern == "Flower":
                # Create a copy of orbital_parameters to avoid modifying the original
                flower_orbital_params = orbital_parameters.copy()
                # Add Flower parameters to the orbital_parameters dictionary
                flower_orbital_params['Np'] = Np # N petals
                flower_orbital_params['Nd'] = Nd # Slots per orbit
                flower_orbital_params['Fn'] = Fn
                flower_orbital_params['Fd'] = Fd # N of orbits (Fn=1)
                flower_orbital_params['Fh'] = Fh
                tau = Np/Nd
                if num_satellites > Nd * Fd:
                    print(f"WARNING: Skipping Flower pattern with {num_satellites} satellites (only {Nd * Fd} slots allowed).")
                    continue
                try:    
                    const = Constellation(num_satellites, flower_orbital_params, epoch, pattern)
                except ValueError as e:
                    print(f"WARNING: Skipping Flower pattern with {num_satellites} satellites: {str(e)}")
                    continue
            
            else:
                const = Constellation(num_satellites, orbital_parameters, epoch, pattern)

            ### Propagation by constellation class function
            result = const.generate_positions_with_time(start_time_datetime, end_time_datetime, step) # ECI Cartesian

            positions = result["positions"] # A 3D numpy array of shape (num_satellites, num_time_steps, 3) with satellite positions in km
            velocities = result["velocities"] # A 3D numpy array of shape (num_satellites, num_time_steps, 3) with satellite velocities in km/s
            time_stamps = result["time_stamps"] # A 1D numpy array of shape (num_time_steps,) with time stamps in seconds since epoch.

            # Calculate the anomalistic period useful for flower
            anomalistic_period = const.estimate_anomalistic_period()

            # Format for display (convert to hours and minutes)
            anomal_period_hours = int(anomalistic_period / 3600)
            anomal_period_minutes = int((anomalistic_period % 3600) / 60)
            anomal_period_seconds = int((anomalistic_period % 3600) / 3600)

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
                
                n_periods = n_periods_ground_tracks  # Number of periods to plot

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

                    ECEF_cartesian_orbit = CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(positions[satellite_considered, :idx_max, :]*1000, time_stamps[:idx_max], reference_time=epoch) # A list of tuples (time, x, y, z), km to m
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
                    # plt.tight_layout(rect=[0.0, 0.00, 1, 0.95])  # Does not leave space at the bottom for the legend
                    plt.subplots_adjust(left=0, bottom=0.15, right=1, top=0.85)
                # Dynamically include orbital parameters in the title and filename
                orbital_params_str = (
                    f"a={orbital_parameters['semi_major_axis']:.0f}km / h_p={((1-orbital_parameters['eccentricity'])*orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS):.0f} km, "
                    f"e={orbital_parameters['eccentricity']}, "
                    f"i={orbital_parameters['inclination']}°, "
                    f"RAAN varies, "
                    f"ω={orbital_parameters['argument_of_periapsis']}°, \n"
                    f'T_sidereal={orbital_period_hours} hr {orbital_period_minutes} min {orbital_period_seconds} s'
                )

                if pattern == 'Walker-Delta':
                    ax.set_title(f"Ground tracks of a {pattern} constellation {orbital_parameters['inclination']}:{num_satellites}/{P}/{F}, during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/ground_tracks/{pattern}/{orbital_parameters['inclination']}-{P}-{F}/{n_periods}T/gt_s{step}_N{num_satellites}-{P}-{F}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}_RAAN{orbital_parameters['RAAN']}.png"
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
                elif pattern == "Flower":
                    ax.set_title(f"Ground tracks of a {pattern} constellation with N={num_satellites}, Np={Np}, Nd={Nd}, Fn={Fn}, Fd={Fd}, Fh={Fh}, during {n_periods} orbital period(s)\n{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}\n({orbital_params_str})")
                    filename = (f"plots/ground_tracks/{pattern}/{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}/{n_periods}T/constellation_ground_tracks_step{step}_N{num_satellites}-{Np}-{Nd}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png")                
                    # Add extra padding at the top specifically for Flower constellation plots
                    plt.subplots_adjust(top=0.80)  # Increase top margin to make room for title
                else:
                    ax.set_title(f"Ground tracks of a {pattern} constellation with {num_satellites} sat(s) during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/ground_tracks/{pattern}/{n_periods}T/constellation_ground_tracks_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename)
                plt.close(fig)

                pass

            if PLOT_INERTIAL_GROUND_TRACKS:
                #############################################
                # Plot inertial ground track (without Earth rotation)
                print(f"Plotting inertial ground tracks of {num_satellites} satellites...")
                
                n_periods = n_periods_ground_tracks  # Number of periods to plot

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
                    plt.tight_layout(rect=[0, 0.00, 1, 0.87])

                # Dynamically include orbital parameters in the title and filename
                orbital_params_str = (
                    f"a={orbital_parameters['semi_major_axis']:.0f}km / h_p={((1-orbital_parameters['eccentricity'])*orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS):.0f} km, "
                    f"e={orbital_parameters['eccentricity']}, "
                    f"i={orbital_parameters['inclination']}°, "
                    f"RAAN varies, "
                    f"ω={orbital_parameters['argument_of_periapsis']}°, \n"
                    f'T_sidereal={orbital_period_hours} hr {orbital_period_minutes} min {orbital_period_seconds} s'
                )
                
                if pattern == "Walker-Delta":
                    ax.set_title(f"{pattern} constellation with {num_satellites} sat(s), P={P}, F={F}, during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/inertial_ground_tracks/{pattern}/{orbital_parameters['inclination']}-{P}-{F}/{n_periods}T/constellation_inertial_ground_tracks_step{step}_N{num_satellites}_P{P}_F{F}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                elif pattern == "Walker-Delta-SingleSatPerPlane":
                    ax.set_title(f"{pattern} constellation with {num_satellites} sat(s), during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/inertial_ground_tracks/{pattern}/{n_periods}T/constellation_inertial_ground_tracks_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                elif pattern == "Walker-Delta-SinglePlane":
                    ax.set_title(f"{pattern} constellation with {num_satellites} sat(s), RAAN={orbital_parameters['RAAN']}°, during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/inertial_ground_tracks/{pattern}/{n_periods}T/constellation_inertial_ground_tracks_step{step}_N{num_satellites}_RAAN{orbital_parameters['RAAN']}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                elif pattern == "Flower":
                    ax.set_title(f"{pattern} constellation with {num_satellites} sat(s), Np={Np}, Nd={Nd}, Fn={Fn}, Fd={Fd}, Fh={Fh}, during {n_periods} orbital period(s)\n{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}\n({orbital_params_str})")
                    filename = f"plots/inertial_ground_tracks/{pattern}/{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}/{n_periods}T/constellation_inertial_ground_tracks_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                else:
                    ax.set_title(f"{pattern} constellation with {num_satellites} sat(s), during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/inertial_ground_tracks/{pattern}/{n_periods}T/constellation_inertial_ground_tracks_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"                
                # ax.set_title(f"Inertial ground tracks of {pattern} constellation with {num_satellites} sat(s) during {n_periods} orbital period(s)\n({orbital_params_str})")
                # filename = f"plots/inertial_ground_tracks/{pattern}/{n_periods}T/constellation_inertial_ground_tracks_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename)
                plt.close(fig)

            if PLOT_INERTIAL_3D:
                #############################################
                # Plot inertial orbit of satellite in 3D (in loop)
                print(f"Plotting inertial 3D orbits of {num_satellites} satellites...")
                
                n_periods = n_periods_3d  # Number of periods to plot

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

                # Set title and filename based on pattern type
                orbital_params_str = (
                    f"a={orbital_parameters['semi_major_axis']:.0f}km / h_p={((1-orbital_parameters['eccentricity'])*orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS):.0f} km, "
                    f"e={orbital_parameters['eccentricity']}, "
                    f"i={orbital_parameters['inclination']}°, "
                    f"RAAN varies, "
                    f"ω={orbital_parameters['argument_of_periapsis']}°, \n"
                    f'T_sidereal={orbital_period_hours} hr {orbital_period_minutes} min {orbital_period_seconds} s'
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
                elif pattern == "Flower":
                    orbital_params_str = (
                        f"a={orbital_parameters['semi_major_axis']:.0f}km / h_p={((1-orbital_parameters['eccentricity'])*orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS):.0f} km, "
                        f"e={orbital_parameters['eccentricity']}, "
                        f"i={orbital_parameters['inclination']}°, "
                        f"RAAN varies, "
                        f"ω={orbital_parameters['argument_of_periapsis']}°, \n"
                        f'T_sidereal={orbital_period_hours} hr {orbital_period_minutes} min {orbital_period_seconds} s, '
                        f'T_flower={tau:.4f}*T_sidereal'
                    )
                    ax.set_title(f"3D inertial visualization of {pattern} constellation with {num_satellites} sat(s), Np={Np}, Nd={Nd}, Fn={Fn}, Fd={Fd}, Fh={Fh}\n{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh},during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/inertial_3D/{pattern}/{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}/{n_periods}T/3D_ECI_Orbit_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"                    
                    plt.subplots_adjust(top=0.85)  # Increase top margin to make room for title
                else:
                    ax.set_title(f"3D inertial visualization of {pattern} constellation with {num_satellites} sat(s) during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/inertial_3D/{pattern}/{n_periods}T/3D_ECI_Orbit_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename)
                plt.close(fig)     

                pass

            if PLOT_ECEF_3D:
                #############################################
                # Plot ECEF (Earth-Centered Earth-Fixed) orbit of satellite in 3D
                print(f"Plotting ECEF 3D orbits of {num_satellites} satellites...")
                
                n_periods = n_periods_ecef3d  # Number of periods to plot (reduced from 3 to avoid too much rotation)
                
                idx_oneperiod = np.searchsorted(time_stamps, orbital_period) # Index of one orbital period
                idx_max = n_periods * idx_oneperiod  # Index of total number of time steps to plot
                
                # Create two figures - one for standard 3D view and one for top-down view
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                fig_topdown = plt.figure(figsize=(10, 10))
                ax_topdown = fig_topdown.add_subplot(111, projection='3d')
                # Create a new figure for top-down snapshot (final positions only)
                fig_snapshot = plt.figure(figsize=(10, 10))
                ax_snapshot = fig_snapshot.add_subplot(111, projection='3d')
                
                # Plot the Earth
                earth_radius = constants.EARTH_RADIUS  # km
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x = earth_radius * np.outer(np.cos(u), np.sin(v))
                y = earth_radius * np.outer(np.sin(u), np.sin(v))
                z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
                # Regular 3D plot Earth
                ax.plot_surface(x, y, z, color='b', alpha=0.1, edgecolor='k', linewidth=0.25)
                
                # Top-down plot Earth
                ax_topdown.plot_surface(x, y, z, color='b', alpha=0.1, edgecolor='k', linewidth=0.25)
                ax_snapshot.plot_surface(x, y, z, color='b', alpha=0.1, edgecolor='k', linewidth=0.25)

                # Add continents as wireframe to show Earth's rotation
                # Simplified continent markers at key longitudes (0°, 90°, 180°, 270°)
                continent_radius = earth_radius * 1.01  # Slightly larger than Earth
                lon_markers = [0, np.pi/2, np.pi, 3*np.pi/2]
                for lon in lon_markers:
                    x_marker = continent_radius * np.cos(lon)
                    y_marker = continent_radius * np.sin(lon)
                    ax.plot([x_marker], [y_marker], [0], 'ro', markersize=4)
                
                # Plot satellite positions in ECEF frame
                for satellite in range(num_satellites):
                    # Extract ECI positions for this satellite
                    satellite_positions_eci = positions[satellite, :idx_max, :]  # ECI positions in km
                    
                    # Convert ECI to ECEF
                    # Convert to meters for the conversion function
                    eci_positions_m = satellite_positions_eci * 1000
                    
                    # Convert to ECEF using the reference time as the start time
                    ecef_positions = CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(
                        eci_positions_m, 
                        time_stamps[:idx_max],
                        reference_time=start_time_datetime
                    )
                    
                    # Convert back to km for plotting
                    ecef_positions_km = ecef_positions / 1000
                    
                    # Create label
                    label = create_label_from_satrec(satellite, const.satrecs[satellite], pattern, num_satellites, P if pattern == 'Walker-Delta' else None)
                    
                    # Plot the satellite trajectory
                    ax.scatter(
                        ecef_positions_km[:, 0],  # X-coordinates
                        ecef_positions_km[:, 1],  # Y-coordinates
                        ecef_positions_km[:, 2],  # Z-coordinates
                        label=label,
                        color=colors[satellite],
                        s=5,  # Size of scatter points
                        alpha=0.8  # Transparency
                    )
                    
                    # Plot the line connecting the scatter points
                    ax.plot(
                        ecef_positions_km[:, 0],
                        ecef_positions_km[:, 1],
                        ecef_positions_km[:, 2],
                        color=colors[satellite],
                        linewidth=1,
                        alpha=0.6,
                        linestyle='-'
                    )

                    # Plot on top-down plot
                    ax_topdown.scatter(
                        ecef_positions_km[:, 0],
                        ecef_positions_km[:, 1],
                        ecef_positions_km[:, 2],
                        label=label,
                        color=colors[satellite],
                        s=5,
                        alpha=0.8
                    )
                    
                    ax_topdown.plot(
                        ecef_positions_km[:, 0],
                        ecef_positions_km[:, 1],
                        ecef_positions_km[:, 2],
                        color=colors[satellite],
                        linewidth=1,
                        alpha=0.6,
                        linestyle='-'
                    )        

                    # Plot the final position
                    ax_snapshot.scatter(
                        ecef_positions_km[-1,0],  # X-coordinate
                        ecef_positions_km[-1,1],  # Y-coordinate
                        ecef_positions_km[-1,2],  # Z-coordinate
                        label=label,
                        color=colors[satellite],
                        s=30,  # Larger size for better visibility
                        alpha=1.0  # Full opacity
                    )                                
                
                # Set plot limits for better visualization
                max_range = np.max(np.abs(positions)) * 1.1  # 10% larger than max ECI positions
                ax.set_xlim([-max_range, max_range])
                ax.set_ylim([-max_range, max_range])
                ax.set_zlim([-max_range, max_range])

                ax_topdown.set_xlim([-max_range, max_range])
                ax_topdown.set_ylim([-max_range, max_range])
                ax_topdown.set_zlim([-max_range, max_range])

                ax_snapshot.set_xlim([-max_range, max_range])
                ax_snapshot.set_ylim([-max_range, max_range])
                ax_snapshot.set_zlim([-max_range, max_range])
                                
                # Add labels and legend
                ax.set_xlabel("X [km]")
                ax.set_ylabel("Y [km]")
                ax.set_zlabel("Z [km]")

                ax_topdown.set_xlabel("X [km]")
                ax_topdown.set_ylabel("Y [km]")
                ax_topdown.set_zlabel("Z [km]")    
                ax_topdown.tick_params(axis='y', pad=15, labelright=False, labelleft=True)  # Move y-axis tick labels to the left, Increase pad value to move further left
                ax_topdown.zaxis.set_ticklabels([])  # Remove Z tick labels
                ax_topdown.zaxis.line.set_visible(False)  # Hide Z axis line
                ax_topdown.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Make Z pane transparent

                ax_snapshot.set_xlabel("X [km]")
                ax_snapshot.set_ylabel("Y [km]")
                ax_snapshot.set_zlabel("Z [km]") 
                ax_snapshot.tick_params(axis='y', pad=15, labelright=False, labelleft=True) # Move y-axis tick labels to the left, Increase pad value to move further left
                ax_snapshot.zaxis.set_ticklabels([]) # Remove Z tick labels
                ax_snapshot.zaxis.line.set_visible(False) # Hide Z axis line
                ax_snapshot.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) # Make Z pane transparent

                # Set the view angle for top-down plot
                ax_topdown.view_init(elev=90, azim=-90)  # Look directly down the z-axis (from North pole)            
                ax_snapshot.view_init(elev=90, azim=-90)    

                # Add legends to both plots
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.00), fontsize="small", ncol=4)
                ax_topdown.legend(loc="upper center", bbox_to_anchor=(0.5, 0.12), fontsize="small", ncol=4) # moving legend box upwards. lot of vertical space from plot
                ax_snapshot.legend(loc="upper center", bbox_to_anchor=(0.5, 0.12), fontsize="small", ncol=4)
                
                # Handle layout based on number of satellites
                if num_satellites > 89:
                    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.2, top=0.9)
                    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.01), fontsize="x-small", ncol=7)
                    ax_topdown.legend(loc="upper center", bbox_to_anchor=(0.5, 0.14), fontsize="x-small", ncol=7) # moving legend box upwards. lot of vertical space from plot
                    ax_snapshot.legend(loc="upper center", bbox_to_anchor=(0.5, 0.14), fontsize="x-small", ncol=7)
                elif num_satellites > 75:
                    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.2, top=0.9)
                    fig_snapshot.subplots_adjust(left=0.12, right=0.88, bottom=0.05, top=0.95)
                    fig_topdown.subplots_adjust(left=0.12, right=0.88, bottom=0.05, top=0.95)
                    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.01), fontsize="x-small", ncol=7)
                    ax_topdown.legend(loc="upper center", bbox_to_anchor=(0.5, 0.14), fontsize="x-small", ncol=7) # moving legend box upwards. lot of vertical space from plot
                    ax_snapshot.legend(loc="upper center", bbox_to_anchor=(0.5, 0.14), fontsize="x-small", ncol=7)                
                elif num_satellites > 49:
                    plt.tight_layout(rect=[0, -0.03, 1, 0.95])
                    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize="x-small", ncol=5)
                    ax_topdown.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize="x-small", ncol=5)
                else:
                    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.2, top=0.9)
                    fig_snapshot.subplots_adjust(left=0.12, right=0.88, bottom=0.2, top=0.9)   

                # Set title and filename based on pattern type
                orbital_params_str = (
                    f"a={orbital_parameters['semi_major_axis']:.0f}km / h_p={((1-orbital_parameters['eccentricity'])*orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS):.0f} km, "
                    f"e={orbital_parameters['eccentricity']}, "
                    f"i={orbital_parameters['inclination']}°, "
                    f"RAAN varies, "
                    f"ω={orbital_parameters['argument_of_periapsis']}°, \n"
                    f'T_sidereal={orbital_period_hours} hr {orbital_period_minutes} min {orbital_period_seconds} s'
                )
                
                if pattern == 'Walker-Delta':
                    ax.set_title(f"3D ECEF visualization of {pattern} constellation {orbital_parameters['inclination']}:{num_satellites}/{P}/{F}, during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/ecef_3D/{pattern}/{orbital_parameters['inclination']}-{P}-{F}/{n_periods}T/3D_ECEF_Orbit_step{step}_N{num_satellites}-{P}-{F}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                
                    ax_topdown.set_title(f"Top-down ECEF view of {pattern} constellation {orbital_parameters['inclination']}:{num_satellites}/{P}/{F}, during {n_periods} orbital period(s)\n({orbital_params_str})")
                    topdown_filename = f"plots/ecef_topdown/{pattern}/{orbital_parameters['inclination']}-{P}-{F}/{n_periods}T/Topdown_ECEF_Orbit_step{step}_N{num_satellites}-{P}-{F}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"   

                    ax_snapshot.set_title(f"Final positions (top-down view) of {pattern} constellation {orbital_parameters['inclination']}:{num_satellites}/{P}/{F}\n({orbital_params_str})\n after {n_periods} orbital period(s)")
                    snapshot_filename = f"plots/ecef_topdown/{pattern}/{orbital_parameters['inclination']}-{P}-{F}/{n_periods}T/Snapshot_ECEF_step{step}_N{num_satellites}-{P}-{F}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"                                     
                elif pattern == 'Walker-Delta-SinglePlane':
                    orbital_params_str = (
                        f"a={orbital_parameters['semi_major_axis']}km / h={orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
                        f"e={orbital_parameters['eccentricity']}, "
                        f"i={orbital_parameters['inclination']}°, "
                        f"RAAN={orbital_parameters['RAAN']}°, "
                        f"ω={orbital_parameters['argument_of_periapsis']}°"
                    )
                    
                    ax.set_title(f"3D ECEF visualization of {pattern} constellation with {num_satellites} sat(s) during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/ecef_3D/{pattern}/{n_periods}T/3D_ECEF_Orbit_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"

                    ax_topdown.set_title(f"Top-down ECEF view of {pattern} constellation with {num_satellites} sat(s), during {n_periods} orbital period(s)\n({orbital_params_str})")
                    topdown_filename = f"plots/ecef_topdown/{pattern}/{n_periods}T/Topdown_ECEF_Orbit_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                elif pattern == "Flower":
                    orbital_params_str = (
                        f"a={orbital_parameters['semi_major_axis']:.0f}km / h_p={((1-orbital_parameters['eccentricity'])*orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS):.0f} km, "
                        f"e={orbital_parameters['eccentricity']}, "
                        f"i={orbital_parameters['inclination']}°, "
                        f"RAAN varies, "
                        f"ω={orbital_parameters['argument_of_periapsis']}°, \n"
                        f'T_sidereal={orbital_period_hours} hr {orbital_period_minutes} min {orbital_period_seconds} s, '
                        f'T_flower={tau:.4f}*T_sidereal'
                    )                    
                    ax.set_title(f"3D ECEF visualization of {pattern} constellation with {num_satellites} sat(s), Np={Np}, Nd={Nd}, Fn={Fn}, Fd={Fd}, Fh={Fh}\n{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}, during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/ecef_3D/{pattern}/{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}/{n_periods}T/3D_ECEF_Orbit_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}_omega{orbital_parameters['argument_of_periapsis']}.png"

                    ax_topdown.set_title(f"Top-down ECEF view of {pattern} constellation with {num_satellites} sat(s), Np={Np}, Nd={Nd}, Fn={Fn}, Fd={Fd}, Fh={Fh}\n{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}, during {n_periods} orbital period(s)\n({orbital_params_str})")
                    topdown_filename = f"plots/ecef_topdown/{pattern}/{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}/{n_periods}T/Topdown_ECEF_Orbit_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}_omega{orbital_parameters['argument_of_periapsis']}.png"

                    ax_snapshot.set_title(f"Final positions (top-down view) of {pattern} constellation with {num_satellites} sat(s), {Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}\n({orbital_params_str})\n after {n_periods} orbital period(s)")
                    snapshot_filename = f"plots/ecef_topdown/{pattern}/{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}/{n_periods}T/Snapshot_ECEF_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"       
                else:
                    ax.set_title(f"3D ECEF visualization of {pattern} constellation with {num_satellites} sat(s) during {n_periods} orbital period(s)\n({orbital_params_str})")
                    filename = f"plots/ecef_3D/{pattern}/{n_periods}T/3D_ECEF_Orbit_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"

                    ax_topdown.set_title(f"Top-down ECEF view of {pattern} constellation with {num_satellites} sat(s), during {n_periods} orbital period(s)\n({orbital_params_str})")
                    topdown_filename = f"plots/ecef_topdown/{pattern}/{n_periods}T/Topdown_ECEF_Orbit_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"

                    ax_snapshot.set_title(f"Final positions (top-down view) of {pattern} constellation with {num_satellites} sat(s)\n({orbital_params_str}\nafter {n_periods} orbital period(s))")
                    snapshot_filename = f"plots/ecef_topdown/{pattern}/{n_periods}T/Snapshot_ECEF_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"

                # Create directories if they don't exist and save figures
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.figure(fig.number)
                plt.savefig(filename)
                
                os.makedirs(os.path.dirname(topdown_filename), exist_ok=True)
                plt.figure(fig_topdown.number)
                plt.savefig(topdown_filename)

                os.makedirs(os.path.dirname(snapshot_filename), exist_ok=True)
                plt.figure(fig_snapshot.number)
                plt.savefig(snapshot_filename)
                
                plt.close(fig)
                plt.close(fig_topdown)
                plt.close(fig_snapshot)

            if PLOT_ECEF_3D and ANIMATE_CONSTELLATION:
                # Generate the animation using the already computed positions
                fps_anim = 60
                ANIMATE_TOPDOWN = False  # change only this one
                ANIMATE_3D = True  # 

                if ANIMATE_TOPDOWN == True:
                    view = "topdown"
                else:
                    view = "3d"

                if pattern == "Flower":
                    const.create_animation(
                        time_stamps=time_stamps,
                        positions=positions,
                        start_time_datetime=start_time_datetime,
                        n_periods=n_periods_ecef3d,
                        filename=f"plots/animations/{pattern}/{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}/{view}_animation_step{step}_{fps_anim}fps/{n_periods}T/N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.mp4",
                        view=view,
                        fps=fps_anim,
                        tail_length=5 # number of previous positions to show in tail
                    )
                else:    
                    const.create_animation(
                        time_stamps=time_stamps,
                        positions=positions,
                        start_time_datetime=start_time_datetime,
                        n_periods=n_periods_ecef3d,
                        filename=f"plots/animations/{pattern}/{view}_animation_step{step}_{fps_anim}fps/{n_periods}T/N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.mp4",
                        view=view,
                        fps=fps_anim,
                        tail_length=5 # number of previous positions to show in tail
                    )

            if COMPUTE_GLOBAL_COVERAGE:
                #############################################
                ### Compute global coverage using grid-based approach
                print(f"Computing global coverage for {num_satellites} satellites...")
                
                n_periods = n_periods_global_coverage  # Number of periods to consider for coverage computation

                SHOW_GROUND_TRACKS_ON_COVERAGE = True # Show satellite ground tracks on coverage plots (for debugging)

                # Create a Coverage object
                coverage_obj = Coverage(orbital_parameters, positions=positions, times=time_stamps)

                # Extract orbital parameters for the subtitle
                orbital_params_str = (
                    f"a={orbital_parameters['semi_major_axis']:.1f}km / h={(orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS):.1f} km, "
                    f"e={orbital_parameters['eccentricity']:.2f}, "
                    f"i={orbital_parameters['inclination']:.2f}°, "
                    f"RAAN varies, "
                    f"ω={orbital_parameters['argument_of_periapsis']:.2f}°"
                )
                # Define coverage parameters for computation
                for masking_angle in masking_angles:
                    for fov in fovs:
                        print(f"\nComputing global coverage with masking angle: {masking_angle}° and FOV: {fov}°")
                        
                        # Compute global coverage
                        global_coverage = coverage_obj.compute_global_coverage(
                            positions, 
                            time_stamps, 
                            orbital_periods=n_periods,  # Consider one orbital period
                            masking_angle=masking_angle, 
                            fov=fov
                        )
                        
                        # Print coverage statistics
                        constellation_coverage = global_coverage['constellation_coverage']
                        print(f"Global constellation coverage: {constellation_coverage:.2f}%")
                        
                        # Print individual satellite coverage
                        if len(global_coverage['single_sat_coverage']) > 1:
                            print(f"Individual satellite coverages:")
                            for sat_idx, sat_coverage in enumerate(global_coverage['single_sat_coverage']):
                                print(f"  Satellite {sat_idx+1}: {sat_coverage:.2f}%")
                            
                            # Calculate and print statistical measures
                            mean_coverage = np.mean(global_coverage['single_sat_coverage'])
                            median_coverage = np.median(global_coverage['single_sat_coverage'])
                            min_coverage = np.min(global_coverage['single_sat_coverage'])
                            max_coverage = np.max(global_coverage['single_sat_coverage'])
                            std_coverage = np.std(global_coverage['single_sat_coverage'])
                            
                            print(f"Coverage statistics:")
                            print(f"  Mean: {mean_coverage:.2f}%")
                            print(f"  Median: {median_coverage:.2f}%")
                            print(f"  Min: {min_coverage:.2f}%")
                            print(f"  Max: {max_coverage:.2f}%")
                            print(f"  Std dev: {std_coverage:.2f}%")
                        
                        # Compute grid metrics
                        grid = global_coverage['grid']
                        grid_density = grid['density']
                        num_lat_cells = len(grid['lat'])
                        num_lon_cells = len(grid['lon'])
                        total_cells = num_lat_cells * num_lon_cells
                        covered_cells = np.sum(global_coverage['coverage_map'])
                        
                        print(f"Grid information:")
                        print(f"  Density: {grid_density:.2f}°")
                        print(f"  Grid size: {num_lat_cells} × {num_lon_cells} = {total_cells} cells")
                        print(f"  Covered cells: {covered_cells} ({covered_cells/total_cells*100:.2f}%)")
                        
                        if pattern == 'Walker-Delta':
                            title = f"Grid-based global coverage of a {pattern} constellation {orbital_parameters['inclination']}:{num_satellites}/{P}/{F}, during {n_periods} orbital period(s)\n({orbital_params_str})\nMasking Angle: {masking_angle}°, FOV: {fov}°, Total Coverage: {constellation_coverage:.2f}%"
                            filename = f"plots/grid_coverage/global/{pattern}/{orbital_parameters['inclination']}-{P}-{F}/{n_periods}T/mask{masking_angle}_fov{fov}/constellation_coverage_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"                            
                        elif pattern == 'Walker-Delta-SinglePlane':
                            orbital_params_str = (
                                f"a={orbital_parameters['semi_major_axis']:.1f}km / h={(orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS):.1f} km, "
                                f"e={orbital_parameters['eccentricity']}, "
                                f"i={orbital_parameters['inclination']}°, "
                                f"RAAN={orbital_parameters['RAAN']}°, "
                                f"ω={orbital_parameters['argument_of_periapsis']}°"
                            )     
                            title = f"Grid-based global coverage of a {pattern} constellation with {num_satellites} sat(s) during {n_periods} orbital period(s)\n({orbital_params_str})\nMasking Angle: {masking_angle}°, FOV: {fov}°, Total Coverage: {constellation_coverage:.2f}%"
                            filename = f"plots/grid_coverage/global/{pattern}/{n_periods}T/mask{masking_angle}_fov{fov}/constellation_coverage_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"                                                        
                        elif pattern == "Flower":
                            title = f"Grid-based global coverage of a {pattern} constellation with {num_satellites} sat(s), Np={Np}, Nd={Nd}, Fn={Fn}, Fd={Fd}, Fh={Fh}, during {n_periods} orbital period(s)\n({orbital_params_str})\nMasking Angle: {masking_angle}°, FOV: {fov}°, Total Coverage: {constellation_coverage:.2f}% ({orbital_params_str})"
                            filename = f"plots/grid_coverage/global/{pattern}/{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}/{n_periods}T/mask{masking_angle}_fov{fov}/constellation_coverage_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"                                                        
                        else:
                            title = f"Grid-based global coverage of a {pattern} constellation with {num_satellites} sat(s) during {n_periods} orbital period(s)\n({orbital_params_str})\nMasking Angle: {masking_angle}°, FOV: {fov}°, Total Coverage: {constellation_coverage:.2f}%"
                            filename = f"plots/grid_coverage/global/{pattern}/{n_periods}T/mask{masking_angle}_fov{fov}/constellation_coverage_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"                                                        
                        # Plot the coverage map with binary colormap (covered or not covered)
                        fig = Coverage.plot_coverage_map(global_coverage, title=title)
                        
                        # Add ground tracks if enabled
                        if SHOW_GROUND_TRACKS_ON_COVERAGE:
                            ax = fig.axes[0]  # Get the main plot axis
                            print("Adding ground tracks to coverage plot for debugging...")
                            
                            # Determine how many time steps to include
                            idx_oneperiod = np.searchsorted(time_stamps, orbital_period)
                            idx_max = idx_oneperiod  # One orbital period
                            
                            for satellite in range(num_satellites):
                                # Convert to ECEF and then to geodetic coordinates
                                ECEF_cartesian_orbit = CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(
                                    positions[satellite, :idx_max, :]*1000, time_stamps[:idx_max])
                                ECEF_geodetic_orbit = [CoordinateConversion.ECEF_cartesian_to_geodetic(x, y, z) 
                                                     for x, y, z in ECEF_cartesian_orbit]
                                
                                # Extract coordinates
                                latitudes, longitudes, altitudes = zip(*ECEF_geodetic_orbit)
                                
                                # Handle longitude discontinuities
                                unwrapped_longitudes = np.unwrap(np.radians(longitudes))
                                normalized_longitudes = np.degrees(unwrapped_longitudes)
                                
                                # Plot the ground track with dotted line
                                ax.plot(
                                    normalized_longitudes,
                                    latitudes,
                                    linestyle=':', 
                                    linewidth=1,
                                    color='red',  # Use a visible color
                                    transform=ccrs.PlateCarree(),
                                    alpha=0.7,  # Slightly transparent
                                    zorder=5  # To ensure it's drawn on top
                                )
                        
                        # Create directory and save the plot
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        fig.savefig(filename)
                        plt.close(fig)

            if COMPUTE_REGIONAL_COVERAGE:
                #############################################
                ### Compute regional coverage using grid-based approach
                
                n_periods = n_periods_regional_coverage  # Number of periods to consider for coverage computation

                SHOW_GROUND_TRACKS_ON_COVERAGE = True # Show satellite ground tracks on coverage plots (for debugging)

                # Create a Coverage object
                coverage_obj = Coverage(orbital_parameters, positions=positions, times=time_stamps)
                

                print(f"\nComputing regional coverage for {num_satellites} satellites for regions: {regions_to_analyze}")

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
                        for region in regions_to_analyze:
                            print(f"  Analyzing {region}...")
                            regional_result = coverage_obj.compute_regional_coverage(
                                positions, 
                                time_stamps, 
                                region, 
                                orbital_periods=1, 
                                masking_angle=masking_angle, 
                                fov=fov
                            )
                    
                            regional_coverage = regional_result['constellation_coverage']
                            print(f"    {region} coverage: {regional_coverage:.2f}%")
                            
                            if pattern == 'Walker-Delta':
                                title = f"Grid-based regional coverage over the region of {region} of a\n{pattern} constellation {orbital_parameters['inclination']}:{num_satellites}/{P}/{F}, during {n_periods} orbital period(s)\n({orbital_params_str})\nMasking Angle: {masking_angle}°, FOV: {fov}°, Total Coverage: {regional_coverage:.2f}%"
                                filename = f"plots/grid_coverage/regional/{pattern}/{orbital_parameters['inclination']}-{P}-{F}/{n_periods}T/mask{masking_angle}_fov{fov}/region_{region}_coverage_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"                                                           
                            elif pattern == 'Walker-Delta-SinglePlane':
                                orbital_params_str = (
                                    f"a={orbital_parameters['semi_major_axis']}km / h={orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
                                    f"e={orbital_parameters['eccentricity']}, "
                                    f"i={orbital_parameters['inclination']}°, "
                                    f"RAAN={orbital_parameters['RAAN']}°, "
                                    f"ω={orbital_parameters['argument_of_periapsis']}°"
                                )     
                                title = f"Grid-based regional coverage over the region of {region} of a\n{pattern} constellation with {num_satellites} sat(s) during {n_periods} orbital period(s)\n({orbital_params_str})\nMasking Angle: {masking_angle}°, FOV: {fov}°, Total Coverage: {regional_coverage:.2f}%"
                                filename = f"plots/grid_coverage/regional/{pattern}/{n_periods}T/mask{masking_angle}_fov{fov}/region_{region}_coverage_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"                                                           
                            elif pattern == "Flower":
                                title = f"Grid-based regional coverage over the region of {region} of a\n{pattern} constellation with {num_satellites} sat(s), Np={Np}, Nd={Nd}, Fn={Fn}, Fd={Fd}, Fh={Fh}, during {n_periods} orbital period(s)\n({orbital_params_str})\nMasking Angle: {masking_angle}°, FOV: {fov}°, Total Coverage: {regional_coverage:.2f}%"
                                filename = f"plots/grid_coverage/regional/{pattern}/{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}/{n_periods}T/mask{masking_angle}_fov{fov}/region_{region}_coverage_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"                                                           
                            else:
                                title = f"Grid-based regional coverage over the region of {region} of a\n{pattern} constellation with {num_satellites} sat(s) during {n_periods} orbital period(s)\n({orbital_params_str})\nMasking Angle: {masking_angle}°, FOV: {fov}°, Total Coverage: {regional_coverage:.2f}%"
                                filename = f"plots/grid_coverage/regional/{pattern}/{n_periods}T/mask{masking_angle}_fov{fov}/region_{region}_coverage_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"                                                           

                            # Plot the regional coverage
                            fig = Coverage.plot_coverage_map(regional_result, title=title)
                            
                            # Add ground tracks if enabled
                            if SHOW_GROUND_TRACKS_ON_COVERAGE:
                                ax = fig.axes[0]  # Get the main plot axis
                                
                                # Determine how many time steps to include
                                idx_oneperiod = np.searchsorted(time_stamps, orbital_period)
                                idx_max = idx_oneperiod  # One orbital period
                                
                                for satellite in range(num_satellites):
                                    # Convert to ECEF and then to geodetic coordinates
                                    ECEF_cartesian_orbit = CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(
                                        positions[satellite, :idx_max, :]*1000, time_stamps[:idx_max])
                                    ECEF_geodetic_orbit = [CoordinateConversion.ECEF_cartesian_to_geodetic(x, y, z) 
                                                            for x, y, z in ECEF_cartesian_orbit]
                                    
                                    # Extract coordinates
                                    latitudes, longitudes, altitudes = zip(*ECEF_geodetic_orbit)
                                    
                                    # Handle longitude discontinuities
                                    unwrapped_longitudes = np.unwrap(np.radians(longitudes))
                                    normalized_longitudes = np.degrees(unwrapped_longitudes)
                                    
                                    # Plot the ground track with dotted line
                                    ax.plot(
                                        normalized_longitudes,
                                        latitudes,
                                        linestyle=':', 
                                        linewidth=1,
                                        color='red',  # Use a visible color
                                        transform=ccrs.PlateCarree(),
                                        alpha=0.7,  # Slightly transparent
                                        zorder=5  # To ensure it's drawn on top
                                    )
                            
                            # Create directory and save the plot
                            os.makedirs(os.path.dirname(filename), exist_ok=True)
                            fig.savefig(filename)
                            plt.close(fig)

            if PLOT_COVERAGE:
                #############################################
                ### Plot coverage for multiple satellites
                print(f"Plotting coverage of {num_satellites} satellites...")

                n_periods = n_periods_plot_coverage  # Number of periods to plot

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
                            ax.set_title(f"Coverage areas of a {pattern} constellation {orbital_parameters['inclination']}:{num_satellites}/{P}/{F}, during {n_periods} orbital period(s)\n({orbital_params_str})\nMasking Angle: {masking_angle}°, FOV: {fov}°, Cap Angle: {cap_angle:.2f}° Instant. Fraction Covered: {100*fraction_covered:.4f}%")
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
                        elif pattern == "Flower":
                            ax.set_title(f"Coverage areas of a {pattern} constellation with {num_satellites} sat(s), Np={Np}, Nd={Nd}, Fn={Fn}, Fd={Fd}, Fh={Fh}, during {n_periods} orbital period(s)\n({orbital_params_str})\nMasking Angle: {masking_angle}°, FOV: {fov}°, Cap Angle: {cap_angle:.2f}°, Instant. Fraction Covered: {100*fraction_covered:.4f}%")
                            filename = f"plots/coverage/{pattern}/{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}/{n_periods}T/constellation_coverage_step{step}_N{num_satellites}_mask{masking_angle}_fov{fov}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"                        
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
                        plt.close(fig)
                        
                        pass

            if COMPUTE_VISIBILITY:
                #############################################
                ### Compute and store visibility/overlap data for all satellites (NO plotting)
                print(f"Computing visibility and overlap for {num_satellites} satellites in {pattern}...")
                n_periods = n_periods_compute_visibility
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

            if PLOT_VISIBILITY and COMPUTE_VISIBILITY:
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
                elif pattern == "Flower":
                    ax1.set_title(f"Satellite Visibility from {gs} Ground Station a {pattern} constellation\nwith {num_satellites} sat(s), Np={Np}, Nd={Nd}, Fn={Fn}, Fd={Fd}, Fh={Fh}, during {n_periods} orbital period(s)\n({orbital_params_str})\nTotal time with at least one visible sat: {total_time_with_any_visible:.2f} s ({total_time_with_any_visible/3600:.2f} hr), {percentage_visible:.2f}%", fontsize=10)
                    filename = f"plots/visibility/{pattern}/{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}/{n_periods}T/visibility_multiple_satellites{gs}_step{step}_N{num_satellites}_el{elevation_threshold}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"                
                else:
                    ax1.set_title(f"Satellite Visibility from {gs} Ground Station a {pattern} constellation\nwith N{num_satellites} sat(s), during {n_periods} orbital period(s)\n({orbital_params_str})\nTotal time with at least one visible sat: {total_time_with_any_visible:.2f} s ({total_time_with_any_visible/3600:.2f} hr), {percentage_visible:.2f}%; {overlap_text}", fontsize=10)
                    filename = f"plots/visibility/{pattern}/{n_periods}T/visibility_multiple_satellites_{gs}_step{step}_N{num_satellites}_el{elevation_threshold}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename, bbox_inches='tight')
                plt.close(fig)

                pass
            
            # END OF NUM_SATELLITES LOOP
            #############################################

        ### STILL IN THE PATTERN LOOP
        if PLOT_DATA_COMPARISON and COMPUTE_VISIBILITY:
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
            elif pattern == "Flower":
                ax1.set_title(f"Visibility and Overlap vs. N of Satellites\n from {gs} Ground Station over {n_periods} orbital period(s)\n({orbital_params_str})\nPattern: {pattern}, Np={Np}, Nd={Nd}, Fn={Fn}, Fd={Fd}, Fh={Fh}")
                filename = f"plots/visibility/{pattern}/{Np}-{Nd}-{num_satellites}-{Fn}-{Fd}-{Fh}/{n_periods}T/DATA_comparison_{gs}_step{step}_el{elevation_threshold}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"            
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
            plt.close(fig)
        
            pass

    ### Pattern Comparison Plot ###
    if PLOT_PATTERN_VISIBILITY_COMPARISON and COMPUTE_VISIBILITY:
        print(f"Plotting comparison of visibility and elevation angle wrt to number of satellites for different patterns: {pattern_list}...")
        
        # Define a richer set of colors using colormap
        num_patterns = len([p for p in pattern_list if isinstance(p, tuple) and p[0] == "Walker-Delta"]) + 2  # Walker-Delta patterns + 2 special patterns
        pattern_colors = []
        
        # Add colors from a colormap - using tab20 for up to 20 distinct colors
        if num_patterns <= 9:
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
                elif pattern_key.startswith("Flower_"):
                    # Extract Np, Nd, Fn, Fd, Fh from pattern_key
                    parts = pattern_key.split('_')
                    Np, Nd, Fn, Fd, Fh = [int(p[2:]) for p in parts[1:]]
                    label = f"Flower constellation with Np={Np}, Nd={Nd}, Fn={Fn}, Fd={Fd}, Fh={Fh}"                    
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
        # Adjust figure size based on pattern count
        fig_height = 10
        fig_width = 10
        
        if len(patterns_to_plot) > 50:
            fig_height = 14
            fig_width = 14
        elif len(patterns_to_plot) > 35:
            fig_height = 12
            fig_width = 12
        elif len(patterns_to_plot) > 25:
            fig_width = 11
        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(fig_width, fig_height), gridspec_kw={'height_ratios': [2, 1]})
        
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
        # Calculate optimal column counts based on pattern counts - limited to max 4 cols for readability
        walker_cols = min(4, max(2, len(walker_delta_lines) // 10 + 2))  # At least 2 columns, at most 4
        special_cols = min(4, max(2, len(special_lines) // 5 + 1))  # At least 2 columns, at most 4
        
        if walker_delta_lines:
            # Remove legend from inside the plot and place it at the bottom
            first_legend = fig.legend(
                handles=walker_delta_lines,
                labels=walker_delta_labels,
                title="Walker-Delta Patterns",
                fontsize=9,  # Slightly larger for better readability with fewer columns
                loc='lower center',
                bbox_to_anchor=(0.5, 0),
                ncol=walker_cols
            )
        
        if special_lines:
            # Calculate appropriate vertical position for second legend
            # The more rows in the first legend, the lower the second legend needs to be
            walker_rows = (len(walker_delta_lines) + walker_cols - 1) // walker_cols if walker_delta_lines else 0
            legend_y_pos = -0.05
            
            # Adjust position based on rows in first legend - more space needed with fewer columns
            if walker_rows > 0:
                if walker_rows <= 2:
                    legend_y_pos = -0.08
                elif walker_rows <= 4:
                    legend_y_pos = -0.11
                elif walker_rows <= 6:
                    legend_y_pos = -0.14
                elif walker_rows <= 8:
                    legend_y_pos = -0.17
                elif walker_rows <= 10:
                    legend_y_pos = -0.20
                else:
                    legend_y_pos = -0.23
                
            second_legend = fig.legend(
                handles=special_lines,
                labels=special_labels,
                title="Special Patterns",
                fontsize=9,  # Slightly larger for better readability with fewer columns
                loc='lower center',
                bbox_to_anchor=(0.5, legend_y_pos),
                ncol=special_cols
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
        
        # Calculate extra space needed for legends based on number of patterns and columns
        legend_space = 0.05  # Default space for legends
        
        # Determine total number of legend rows needed (based on number of patterns and columns)
        total_patterns = len(patterns_to_plot)
        wd_patterns = len(walker_delta_lines)
        special_patterns = len(special_lines)
        
        # Calculate number of columns that will be used - limited to max 4 cols for readability
        walker_cols = min(4, max(2, wd_patterns // 10 + 2))  # At least 2 columns, at most 4
        special_cols = min(4, max(2, special_patterns // 5 + 1))  # At least 2 columns, at most 4
        
        # Calculate rows needed (ceiling division)
        walker_rows = (wd_patterns + walker_cols - 1) // walker_cols if wd_patterns > 0 else 0
        special_rows = (special_patterns + special_cols - 1) // special_cols if special_patterns > 0 else 0
        
        # Adjust legend space based on number of rows
        total_rows = walker_rows + special_rows
        
        if total_rows > 15:
            legend_space = 0.30  # Increased for more rows due to fewer columns
        elif total_rows > 12:
            legend_space = 0.25
        elif total_rows > 9:
            legend_space = 0.20
        elif total_rows > 6:
            legend_space = 0.15
        elif total_rows > 4:
            legend_space = 0.11
        elif total_rows > 2:
            legend_space = 0.07
        elif total_rows > 0:
            legend_space = 0.05
        
        # Add extra space for titles and padding between legend groups
        if special_lines and walker_delta_lines:
            legend_space += 0.05
            
        print(f"Debug: Number of patterns: {total_patterns} (WD: {wd_patterns}, Special: {special_patterns})")
        print(f"Debug: Legend rows: {total_rows} (WD: {walker_rows} in {walker_cols} cols, Special: {special_rows} in {special_cols} cols)")
        print(f"Debug: Legend space: {legend_space}")
            
        # Adjust spacing between subplots - leave space at bottom for legends
        plt.tight_layout(rect=[0, legend_space, 1, 0.98])
        
        # Save the comparison plot with secondary axis
        filename = f"plots/visibility/PatternComparison/{n_periods}T/PatternComparison_{gs}_step{step}_el{elevation_threshold}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # For large numbers of patterns, don't use bbox_inches='tight' to avoid cropping legends
        if len(patterns_to_plot) > 25:
            # Increase figure size for better legend accommodation
            fig.set_size_inches(12, 12)
            # Save without tight bbox to ensure legends are included
            plt.savefig(filename, dpi=300)
        else:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        