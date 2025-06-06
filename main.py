from utilities import CoordinateConversion, orbiter, TimeConversion
from utilities_constellations import Constellation
from utilities_coverage import Coverage
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

import os # to create directories if they do not exist

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")

### INPUTS ###
# num_satellites_list = [5]
# num_satellites_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,35,40,50,60,70,80,90,100]  ù
num_satellites_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]  

# Includes derivatives
orbital_parameters = {
    'semi_major_axis':     6800.0,    # km
    'eccentricity':        0.000,    # unitless
    'inclination':         50.0,     # deg
    'RAAN':                0.0,     # deg
    'argument_of_periapsis': 250.0,  # deg     
    'true_anomaly':        0.0,      # deg
    'ndot':                1e-8,     # rev/day²
    'nddot':               1e-12,    # rev/day³
    'bstar':               2e-5      # 1/er
}
epoch = datetime(2025,5,5,0,0,0)

### INPUT for visibility plots ###
gs = "Fucino"  # Ground station location

### INPUTS for coverage plots ###
masking_angles = [5.0,10.0,15.0]  # degrees, the larger the smaller the coverage area. Same as elevation angle threshold. Usually 10-15 deg min
# masking_angles = [20.0]  # degrees, the larger the smaller the coverage area. Same as elevation angle threshold
fovs = [20,70,120,180] # degrees, field of view of the sensor. If too low is limiting, if it's large then the masking angle is limiting
# fovs = [170] # degrees, field of view of the sensor

# Define propagation window & step. Make sure is at least the number of periods to be plotted
start_time_datetime = datetime(2025, 5, 5, 12, 00, 00, 00, tzinfo=timezone.utc) # my input
end_time_datetime =   datetime(2025, 5, 15, 12, 00, 00, 00, tzinfo=timezone.utc) # my input
step = 10  # Time step input

# Swtiches for plotting
PLOT_GROUND_TRACKS = False
PLOT_COVERAGE = False
COMPUTE_VISIBILITY = True
PLOT_VISIBILITY = COMPUTE_VISIBILITY
PLOT_DATA_COMPARISON = COMPUTE_VISIBILITY


# Estimate one period index to stop ground track after one orbit
a = orbital_parameters['semi_major_axis']  # km from Earth's center
orbital_period = 2 * np.pi * np.sqrt(a**3 / constants.GRAVITATIONAL_CONSTANT)

orbital_period_hours = int(orbital_period // 3600)
orbital_period_minutes = int((orbital_period % 3600) // 60)
orbital_period_seconds = int(orbital_period % 60)
print(f"Orbital period: {orbital_period_hours} hours, {orbital_period_minutes} minutes, {orbital_period_seconds} seconds ({orbital_period/3600:.4f} hr)")

# visibility data analysis
total_time_with_any_visible_list = []
overlap_percentage_list = []
num_satellites_record = []

if __name__ == "__main__":
    # Setting ground station parameters based on the selected ground station
    if gs == "Japan":
        gs_lat = 45.0      # degrees
        gs_lon = 140.0       # degrees
        gs_alt = 0.0       # meters
    elif gs == "AFB_Andersen":
        gs_lat = 13.6      # degrees
        gs_lon = 144.9       # degrees
        gs_alt = 0.0       # meters
    elif gs == "AFB_Hickam":
        gs_lat = 21.3      # degrees
        gs_lon = -157.9      # degrees
        gs_alt = 0.0       # meters
    elif gs == "Kiruna":
        gs_lat = 67.9      # degrees
        gs_lon = 20.4       # degrees
        gs_alt = 0.0       # meters
    elif gs == "AFB_Elmendorf":
        gs_lat = 61.2      # degrees
        gs_lon = -149.9      # degrees
        gs_alt = 0.0       # meters
    elif gs == "Svalbard":
        gs_lat = 78.2      # degrees
        gs_lon = 15.6       # degrees
        gs_alt = 0.0       # meters
    elif gs == "Troll_Antarctica":
        gs_lat = -72.0     # degrees
        gs_lon = 2.5        # degrees
        gs_alt = 0.0       # meters
    elif gs == "Fucino":
        gs_lat = 42.0      # degrees
        gs_lon = 13.6       # degrees
        gs_alt = 0.0       # meters
    
    gs_geodetic = (gs_lat, gs_lon, gs_alt)

    for num_satellites in num_satellites_list:
        print(f"Processing constellation with {num_satellites} satellites...")
        # Build the constellation
        const = Constellation(num_satellites, orbital_parameters, epoch)

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
            
            n_periods = 5  # Number of periods to plot

            idx_oneperiod = np.searchsorted(time_stamps, orbital_period) # Index of one orbital period
            idx_max = n_periods * idx_oneperiod  # Index of total number of time steps to plot

            fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
            ax.stock_img()
            ax.coastlines()
            ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")

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

                ax.plot(
                    normalized_longitudes,
                    latitudes,
                    marker="o",
                    markersize=2,
                    linestyle="-",
                    color=colors[satellite],  # Assign unique color
                    label=f"Satellite {satellite + 1} (RAAN={orbital_parameters['RAAN'] + satellite*(360/num_satellites):.2f}°)",  # Include RAAN in label
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

            ax.set_title(f"Ground tracks of {num_satellites} satellite(s) during {n_periods} orbital period(s)\n({orbital_params_str})")
            filename = f"plots/ground_tracks/{n_periods}T/constellation_ground_tracks_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
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
                f"RAAN varies by satellite, "
                f"ω={orbital_parameters['argument_of_periapsis']}°"
            )

            for masking_angle in masking_angles:
                for fov in fovs:
                    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
                    ax.stock_img()
                    ax.coastlines()
                    ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")

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

                    # Create custom legend entries with opaque colors
                    legend_handles = [
                        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=8)
                        for i in range(num_satellites)
                    ]
                    legend_labels = [
                        f"Satellite {i + 1} (RAAN={orbital_parameters['RAAN'] + i * (360 / num_satellites):.1f}°)"
                        for i in range(num_satellites)
                    ]

                    # Add legend to the plot
                    # ax.legend(handles=legend_handles, labels=legend_labels, loc="lower left", fontsize="small", ncol=2)
                    ax.set_title(f"Coverage areas of {num_satellites} satellites during {n_periods} orbital period(s)\n({orbital_params_str})\n Masking Angle: {masking_angle}°, FOV: {fov}°, Cap Angle: {cap_angle:.2f}°Instant. Fraction Covered: {100*fraction_covered:.4f}%")


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
                    filename = f"plots/coverage/{n_periods}T/constellation_coverage_step{step}_N{num_satellites}_mask{masking_angle}_fov{fov}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    plt.savefig(filename)
                    
                    pass

        if COMPUTE_VISIBILITY:
            #############################################
            ### VISIBILITY PLOTS
            n_periods = 25
            idx_oneperiod = np.searchsorted(time_stamps, orbital_period) # Index of one orbital period
            idx_max = n_periods * idx_oneperiod  # Index of total number of time steps to plot
            
            elevation_threshold = 10.0  # degrees

            # Extract orbital parameters for the subtitle
            orbital_params_str = (
                f"a={orbital_parameters['semi_major_axis']}km / h={orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
                f"e={orbital_parameters['eccentricity']}, "
                f"i={orbital_parameters['inclination']}°, "
                f"RAAN varies by satellite, "
                f"ω={orbital_parameters['argument_of_periapsis']}°"
            )

            # #############################################
            # ### Sample visibility plot of one satellite
            
            # satellite_index = 4
        
            # visibility = []
            # elevations = []
            # times = []
            # for idx in range(idx_max):
            #     sat_position = positions[satellite_index, idx, :]  # km
            #     sat_time = time_stamps[idx]
            #     # Convert to geographic (lat, lon, r)
            #     ECEF_cartesian_orbit = CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(sat_position*1000, sat_time) # A list of tuples (time, x, y, z), km to m

            #     # Check visibility
            #     vis, elev = Coverage.is_visible_cartesian_ECEF(ECEF_cartesian_orbit, gs_geodetic, elevation_threshold)
            #     visibility.append(int(np.asarray(vis).item()))
            #     elevations.append(elev)
            #     times.append(sat_time / 3600.0)  # convert to hours if desired

            pass

        if PLOT_VISIBILITY:
            # plt.figure(figsize=(10, 4))
            # plt.plot(times, visibility, drawstyle='steps-post')
            # plt.xlabel("Time [hours]")
            # plt.ylabel(f"Visible (elev > {elevation_threshold}°)")
            # plt.title(f"Satellite N{satellite_index+1} Visibility from {gs} Ground Station")
            # plt.yticks([0, 1], ["No", "Yes"])
            # plt.grid(True)
            # plt.tight_layout()
            # # Save the plot
            # filename = f"plots/visibility/{n_periods}T/visibility_1sat(N{satellite_index+1})_{gs}_step{step}_el{elevation_threshold}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
            # os.makedirs(os.path.dirname(filename), exist_ok=True)
            # plt.savefig(filename)

            # # Calculate total visible seconds for the satellite
            # visible_count = sum(visibility)  # Number of time steps where visible
            # total_visible_seconds = visible_count * step  # step is your time step in seconds
            # total_times = len(times) * step  # Total time in hours

            # print(f"Total visible time for sat N{satellite_index+1} over {total_times:.2f} s ({total_times/3600:.4f} hr): {total_visible_seconds} s ({total_visible_seconds/3600:.4f} hr), {total_visible_seconds*100/total_times:.4f}%")

            #############################################
            ### Visibility and Elevation plots for multiple satellites
            print(f"Plotting visibility and elevation angle of {num_satellites} satellites...")

            fig = plt.figure(figsize=(12, 7))
            gs_plot = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.05)
            ax1 = fig.add_subplot(gs_plot[0])
            ax2 = fig.add_subplot(gs_plot[1], sharex=ax1)
        
            # Store visibility for all satellites at all time steps
            all_visibilities = []

            for satellite_index in range(num_satellites):
                visibility = []
                elevations = []
                times = []
                for idx in range(idx_max):
                    # Get satellite state from ECI Cartesian sgp4
                    sat_position = positions[satellite_index, idx, :]  # km 
                    sat_time = time_stamps[idx]

                    ECEF_cartesian_orbit = CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(sat_position*1000, sat_time) # A list of tuples (time, x, y, z), km to m
                    
                    # Check visibility
                    vis, elev = Coverage.is_visible_cartesian_ECEF(ECEF_cartesian_orbit, gs_geodetic, elevation_threshold)

                    visibility.append(int(np.asarray(vis).item()))
                    elevations.append(elev)
                    times.append(sat_time / 3600.0)  # convert to hours if desired
                all_visibilities.append(visibility)
                # Plot visibility (dashed line)
                ax1.plot(times, visibility, drawstyle='steps-post', color=colors[satellite_index], 
                        linestyle='--', label=f"Satellite {satellite_index + 1} (RAAN={orbital_parameters['RAAN'] + satellite_index * (360 / num_satellites):.1f}°)")
                # Plot elevation (solid line)
                ax2.plot(times, elevations, color=colors[satellite_index], linestyle='-')

                # Calculate total visible seconds for the satellite
                visible_count = sum(visibility)  # Number of time steps where visible
                total_visible_seconds = visible_count * step  # step is your time step in seconds
                total_times = len(times) * step  # Total time in hours
                # print(f"Total visible time for sat N{satellite_index+1} over {total_times:.2f} s ({total_times/3600:.4f} hr): {total_visible_seconds} s ({total_visible_seconds/3600:.4f} hr), {total_visible_seconds*100/total_times:.4f}%")
            
            # Convert to numpy array for easy manipulation
            all_visibilities = np.array(all_visibilities)  # shape: (num_satellites, num_time_steps)
            # Total visible time summed over all satellites (double-counts if multiple are visible at once)
            total_visible_seconds_all = np.sum(all_visibilities) * step
            print(f"Total visible time from {gs} summed over all satellites: {total_visible_seconds_all} s ({total_visible_seconds_all/3600:.2f} hr)")
            # Time with at least one satellite visible
            # For each time step, check if any satellite is visible
            any_visible_per_timestep = np.any(all_visibilities, axis=0)  # shape: (num_time_steps,)
            total_time_with_any_visible = np.sum(any_visible_per_timestep) * step
            print(f"Total time from {gs} with at least one satellite visible: {total_time_with_any_visible:.2f} s ({total_time_with_any_visible/3600:.2f} hr), {total_time_with_any_visible * 100 / (len(times) * step):.2f}%")
            print(f"Constellation overlap percentage from {gs}: {((total_visible_seconds_all -total_time_with_any_visible)/total_time_with_any_visible)*100:.2f}%")

            ax1.set_ylabel(f"Visible (elev > {elevation_threshold}°)")
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels(["No", "Yes"])
            ax1.set_title(f"Satellite Visibility from {gs} Ground Station for {num_satellites} Satellite(s)\n({orbital_params_str})")
            fig.suptitle(f'Total time with at least one visible sat: {total_time_with_any_visible:.2f} s ({total_time_with_any_visible/3600:.2f} hr), {total_time_with_any_visible * 100 / (len(times) * step):.2f}%; Overlap: {((total_visible_seconds_all -total_time_with_any_visible)/total_time_with_any_visible)*100:.2f}%', fontsize="medium", y=0.97)
            ax1.grid(True)
            # ax1.legend(loc="best", fontsize="small", ncol=2)
            plt.setp(ax1.get_xticklabels(), visible=False)

            ax2.set_xlabel("Time [hours]")
            ax2.set_ylabel("Elevation [deg]")
            # ax2.set_title("Elevation Angle")
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
            # # Use constrained_layout instead of tight_layout for better compatibility with gridspec
            # fig.subplots_adjust(
            #     left=0.08, right=0.98, top=0.93, bottom=0.35, hspace=0.15
            # )

            # Only show legend for threshold line if not already present
            handles, labels = ax2.get_legend_handles_labels()
            if handles:
                ax2.legend(loc="upper right", fontsize="small")

            plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space at the bottom for the legend
            
            # Save the plot
            filename = f"plots/visibility/{n_periods}T/visibility_multiple_satellites_{gs}_step{step}_N{num_satellites}_el{elevation_threshold}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, bbox_inches='tight')

            # Append results to lists for further analysis
            total_time_with_any_visible_list.append(total_time_with_any_visible / 3600)  # convert to hours if you want
            overlap_percentage_list.append((total_visible_seconds_all - total_time_with_any_visible) / total_time_with_any_visible * 100)
            num_satellites_record.append(num_satellites)

            pass
        
        # END OF NUM_SATELLITES LOOP
        #############################################

    if PLOT_DATA_COMPARISON:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 7), gridspec_kw={'height_ratios': [2, 1]})

        # Top plot: total time with at least one satellite visible
        ax1.plot(num_satellites_record, total_time_with_any_visible_list, marker='o')
        ax1.set_ylabel("Total time with\nat least one visible sat [hr]")
        ax1.grid(True)
        ax1.set_title(f"Visibility and Overlap vs. N of Satellites\n from {gs} Ground Station over {n_periods} orbital period(s)\n({orbital_params_str})")

        # Bottom plot: overlap percentage
        ax2.plot(num_satellites_record, overlap_percentage_list, marker='o', color='orange')
        ax2.set_xlabel("Number of satellites in constellation")
        ax2.set_ylabel("Overlap percentage [%]")
        # ax2.set_yscale('log') # There are important 0 values, so log scale is not appropriate
        ax2.grid(True)

        plt.tight_layout()
        # Save the plot
        filename = f"plots/visibility/{n_periods}T/DATA_comparison_{gs}_step{step}_el{elevation_threshold}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    
        pass