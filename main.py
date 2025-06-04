from utilities import CoordinateConversion, orbiter, TimeConversion
from utilities_constellations import Constellation
from utilities_coverage import Coverage
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
import cartopy.crs as ccrs
from sgp4.api import Satrec, jday, WGS84
from matplotlib.colors import hsv_to_rgb
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import constants

# atmosense, darpa, raytracing mit,
### INPUTS ###
num_satellites = 10  # Number of satellites in the constellation

# Usage: include derivatives
orbital_parameters = {
    'semi_major_axis':     7800.0,    # km
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

gs = "Fucino"  # Ground station location

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

if __name__ == "__main__":
    # Build the constellation
    const = Constellation(num_satellites, orbital_parameters, epoch)

    # Define propagation window & step
    start_time_datetime = datetime(2025, 5, 5, 12, 00, 00, 00, tzinfo=timezone.utc) # my input
    end_time_datetime =   datetime(2025, 5, 6, 12, 00, 00, 00, tzinfo=timezone.utc) # my input
    step = 10  # Time step input

    ### Propagation by constellation class function
    result = const.generate_positions_with_time(start_time_datetime, end_time_datetime, step) # ECI Cartesian

    positions = result["positions"] # A 3D numpy array of shape (num_satellites, num_time_steps, 3) with satellite positions in km
    velocities = result["velocities"] # A 3D numpy array of shape (num_satellites, num_time_steps, 3) with satellite velocities in km/s
    time_stamps = result["time_stamps"] # A 1D numpy array of shape (num_time_steps,) with time stamps in seconds since epoch.

    # # Example: Access position of satellite 0 at time step 10
    # satellite_index = 0
    # time_step_index = 10
    # print(f"Time: {time_stamps[time_step_index]} seconds")
    # print(f"Position: {positions[satellite_index, time_step_index]}")
    # print(f"Velocity: {velocities[satellite_index, time_step_index]}")

    # #####################################
    # ### PLOTTING ########################
    # #####################################
    
    # Estimate one period index to stop ground track after one orbit
    a = orbital_parameters['semi_major_axis']  # km from Earth's center
    orbital_period = 2 * np.pi * np.sqrt(a**3 / constants.GRAVITATIONAL_CONSTANT)

    orbital_period_hours = int(orbital_period // 3600)
    orbital_period_minutes = int((orbital_period % 3600) // 60)
    orbital_period_seconds = int(orbital_period % 60)
    print(f"Orbital period: {orbital_period_hours} hours, {orbital_period_minutes} minutes, {orbital_period_seconds} seconds")

    idx_max = np.searchsorted(time_stamps, orbital_period)

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.stock_img()
    ax.coastlines()
    ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")

    # colors = cm.tab10(np.linspace(0, 1, num_satellites))  # Unique color for each satellite
    # colors = cm.plasma(np.linspace(0, 1, num_satellites))  # Use the 'plasma' colormap
    colors = [hsv_to_rgb((i / num_satellites, 0.8, 0.9)) for i in range(num_satellites)]    
    # colors = cm.tab20(np.linspace(0, 1, num_satellites))

    # for satellite in range(num_satellites):
    #     satellite_considered = satellite
    #     geographic_orbit = CoordinateConversion.ECI_cartesian_to_geographic(positions[satellite_considered, :idx_max, :], time_stamps[:idx_max]) # A list of tuples (time, latitude, longitude, radial_distance)

    #     ECEF_cartesian_orbit = CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(positions[satellite_considered, :idx_max, :], time_stamps[:idx_max]) # A list of tuples (time, x, y, z)
    #     ECEF_geodetic_orbit = [CoordinateConversion.ECEF_cartesian_to_geodetic(x, y, z) for x, y, z in ECEF_cartesian_orbit] # A list of tuples (time, latitude, longitude, altitude)

    #     # Extract data
    #     times = [entry[0] for entry in geographic_orbit]
    #     latitudes = [entry[1] for entry in geographic_orbit]
    #     longitudes = [entry[2] for entry in geographic_orbit]
        
    #     # Handle longitude discontinuities
    #     unwrapped_longitudes = np.unwrap(np.radians(longitudes))  # Unwrap in radians
    #     normalized_longitudes = np.degrees(unwrapped_longitudes)  # Convert back to degrees

    #     #############################################
    #     # Plot ground track for this satellite (in loop)
    #     ax.plot(
    #         normalized_longitudes,
    #         latitudes,
    #         marker="o",
    #         markersize=2,
    #         linestyle="-",
    #         color=colors[satellite],  # Assign unique color
    #         label=f"Satellite {satellite + 1} (RAAN={orbital_parameters['RAAN'] + satellite*(360/num_satellites)}°)",  # Include RAAN in label
    #         transform=ccrs.PlateCarree(),
    #     )

    # ax.legend(loc="lower left", fontsize="small", ncol=2)
    # # ax.set_title("Ground Tracks of All Satellites")
    # # plt.savefig("plots/constellation_ground_tracks_sequential.png")
    # # Dynamically include orbital parameters in the title and filename
    # orbital_params_str = (
    #     f"a={orbital_parameters['semi_major_axis']}km / h={orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
    #     f"e={orbital_parameters['eccentricity']}, "
    #     f"i={orbital_parameters['inclination']}°, "
    #     f"RAAN varies by satellite, "
    #     f"ω={orbital_parameters['argument_of_periapsis']}°"
    # )

    # ax.set_title(f"Ground tracks of {num_satellites} satellites during 1 orbital period\n({orbital_params_str})")
    # filename = f"plots/constellation_ground_tracks_step{step}_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
    # plt.savefig(filename)

    # #############################################
    # ### Plot coverage for multiple satellites
    # fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    # ax.stock_img()
    # ax.coastlines()

    # # Extract orbital parameters for the subtitle
    # orbital_params_str = (
    #     f"a={orbital_parameters['semi_major_axis']}km / h={orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
    #     f"e={orbital_parameters['eccentricity']}, "
    #     f"i={orbital_parameters['inclination']}°, "
    #     f"RAAN varies by satellite, "
    #     f"ω={orbital_parameters['argument_of_periapsis']}°"
    # )

    # # Loop through all satellites
    # elevation_angle = 75.0 # degrees, the larger the smaller the coverage area
    # # consider this a 90- half swath angle

    # for satellite_index in range(num_satellites):
    #     for index in range(0, idx_max): # up until idx_max for one orbital period
    #         state = {
    #             "position": positions[satellite_index, index, :],  # Position of the satellite
    #             "velocity": velocities[satellite_index, index, :],  # Velocity of the satellite
    #             "time": time_stamps[index],  # Time stamp
    #         }

    #         # Instantiate the Coverage class
    #         coverage = Coverage(orbital_parameters, state)

    #         # Compute sample coverage
    #         coverage_points = coverage.compute_sample_coverage(elevation_angle=elevation_angle)

    #         # Extract latitudes and longitudes
    #         if coverage_points:  # Ensure there are valid coverage points
    #             lats, lons = zip(*coverage_points)

    #             # Plot coverage points for this satellite and time step
    #             ax.scatter(
    #                 lons, lats, s=1, color=colors[satellite_index], alpha=0.1, transform=ccrs.PlateCarree()
    #             )

    # # Create custom legend entries with opaque colors
    # legend_handles = [
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=8)
    #     for i in range(num_satellites)
    # ]
    # legend_labels = [
    #     f"Satellite {i + 1} (RAAN={orbital_parameters['RAAN'] + i * (360 / num_satellites):.1f}°)"
    #     for i in range(num_satellites)
    # ]

    # # Add legend to the plot
    # ax.legend(handles=legend_handles, labels=legend_labels, loc="lower left", fontsize="small", ncol=2)
    # ax.set_title(f"Coverage areas of {num_satellites} satellites during 1 orbital period\n({orbital_params_str}), Elevation Angle: {elevation_angle}°")

    # # Save the plot
    # filename = f"plots/coverage_multiple_satellites_step{step}_N{num_satellites}_el{elevation_angle}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
    # plt.savefig(filename)


    # #############################################
    # ### Sample coverage plot of one satellite
    # satellite_index = 0

    # # Create a single figure and axes for the plot
    # fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    # ax.stock_img()
    # ax.coastlines()

    # # Extract orbital parameters for the subtitle
    # orbital_params_str = (
    #     f"a={orbital_parameters['semi_major_axis']}km / h={orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
    #     f"e={orbital_parameters['eccentricity']}, "
    #     f"i={orbital_parameters['inclination']}°, "
    #     f"RAAN: {orbital_parameters['RAAN']}°, "
    #     f"ω={orbital_parameters['argument_of_periapsis']}°, "
    #     f"True Anomaly: {orbital_parameters['true_anomaly']}°"
    # )

    # # Loop through all time steps and add coverage points to the same plot
    # for index in range(0, idx_max):
    #     state = {
    #         "position": positions[satellite_index, index, :],  # Ensure it's a 2D array
    #         "velocity": velocities[satellite_index, index, :],  # km/s
    #         "time": time_stamps[index],  # seconds since epoch
    #     }

    #     # Instantiate the Coverage class
    #     coverage = Coverage(orbital_parameters, state)

    #     # Compute sample coverage
    #     elevation_angle = 99.0 # degrees, the larger the smaller the coverage area
    #     coverage_points = coverage.compute_sample_coverage(elevation_angle=elevation_angle)

    #     # Extract latitudes and longitudes
    #     lats, lons = zip(*coverage_points)

    #     # Plot coverage points for this time step
    #     ax.scatter(lons, lats, s=1, color="blue", alpha=0.5, transform=ccrs.PlateCarree())

    # # Add legend and title
    # ax.legend(["Coverage Area"], loc="lower left")
    # ax.set_title(f"Coverage area of a single satellite (N. {satellite_index+1}) during 1 orbital period\n({orbital_params_str}), Elevation Angle: {elevation_angle}°")

    # # Save the single plot
    # filename = f"plots/coverage_sample_step{step}_1sat(N{satellite_index+1})_el{elevation_angle}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
    # plt.savefig(filename)


    #############################################
    ### Sample visibility plot of one satellite
    
    satellite_index = 4
    elevation_threshold = 10.0  # degrees

    visibility = []
    elevations = []
    times = []

    idx_max_visibility = 1*idx_max

    for idx in range(idx_max_visibility):
        
        # Get satellite state
        sat_position = positions[satellite_index, idx, :]  # km
        sat_time = time_stamps[idx]
        # Convert to geographic (lat, lon, r)
        ECEF_cartesian_orbit = CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(sat_position*1000, sat_time) # A list of tuples (time, x, y, z), km to m

        # Check visibility
        vis, elev = Coverage.is_visible_cartesian_ECEF(ECEF_cartesian_orbit, gs_geodetic, elevation_threshold)
        visibility.append(int(vis))
        elevations.append(elev)
        times.append(sat_time / 3600.0)  # convert to hours if desired

    plt.figure(figsize=(10, 4))
    plt.plot(times, visibility, drawstyle='steps-post')
    plt.xlabel("Time [hours]")
    plt.ylabel(f"Visible (elev > {elevation_threshold}°)")
    plt.title(f"Satellite N{satellite_index+1} Visibility from {gs} Ground Station")
    plt.yticks([0, 1], ["No", "Yes"])
    plt.grid(True)
    plt.tight_layout()
    # Save the plot
    filename = f"plots/visibility_1sat(N{satellite_index+1})_{gs}_step{step}_el{elevation_threshold}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
    plt.savefig(filename)

    #############################################
    ### Visibility and Elevation plots for multiple satellites

    fig = plt.figure(figsize=(12, 7))
    gs_plot = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs_plot[0])
    ax2 = fig.add_subplot(gs_plot[1], sharex=ax1)

    for satellite_index in range(num_satellites):
        visibility = []
        elevations = []
        times = []
        for idx in range(idx_max_visibility):
            # Get satellite state from ECI Cartesian sgp4
            sat_position = positions[satellite_index, idx, :]  # km 
            sat_time = time_stamps[idx]

            ECEF_cartesian_orbit = CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(sat_position*1000, sat_time) # A list of tuples (time, x, y, z), km to m
            
            # Check visibility
            vis, elev = Coverage.is_visible_cartesian_ECEF(ECEF_cartesian_orbit, gs_geodetic, elevation_threshold)

            visibility.append(int(vis))
            elevations.append(elev)
            times.append(sat_time / 3600.0)  # convert to hours if desired
        # Plot visibility (dashed line)
        ax1.plot(times, visibility, drawstyle='steps-post', color=colors[satellite_index], 
                 linestyle='--', label=f"Satellite {satellite_index + 1} (RAAN={orbital_parameters['RAAN'] + satellite_index * (360 / num_satellites):.1f}°)")
        # Plot elevation (solid line)
        ax2.plot(times, elevations, color=colors[satellite_index], linestyle='-')

    ax1.set_ylabel(f"Visible (elev > {elevation_threshold}°)")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["No", "Yes"])
    ax1.set_title(f"Satellite Visibility from {gs} Ground Station ({num_satellites} Satellites)")
    ax1.grid(True)
    ax1.legend(loc="best", fontsize="small", ncol=2)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.set_xlabel("Time [hours]")
    ax2.set_ylabel("Elevation [deg]")
    # ax2.set_title("Elevation Angle")
    ax2.grid(True)
    # Add dashed horizontal line for elevation threshold
    ax2.axhline(elevation_threshold, color='gray', linestyle='--', linewidth=1, label=f"Threshold ({elevation_threshold}°)")

    # Use constrained_layout instead of tight_layout for better compatibility with gridspec
    fig.subplots_adjust(hspace=0.08)

    # Only show legend for threshold line if not already present
    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend(loc="upper right", fontsize="small")

    # Save the plot
    filename = f"plots/visibility_multiple_satellites_{gs}_step{step}_N{num_satellites}_el{elevation_threshold}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"
    plt.savefig(filename)