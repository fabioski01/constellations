from utilities import CoordinateConversion, orbiter, TimeConversion
from utilities_constellations import Constellation
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
import cartopy.crs as ccrs
from sgp4.api import Satrec, jday, WGS84
from matplotlib.colors import hsv_to_rgb

if __name__ == "__main__":
    # # Read TLEs
    # tle_file = "tle.txt"  # Replace with the path to TLE file
    # satellite = orbiter.read_tle_from_file(tle_file)

    # # Define time range for propagation (start, end, and step in seconds)
    # positions, velocities, times = orbiter.sgp4_propagator(satellite, start_time_datetime, step, end_time_datetime)

    # keplerian_elements = [CoordinateConversion.cartesian_to_keplerian(pos, vel) for pos, vel in zip(positions, velocities)]

    num_satellites = 5  # Number of satellites in the constellation

    # Usage: include derivatives
    orbital_parameters = {
        'semi_major_axis':     6771.0,    # km
        'eccentricity':        0.000,    # unitless
        'inclination':         60.0,     # deg
        'RAAN':                15.0,     # deg
        'argument_of_periapsis': 250.0,  # deg     
        'true_anomaly':        0.0,      # deg
        'ndot':                1e-8,     # rev/day²
        'nddot':               1e-12,    # rev/day³
        'bstar':               2e-5      # 1/er
    }
    epoch = datetime(2025,5,5,0,0,0)

    # Build the constellation
    const = Constellation(num_satellites, orbital_parameters, epoch)

    # Define propagation window & step
    # datetime.datetime.now(datetime.UTC)
    start_time_datetime = datetime(2025, 5, 5, 12, 00, 00, 00, tzinfo=timezone.utc) # my input
    end_time_datetime =   datetime(2025, 5, 5, 13, 37, 8, 00, tzinfo=timezone.utc) # my input
    step = 100  # Time step input
    # start = epoch
    # end   = epoch + timedelta(hours=2)  # propagate 2 hours
    # step  = 60                          # seconds

    # Convert start/end to Julian dates + fraction
    start_jd, start_fr = TimeConversion.datetime_to_sgp4time(start_time_datetime)
    end_jd, end_fr = TimeConversion.datetime_to_sgp4time(end_time_datetime)  
    # start_jd, start_fr = jday(start.year, start.month, start.day,
    #                         start.hour, start.minute, start.second)
    # end_jd,   end_fr   = jday(end.year,   end.month,   end.day,
    #                         end.hour,   end.minute,   end.second)

    # Build the Julian date array
    jd_array = np.arange(start_jd + start_fr,
                        end_jd   + end_fr,
                        step / 86400.0)  # seconds → days

    # Pre-allocate result arrays
    N = len(const.satrecs)
    M = len(jd_array)
    positions_all  = np.zeros((N, M, 3))
    velocities_all = np.zeros((N, M, 3))
    times_seconds = np.arange(M) * step    # sec since epoch

    # Propagate each Satrec
    for i, sat in enumerate(const.satrecs):
        for j, jd in enumerate(jd_array):
            err, r, v = sat.sgp4(jd, 0)
            if err:
                raise RuntimeError(f"SGP4 error {err} at sat {i}, step {j}")
            positions_all[i, j, :]  = r   # [x, y, z] in km
            velocities_all[i, j, :] = v   # [vx, vy, vz] in km/s

    # - `positions_all` is shape (satnum, M, 3)
    # - `velocities_all` is shape (satnum, M, 3)

    # # Access example: position of satellite #2 at time index 10:
    # print( positions_all[2, 10] )

    # #####################################
    # ### PLOTTING ########################
    # #####################################
    
    # Estimate one period index to stop ground track after one orbit
    a = orbital_parameters['semi_major_axis']  # km from Earth's center
    GRAVITATIONAL_CONSTANT = 398600.4418  # Earth grav. parameter [km³/s²]
    EARTH_RADIUS = 6371.0  # km
    orbital_period = 2 * np.pi * np.sqrt(a**3 / GRAVITATIONAL_CONSTANT)

    orbital_period_hours = int(orbital_period // 3600)
    orbital_period_minutes = int((orbital_period % 3600) // 60)
    orbital_period_seconds = int(orbital_period % 60)
    print(f"Orbital period: {orbital_period_hours} hours, {orbital_period_minutes} minutes, {orbital_period_seconds} seconds")

    idx_max = np.searchsorted(times_seconds, orbital_period)

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.stock_img()
    ax.coastlines()
    ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")

    # colors = cm.tab10(np.linspace(0, 1, num_satellites))  # Unique color for each satellite
    # colors = cm.plasma(np.linspace(0, 1, num_satellites))  # Use the 'plasma' colormap
    colors = [hsv_to_rgb((i / num_satellites, 0.8, 0.9)) for i in range(num_satellites)]    
    # colors = cm.tab20(np.linspace(0, 1, num_satellites))

    for satellite in range(num_satellites):
        satellite_considered = satellite
        geographic_orbit_MOD = CoordinateConversion.cartesian_to_geographic_mod(
            positions_all[satellite_considered, :, :], times_seconds
        )

        # Extract data
        times = [entry[0] for entry in geographic_orbit_MOD]
        latitudes = [entry[1] for entry in geographic_orbit_MOD]
        longitudes = [entry[2] for entry in geographic_orbit_MOD]
        # Handle longitude discontinuities
        unwrapped_longitudes = np.unwrap(np.radians(longitudes))  # Unwrap in radians
        normalized_longitudes = np.degrees(unwrapped_longitudes)  # Convert back to degrees

        # Plot ground track for this satellite (in loop)
        ax.plot(
            normalized_longitudes,
            latitudes,
            marker="o",
            markersize=2,
            linestyle="-",
            color=colors[satellite],  # Assign unique color
            label=f"Satellite {satellite + 1} (RAAN={orbital_parameters['RAAN'] + satellite*(360/num_satellites)}°)",  # Include RAAN in label
            transform=ccrs.PlateCarree(),
        )

    ax.legend(loc="lower left", fontsize="small", ncol=2)
    # ax.set_title("Ground Tracks of All Satellites")
    # plt.savefig("plots/constellation_ground_tracks_sequential.png")
    # Dynamically include orbital parameters in the title and filename
    orbital_params_str = (
        f"a={orbital_parameters['semi_major_axis']}km / h={orbital_parameters['semi_major_axis']-EARTH_RADIUS} km, "
        f"e={orbital_parameters['eccentricity']}, "
        f"i={orbital_parameters['inclination']}°, "
        f"RAAN varies by satellite, "
        f"ω={orbital_parameters['argument_of_periapsis']}°"
    )

    ax.set_title(f"Ground Tracks of {num_satellites} Satellites\n({orbital_params_str})")
    filename = f"plots/constellation_ground_tracks_N{num_satellites}_a{orbital_parameters['semi_major_axis']}_e{orbital_parameters['eccentricity']}_i{orbital_parameters['inclination']}.png"

    plt.savefig(filename)

    # # Convert to geographic coordinates
    # for satellite in range(num_satellites):
    #     satellite_considered = satellite  # Index of the satellite to consider
    #     geographic_orbit_MOD = CoordinateConversion.cartesian_to_geographic_mod(positions_all[satellite_considered, :, :], times_seconds)

    #     # Extract data
    #     times = [entry[0] for entry in geographic_orbit_MOD]
    #     latitudes = [entry[1] for entry in geographic_orbit_MOD]
    #     longitudes = [entry[2] for entry in geographic_orbit_MOD]
    #     # altitudes = [entry[3] for entry in geographic_orbit_MOD] # in m
    #     radial_distances = [entry[3] for entry in geographic_orbit_MOD] # in m
    #     # radial_distances = [entry[4] for entry in geographic_orbit_MOD] # in m

    #     # Normalize longitude to [-180, 180] for better ground track visualization
    #     normalized_longitudes = [(lon + 180) % 360 - 180 for lon in longitudes]

    #     # Normalize time for color gradient (0 to 1)
    #     norm_times = (np.array(times) - min(times)) / (max(times) - min(times))
    #     colors = cm.plasma(norm_times)  # Use a colormap (e.g., 'plasma', 'viridis', etc.)

    #     # Plot with Earth map background
    #     fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})

    #     # Add Earth map features
    #     ax.stock_img()  # Add a simple Earth map
    #     ax.coastlines()  # Add coastlines
    #     ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")

    #     # Plot ground track
    #     for i in range(len(latitudes) - 1):
    #         ax.plot(
    #             normalized_longitudes[i:i + 2],
    #             latitudes[i:i + 2],
    #             color=colors[i],
    #             linewidth=0,  # Set line width to 0
    #             marker="o",
    #             markersize=2,
    #             transform=ccrs.PlateCarree(),  # Transform coordinates to map projection
    #         )

    #     # Add color bar to show time progression
    #     sm = cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=min(times), vmax=max(times)))
    #     sm.set_array([])  # Necessary for ScalarMappable to work properly
    #     cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", label="Time (s)")

    #     # Labels and Title
    #     ax.set_title(f"Ground Track with time legend, satellite n. {satellite_considered+1}")
    #     plt.savefig(f'plots/corotating_gradient_with_map_mod_{satellite_considered}.png')