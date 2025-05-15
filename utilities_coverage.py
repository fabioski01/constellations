import numpy as np
from utilities import CoordinateConversion
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import constants

"""
Theory from Wertz. Earth coverage refers to the part of  the Earth that a s/c instrument (sensor, antenna, etc.) 
can see at one instant or over an extended period of time. 

Two critical distinctions are:
1. Instananeous field of view (IFOV): The area on the ground that is viewed by the sensor at one instant in time.
   It is a function of the sensor's field of view (FOV) and the altitude of the satellite.
2. Access area: the total area on the ground that could potentially be seen at any moment by turning the s/c or instrument.

In the case of a truly omni-directional antenna, the IFOV and the access area are the same. Most of the times, they are not.

The second important distinction is between the area which can be seen at any one instant vs the rate at which new land
 comes into view as the sc and instrumentation move. In GEO, the instanteneous area is typically the most important 
because the sc is nearly stationary relative to the Earth's surface. In LEO, satellites are moving rapidly over 
 the surface, so the rate at which new land appears is usually critical.

The two distinctions lead to four key parameters for Earth's coverage.
1) Footprint area (F_A, also FOV area or instantaneous coverage area): area that a specific instrument or antenna can see
 at any instant.
2) Instananeous Access Area (IAA): all the area that the instrument or antenna could potentially see at any instant, 
if it were scanned through its normal range of orientations.
3) Area Coverage Rate (ACR): the rate at which the instrument or antenna is sensing or accessing new land.
4) Area Access Rate (AAR): the rate at which new land is coming into the sc's access area.

The ACR and AAR are the same for an instrument which covers all of the area available to it as the sc moves along.
Generally, the access area and the access rates depend only on the orbit and limiting geometry of the system, 
so they are easily derived without knowing the sc design.
On the other hand, the actual area coverqage rate during sc operations may depend on sc system budgets and mission operations

To perform coverage assessment, there is first an analytic assesment with approximate formulas for coverage parameters 
as function of mission variables, and secondly numerical simulations to provide coverage figures of merit.
"""

class Coverage:
    """
    A class to compute and handle satellite coverage on Earth.

    Attributes:
        orbital_parameters (dict): Dictionary containing orbital parameters of the satellite.
        state (dict): Dictionary containing the satellite's position, velocity, and time.
    """

    def __init__(self, orbital_parameters, state):
        """
        Initialize the Coverage object with orbital parameters and state.

        Args:
            orbital_parameters (dict): Orbital parameters of the satellite, including:
                - semi_major_axis (float): Semi-major axis in km.
                - eccentricity (float): Orbital eccentricity.
                - inclination (float): Orbital inclination in degrees.
                - RAAN (float): Right Ascension of the Ascending Node in degrees.
                - argument_of_periapsis (float): Argument of periapsis in degrees.
                - true_anomaly (float): True anomaly in degrees.
            state (dict): Satellite state containing:
                - position (numpy array): Satellite position [x, y, z] in km.
                - velocity (numpy array): Satellite velocity [vx, vy, vz] in km/s.
                - time (float): Time in seconds since epoch.
        """
        self.orbital_parameters = orbital_parameters
        self.state = state

    def compute_sample_coverage(self, elevation_angle=10.0):
        """
        Compute the coordinates of the area covered by the satellite at a specific instant.

        Args:
            elevation_angle (float): Minimum elevation angle (in degrees) for coverage. Default is 10°.

        Returns:
            list of tuple: List of (latitude, longitude) tuples representing the covered area.
        """
        EARTH_RADIUS = 6371.0  # Earth's radius in km

        # Extract satellite position
        sat_position = self.state["position"]
        # if not isinstance(sat_position, (list, np.ndarray)) or len(sat_position) != 3:
        #     raise ValueError(f"Expected a 3D position vector, got {sat_position}")
        sat_times = self.state["time"]
        results = CoordinateConversion.cartesian_to_geographic(sat_position, sat_times)
        print(results)

        # Access the first (and only) tuple in the results list
        if len(results) == 0:
            raise ValueError("No results returned from cartesian_to_geographic")
        time, sat_lat, sat_lon, sat_radial = results[0]  # Unpack the single tuple

        # print(f"sat_lat: {sat_lat}, sat_lon: {sat_lon}, sat_radial: {sat_radial}")
    
        sat_radial_km = sat_radial/1000.0  # Convert to km
        # Compute the coverage radius on Earth's surface
        coverage_radius = np.arccos(EARTH_RADIUS / (sat_radial_km)) * (180 / np.pi)

        # print(f"Coverage radius: {coverage_radius} degrees")

        # Generate coverage points (latitude, longitude)
        coverage_points = []
        for lat_offset in np.linspace(-coverage_radius, coverage_radius, 50):
            for lon_offset in np.linspace(-coverage_radius, coverage_radius, 50):
                lat = sat_lat + lat_offset
                lon = sat_lon + lon_offset

                # Normalize longitude to [-180, 180]
                lon = (lon + 180) % 360 - 180

                if -90 <= lat <= 90 and -180 <= lon <= 180:  # Ensure valid coordinates
                    coverage_points.append((lat, lon))
        
        # print("Coverage points:", coverage_points)
        return coverage_points

    def compute_global_coverage(self, time_window):
        """
        Compute the global coverage of the satellite over a given time window.

        Args:
            time_window (float): Time window in seconds.

        Returns:
            list of tuple: List of (latitude, longitude) tuples representing the globally covered area.
        """
        # Placeholder for global coverage computation
        pass

    def plot_coverage(self, coverage_points):
        """
        Plot the coverage area on a map.

        Args:
            coverage_points (list of tuple): List of (latitude, longitude) tuples representing the covered area.
        """

        # Extract orbital parameters for the subtitle
        orbital_params_str = (
            f"a={self.orbital_parameters['semi_major_axis']}km / h={self.orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
            f"e={self.orbital_parameters['eccentricity']}, "
            f"i={self.orbital_parameters['inclination']}°, "
            f"RAAN: {self.orbital_parameters['RAAN']}°, "
            f"ω={self.orbital_parameters['argument_of_periapsis']}°,"
            f"True Anomaly: {self.orbital_parameters['true_anomaly']}°"
        )
        
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.stock_img()
        ax.coastlines()

        # Extract latitudes and longitudes
        lats, lons = zip(*coverage_points)

        # Plot coverage points
        ax.scatter(lons, lats, s=1, color="blue", label="Coverage Area", transform=ccrs.PlateCarree())
        ax.legend(loc="lower left")
        ax.set_title(f"Satellite Coverage Area\n({orbital_params_str})")
        filename = f"plots/coverage_sample_1sat_a{self.orbital_parameters['semi_major_axis']}_e{self.orbital_parameters['eccentricity']}_i{self.orbital_parameters['inclination']}.png"
        plt.savefig(filename)
