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

    def compute_sample_coverage(self, masking_angle=80.0, fov=120.0):
        """
        Compute the coordinates of the area covered by the satellite at a specific instant.

        Args:
            masking_angle (float): Minimum elevation angle (in degrees) for coverage, the same as the elevation threshold. Default is 80°. The larger the elevation angle, the smaller the coverage area.
            fov (float): Field of view of the satellite sensor in degrees. Default is 120°, which is slightly more than 2 rad, which gives gamma>1>rho, and does not limit the cap angle. 

        Returns:
            list of tuple: List of (latitude, longitude) tuples representing the covered area.
            float: Fraction of Earth covered by the satellite.
            float: Cap angle in degrees.

        Notes:
            From Earth Coverage by Satellites in Circular Orbit, Alan R. Washburn, Department of Operations Research, Naval Postgraduate School
            - R = radius of Earth (6378 km)
            - r = radius of orbit
            - alpha = cap angle, the central Earth angle. Angle between the satellite's position (r) and the edge of the coverage area (R).
            - beta = masking angle, same as the elevation threshold.
            - 2gamma = field of view
            Coverage is often limited by a masking angle (BETA), which is the minimum elevation angle (elevation threshold) that the satellite's sensor can detect a target on the ground. 
            A target is visible to the satellite only if the satellite's elevation, as viewed by the target, is at least BETA (the masking angle). 
            The relation between ALPHA and BETA is that, with RHO = R/r < 1, using the law of sines on the sides r and R, plus the fact that
            `GAMMA + ALPHA = BETA = pi/2
            we have the following relations:
                ALPHA = arccos(RHO*cosBETA) - BETA (with 0 < BETA < pi/2)
            Coverage may also be limited by a sensor's field of view (2GAMMA), rather than a masking angle (or elevation threshold, BETA), in  which case,
                ALPHA = arcsin((sinGAMMA) / RHO) - GAMMA; with sinGAMMA <= RHO
            #If sinGAMMA>RHO, the sensor's field of view is not limiting and the formula for ALPHA(BETA) applies with BETA=0.
            A cap of size ALPHA covers a fraction f of Earth, where
                f = (1 - cos(ALPHA)) / 2; with 0 < ALPHA < pi/2
        """
        # Extract satellite position
        sat_position = self.state["position"] # in km
        sat_times = self.state["time"]

        ECEF_cartesian_orbit = CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(sat_position*1000, sat_times) # km to m
        ECEF_geodetic_orbit = [CoordinateConversion.ECEF_cartesian_to_geodetic(x, y, z) for x, y, z in ECEF_cartesian_orbit]
        # Access the first (and only) tuple in the results list
        if len(ECEF_geodetic_orbit) == 0:
            raise ValueError("No results returned from cartesian_to_geographic")
        sat_lat, sat_lon, altitudes = zip(*ECEF_geodetic_orbit)  # Unpack the single tuple

        sat_altitude_km = altitudes[0] / 1000.0  # altitudes is in meters, [0] to get the first (and only) altitude value as float

        beta = np.radians(masking_angle)
        gamma = np.radians(fov / 2.0)
        rho = constants.EARTH_RADIUS / (constants.EARTH_RADIUS + sat_altitude_km)  # RHO = R/r, with R the radius of Earth and r the radius of orbit

        if np.sin(gamma) > rho:
            # If the field of view is not limiting, use the masking angle formula
            # print("Field of view is not limiting, using masking angle formula.")
            # Compute the cap angle alpha with the masking angle BETA
            alpha = np.arccos(rho * np.cos(beta)) - beta
        elif np.sin(gamma) <= rho:
            # If the field of view is limiting, use the field of view formula
            # print("Field of view is limiting, using field of view formula.")
            # Compute the cap angle alpha with the field of view GAMMA
            alpha = np.arcsin(np.sin(gamma) / rho) - gamma

        cap_angle = np.degrees(alpha)
        # print(f"Cap angle: {cap_angle} degrees")

        fraction_covered = (1 - np.cos(alpha)) / 2
        # print(f"Fraction of Earth covered: {fraction_covered:.4f}")

        # Convert satellite subpoint to radians
        lat0 = np.radians(sat_lat[0])
        lon0 = np.radians(sat_lon[0])

        coverage_points = []
        num_points = 100  # Number of points to draw the circle

        for theta in np.linspace(0, 2 * np.pi, num_points):
            # Spherical law of cosines for latitude
            lat = np.arcsin(np.sin(lat0) * np.cos(alpha) + np.cos(lat0) * np.sin(alpha) * np.cos(theta))
            # Spherical law of cosines for longitude
            lon = lon0 + np.arctan2(np.sin(theta) * np.sin(alpha) * np.cos(lat0),
                                    np.cos(alpha) - np.sin(lat0) * np.sin(lat))
            # Convert back to degrees
            lat_deg = np.degrees(lat)
            lon_deg = np.degrees(lon)
            # Normalize longitude to [-180, 180]
            lon_deg = (lon_deg + 180) % 360 - 180
            if -90 <= lat_deg <= 90 and -180 <= lon_deg <= 180:
                coverage_points.append((lat_deg, lon_deg))

        return coverage_points, fraction_covered, cap_angle

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

    # def plot_coverage(self, coverage_points):
    #     """
    #     Plot the coverage area on a map.

    #     Args:
    #         coverage_points (list of tuple): List of (latitude, longitude) tuples representing the covered area.
    #     """

    #     # Extract orbital parameters for the subtitle
    #     orbital_params_str = (
    #         f"a={self.orbital_parameters['semi_major_axis']}km / h={self.orbital_parameters['semi_major_axis']-constants.EARTH_RADIUS} km, "
    #         f"e={self.orbital_parameters['eccentricity']}, "
    #         f"i={self.orbital_parameters['inclination']}°, "
    #         f"RAAN: {self.orbital_parameters['RAAN']}°, "
    #         f"ω={self.orbital_parameters['argument_of_periapsis']}°,"
    #         f"True Anomaly: {self.orbital_parameters['true_anomaly']}°"
    #     )
        
    #     fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    #     ax.stock_img()
    #     ax.coastlines()

    #     # Extract latitudes and longitudes
    #     lats, lons = zip(*coverage_points)

    #     # Plot coverage points
    #     ax.scatter(lons, lats, s=1, color="blue", label="Coverage Area", transform=ccrs.PlateCarree())
    #     ax.legend(loc="lower left")
    #     ax.set_title(f"Satellite Coverage Area\n({orbital_params_str})")
    #     filename = f"plots/coverage_sample_1sat_a{self.orbital_parameters['semi_major_axis']}_e{self.orbital_parameters['eccentricity']}_i{self.orbital_parameters['inclination']}.png"
    #     plt.savefig(filename)

    @staticmethod
    def is_visible_geodetic_ECEF(sat_geodetic, gs_geodetic, elevation_threshold_deg):
        """
        Check if a satellite is visible from a ground station given geodetic coordinates and elevation threshold.

        Args:
            sat_geodetic (tuple): (lat, lon, alt) of satellite in degrees and meters.
            gs_geodetic (tuple): (lat, lon, alt) of ground station in degrees and meters.
            elevation_threshold_deg (float): Minimum elevation angle in degrees.

        Returns:
            bool: True if visible, False otherwise.
            float: Elevation angle in degrees.
        """
        sat_ecef = CoordinateConversion.geodetic_to_ECEF_cartesian(*sat_geodetic)
        gs_ecef = CoordinateConversion.geodetic_to_ECEF_cartesian(*gs_geodetic)
        # Vector from ground station to satellite
        rho = sat_ecef - gs_ecef
        rho_norm = rho / np.linalg.norm(rho)
        gs_norm = gs_ecef / np.linalg.norm(gs_ecef)  # Local zenith
        
        # Angle between rho and zenith
        cos_alpha = np.dot(rho_norm, gs_norm)
        alpha = np.arccos(cos_alpha)
        elevation = np.degrees(np.pi/2 - alpha)

        return elevation >= elevation_threshold_deg, elevation
    
    @staticmethod
    def is_visible_cartesian_ECEF(sat_ecef_cartesian, gs_geodetic, elevation_threshold_deg):
        """
        Check if a satellite is visible from its cartesian ECEF position, the ground station given its geodetic coordinates, and elevation threshold.

        Args:
            sat_ecef_cartesian (np.ndarray): ECEF position vector [x, y, z] in meters.
            gs_geodetic (tuple): (lat, lon, alt) of ground station in degrees and meters.
            elevation_threshold_deg (float): Minimum elevation angle in degrees.

        Returns:
            bool: True if visible, False otherwise.
            float: Elevation angle in degrees.
        """
        gs_ecef = CoordinateConversion.geodetic_to_ECEF_cartesian(*gs_geodetic)

        # Vector from ground station to satellite
        rho = sat_ecef_cartesian - gs_ecef
        # print(f'norm gs_ecef: {np.linalg.norm(gs_ecef)}')
        # print(f'norm sat_ecef_cartesian: {np.linalg.norm(sat_ecef_cartesian)}')
        # print(f'norm rho: {np.linalg.norm(rho)}')

        rho_norm = rho / np.linalg.norm(rho)
        gs_norm = gs_ecef / np.linalg.norm(gs_ecef)  # Local zenith

        # Angle between rho and zenith
        cos_alpha = np.dot(rho_norm, gs_norm)
        alpha = np.arccos(cos_alpha)
        elevation = np.degrees(np.pi/2 - alpha)

        # if elevation >= elevation_threshold_deg:
        #     print(f'norm rho: {np.linalg.norm(rho)}')

        return elevation >= elevation_threshold_deg, elevation
    

def get_ground_station_coordinates(station_name):
    """
    Returns the geodetic coordinates for a given ground station.
    
    Args:
        station_name (str): Name of the ground station
        
    Returns:
        tuple: (latitude [deg], longitude [deg], altitude [m])
    """
    ground_stations = {
        "Japan": (45.0, 140.0, 0.0),
        "AFB_Andersen": (13.6, 144.9, 0.0),
        "AFB_Hickam": (21.3, -157.9, 0.0),
        "Kiruna": (67.9, 20.4, 0.0),
        "AFB_Elmendorf": (61.2, -149.9, 0.0),
        "Svalbard": (78.2, 15.6, 0.0),
        "Troll_Antarctica": (-72.0, 2.5, 0.0),
        "Fucino": (42.0, 13.6, 0.0)
    }
    
    if station_name in ground_stations:
        return ground_stations[station_name]
    else:
        raise ValueError(f"Unknown ground station name: {station_name}")