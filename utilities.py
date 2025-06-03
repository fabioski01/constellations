from sgp4.api import Satrec
import numpy as np
from datetime import datetime, timezone
from math import trunc
import time
import constants

class orbiter:
    def __init__(self, tle_file):
        self.satellite = self.read_tle_from_file(tle_file)

    @staticmethod
    def read_tle_from_file(filename_TLE):
        '''    Function that:
        -reads one file with TLE data

        Args:

        filename_TLE : string
            Path and filename of the TLE data (in .txt)

        Returns:

        satellite: sgp4.wrapper.Satrec object
            object with satellite data readable by sgp4 Satrec functions
        '''
        print('Reading TLE')
        with open(filename_TLE, 'r') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 3):  # Read in blocks of 3 lines
                name = lines[i].strip()  # Satellite name
                line1 = lines[i + 1].strip()
                line2 = lines[i + 2].strip()
                # Create a satellite object
                satellite = Satrec.twoline2rv(line1, line2)
        return satellite

    @staticmethod
    def sgp4_propagator(satellite, start_time, step, end_time):
        '''    Function that:
        - generates a time domain, from specified start, stop and time step
        - integrates the orbit over this time domain (i.e., call the relevant routine in the SGP4 library)
        - stores the resulting the time series of position and velocity in a suitable data structure, e.g. a NumPy array.
        
        Args:
        
        satellite : sgp4.wrapper.Satrec object
            Object containing satellite data from TLE, readable by sgp4 functions.

        start_time : datetime object
            Start time as a datetime object
        
        step : float
            Time step in seconds for the propagation.
            Example: 60 for a 1-minute interval between points.
        
        end_time : tuple
            End time as a datetime object
        
        Returns:
        
        positions : numpy.ndarray
            An array of shape (N, 3) containing the satellite's position in the Earth-Centered
            Inertial (ECI) frame. Each row represents the (x, y, z) coordinates in kilometers.
        
        velocities : numpy.ndarray
            An array of shape (N, 3) containing the satellite's velocity in the Earth-Centered
            Inertial (ECI) frame. Each row represents the (vx, vy, vz) components in kilometers
            per second.
        
        times : numpy.ndarray
            An array of Julian dates representing the times at which the positions and velocities
            were computed. The Julian dates include fractional days for sub-day precision.
        '''

        # Convert start and end times to Julian date. JD is Julian Day, FR is Fractional Part (in days)
        print(f"Propagating sgp4 with start {start_time} step {step}, end {end_time}")
        start_jd, start_fr = TimeConversion.datetime_to_sgp4time(start_time)
        end_jd, end_fr = TimeConversion.datetime_to_sgp4time(end_time)
        print(f"sgp4 propagation done")    

        # Generate time steps in Julian date format
        jd_range = np.arange(start_jd + start_fr, end_jd + end_fr, step / 86400.0)  # Convert seconds to days

        # Propagate the orbit
        positions = []
        velocities = []
        times = []

        for jd in jd_range:
            e, r, v = satellite.sgp4(jd, 0)  # SGP4 propagation
            if e == 0:  # Check for successful computation
                positions.append(r)
                velocities.append(v)
                times.append(jd)
            else:
                print(f"Propagation error at JD {jd}: Error code {e}")

        # Convert results to arrays for further analysis
        positions = np.array(positions)  # Shape: (N, 3) for x, y, z
        velocities = np.array(velocities)  # Shape: (N, 3) for vx, vy, vz
        times = np.array(times)  # Julian dates

        return positions, velocities, times

class TimeConversion:

    @staticmethod
    def datetime_to_sgp4time(datetime_input):
        """Return two floats that, when added, produce the specified Julian date.

        The first float returned gives the date, while the second float
        provides an additional offset for the particular hour, minute, and
        second of that date.  Because the second float is much smaller in
        magnitude it can, unlike the first float, be accurate down to very
        small fractions of a second.

        Note that the first float, which gives the moment of midnight that
        commences the given calendar date, always carries the fraction
        ``.5`` because Julian dates begin and end at noon.  This made
        Julian dates more convenient for astronomers in Europe, by making
        the whole night belong to a single Julian date.

        This function is a modification of jday to get a datetime object as input.
        The function jday is a simple translation to Python of the C++ routine
        ``jday()`` in Vallado's ``SGP4.cpp``.
        
        Args:

        datetime : datetime object
            
        Returns:

        jd : float
            explained above
        fr : float
            explained above
        """
            
        print(f"Converting datetime {datetime_input}")
        
        if not isinstance(datetime_input, datetime):
            raise TypeError(f"Expected datetime object, got {type(datetime_input)}")

        year, mon, day, hr, minute, sec = datetime_input.year, datetime_input.month, datetime_input.day, datetime_input.hour, datetime_input.minute, datetime_input.second

        jd = (367.0 * year
            - 7 * (year + ((mon + 9) // 12.0)) * 0.25 // 1.0
        + 275 * mon / 9.0 // 1.0
        + day
            + 1721013.5)
        fr = (sec + minute * 60.0 + hr * 3600.0) / 86400.0;
        return jd, fr

    @staticmethod
    def datetime_to_mjd(datetime):
        '''
        From slides
        C = trunc((M - 14)/12) (1)
        JD0 = D - 32075 + trunc(1461 * (Y + 4800 + C)/4) (2)
        JD0+ = trunc(367 * (M - 2 - C * 12)/12) (3)
        JD0- = trunc(3 * (trunc((Y + 4900 + C)/100))/4) (4)
        JD = JD0 + F - 0.5 (5)
        MJD = JD - 2400000.5
        '''
        Y, M, D, F = datetime.year, datetime.month, datetime.day, datetime.hour/24
        C = trunc((M - 14)/12)
        JD0 = D - 32075 + trunc(1461 * (Y + 4800 + C)/4)
        JD0 += trunc(367 * (M - 2 - C * 12)/12)
        JD0 -= trunc(3 * (trunc((Y + 4900 + C)/100))/4)
        # JD0, F = datetime_to_sgp4time(datetime)
        JD = JD0 + F - 0.5
        MJD = JD - 2400000.5

        return MJD

class CoordinateConversion:

    @staticmethod
    def cartesian_to_keplerian(r, v, mu=3.986004418e5):
        """
        Converts Cartesian state vectors (position and velocity) into Keplerian orbital elements.

        Parameters:
        -----------
        r : list or numpy.ndarray
            The position vector in Cartesian coordinates [x, y, z], in km.
        v : list or numpy.ndarray
            The velocity vector in Cartesian coordinates [vx, vy, vz], in kilometers per second.
        mu : float, optional
            Standard gravitational parameter of the central body (default is 398600.4418 km^3/s^2 for Earth).

        Returns:
        --------
        dict
            A dictionary containing the following Keplerian orbital elements:
            - "semi_major_axis" : float
                Semi-major axis (a), in kilometers.
            - "eccentricity" : float
                Orbital eccentricity (e), unitless.
            - "inclination" : float
                Inclination (i) in degrees.
            - "RAAN" : float
                Right Ascension of the Ascending Node (Ω) in degrees.
            - "argument_of_periapsis" : float
                Argument of periapsis (ω) in degrees.
            - "true_anomaly" : float
                True anomaly (theta) in degrees.

        Assumptions:
        ------------
        - The central body is assumed to be spherically symmetric.
        - The position and velocity vectors are given in an inertial reference frame.

        Notes:
        ------
        - The function handles elliptical orbits but does not account for hyperbolic or parabolic trajectories.
        - All angles are returned in degrees for compatibility with most orbital mechanics software.

        Example:
        --------
        >>> r = [7000, 0, 0]  # Position in km
        >>> v = [0, 7.5, 1]   # Velocity in km/s
        >>> cartesian_to_keplerian(r, v)
        {
            "semi_major_axis": 7078.137,
            "eccentricity": 0.001,
            "inclination": 1.234,
            "RAAN": 45.678,
            "argument_of_periapsis": 78.901,
            "true_anomaly": 123.456
        }

        """

        # Magnitudes of r and v
        r_mag = np.linalg.norm(r) # m
        v_mag = np.linalg.norm(v) # m/s
        
        # Angular momentum vector
        h = np.cross(r, v) # m^2/s
        h_mag = np.linalg.norm(h)
        
        # Eccentricity vector
        e_vec = (np.cross(v, h) / mu) - (r / r_mag)
        e = np.linalg.norm(e_vec)
        
        # Semi-major axis
        a = 1 / ((2 / r_mag) - (v_mag**2 / mu))
        
        # Inclination
        i = np.arccos(h[2] / h_mag) # z component of h_vec
        
        # Node vector
        n = np.cross([0, 0, 1], h) # Ascending node vector (unit vector aligned with the z-direction cross h):
        n_mag = np.linalg.norm(n)
        
        # Right Ascension of Ascending Node (RAAN)
        Omega = np.arctan2(n[1], n[0])
        if Omega < 0:
            Omega += 2 * np.pi
        
        # Argument of Periapsis
        omega = np.arctan2(
            np.dot(np.cross(n, e_vec), h) / h_mag,
            np.dot(n, e_vec)
        )
        if omega < 0:
            omega += 2 * np.pi
        
        # True Anomaly
        nu = np.arctan2(
            np.dot(np.cross(e_vec, r), h) / h_mag,
            np.dot(e_vec, r)
        )
        if nu < 0:
            nu += 2 * np.pi
        
        return {
            "semi_major_axis": float(a),
            "eccentricity": float(e),
            "inclination": float(np.degrees(i)),
            "RAAN": float(np.degrees(Omega)),
            "argument_of_periapsis": float(np.degrees(omega)),
            "true_anomaly": float(np.degrees(nu)),
        }    

    @staticmethod
    def keplerian_to_cartesian(elements, mu=3.986004418e5):
        """
        Convert Keplerian elements to Cartesian position and velocity vectors.
        
        Parameters:
            elements (list of dict): A list of dictionaries containing Keplerian elements:
                - semi_major_axis (a) in km
                - eccentricity (e)
                - inclination (i) in degrees
                - RAAN (Ω) in degrees
                - argument_of_periapsis (ω) in degrees
                - true_anomaly (theta) in degrees
            mu (float): Gravitational parameter of the central body in km^3/s^2. Default is the value for Earth.
        
        Returns:
            list of tuple: Each tuple contains:
                - position (numpy array): Position vector [x, y, z] in km.
                - velocity (numpy array): Velocity vector [vx, vy, vz] in km/s.
        """
        positions = []
        velocities = []
        for elem in elements:
            # Extract Keplerian elements
            a = elem["semi_major_axis"]
            e = elem["eccentricity"]
            i = np.radians(elem["inclination"])  # Convert to radians
            raan = np.radians(elem["RAAN"])      # Convert to radians
            omega = np.radians(elem["argument_of_periapsis"])  # Convert to radians
            nu = np.radians(elem["true_anomaly"])  # Convert to radians
            
            # Semi-latus rectum
            p = a * (1 - e**2)
            
            # Position in the orbital plane
            r = p / (1 + e * np.cos(nu))  # Radius
            x_orb = r * np.cos(nu)
            y_orb = r * np.sin(nu)
            z_orb = 0
            
            # Velocity in the orbital plane
            h = np.sqrt(mu * p)  # Specific angular momentum
            vx_orb = -mu / h * np.sin(nu)
            vy_orb = mu / h * (e + np.cos(nu))
            vz_orb = 0
            
            # Orbital plane to ECI transformation matrix
            cos_omega = np.cos(omega)
            sin_omega = np.sin(omega)
            cos_raan = np.cos(raan)
            sin_raan = np.sin(raan)
            cos_i = np.cos(i)
            sin_i = np.sin(i)
            
            rotation_matrix = np.array([
                [cos_raan * cos_omega - sin_raan * sin_omega * cos_i,
                -cos_raan * sin_omega - sin_raan * cos_omega * cos_i,
                sin_raan * sin_i],
                [sin_raan * cos_omega + cos_raan * sin_omega * cos_i,
                -sin_raan * sin_omega + cos_raan * cos_omega * cos_i,
                -cos_raan * sin_i],
                [sin_omega * sin_i,
                cos_omega * sin_i,
                cos_i]
            ])
            
            # Transform position and velocity to ECI
            position_orb = np.array([x_orb, y_orb, z_orb])
            velocity_orb = np.array([vx_orb, vy_orb, vz_orb])
            position_eci = rotation_matrix @ position_orb
            velocity_eci = rotation_matrix @ velocity_orb
            
            # Append the results to respective lists
            positions.append(position_eci)
            velocities.append(velocity_eci)

        # Convert results to arrays for further analysis
        positions = np.array(positions)  # Shape: (N, 3) for x, y, z
        velocities = np.array(velocities)  # Shape: (N, 3) for vx, vy, vz

        return positions, velocities

    @staticmethod
    def ECI_cartesian_to_geographic(positions, times, reference_time=None):
        """
        Convert Cartesian positions to geographic latitude, longitude, and radial distance,
        accounting for Earth's rotation using ERA and GMST.
        
        Args:
            positions (np.ndarray): Nx3 array of Cartesian positions (x, y, z) in km.
            times (np.ndarray): 1D array of time values in seconds (since reference_time).
            reference_time (datetime, optional): The reference time (used to calculate GMST).
        
        Returns:
            list: A list of tuples (time, latitude, longitude, radial_distance).

        Notes:
            - Latitude is geocentric latitude (for spherical Earth model), and not geodetic (for ellipsoidal Earth model).
            - Longitude is correct for both geographic and geodetic systems.
            - Radial distance is from Earth's constant radius, not accounting for Earth's ellipsoidal shape.
        """
        
        # Default reference time as current time if not provided
        if reference_time is None:
            reference_time = datetime.utcnow()

        # GMST computation for each time step based on the alternative formula
        def calculate_gmst(time_in_seconds):
            # Calculate Julian Date (JD)
            jd = 2451545.0 + (time_in_seconds / 86400.0)  # Convert from seconds to Julian days
            D_UT = jd - 2451545.0  # Number of days since J2000.0

            # GMST in hours at 0h UT
            gmst_hours = (constants.GMST_CONSTANT + constants.EARTH_OMEGA * D_UT) % 24.0  # Normalize to [0, 24)
            return gmst_hours

        results = []
       
        # Ensure positions is a 2D array
        if isinstance(positions, np.ndarray) and positions.ndim == 1:
            positions = positions.reshape(1, -1)  # Reshape to 2D array if it's 1D
        # Ensure times is a 1D array
        if isinstance(times, (float, int)):
            times = np.array([times])  # Convert scalar to 1D array
        elif isinstance(times, np.ndarray) and times.ndim == 0:
            times = times.reshape(1)  # Reshape scalar array to 1D

        if len(positions) != len(times):
            raise ValueError("Positions and times must have the same length")
        
        # print(f"Positions shape: {positions.shape}, Times shape: {times.shape}")
        # print(f"Positions: {positions}, Times: {times}")

        for i, position in enumerate(positions):
            x, y, z = position*1000

            # Radial distance
            r = np.sqrt(x**2 + y**2 + z**2)

            # Latitude  
            latitude = np.degrees(np.arcsin(z / r))

            # Longitude (adjust for Earth's rotation)
            longitude = np.degrees(np.arctan2(y, x))

            # longitude, latitude, altitude = transformer.transform(x, y, z)

            # Calculate GMST (Greenwich Mean Sidereal Time) in hours
            time_in_seconds = times[i]
            gmst = calculate_gmst(time_in_seconds)
            
            # Convert GMST (in hours) to degrees (1 hour = 15 degrees)
            era = gmst * 15.0

            # Adjust longitude with ERA for Earth's rotation
            rotated_longitude = longitude - era
            rotated_longitude = (rotated_longitude + 360) % 360  # Normalize to [0, 360)

            # Store results as a tuple (time, latitude, longitude, radial_distance)
            results.append((times[i], latitude, rotated_longitude, r))
            # print(f"Appending to results: time={times[i]}, latitude={latitude}, longitude={rotated_longitude}, radial_distance={r}")

        return results
    
    @staticmethod
    def ECI_cartesian_to_ECEF_cartesian(eci_positions, times, reference_time=None):
        """
        Convert ECI (Earth-Centered Inertial) Cartesian positions to ECEF (Earth-Centered, Earth-Fixed) Cartesian positions.

        Args:
            eci_positions (np.ndarray): Nx3 array of ECI positions (x, y, z) in meters.
            times (np.ndarray): 1D array of time values in seconds (since reference_time).
            reference_time (datetime, optional): The reference time (used to calculate GMST).

        Returns:
            np.ndarray: Nx3 array of ECEF positions (x, y, z) in meters.

        Notes:
            - This function rotates the ECI position vector about the Z-axis by the GMST angle to obtain ECEF coordinates.
            - Assumes positions are given in meters.
        """

        if reference_time is None:
            reference_time = datetime.utcnow()

        def calculate_gmst(time_in_seconds):
            # Calculate Julian Date (JD)
            jd = 2451545.0 + (time_in_seconds / 86400.0)  # Convert from seconds to Julian days
            D_UT = jd - 2451545.0  # Number of days since J2000.0
            # GMST in hours at 0h UT
            gmst_hours = (constants.GMST_CONSTANT + constants.EARTH_OMEGA * D_UT) % 24.0  # Normalize to [0, 24)
            return gmst_hours

        # Ensure positions is a 2D array
        eci_positions = np.atleast_2d(eci_positions)
        # Ensure times is a 1D array
        times = np.atleast_1d(times)
        if len(eci_positions) != len(times):
            raise ValueError("Positions and times must have the same length")

        ecef_positions = np.zeros_like(eci_positions)

        for i, position in enumerate(eci_positions):
            # Calculate GMST for this time
            gmst = calculate_gmst(times[i])
            # Convert GMST (in hours) to radians (1 hour = 15 degrees)
            theta = np.radians(gmst * 15.0)
            # Rotation matrix about Z-axis
            R = np.array([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            # Apply rotation
            ecef_positions[i] = R @ position

        return ecef_positions
    
    @staticmethod
    def ECEF_cartesian_to_geodetic(x, y, z):
        """
        Convert ECEF (Earth-Centered, Earth-Fixed) Cartesian coordinates to geodetic coordinates (latitude, longitude, altitude)
        using the WGS84 ellipsoid model.

        Args:
            x (float): X coordinate in meters (ECEF).
            y (float): Y coordinate in meters (ECEF).
            z (float): Z coordinate in meters (ECEF).

        Returns:
            tuple: (latitude, longitude, altitude)
                - latitude (float): Geodetic latitude in degrees.
                - longitude (float): Longitude in degrees.
                - altitude (float): Altitude above the WGS84 ellipsoid in meters.

        Notes:
            - Geodetic latitude is the angle between the equatorial plane and the normal to the reference ellipsoid.
            - Longitude is the angle from the prime meridian.
            - Altitude is measured above the reference ellipsoid.
            - This function uses the Bowring's method for the conversion.
        """
        # WGS84 ellipsoid constants
        a = 6378137.0                # Semi-major axis [m]
        e2 = 6.69437999014e-3        # First eccentricity squared
        b = a * np.sqrt(1 - e2)      # Semi-minor axis [m]
        ep = np.sqrt((a**2 - b**2) / b**2)  # Second eccentricity

        # Distance from Z axis
        p = np.sqrt(x**2 + y**2)

        # Parametric (or reduced) latitude
        th = np.arctan2(a * z, b * p)

        # Longitude calculation
        lon = np.arctan2(y, x)

        # Latitude calculation (Bowring's formula)
        lat = np.arctan2(z + ep**2 * b * np.sin(th)**3,
                        p - e2 * a * np.cos(th)**3)

        # Radius of curvature in the prime vertical
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)

        # Altitude above ellipsoid
        alt = p / np.cos(lat) - N

        # Convert latitude and longitude from radians to degrees
        lat = np.degrees(lat)
        lon = np.degrees(lon)

        return lat, lon, alt

    @staticmethod
    def geodetic_to_ECEF_cartesian(lat, lon, alt):
        """
        Convert geodetic coordinates (latitude, longitude, altitude) to ECEF (Earth-Centered, Earth-Fixed) Cartesian coordinates.

        Args:
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.
            alt (float): Altitude above the WGS84 ellipsoid in meters.

        Returns:
            np.ndarray: ECEF position vector [x, y, z] in meters.

        Notes:
            - Uses the WGS84 ellipsoid model.
            - Latitude and longitude should be in degrees; altitude in meters.
            - ECEF is a right-handed coordinate system with:
                x-axis passing through the intersection of the Equator and Prime Meridian,
                y-axis passing through the Equator at 90°E longitude,
                z-axis passing through the North Pole.
        """
        # WGS84 ellipsoid constants
        a = 6378137.0  # semi-major axis [m]
        e2 = 6.69437999014e-3  # first eccentricity squared

        # Convert latitude and longitude from degrees to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # Prime vertical radius of curvature
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)    # X coordinate ECEF (m)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)    # Y coordinate ECEF (m)
        z = (N * (1 - e2) + alt) * np.sin(lat_rad)           # Z coordinate ECEF (m)
        return np.array([x, y, z])

class ExtendedCoordinateConversion(CoordinateConversion):
    @staticmethod
    def cartesian_to_spherical(positions):
        import numpy as np
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        lon = np.arctan2(y, x)       
        lat = np.arcsin(z / r)
        return r, lon, lat

class propagator:
    def __init__(self, mass=5.972e24, gravitational_constant=6.67430e-11):
        """
        Initializes the propagator with the mass of the gravitational body.
        
        Args:
            mass (float): Mass of the gravitational body (in kg). Default is Earth's mass.
            gravitational_constant (float): Gravitational constant (in m^3 kg^-1 s^-2). Default is the standard value.
        """
        self.mass = mass
        self.G = gravitational_constant

    @staticmethod
    def gravitational_acceleration(position, mass=5.972e24):
        """
        Compute the 3D gravitational acceleration due to a point mass.
        
        Args:
            position (np.ndarray): A 3-element array or list [x, y, z] representing the position vector (in meters).
            mass (float): The mass of the gravitational body (in kilograms). Defaults to the mass of the Earth.
        
        Returns:
            np.ndarray: A 3-element array representing the gravitational acceleration vector (in m/s^2).
        """
        # Gravitational constant (m^3 kg^-1 s^-2)
        G = 6.67430e-11
        
        # Compute the norm of the position vector
        r = np.linalg.norm(position)
        
        if r == 0:
            raise ValueError("Position vector cannot be the zero vector.")
        
        # Gravitational acceleration vector
        acceleration = -G * mass / r**3 * np.array(position)
        
        return acceleration

    def euler_integrator(initial_state, time_step, stop_time, acceleration_function):
        """
        Integrates the spacecraft's state using the Euler method and also measures the total computation time.
        
        Args:
            initial_state (list): Initial state vector [x, y, z, vx, vy, vz] in meters and m/s.
            time_step (float): Time step for integration in seconds.
            stop_time (float): Total simulation time in seconds.
            acceleration_function (function): Function to compute the 3D gravitational acceleration based on position.
        
        Returns:
            tuple: (positions, velocities, times, total_cpu_time, avg_time_per_step)
                - positions (numpy array): Array of positions over time.
                - velocities (numpy array): Array of velocities over time.
                - times (numpy array): Array of time steps.
                - total_cpu_time (float): Total computation time in seconds.
                - avg_time_per_step (float): Average computation time per integration step in seconds.
        """
        # Record the start time
        start_time = time.time()
        
        # Initialize state
        state = initial_state
        current_time = 0.0

        # Initialize lists to store time, position, and velocity
        times = [current_time]  # Include initial time
        positions = [state[:3]]  # Include initial position
        velocities = [state[3:]]  # Include initial velocity

        # Perform Euler integration until stop_time is reached
        while current_time < stop_time:
            # Compute gravitational acceleration (a function you define)
            acceleration = acceleration_function(state[:3])  # Assuming first 3 are position
            
            # Update position and velocity using Euler method
            state[3:] = [state[3] + acceleration[0] * time_step,
                        state[4] + acceleration[1] * time_step,
                        state[5] + acceleration[2] * time_step]
            state[:3] = [state[0] + state[3] * time_step,
                        state[1] + state[4] * time_step,
                        state[2] + state[5] * time_step]
            
            # Store the current state (time, position, velocity)
            times.append(current_time)
            positions.append(state[:3])  # Position: x, y, z
            velocities.append(state[3:])  # Velocity: vx, vy, vz
            
            # Increment time
            current_time += time_step

        # Convert lists to numpy arrays for easier manipulation later
        times = np.array(times)
        positions = np.array(positions)
        velocities = np.array(velocities)

        # Record the end time
        end_time = time.time()

        # Calculate total CPU time (in seconds)
        total_cpu_time = end_time - start_time

        # Calculate the average computational time per integration step
        num_steps = len(times)
        avg_time_per_step = total_cpu_time / num_steps if num_steps > 0 else 0

        # Return the separate arrays for positions, velocities, and times, along with the CPU timing information
        return positions, velocities, times, total_cpu_time, avg_time_per_step
    
    @staticmethod
    def runge_kutta_integrator(self, initial_state, time_step, stop_time, acceleration_function):
        """
        Integrates the spacecraft's state using the 4th-order Runge-Kutta method.

        Args:
            initial_state (list): Initial state vector [x, y, z, vx, vy, vz] in meters and m/s.
            time_step (float): Time step for integration in seconds.
            stop_time (float): Total simulation time in seconds.
            acceleration_function (function): Function to compute the 3D gravitational acceleration based on position.

        Returns:
            tuple: (positions, velocities, times)
                - positions (numpy array): Array of positions over time.
                - velocities (numpy array): Array of velocities over time.
                - times (numpy array): Array of time steps.
        """
        print(f'RK initial state {initial_state}')
        # Initialize state
        state = np.array(initial_state, dtype=np.float64)
        current_time = 0.0

        # Initialize lists to store time, position, and velocity
        times = [current_time]
        positions = [state[:3]]
        velocities = [state[3:]]

        # Integration loop
        while current_time < stop_time:
            # Compute Runge-Kutta coefficients
            k1 = propagator.compute_derivatives(state, acceleration_function)
            k2 = propagator.compute_derivatives(state + 0.5 * time_step * k1, acceleration_function)
            k3 = propagator.compute_derivatives(state + 0.5 * time_step * k2, acceleration_function)
            k4 = propagator.compute_derivatives(state + time_step * k3, acceleration_function)

            # Update the state
            state = state + (time_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Store the new state
            current_time += time_step
            times.append(current_time)
            positions.append(state[:3])
            velocities.append(state[3:])

        # Convert lists to numpy arrays
        times = np.array(times)
        positions = np.array(positions)
        velocities = np.array(velocities)

        return positions, velocities, times

    @staticmethod
    def compute_derivatives(self, state, acceleration_function):
        """
        Computes the derivatives (velocity and acceleration) for the state vector.

        Args:
            state (numpy array): State vector [x, y, z, vx, vy, vz].
            acceleration_function (function): Function to compute the 3D gravitational acceleration based on position.

        Returns:
            numpy array: Derivatives [vx, vy, vz, ax, ay, az].
        """
        position = state[:3]
        velocity = state[3:]
        acceleration = acceleration_function(position)
        return np.concatenate((velocity, acceleration)) 

class ExtendedPropagator(propagator):
    @staticmethod
    def euler_integrator_mod(initial_state, time_step, stop_time, acceleration_function):
        """
        Integrates the spacecraft's state using the Euler method and also measures the total computation time.
        
        Args:
            initial_state (list): Initial state vector [x, y, z, vx, vy, vz] in meters and m/s.
            time_step (float): Time step for integration in seconds.
            stop_time (float): Total simulation time in seconds.
            acceleration_function    (function): Function to compute the 3D gravitational acceleration based on position.
        
        Returns:
            tuple: (positions, velocities, times, total_cpu_time, avg_time_per_step)
                - positions (numpy array): Array of positions over time.
                - velocities (numpy array): Array of velocities over time.
                - times (numpy array): Array of time steps.
                - total_cpu_time (float): Total computation time in seconds.
                - avg_time_per_step (float): Average computation time per integration step in seconds.
        """
        # Record the start time
        start_time = time.time()
        
        # Initialize state
        state = initial_state
        current_time = 0.0

        # Initialize lists to store time, position, and velocity
        times = [current_time]  # Include initial time
        positions = [state[:3]]  # Include initial position
        velocities = [state[3:]]  # Include initial velocity

        # Perform Euler integration until stop_time is reached
        while current_time < stop_time:
            # Now pass the entire 6D state instead of state[:3]
            acceleration = acceleration_function(state)  
            # acceleration should be a 3-element array: [ax, ay, az]
            
            # Update position and velocity using Euler method
            # velocity = velocity + a * dt
            state[3] += acceleration[0] * time_step
            state[4] += acceleration[1] * time_step
            state[5] += acceleration[2] * time_step

            # position = position + velocity * dt
            state[0] += state[3] * time_step
            state[1] += state[4] * time_step
            state[2] += state[5] * time_step
            
            # Store the current state (time, position, velocity)
            times.append(current_time)
            positions.append(state[:3])  # Position: x, y, z
            velocities.append(state[3:])  # Velocity: vx, vy, vz
            
            # Increment time
            current_time += time_step

        # Convert lists to numpy arrays for easier manipulation later
        times = np.array(times)
        positions = np.array(positions)
        velocities = np.array(velocities)

        # Record the end time
        end_time = time.time()

        # Calculate total CPU time (in seconds)
        total_cpu_time = end_time - start_time

        # Calculate the average computational time per integration step
        num_steps = len(times)
        avg_time_per_step = total_cpu_time / num_steps if num_steps > 0 else 0

        # Return the separate arrays for positions, velocities, and times, along with the CPU timing information
        return positions, velocities, times, total_cpu_time, avg_time_per_step

class disturbances:
    """
    A collection of static methods for different orbital disturbances.
    """
    def __init__(self, sun_position, moon_position, mu_sun, mu_moon):
        self.sun_position_m = sun_position # this is needed because the srp acceleration needs the sun_position, but it's not in the integrator
        self.moon_position_m = moon_position
        self.mu_sun = mu_sun
        self.mu_moon = mu_moon

    @staticmethod
    def j2_acceleration(position, mu=3.986004418e14, R=6371e3, J2=1.083e-3):
        """
        Computes the gravitational disturbance from J2 in the inertial frame.

        Parameters
        ----------
        position : array-like of length 3
            [x, y, z] in meters (m).
        mu : float, optional
            Earth's gravitational parameter (m^3/s^2). Default is 3.986004418e14.
        R : float, optional
            Earth's mean radius (m). Default is 6371e3.
        J2 : float, optional
            Earth's J2 coefficient. Default is 1.083e-3.

        Returns
        -------
        numpy.ndarray of shape (3,)
            The J2 disturbing acceleration vector [ax, ay, az] in m/s^2.
        """
        x, y, z = position
        r_sq = x**2 + y**2 + z**2
        r = np.sqrt(r_sq)
        if r == 0:
            raise ValueError("Position vector cannot be the zero vector.")

        z2 = z**2
        # r^5 = (r^2)(r^3) = r_sq * r^3
        r5 = r_sq * (r**3)

        # Common factor in J2 acceleration
        factor = 1.5 * J2 * mu * R**2 / r5

        # Earth’s rotation axis is assumed to align with z-axis, so:
        #   a_j2_x = factor * x * (5*z^2 / r^2 - 1)
        #   a_j2_y = factor * y * (5*z^2 / r^2 - 1)
        #   a_j2_z = factor * z * (5*z^2 / r^2 - 3)
        z_term = (5.0 * z2 / r_sq) - 1.0

        a_j2_x = factor * x * z_term
        a_j2_y = factor * y * z_term
        a_j2_z = factor * z * ((5.0 * z2 / r_sq) - 3.0)

        return np.array([a_j2_x, a_j2_y, a_j2_z])
    
    @staticmethod
    def total_acceleration_j2(position):
        """
        Returns total acceleration = central gravity + J2 disturbance.
        position: [x, y, z] in meters.
        """
        a_central = propagator.gravitational_acceleration(position) 
        a_j2      = disturbances.j2_acceleration(position)          
        return a_central + a_j2
    
    @staticmethod
    def compute_eta_doornbos(r_earth_sat, r_earth_sun, R_earth=6371e3, R_sun=695700e3):
        """
        Computes eta = h_g / R_p using Doornbos eqs. (25)-(30).
        
        Parameters
        ----------
        r_earth_sat : np.ndarray (3,)
            Satellite position wrt Earth's center [m].
        r_earth_sun : np.ndarray (3,)
            Sun position wrt Earth's center [m].
        R_earth : float
            Earth's radius [m]. Considering mean radius of 6371 km
        R_sun : float
            Sun's physical radius [m]. Considering 695700 km
        
        Returns
        -------
        float
            eta = h_g / R_p
        """
        # (25) r_os = r_earth_sat - r_earth_sun 
        # but we only need e_os = r_os / ||r_os||
        r_os = r_earth_sat - r_earth_sun
        norm_os = np.linalg.norm(r_os)
        if norm_os == 0:
            # Degenerate: satellite-sun at same coords?
            return 0.0
        
        e_os = r_os / norm_os  # unit vector from Sun to sat
        
        # (26) r_ps = ( e_os · r_earth_sat ) e_os
        dot_prod = np.dot(e_os, r_earth_sat)
        r_ps = dot_prod * e_os
        r_ps_norm = np.linalg.norm(r_ps)
        
        # (27) r_earth_p = r_earth_sat - r_ps
        r_earth_p = r_earth_sat - r_ps
        r_earth_p_norm = np.linalg.norm(r_earth_p)
        
        # (28) h_g = r_earth_p - R_earth
        h_g = r_earth_p_norm - R_earth
        
        # (29) R_p = (r_ps / r_os) * R_sun
        # note that r_ps = ||r_ps||, r_os = ||r_os||
        R_p = (r_ps_norm / norm_os) * R_sun
        
        # (30) eta = h_g / R_p
        if R_p == 0:
            return 0.0
        
        eta = h_g / R_p

        return eta

    @staticmethod
    def shadow_function_fg(eta):
        """
        Doornbos eq. (32):
            f_g(eta) = 1 
                    - (1/pi)*arccos(eta)
                    + (eta/pi)*sqrt(1-eta^2),  for -1 <= eta < 1
                    = 0, for eta < -1
                    = 1, for eta >= 1 
        """
        value = 1 - 1/np.pi * np.arccos(eta) + eta/np.pi * np.sqrt(1 - eta**2)
        return value

    def srp_acceleration(
        self,
        state,
        sun_position,
        B_R=0.02,          # [m^2/kg] ballistic coefficient
        solar_flux=1361.0, # [W/m^2] (at 1 AU)
        c=299792458.0      # speed of light [m/s]
    ):
        """
        Compute SRP acceleration [m/s^2] using Doornbos (2012) penumbra/umbra model,
        treating B_R separately from P.

        Parameters
        ----------
        state : array-like of length 6
            [x, y, z, vx, vy, vz] of satellite, in an Earth-centered inertial frame [m, m/s].
        sun_position : array-like of length 3
            [X_sun, Y_sun, Z_sun] in the same inertial frame [m].
        B_R : float
            Radiation-pressure ballistic coefficient [m^2/kg].
        solar_flux : float
            Solar flux [W/m^2], default ~1361 W/m^2 at 1 AU.
        c : float
            Speed of light in m/s.

        Returns
        -------
        np.ndarray of shape (3,)
            The SRP acceleration vector [ax, ay, az] in m/s^2.
        """
        sat_pos = np.array(state[:3], dtype=float)
        sun_pos = np.array(sun_position, dtype=float)

        # Vector from satellite to Sun
        r_vec = sun_pos - sat_pos
        dist = np.linalg.norm(r_vec)
        if dist < 1e-9:
            # Degenerate: sat and Sun at the same point => no well-defined SRP
            return np.zeros(3, dtype=float)

        # Unit direction from satellite to Sun
        e_hat = r_vec / dist

        # Compute geometry factor eta
        eta = self.compute_eta_doornbos(sat_pos, sun_pos)
        # print(eta) # debug
        # Piecewise function
        if eta < -1:
            # Case with full shadow
            a_srp = np.zeros(3)
        elif -1 <= eta < 1:
            # Case with partial illumination
            f_g = self.shadow_function_fg(eta)
            # Nominal radiation pressure P = W / c [N/m^2]
            P = solar_flux / c  # [N/m^2 per unit mass, effectively m/s^2]
            # SRP acceleration
            a_srp = f_g* B_R * P * e_hat
        elif eta >= 1:
            # Nominal radiation pressure P = W / c [N/m^2]
            P = solar_flux / c  # [N/m^2 per unit mass, effectively m/s^2]
            # Case with full illumination
            a_srp = B_R * P * e_hat

        return a_srp
    
    def total_acceleration_srp(self, state):
        """
        Returns total acceleration = central gravity + SRP disturbance.
        position: [x, y, z] in meters.
        """
        position = [state[0], state[1], state[2]]
        a_central = propagator.gravitational_acceleration(position)
        a_srp      = disturbances.srp_acceleration(self, state, self.sun_position_m)
        return a_central + a_srp
    
    @staticmethod
    def atmospheric_drag_acceleration(
        state,
        ballistic_coeff=0.01,      # B_D [m^2/kg]
        rho_0=1.21e-18,           # mean density at reference altitude [kg/m^3]
        H=9.6e6,                  # scale height [m]
        ref_height=1.5e7,         # reference altitude [m] (15000 km)
        R_earth=6371e3,           # Earth radius [m]
        omega_earth=7.2921159e-5  # Earth's rotation rate [rad/s]
    ):
        """
        Compute the atmospheric drag acceleration [m/s^2] for a satellite in Earth-centered 
        inertial (ECI) coordinates, assuming an exponential density atmosphere rotating with Earth.
        
        Parameters
        ----------
        state : array-like, length 6
            [x, y, z, vx, vy, vz] in ECI [m, m/s].
        ballistic_coeff : float
            The ballistic coefficient B_D = C_D * A / m [m^2/kg].
            Here, 0.01 m^2/kg is assumed as given.
        rho_0 : float
            Mean density [kg/m^3] at the reference altitude. (1.21e-18)
        H : float
            Scale height [m]. (9.6e6 => 9600 km)
        ref_height : float
            Reference altitude [m] where density = rho_0. (1.5e7 => 15000 km)
        R_earth : float
            Earth radius [m]. (6371e3 by default)
        omega_earth : float
            Earth's rotation rate [rad/s]. (~7.2921159e-5)
        
        Returns
        -------
        np.ndarray of shape (3,)
            Drag acceleration in ECI frame [ax, ay, az] [m/s^2].
        """
        # Extract position and velocity in ECI
        x, y, z, vx, vy, vz = state
        r_sat = np.array([x, y, z], dtype=float)
        v_sat = np.array([vx, vy, vz], dtype=float)
        
        # 1) Compute altitude (above Earth's surface or ref altitude).
        #    Approach A: altitude above Earth's surface
        #        alt = ||r_sat|| - R_earth
        #    Approach B: altitude above the reference layer
        #        alt = ||r_sat|| - (R_earth + ref_height)
        #    The latter is consistent with direct usage in the exponent:
        
        r_norm = np.linalg.norm(r_sat)
        # altitude above Earth's surface:
        alt_surf = r_norm - R_earth  
        # altitude above reference altitude:
        alt = alt_surf - (ref_height)  # how far above/below that 15k km layer
        
        # 2) Compute density from the exponential model
        rho = rho_0 * np.exp(-alt / H) if alt >= -ref_height else 0.0
        
        # 3) Compute the velocity of the rotating atmosphere
        #    Earth's rotation vector (about z-axis, ignoring precession or nutation):
        omega_vec = np.array([0.0, 0.0, omega_earth], dtype=float)
        
        # Relative velocity of satellite w.r.t. atmosphere
        v_atmo = np.cross(omega_vec, r_sat)  # rotation velocity of atmosphere at position r_sat
        v_rel = v_sat - v_atmo
        
        # 4) Drag acceleration
        v_rel_norm = np.linalg.norm(v_rel)
        # if v_rel_norm < 1e-10 or rho <= 1e-30:
        #     # If extremely small relative velocity or negligible density => no drag
        #     return np.zeros(3)
        
        # a_drag = -1/2 * B_D * rho * |v_rel| * v_rel
        # sign is negative b/c drag opposes v_rel
        a_drag = -0.5 * ballistic_coeff * rho * v_rel_norm * v_rel
        
        return a_drag
    
    def total_acceleration_drag(self, state):
        """
        Returns total acceleration = central gravity + drag disturbance.
        position: [x, y, z] in meters.
        """
        pos = state[:3]
        a_central = propagator.gravitational_acceleration(pos)
        a_drag = self.atmospheric_drag_acceleration(state)
        return a_central + a_drag

    @staticmethod
    def tbp_acceleration_two_bodies(state,
                                    body1_position,
                                    body2_position,
                                    mu_body1,
                                    mu_body2):
        """
        Computes the 3D third-body acceleration due to two external bodies (Body 1 and Body 2).

        Parameters
        ----------
        state : array-like of length 6
            [x, y, z, vx, vy, vz] of the satellite in an Earth-centered inertial frame [m, m/s].
        body1_position : array-like of length 3
            [X1, Y1, Z1] in ECI [m], position of Body 1 wrt Earth.
            (For example, the Sun's position wrt Earth.)
        body2_position : array-like of length 3
            [X2, Y2, Z2] in ECI [m], position of Body 2 wrt Earth.
            (For example, the Moon's position wrt Earth.)
        mu_body1 : float
            Gravitational parameter GM of Body 1 [m^3/s^2].
            (For example, mu_sun ~ 1.32712440018e20 for the Sun.)
        mu_body2 : float
            Gravitational parameter GM of Body 2 [m^3/s^2].
            (For example, mu_moon ~ 4.9048695e12 for the Moon.)

        Returns
        -------
        np.ndarray of shape (3,)
            The total third-body acceleration from both bodies [ax, ay, az] in m/s^2..
        """

        # Extract satellite position from the state
        r_sat = np.array(state[:3], dtype=float)

        # Convert the positions of the two bodies into numpy arrays
        r_body1 = np.array(body1_position, dtype=float)
        r_body2 = np.array(body2_position, dtype=float)

        sat_body1 = r_body1 - r_sat
        sat_body2 = r_body2 - r_sat
        acc_body1 = mu_body1 * (sat_body1/(np.linalg.norm(sat_body1)**3)
                            - body1_position/(np.linalg.norm(body1_position)**3))
        acc_body2 = mu_body2 * (sat_body2/(np.linalg.norm(sat_body2)**3)
                            - body2_position/(np.linalg.norm(body2_position)**3))
        acc = acc_body1 + acc_body2
        return acc

    def total_acceleration_tbp(self, state):
        """
        Returns total acceleration = Earth central gravity + third-body perturbations (Sun + Moon).
        """
        pos = state[:3]
        a_central = propagator.gravitational_acceleration(pos)  # Earth point-mass
        a_tbp = self.tbp_acceleration_two_bodies(state, self.sun_position_m, self.moon_position_m, self.mu_sun, self.mu_moon)
        return a_central + a_tbp