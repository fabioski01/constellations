from utilities import CoordinateConversion
import numpy as np
from datetime import datetime
from sgp4.api import Satrec, WGS84, jday

class Constellation:
    """
    A class to represent a simple satellite constellation.

    Attributes:
        num_satellites (int): Number of satellites in the constellation.
        orbital_parameters (dict): Orbital parameters for the satellites.
    """

    def __init__(self, num_satellites, orbital_parameters, epoch):
        """
        Initializes the Constellation object with orbital parameters.

        Args:
            num_satellites (int): Number of satellites in the constellation.
            orbital_parameters (dict): Dictionary containing orbital parameters as keys:
                - semi_major_axis (a) in km
                - eccentricity (e)
                - inclination (i) in degrees
                - RAAN (Ω) in degrees
                - argument_of_periapsis (ω) in degrees
                - true_anomaly (theta) in degrees (initial value)
                - *optional* ndot (rev/day²), nddot (rev/day³), bstar (1/ER)
        """
        self.num_satellites = num_satellites
        self.orbital_parameters = orbital_parameters
        self.epoch = epoch
        self.satrecs = self._build_satrecs()

        # print(orbital_parameters)
        # print(epoch)

    # def generate_positions(self):
    #     """
    #     Generates the Cartesian positions of all satellites in the constellation.

    #     The satellites are evenly distributed along the orbit based on their true anomalies.

    #     Returns:
    #         np.ndarray: A 2D array of shape (num_satellites, 3) containing the Cartesian
    #                     positions of the satellites in the Earth-Centered Inertial (ECI) frame.
    #                     Each row corresponds to the (x, y, z) coordinates of a satellite.
    #     """
    #     positions = []
    #     velocities = []
    #     for i in range(self.num_satellites):
    #         # Update the true anomaly for each satellite
    #         self.orbital_parameters["true_anomaly"] = 360 * i / self.num_satellites
    #         # Use the existing keplerian_to_cartesian function
    #         pos, vel = CoordinateConversion.keplerian_to_cartesian([self.orbital_parameters])
    #         positions.append(pos[0])  # Extract the position vector
    #         velocities.append(vel[0])  # Extract the velocity vector
    #     return np.array(positions), np.array(velocities)
    
    def generate_positions_with_time(self, start_time, end_time, time_step):
        """
        Generates the ECI TEME (True Equator Mean Equinox) Cartesian positions, velocities, and time stamps of all satellites in the constellation.

        Args:
            start_time (datetime): Start time of the propagation.
            end_time (datetime): End time of the propagation.
            time_step (float): Time step in seconds.

        Returns:
            dict: A dictionary containing:
                - "positions": A 3D numpy array of shape (num_satellites, num_time_steps, 3) with satellite positions in ECI TEME (True Equator Mean Equinox).
                - "velocities": A 3D numpy array of shape (num_satellites, num_time_steps, 3) with satellite velocities in ECI TEME (True Equator Mean Equinox).
                - "time_stamps": A 1D numpy array of shape (num_time_steps,) with time stamps in seconds since epoch.
        """

        # Convert start and end times to Julian dates
        start_jd, start_fr = jday(
            start_time.year, start_time.month, start_time.day,
            start_time.hour, start_time.minute, start_time.second
        )
        end_jd, end_fr = jday(
            end_time.year, end_time.month, end_time.day,
            end_time.hour, end_time.minute, end_time.second
        )

        # Compute the number of time steps
        num_time_steps = int((end_jd + end_fr - start_jd - start_fr) * 86400 / time_step) + 1

        # Pre-allocate arrays for positions, velocities, and time stamps
        positions = np.zeros((self.num_satellites, num_time_steps, 3))
        velocities = np.zeros((self.num_satellites, num_time_steps, 3))
        time_stamps = np.linspace(0, (num_time_steps - 1) * time_step, num_time_steps)

        # Propagate each satellite
        for i, sat in enumerate(self.satrecs):
            for j in range(num_time_steps):
                jd = start_jd + (j * time_step) / 86400.0
                fr = start_fr
                err, r, v = sat.sgp4(jd, fr)
                if err != 0:
                    raise RuntimeError(f"SGP4 error {err} for satellite {i} at time step {j}")
                positions[i, j, :] = r  # Position [x, y, z] in km
                velocities[i, j, :] = v  # Velocity [vx, vy, vz] in km/s

        return {
            "positions": positions,
            "velocities": velocities,
            "time_stamps": time_stamps
        }    

    def _build_satrecs(self):
        """
        Construct and initialize a Satrec object for each satellite in the constellation.

        Translates inputs of classical orbital elements into the raw parameters that SGP4's Satrec needs,
        including any derivatives for perturbations etc. (ndot, nddot, bstar).

        Steps:
        1. Convert the epoch (a Python datetime) to SGP4's Julian date + fraction
           using `jday()`.
        2. Unpack orbital parameters:
           - semi-major axis `a` [km]
           - eccentricity `ecco`
           - inclination `inclo`, RAAN `nodeo0`,
             argument of periapsis `argpo`, true anomaly `f0` (all in degrees → radians)
        3. Compute the mean motion (rev/day) from µ and `a`.
        4. Convert true anomaly `f0` → eccentric anomaly `E` → mean anomaly `mo`.
        5. Pull in your derivatives `ndot`, `nddot`, and drag coefficient `bstar` (defaulting to 0 if not specified).
        6. Loop over `num_satellites`, optionally phase each one by tweaking
           `nodeo` or `mo`, then call `sat.sgp4init(...)` with:
             - gravity model (`WGS84`)
             - opsmode (`'i'` for “imported” mode)
             - satellite ID (`i`)
             - epoch (jd, fr)
             - bstar, ndot, nddot
             - ecco, argpo, inclo, mo, mean_motion, nodeo
        7. Collect and return the list of fully-initialized Satrec objects.

        Returns
        -------
        List[Satrec]
            A list of `sgp4.api.Satrec` instances, each ready for propagation via
            `sat.sgp4(jd, fr)`.
        """

        satrecs = []
        # epoch → JD + fraction
        jd, fr = jday(
            self.epoch.year, self.epoch.month, self.epoch.day,
            self.epoch.hour, self.epoch.minute, self.epoch.second
        )

        # unpack classical elements
        a      = self.orbital_parameters['semi_major_axis']
        ecco   = self.orbital_parameters['eccentricity']
        inclo  = np.deg2rad(self.orbital_parameters['inclination'])
        nodeo0 = np.deg2rad(self.orbital_parameters['RAAN'])
        argpo  = np.deg2rad(self.orbital_parameters['argument_of_periapsis'])
        f0     = np.deg2rad(self.orbital_parameters['true_anomaly'])

        GRAVITATIONAL_CONSTANT = 398600.4418  # Earth grav. parameter [km³/s²]
        # compute mean_motion and mean anomaly
        n_rad_s    = np.sqrt(GRAVITATIONAL_CONSTANT / a**3)
        mean_motion = n_rad_s / (2*np.pi) * 86400.0  # rev/day
        E = 2*np.arctan(np.tan(f0/2) * np.sqrt((1 - ecco)/(1 + ecco)))
        mo = E - ecco*np.sin(E)

        # pull derivatives from input (or default to 0)
        ndot  = self.orbital_parameters.get('ndot',  0.0)
        nddot = self.orbital_parameters.get('nddot', 0.0)
        bstar = self.orbital_parameters.get('bstar', 0.0)

        # jd, fr from jday(...)
        epoch_days = (jd + fr) - 2433281.5

        for satellite_code in range(self.num_satellites):
            # simple per‐satellite phasing, adjusting nodeo or mo
            nodeo = nodeo0 + satellite_code*(2*np.pi/self.num_satellites)
            # mo    = mo +    satellite_code*(2*np.pi/self.num_satellites)
            # argpo    = argpo +    satellite_code*(2*np.pi/self.num_satellites)
            # nodeo = nodeo0
            
            # sgp4 expects mean motion in rad/min, not rev/day
            # μ in km³/s², a in km
            n_rad_s  = np.sqrt(GRAVITATIONAL_CONSTANT / a**3)      # rad / s
            no_kozai = n_rad_s * 60.0         # rad / minute

            sat = Satrec()
            sat.sgp4init(
                WGS84,                # 1. gravity model
                'i',                  # 2. opsmode
                satellite_code,       # 3. satnum called in loop
                epoch_days,           # 4. single epoch (float)
                bstar,                # 5. B* drag
                ndot,                 # 6. 1st derivative of mean motion
                nddot,                # 7. 2nd derivative
                ecco,                 # 8. eccentricity
                argpo,                # 9. argument of perigee
                inclo,                # 10. inclination
                mo,                   # 11. mean anomaly
                no_kozai,             # 12. mean motion (radians/minute)
                nodeo                 # 13. RAAN
            )

            satrecs.append(sat)
        return satrecs