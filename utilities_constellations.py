from utilities import CoordinateConversion
import numpy as np
from datetime import datetime
from sgp4.api import Satrec, WGS84, jday
import constants

class Constellation:
    """
    A class to represent a simple satellite constellation.

    Attributes:
        num_satellites (int): Number of satellites in the constellation.
        orbital_parameters (dict): Orbital parameters for the satellites.
        pattern (str): Constellation pattern (e.g., "Walker-Delta-SinglePlane", "Walker-Delta").
        P (int): Number of orbital planes (for pattern "Walker-Delta").
        F (int): Phasing between satellites in adjacent planes (for pattern "Walker-Delta").
    """

    def __init__(self, num_satellites, orbital_parameters, epoch, pattern, P=1, F=1):
        """
        Initializes the Constellation object with orbital parameters.

        Args:
            num_satellites (int): Number of satellites in the constellation. (T in Walker delta pattern notation)
            orbital_parameters (dict): Dictionary containing orbital parameters as keys:
                - semi_major_axis (a) in km
                - eccentricity (e)
                - inclination (i) in degrees
                - RAAN (Ω) in degrees
                - argument_of_periapsis (ω) in degrees
                - true_anomaly (theta) in degrees (initial value)
                - *optional* ndot (rev/day²), nddot (rev/day³), bstar (1/ER)
            epoch (datetime): Epoch for the orbital elements.
            pattern (str): Constellation pattern (default is "Walker-Delta-SinglePlane").
            P (int): (default is 1, for pattern "Walker-Delta") Number of orbital planes evenly spaced in node.
            F (int): (default is 1, for pattern "Walker-Delta") Relative spacing between satellites in adjacent planes, must be between 0 to (P-1).
        """
        self.num_satellites = num_satellites
        self.orbital_parameters = orbital_parameters
        self.epoch = epoch
        self.pattern = pattern

    #     self.satrecs = self._build_satrecs()

    # def set_walker_parameters(self, P, F):
        """
        Sets the Walker-Delta parameters (P and F) after initialization.

        Args:
            P (int): Number of orbital planes.
            F (int): Phasing between satellites in adjacent planes.

        Notes:
        Walker-Delta pattern contains a total number of T satellites, evenly distributed in P orbital planes, with S satellites
        in each plane (S=T/P).
        All of the orbits are assumed to be at the same inclination i wrt equator. The reference plane need not be the Earth's ]
        equator, however, since orbit perturbations depend on the inclination relative to the equator, this is the most
        practical choice.
        Unlike the streets of coverage patterns, the ascending nodes of the P orbital planes in Walker patterns are uniformly
        distributed around the equator at intervals of 360/P degrees.
        Within each orbital plane, the T/P satellites are evenly distributed along the orbit at intervals of 360/(T/P) degrees.
        The relative phase between the satellites in adjacent orbit planes is defined by the phase diference, which in a
        constellation is the angle in the direction of motion from the ascending node to the nearest satellite at a time
        when a satellite in the next mosst westerly plane is at its ascending node.
        In order for all the orbit planes to have the same relationship to each other, the phase difference must be an
        integral multiple, F, of 360/T degrees, where F can be any integer from 0 to P-1.
        Usually such constellation is written in shorthand notation as i:T/P/F.

        Defining a patern unit PU=360deg/T, then:
        - planes are spaced at intervals of S PUs in node (node spacing)
        - satellites are spaced at intervals of P PUs within each plane (in plane spacing)

        If a satellite is at ist ascending node, the next most easterly satellite will be F PUs past the node.
        F is an integer which can take on any value from 0 to P-1.
        """
        self.P = P
        self.F = F

        # Validate inputs for Walker-Delta
        if self.pattern == "Walker-Delta" and self.num_satellites % self.P != 0:
            raise ValueError("Total number of satellites (T) must be divisible by the number of planes (P).")
        if self.pattern == "Walker-Delta" and self.num_satellites % self.F != 0:
            raise ValueError("Total number of satellites (T) must be divisible by phasing between satellites in adjacent planes (F).")
        
        self._satrecs = self._build_satrecs()

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

        # compute mean_motion and mean anomaly
        n_rad_s    = np.sqrt(constants.GRAVITATIONAL_CONSTANT / a**3)
        mean_motion = n_rad_s / (2*np.pi) * 86400.0  # rev/day
        E = 2*np.arctan(np.tan(f0/2) * np.sqrt((1 - ecco)/(1 + ecco)))
        mo = E - ecco*np.sin(E)

        # pull derivatives from input (or default to 0)
        ndot  = self.orbital_parameters.get('ndot',  0.0)
        nddot = self.orbital_parameters.get('nddot', 0.0)
        bstar = self.orbital_parameters.get('bstar', 0.0)

        # jd, fr from jday(...)
        epoch_days = (jd + fr) - 2433281.5

        # sgp4 expects mean motion in rad/min, not rev/day
        # μ in km³/s², a in km
        n_rad_s  = np.sqrt(constants.GRAVITATIONAL_CONSTANT / a**3)      # rad / s
        no_kozai = n_rad_s * 60.0         # rad / minute
        
        if self.pattern == "Walker-Delta-SingleSatPerPlane":
            for satellite_code in range(self.num_satellites):
                # simple per‐satellite phasing, adjusting nodeo or mo
                nodeo = nodeo0 + satellite_code*(2*np.pi/self.num_satellites)   # RAAN
                # mo = mo + satellite_code*(2*np.pi/self.num_satellites)    # mean anomaly
                # argpo = argpo + satellite_code*(2*np.pi/self.num_satellites)  # argument of perigee
                # nodeo = nodeo0

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

        elif self.pattern == "Walker-Delta-SinglePlane":
            for satellite_code in range(self.num_satellites):
                # simple per‐satellite phasing, adjusting nodeo or mo
                satellite_mo = mo + satellite_code*(2*np.pi/self.num_satellites)    # mean anomaly
                # debug
                # satellite_mo = mo + 0.1*satellite_code
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
                    satellite_mo,                   # 11. mean anomaly
                    no_kozai,             # 12. mean motion (radians/minute)
                    nodeo0                # 13. RAAN
                )
                
                # print(sat.nodeo, sat.mo, sat.argpo, sat.inclo, sat.ecco, sat.bstar)
                # print(epoch_days)
                # print(f"Satellite {satellite_code}: Mean Anomaly = {np.rad2deg(satellite_mo):.2f}°")
                satrecs.append(sat)            
                
        elif self.pattern == "Walker-Delta":
            # Multiple planes with phasing
            S = self.num_satellites // self.P # Number of satellites per plane
            for plane in range(self.P):
                for satellite_in_plane in range(S):
                    nodeo = nodeo0 + plane * (2 * np.pi / self.P)  # RAAN for the plane
                    # mo_plane = mo + (satellite_in_plane * self.F * (2 * np.pi / self.num_satellites))  # Phasing
                    mo_sat = mo + (satellite_in_plane * (2 * np.pi / S))  # Phasing within the plane
                    sat = Satrec()
                    sat.sgp4init(
                        WGS84,                # 1. gravity model
                        'i',                  # 2. opsmode
                        satellite_in_plane,       # 3. satnum called in loop
                        epoch_days,           # 4. single epoch (float)
                        bstar,                # 5. B* drag
                        ndot,                 # 6. 1st derivative of mean motion
                        nddot,                # 7. 2nd derivative
                        ecco,                 # 8. eccentricity
                        argpo,                # 9. argument of perigee
                        inclo,                # 10. inclination
                        mo_sat,               # 11. mean anomaly
                        no_kozai,             # 12. mean motion (radians/minute)
                        nodeo                 # 13. RAAN
                    )
                    # sat = self._initialize_satrec(epoch_days, bstar, ndot, nddot, ecco, argpo, inclo, mo_plane, mean_motion, nodeo)
                    # print(sat)
                    satrecs.append(sat)

        else:
            raise ValueError(f"Unknown constellation pattern: {self.pattern}")
        return satrecs
    
    # @staticmethod
    def _initialize_satrec(self, epoch_days, bstar, ndot, nddot, ecco, argpo, inclo, mo, mean_motion, nodeo):
        """
        Helper function to initialize a Satrec object.

        Args:
            epoch_days (float): Epoch in Julian days.
            bstar (float): Drag coefficient.
            ndot (float): First derivative of mean motion.
            nddot (float): Second derivative of mean motion.
            ecco (float): Eccentricity.
            argpo (float): Argument of perigee.
            inclo (float): Inclination.
            mo (float): Mean anomaly.
            mean_motion (float): Mean motion in rev/day.
            nodeo (float): RAAN.

        Returns:
            Satrec: Initialized satellite object.
        """
        sat = Satrec()
        sat.sgp4init(
            WGS84,                # Gravity model
            'i',                  # Opsmode
            0,                    # Satellite number
            epoch_days,           # Epoch
            bstar,                # Drag coefficient
            ndot,                 # First derivative of mean motion
            nddot,                # Second derivative of mean motion
            ecco,                 # Eccentricity
            argpo,                # Argument of perigee
            inclo,                # Inclination
            mo,                   # Mean anomaly
            mean_motion * (2 * np.pi / 86400.0),  # Mean motion in rad/min
            nodeo                 # RAAN
        )
        return sat
    
    @property
    def satrecs(self):
        """Access the satrec objects."""
        return self._satrecs if hasattr(self, '_satrecs') else []
    
def create_label_from_satrec(satellite_index, satrec, pattern, num_satellites, P=None):
    """
    Create satellite label from satrec orbital elements.

    Args:
        satellite_index: Index of the satellite (0-based)
        satrec: SGP4 Satrec object
        pattern: Constellation pattern
        num_satellites: Total number of satellites
        P: Number of planes (for Walker-Delta)

    Returns:
        str: Formatted label string
    """
    # Extract orbital elements from satrec (all in radians, convert to degrees)
    raan_deg = np.rad2deg(satrec.nodeo) % 360
    mo_deg = np.rad2deg(satrec.mo) % 360

    # Format to remove unnecessary decimals
    raan_formatted = f"{raan_deg:.2f}".rstrip('0').rstrip('.')
    mo_formatted = f"{mo_deg:.2f}".rstrip('0').rstrip('.')

    if pattern == 'Walker-Delta':
        if P is not None:
            S = num_satellites // P
            plane_index = satellite_index // S
            satellite_in_plane = satellite_index % S
            label = f"Sat-{satellite_index + 1}, P-{plane_index+1}-S{satellite_in_plane+1} (Ω={raan_formatted}°, M={mo_formatted}°)"
        else:
            label = f"Sat-{satellite_index + 1} (Ω={raan_formatted}°, M={mo_formatted}°)"

    elif pattern == 'Walker-Delta-SinglePlane':
        label = f"Sat-{satellite_index + 1} (M={mo_formatted}°)"

    else:
        label = f"Sat {satellite_index + 1} (Ω={raan_formatted}°)"

    return label