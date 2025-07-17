from utilities import CoordinateConversion
import numpy as np
from datetime import datetime
from sgp4.api import Satrec, WGS84, jday
import constants

class Constellation:
    """
    A class to represent a simple satellite constellation.

    Supports Walker-Delta and Flower constellation patterns.

    Attributes:
        num_satellites (int): Number of satellites in the constellation.
        orbital_parameters (dict): Orbital parameters for the satellites.
        epoch (datetime): Epoch for the orbital elements.
        pattern (str): Constellation pattern ("Walker-Delta-SinglePlane", "Walker-Delta", "Flower").

        Additional patterns parameters:
        P (int): Number of orbital planes (for Walker-Delta).
        F (int): Phasing between satellites in adjacent planes (for Walker-Delta).
        Np (int): Number of petals (Flower pattern).
        Nd (int): Number of sidereal days for ground track repeat (Flower pattern).
        Fn (int): Satellite phasing integer (Flower pattern, relatively prime to Fd).
        Fd (int): Day phasing integer (Flower pattern, relatively prime to Fn).
        Fh (int): Step phasing integer (Flower pattern, user chosen, 0 <= Fh < Nd).
    """

    def __init__(
        self,
        num_satellites,
        orbital_parameters,
        epoch,
        pattern,
        P=1,
        F=1,
        Np=1,
        Nd=1,
        Fn=1,
        Fd=1,
        Fh=1
    ):
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
            Np (int): (default is 1, for pattern "Flower") Number of petals (lobes in ground track).
            Nd (int): (default is 1, for pattern "Flower") Number of sidereal days for ground track repeat.
            Fn (int): (default is 1, for pattern "Flower") Phase number, phasing integer, must be relatively prime to Fd.
            Fd (int): (default is 1, for pattern "Flower") Phase denominator, phasing integer, must be relatively prime to Fn.
            Fh (int): (default is 1, for pattern "Flower") Step phasing integer, must be in the range [0, Nd-1], can be non constant.
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
        
        """
        Flower Constellation pattern (Mortari et al.):
        - Ns: Number of satellites. Ns <= Fd*Nd. A total of Ns distributed into Fd orbits, Nd sat for each orbit.
        - Np: Number of petals (lobes in ground track). Note that Np indicates the number of revolutions required to
            complete one period of repetition, and corresponds to the number of petals that appear in ECF.
        - Nd: Number of sidereal days for ground track repeat. This is also equal to the numbr of satellites that
            can be placed in a single orbit. For example, if one wishes to place four satellites into a single orbit, then
            Nd => 4.
        - Fn, Fd, Fh: Integer phasing parameters
            * Fn: Phase num(ber?), phasing integer (relatively prime to Fd)
            * Fd: Phase denominator, phasing integer (relatively prime to Fn). Fd represents the total number of distinct orbits
            used to build the constellation.
            * Fh: Hour phasing integer (user chosen, 0 <= Fh < Nd). This does not have necesarily be constant, since compatible
            orbits have Nd admissible orbital locations displaced one from another by the mean anomaly deltaM=2pi/Nd. Fh can be
            set as a rule as Fh=constant, or Fh=(Fh0 + k)*mod(Nd), or in general Fh = f(k)*mod(Nd), where f(k) identifies an
            assigned numerical sequence (e.g., Fibonacci's sequence) we obtain "harmonic" satekkute distributions and dynamics.
            However, a specifically ordered sequence is not required by theory and even a random distribution such as Fh=rand(Nd)
            identifies a function providing nonuniform random integer numbers in the range [0, Nd-1] is admissible. However with
            a f(k) random function, the harmonic dynamics and distribution is, in general, lost.
        
        - All satellites share argument of perigee, period, perigee height, eccentricity, inclination.
        - Regarding inclination:
            Some interesting inclinations are: 0°, 63.4°, 135°, 165°. In some cases it is desirable to select the orbit inclination
            to be one of the critical inclinations in an effort to freeze the constellation, which will prevent the line of apsides from
            shifting over time. When inclination is set to 0°, patterns are formed (e.g. a five-pointed star). With a 63.4° inclination the
            pattern is generally sparse, when it is set to 135° it resembles a tear-drop with 165° it is roughly circular. Generally,
            an inclination of 63.4° is used.
        - Regarding the argument of perigee:
            Usually it is set at 270°. Setting it to 90° the constellations would be mirror images wrt Earth's equator. Adjusting the
            argument of peirfee is one way od reorienting a flower constellation, except that the axis of symmetry is still coincident 
            with the Earth's spin axis. This means that setting omega=45°, 135°, 225°, or 315° results in a constellation that looks
            more like a fan blade than a flower. Notice that there are some critical points: argument of perigee equal to 90° or 270°
            create flower constellations with the highest apogee altitudes, while 0° and 180° create flower contellations that are symmetric
            wrt the plane perpendicular to the axis of symmetry (the eqautor) resulting in lowest pogee altitudes. Values of 45°, 135°,
            225°, or 315° results in a blend between the two extremes.
        - The height of perigee, related to eccentricity, is usually set to 600 km to reduce drag effects.
        - To obtain a repeating ground track that repeats after the satellites completes Np revolutions over Nd, the relationship is
            Tr = Np * T_omega = Nd * T_omegaG, with Tr period of repetition, T_omega is the Greenwhich nodal period, and T(?) is
            the anomalistic period (from perigee to perigee).
        - Node lines (RAANs) are displaced along the equatorial plane.
        
        - Generally they are identified with numbers Np-Nd-Ns, such as 8-1-9, or 4-1-4. If more numbers are present, they identify the
            phasing integers. For instance 8-1-9-1-9 (Np-Nd-Ns-Fn-Fd) has 8 petals, 1 sat per orbit, 9 sats, 1 Fn, 9 orbits.
        - The ratio Tau=Np/Nd must be reasonable. For large Np, given Nd, the anomalistic period goes to zero (physically unrealizable).
            The ratio must be above a certain lower bound, which function of some minimum allowable perigee altitude. For large Nd, given Np,
            the anomalistic period goes to infinity. Varying Tau has a direct impact on the anomalistic orbit period. For instance a
            3-1 flower constellation (Np-Nd) has a 8 hr orbit period, corresponding to Tau=0.333. For a Tau=1 the anomalistic period will
            be 24 hr (not including perturbations). For example, if one desires an 8 hr orbit, then Tau=0.3342 based on a sidereal period 
            of 23 hr 56 min 4.09 s, the best combinations of Nd/Np are 257/769, 129/386, 258/772, 127/380. If we limit the numbers to a
            certain range, some guesses (less accurate) are 1/3, 2/6, 3/9. A common constellation used is the 769-257 (769 petals, 257 days).
            Note that if you perturb Tau slightly, you can identify alternative constellations which might offer additional phasing possibilities.
            Usually this is done to allow for larger Nd, since this limits the number of satellites per orbit.
        - The number of available slots for satellites is limited by Ns <= Fd*Nd. This implies that as the nuber of days to repeat the
            flower constellation increases, there are more slots open for satellites to fill. Therefore, the potential number and kind
            of flower constellation types correspondingly increases. A total of Ns distributed into Fd orbits, Nd sat for each orbit.
        - Fn and Fd (phase number and phase denominator) are related to the distribution of the right ascension of the ascending node (RAAN).
            When Fn = 1, then Fd provides the number of orbit slots available to place satellites into. In other words, if one desires to place
            four satellites into four separate orbits, then Fd=4. This holds for any value of Nd. However, if one uses a value of Fd that is
            less than Ns, then only Fd*Nd satellites can be accomodated. If Fd is increased beyond Ns, the spacing in ECI orbits shrinks.
            Thus, when Fd>Ns empty orbit slots are left. For a flower constellation 3-1-4 (4 sats, 3 petals, 1 sat per orbit), if one selects
            Fn=1 and Fd=7, then one will get four satellites in four different orbits that are arrayed as if there were seven orbits.
        - Np and Nd must be relatively prime, Fn and Fd must be relatively prime.

        - Mathematical rules:
            RAAN_k = 2π(Fn/Fd) * k
            M_{k+1} = M_k - 2π(Fn*Np + Fd*Fh)/(Fd*Nd)
            Tr = Np * T_omega = Nd * T_omegaG
            Tau=Np/Nd
            Ns <= Fd*Nd

        - Examples:
            Basic flowers: constellations of sats whose patterns are the relative orbit itself. E.g.: 3-1-4, 3-2-4
            Planar patterns: when orbit inclination is set to 0° or 180°. These tend to spin the most rapidly. In some cases the patterns are
                of a fixed shape abd merely spin about the axis of symmetry. In some cases they start out in one shape but then reform themselves
                into a never-ending cyclical process. E.g.: "Tre Lacci", 31-11-30-7-10, i=180°, ω=270° (a=21133.149 km, e=0.2723216, hp=9000 km); 
                as time goes this pattern from three groups of satellites curl in on themselves forming a ball, then un-curls themselves into the
                original formation.
            Rings: the rings are not necessarily perfect circles. Rings can be generated with Ns = Np+Nd and Fn=1, Fd=Ns. E.g.: 8-1-90-1-90,
                i=165°, ω=270°, hp=3000 km; these rings spin on their own axis of symmetry while rigidly rotating about the flower constellation axis
                of symmetry.
            Straight lines of satellites: not physically realizable due to the intersecting relative orbit path which ultimately leads to high
                collision probabilities. It is still possible to generate a straight line of sats which rigidly rotates about the axis of symmetry.
                They regularly meet at the intersections of the relative orbits as they rotate about Earth. These lines are formed by collapsing the
                ring formation upon itself. By making the inclination of a ring formation almost 180° and setting the hp such that circular orbits
                result, one can get a straight line formation. E.g.: 4-1-5-1-5, i=174.8°, ω=270°, hp=10354 km (a=16732.862 km, e=0.0000434).
            Figure 8's: for instance a basic figure 8 is 12-1-13-1-13, i=116.6°, ω=270°, hp=1666 km (a=8044.321 km, e=0.0000). Another one can
                contain 3 figure 8 in basic configuration and is 37-18-57-6-19, i=63.4°, ω=270°, hp=19702 km (a=26080.990 km, e=0.0000327).
            Helixes: one can generate spirals that resemble a "Slinky" toy as it collapses and expands about the Earth, or one can generate multiple
                helixes that are either intertwined or separated. E.g. 15-7-49-23-49, i=63.4°, ω=270°, hp=9000 km (a=25367.718 km, e=0.3937911). In this
                examples the two helixes are intertwined about their axis of symmetry and descend from apogee. They collapse in themselves into a flat
                ring at perigee, then they spiral back up to apogee in a cyclical process.
        """

        # Validate inputs for Flower constellation
        if self.pattern == "Flower":
            required_keys = ['Np', 'Nd', 'Fn', 'Fd', 'Fh']
            for key in required_keys:
                if key not in self.orbital_parameters or self.orbital_parameters[key] is None:
                    raise ValueError(f"Flower pattern requires '{key}' in orbital_parameters.")

            Np = self.orbital_parameters['Np']
            Nd = self.orbital_parameters['Nd']
            Fn = self.orbital_parameters['Fn']
            Fd = self.orbital_parameters['Fd']
            Fh = self.orbital_parameters['Fh']

            # Check relatively prime
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return a

            if gcd(Np, Nd) != 1:
                raise ValueError("Np and Nd must be relatively prime for Flower constellation.")
            if gcd(Fn, Fd) != 1:
                raise ValueError("Fn and Fd must be relatively prime for Flower constellation.")
            if not (0 <= Fh < Nd):
                raise ValueError("Fh (phasing step) must be in the range [0, Nd-1].")

            self.Np = Np
            self.Nd = Nd
            self.Fn = Fn
            self.Fd = Fd
            self.Fh = Fh

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

        elif self.pattern == "Flower":
            # User must provide: Ns, Np, Nd, Fn, Fd, Fh
            Ns = self.num_satellites
            Np = self.orbital_parameters['Np']
            Nd = self.orbital_parameters['Nd']
            Fn = self.orbital_parameters['Fn']
            Fd = self.orbital_parameters['Fd']
            Fh = self.orbital_parameters['Fh']
            mo_0 = mo  # initial mean anomaly

            # Assume Np and Nd are relatively prime, Fn and Fd are relatively prime
            mean_anomaly = mo_0
            for k in range(Ns):
                raan = 2 * np.pi * (Fn / Fd) * k
                # For k=0, mean_anomaly is mo_0
                if k == 0:
                    ma = mean_anomaly
                else:
                    ma = mean_anomaly - 2 * np.pi * (Fn * Np + Fd * Fh) / (Fd * Nd)
                    mean_anomaly = ma  # update for next satellite

                sat = Satrec()
                sat.sgp4init(
                    WGS84,
                    'i',
                    k,
                    epoch_days,
                    bstar,
                    ndot,
                    nddot,
                    ecco,
                    argpo,
                    inclo,
                    ma,
                    no_kozai,
                    raan
                )
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

    def estimate_anomalistic_period(self):
        """
        Calculate the anomalistic period (time between successive perigee passages)
        including J2 perturbation effects for this constellation's orbit
        
        Returns:
            Period in seconds
        """
        # Extract orbital parameters
        semi_major_axis = self.orbital_parameters['semi_major_axis']  # km
        eccentricity = self.orbital_parameters['eccentricity']        # unitless
        inclination = self.orbital_parameters['inclination']          # deg
        
        # Basic Keplerian period
        mu = constants.GRAVITATIONAL_CONSTANT  # km³/s²
        T_kepler = 2 * np.pi * np.sqrt(semi_major_axis**3 / mu)
        
        # For highly accurate calculations, include J2 perturbation effects
        J2 = 0.00108263  # Earth's J2 coefficient
        R_earth = constants.EARTH_RADIUS  # Earth's equatorial radius in km
        inc_rad = np.radians(inclination)
        
        # Calculate semi-latus rectum
        p = semi_major_axis * (1 - eccentricity**2)
        
        # Rate of precession of argument of periapsis due to J2
        n = np.sqrt(mu / semi_major_axis**3)  # Mean motion
        d_omega_dt = (3/2) * n * J2 * (R_earth/p)**2 * (5 * np.cos(inc_rad)**2 - 1)
        
        # Anomalistic period accounting for precession
        T_anomalistic = 2 * np.pi / (n + d_omega_dt)
        
        return T_anomalistic        
    
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