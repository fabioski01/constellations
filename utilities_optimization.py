"""
# Satellite Constellation Optimizer

This module provides tools for optimizing satellite constellation configurations based on multiple
weighted objectives. The optimizer can balance trade-offs between:

## Coverage Metrics
- Local coverage (ground station visibility)
- Regional coverage (specific geographical area)
- Global coverage (Earth-wide coverage estimation)

## Cost Factors
- Satellite costs, payload (depending on instrument type, mass, power, design life) and spacecraft (depending on mass) (including economies of scale)
-  (DEACTIVATED) Operational costs (station-keeping, deorbiting, communication)
- Launch costs (based on altitude, inclination, launch vehicle capacity, orbital planes)

## Science Requirements
- Target altitude optimization
- Inclination preferences for specific mission types

## Features
- Modular objective function with configurable weights
- Support for various constellation patterns (Walker-Delta, etc.)
- Grid-based coverage estimation for regions and global areas
- Realistic cost models with economies of scale
- Science requirement scoring based on mission profile
- Advanced visualization of optimization results

## Usage Examples
```python
# Example: Optimize for regional coverage
optimizer = ConstellationOptimizer(
    pattern_type="Walker-Delta",
    regional_weight=0.7,     # Emphasize regional coverage
    global_weight=0.2,       # Some global coverage
    local_weight=0.1,        # Minimal ground station requirements
    satellite_cost_weight=0.5,
    launch_cost_weight=0.3,
    operational_cost_weight=0.2,
    science_weight=0.0       # No specific science requirements
)

# Run the optimization
orbital_params, pattern_params, num_satellites, score = optimizer.optimize(
    method='differential_evolution',
    max_iter=50,
    popsize=15
)

# Analyze the results 
optimizer.analyze_component_scores(plot=True)
```

## References
- Walker, J.G. "Satellite Constellations" Journal of the British Interplanetary Society, 1984
- Wertz, J.R. "Orbit and Constellation Design and Management", 2001
- Larson, W. and Wertz, J. "Space Mission Analysis and Design", 1999

## Author
Fabio Meloni - TU Delft MSc Student - NASA Jet Propulsion Laboratory Visiting Researcher
"""

import numpy as np
from scipy import optimize
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Import from existing files
from utilities_constellations import Constellation
from utilities_coverage import Coverage, get_ground_station_coordinates
import constants

class ConstellationOptimizer:
    """
    Optimizer for satellite constellation parameters based on multiple weighted objectives.
    
    This class provides a comprehensive framework for optimizing satellite constellation designs
    by balancing multiple competing objectives:
    
    1. Coverage metrics: local (ground station), regional (specific area), and global coverage
    2. Cost factors: spacecraft, payload, and launch costs
    3. Science requirements: orbital parameters optimized for specific mission requirements
    
    The optimizer uses evolutionary algorithms (differential evolution by default) to find
    constellation configurations that maximize the weighted combination of these objectives.
    Parameters that can be optimized include:
    
    - Number of satellites
    - Orbital parameters (altitude, inclination, eccentricity, etc.)
    - Constellation pattern parameters (e.g., number of planes and phasing for Walker-Delta)
    
    The modular design allows for easy extension with new objective components or
    modification of existing scoring functions to meet specific mission needs.
    """
    
    def __init__(self, pattern_type="Walker-Delta", 
                 start_time=None, end_time=None, step=1500,
                 gs=None, elevation_threshold=10.0,
                 # Coverage weights
                 local_weight=0.0, regional_weight=0.0, global_weight=0.9,
                 # Cost weights
                 satellite_cost_weight=0.2, operational_cost_weight=0.1, launch_cost_weight=0.1,
                 # Science requirements
                 science_weight=0.9, target_altitude=500.0, target_altitude_tolerance=50.0,
                target_inclination=50.0, target_inclination_tolerance=10.0,
                 # Regional coverage area
                 regional_area=None,
                 # New: launch and payload inputs
                 sat_mass_kg=130.0,
                 payloads=[{'type': 'fields', 'cat': 'EO', 'mass_kg': 12.0, 'power_w': 35.0},
                           {'type': 'particle', 'cat': 'EO', 'mass_kg': 9.0, 'power_w': 22.0},],
                 dv_deployment_m_s: float = 100.0,
                 T_deployment_days: float = 180.0):
        """
        Initialize the optimizer with baseline parameters.
        
        Parameters:
        -----------
        pattern_type : str
            Type of constellation pattern ("Walker-Delta", "Walker-Delta-SinglePlane", etc.)
        start_time : datetime
            Start time for simulation (default: current time)
        end_time : datetime
            End time for simulation (default: start_time + 1 day)
        step : int
            Time step in seconds for propagation
        gs : str
            Ground station location for local visibility
        elevation_threshold : float
            Minimum elevation angle (degrees) for visibility
            
        Coverage Weights:
        ----------------
        local_weight : float (0-1)
            Weight for local visibility from ground station
        regional_weight : float (0-1)
            Weight for regional coverage (e.g., Western US)
        global_weight : float (0-1)
            Weight for global coverage estimate
            
        Cost Weights:
        ------------
        satellite_cost_weight : float (0-1)
            Weight for satellite cost (more satellites = higher cost)
        operational_cost_weight : float (0-1)
            Weight for operational costs (lower orbit = quicker deorbiting = lower cost)
        launch_cost_weight : float (0-1)
            Weight for launch costs (higher orbit = higher cost)
            knobs for launch cost model of deployment:
             - dv_deployment_m_s : float (higher means lower launches for depl needed)
             - t deployment: float (higher means lower launches for depl needed)

            
        Science Requirements:
        -------------------
        science_weight : float (0-1)
            Weight for science requirements
        target_altitude : float
            Target altitude in km for science instruments
        target_altitude_tolerance : float
            Tolerance range for target altitude in km
        target_inclination : float
            Target inclination in degrees for science requirements (default: 50.0)
        target_inclination_tolerance : float
            Tolerance range for target inclination in degrees (default: 10.0)
            
        Regional Coverage Area:
        ---------------------
        regional_area : list of tuples
            List of (lat, lon) coordinates defining the region of interest

        Spacecraft Mass:
        ---------------------
        sat_mass_kg : float
            Mass of the satellite in kg (default: 130.0 kg)

        Payloads
        ---------------------
        payloads : list of dicts
            List of payloads with their type, environmental category, mass, and power requirements.
        """
        self.pattern_type = pattern_type
        self.step = step
        self.gs = gs
        self.elevation_threshold = elevation_threshold
        
        # Coverage weights
        self.local_weight = local_weight
        self.regional_weight = regional_weight
        self.global_weight = global_weight
        
        # Cost weights
        self.satellite_cost_weight = satellite_cost_weight
        self.operational_cost_weight = operational_cost_weight
        self.launch_cost_weight = launch_cost_weight
        
        # Science requirements
        self.science_weight = science_weight
        self.target_altitude = target_altitude
        self.target_altitude_tolerance = target_altitude_tolerance
        self.target_inclination = target_inclination
        self.target_inclination_tolerance = target_inclination_tolerance

        # Store spacecraft mass and payload definition for cost models
        self.sat_mass_kg = float(sat_mass_kg)
        self.payloads = payloads
        # launch deployment knobs
        self.dv_deployment_m_s = float(dv_deployment_m_s)
        self.T_deployment_days = float(T_deployment_days)

        from utilities_coverage import get_region_coordinates, get_available_regions

        if regional_area is None:
            # Default region (Western US)
            self.regional_area = [
                (32.0, -125.0),  # Southwest corner
                (32.0, -105.0),  # Southeast corner
                (49.0, -105.0),  # Northeast corner
                (49.0, -125.0)   # Northwest corner
            ]
        elif isinstance(regional_area, str):
            # If regional_area is a string, look up the coordinates
            try:
                self.regional_area = get_region_coordinates(regional_area)
                print(f"Using predefined region: {regional_area}")
            except Exception as e:
                available_regions = get_available_regions()
                print(f"Error: Region '{regional_area}' not found. Available regions: {available_regions}")
                # Fall back to default
                self.regional_area = [
                    (32.0, -125.0),  # Southwest corner
                    (32.0, -105.0),  # Southeast corner
                    (49.0, -105.0),  # Northeast corner
                    (49.0, -125.0)   # Northwest corner
                ]
                print("Using default Western US region instead.")
        else:
            # Assume it's already a list of coordinates
            self.regional_area = regional_area
        
        # Set default times if not provided
        if start_time is None:
            self.start_time = datetime(2025, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
        else:
            self.start_time = start_time
            
        if end_time is None:
            self.end_time = datetime(2025, 5, 6, 12, 0, 0, tzinfo=timezone.utc)
        else:
            self.end_time = end_time
        
        # Get ground station coordinates
        self.gs_lat, self.gs_lon, self.gs_alt = get_ground_station_coordinates(gs)
        self.gs_geodetic = (self.gs_lat, self.gs_lon, self.gs_alt)
        
        # Store best configuration found
        self.best_params = None
        self.best_score = -np.inf
        self.optimization_history = []
        
        # Initialize operational cost details storage for debugging
        self._operational_cost_details = {}
        
    def _create_orbital_parameters(self, params):
        """
        Create orbital parameters dictionary from optimization parameters.
        
        Parameters:
        -----------
        params : list or array
            Parameters to optimize [num_satellites, semi_major_axis, eccentricity, inclination, 
                                   RAAN, argument_of_periapsis, P, F]
            or [num_satellites, semi_major_axis, eccentricity, inclination, 
                RAAN, argument_of_periapsis, Np, Nd, Fn, Fd, Fh] for Flower                           
        
        Returns:
        --------
        num_satellites : int
            Number of satellites in constellation
        orbital_parameters : dict
            Dictionary of orbital parameters
        pattern_params : dict
            Dictionary of pattern parameters (P, F)
        """
        # Extract parameters (continuous and discrete)
        num_satellites = int(round(params[0]))  # Number of satellites (integer)
        perigee_height = params[1] # using perigee height instead of semi-major axis in km
        eccentricity = params[2]
        inclination = params[3]
        RAAN = params[4]
        arg_periapsis = params[5]

        # Calculate semi-major axis from perigee height and eccentricity
        semi_major_axis = (perigee_height + constants.EARTH_RADIUS) / (1 - eccentricity)
        
        # Create orbital parameters dictionary
        orbital_parameters = {
            'semi_major_axis': semi_major_axis,    # km
            'eccentricity': eccentricity,          # unitless
            'inclination': inclination,            # deg
            'RAAN': RAAN,                          # deg
            'argument_of_periapsis': arg_periapsis,# deg     
            'true_anomaly': 0.0,                   # deg
            'ndot': 1e-8,                          # rev/day²
            'nddot': 1e-12,                        # rev/day³
            'bstar': 2e-5                          # 1/er
        }
        
        # Extract pattern-specific parameters
        pattern_params = {}
        if self.pattern_type == "Walker-Delta":
            # Be defensive: params may be shorter depending on caller; provide sensible defaults
            try:
                p_val = int(round(params[6]))
            except Exception:
                p_val = 1
            try:
                f_val = int(round(params[7]))
            except Exception:
                f_val = 1

            pattern_params['P'] = max(1, p_val)  # Number of planes must be at least 1
            # Phasing parameter should be at least 1 and less than P (will be clamped later if needed)
            pattern_params['F'] = max(1, f_val)
        
            # Walker-Delta requires: 
            # 1. T (total satellites) divisible by P (number of planes)
            # 2. T (total satellites) divisible by F (phasing parameter)
            # these constraints are checked in the objective function
        
        elif self.pattern_type == "Flower":
            # Defensive parsing: optimization callers may not always provide all Flower parameters
            # Defaults chosen to form a minimal valid Flower configuration
            defaults = {'Np': 1, 'Nd': 1, 'Fn': 1, 'Fd': 1, 'Fh': 0}
            keys = ['Np', 'Nd', 'Fn', 'Fd', 'Fh']
            for i, k in enumerate(keys):
                idx = 6 + i
                try:
                    pattern_params[k] = int(round(params[idx]))
                except Exception:
                    pattern_params[k] = defaults[k]

            # Ensure parameters are valid ranges
            pattern_params['Np'] = max(1, pattern_params['Np'])
            pattern_params['Nd'] = max(1, pattern_params['Nd'])
            pattern_params['Fn'] = max(1, pattern_params['Fn'])
            pattern_params['Fd'] = max(1, pattern_params['Fd'])
            # Fh must be in [0, Nd-1]
            pattern_params['Fh'] = max(0, pattern_params['Fh'])
            if pattern_params['Fh'] >= pattern_params['Nd']:
                pattern_params['Fh'] = pattern_params['Nd'] - 1
            # Also include these in the orbital_parameters dict so callers that
            # expect Flower keys (e.g. Constellation) find them there.
            orbital_parameters['Np'] = pattern_params['Np']
            orbital_parameters['Nd'] = pattern_params['Nd']
            orbital_parameters['Fn'] = pattern_params['Fn']
            orbital_parameters['Fd'] = pattern_params['Fd']
            orbital_parameters['Fh'] = pattern_params['Fh']
            
        return num_satellites, orbital_parameters, pattern_params
    
    def objective_function(self, params):
        """
        Modular objective function to maximize combined score of:
        - Local visibility (time satellite is visible from ground station)
        - Regional coverage (coverage of a specific region)
        - Global coverage (estimated coverage of the entire Earth)
        - Satellite cost (fewer satellites is better)
        - Operational cost (lower altitude for easier deorbiting)
        - Launch cost (lower altitude and inclination is better)
        - Science requirements (proximity to target altitude)
        
        Returns negative value because optimizers typically minimize functions.
        
        Parameters:
        -----------
        params : list or array
            Parameters to optimize [num_satellites, semi_major_axis, eccentricity, inclination, 
                                   RAAN, argument_of_periapsis, P, F]
        
        Returns:
        --------
        score : float
            Negative of the weighted score (higher is better)
        """
        try:
            # Create parameter dictionaries
            num_satellites, orbital_parameters, pattern_params = self._create_orbital_parameters(params)
            
            # Check parameter validity
            if num_satellites < 1 or num_satellites > 200:  # Set reasonable satellite count bounds
                print(f"Warning: Invalid number of satellites {num_satellites}, must be between 1 and 200.")
                return 1e10  # Invalid number of satellites
                
            if orbital_parameters['eccentricity'] < 0 or orbital_parameters['eccentricity'] >= 1:
                print(f"Warning: Invalid eccentricity {orbital_parameters['eccentricity']}, must be in [0, 1).")
                return 1e10  # Invalid eccentricity
                
            if orbital_parameters['semi_major_axis'] < constants.EARTH_RADIUS + 200:
                print(f"Warning: Semi-major axis {orbital_parameters['semi_major_axis']} too low, may not be a valid orbit.")
                return 1e10  # Too low orbit
            
            # Create the constellation
            epoch = datetime(2025, 5, 5, 0, 0, 0)
            
            # For Walker-Delta, handle P and F
            if self.pattern_type == "Walker-Delta":
                P = pattern_params['P']
                F = pattern_params['F']
                
                # Validate P and F
                if P < 1 or F < 1 or F >= P:
                    # print(f"Warning: Invalid Walker-Delta parameters P={P}, F={F}.")
                    return 1e10
                
                # Check if number of satellites is compatible with P
                if num_satellites < P or num_satellites % P != 0:
                    # print(f"Warning: Number of satellites {num_satellites} must be divisible by number of planes (P={P}).")
                    return 1e10
                    
                # Check if number of satellites (T) is divisible by F
                if num_satellites % F != 0:
                    # print(f"Warning: Number of satellites {num_satellites} must be divisible by phasing (F={F}).")
                    return 1e10  # Total number of satellites (T) must be divisible by phasing (F)
                    
                # Create constellation
                const = Constellation(
                    num_satellites, orbital_parameters, epoch, 
                    self.pattern_type, P=P, F=F
                )
            
            elif self.pattern_type == "Flower":
                # Get Flower constellation parameters
                Np = pattern_params['Np']
                Nd = pattern_params['Nd']
                Fn = pattern_params['Fn']
                Fd = pattern_params['Fd']
                Fh = pattern_params['Fh']
                
                # Validate Flower constellation parameters
                if Np < 1:
                    print(f"Warning: Invalid number of petals Np={Np}, must be at least 1.")
                    return 1e10
                    
                if Nd < 1:
                    print(f"Warning: Invalid number of days Nd={Nd}, must be at least 1.")
                    return 1e10
                
                # Check if Fn and Fd are relatively prime (gcd = 1)
                def gcd(a, b):
                    while b:
                        a, b = b, a % b
                    return a
                    
                if gcd(Fn, Fd) != 1:
                    # print(f"Warning: Fn={Fn} and Fd={Fd} must be relatively prime (no common factors).")
                    return 1e10
                    
                if Fh < 0 or Fh >= Nd:
                    print(f"Warning: Step phasing Fh={Fh} must be in range [0, {Nd-1}].")
                    return 1e10
                
                # Add Flower parameters to orbital_parameters dictionary as well as passing as keywords args to ensure compatibility with the Constellation class
                orbital_parameters = orbital_parameters.copy()
                orbital_parameters['Np'] = Np
                orbital_parameters['Nd'] = Nd
                orbital_parameters['Fn'] = Fn
                orbital_parameters['Fd'] = Fd
                orbital_parameters['Fh'] = Fh

                # For Flower constellations, create with appropriate parameters
                const = Constellation(
                    num_satellites, orbital_parameters, epoch,
                    self.pattern_type, Np=Np, Nd=Nd, Fn=Fn, Fd=Fd, Fh=Fh
                )            
                        
            else:
                const = Constellation(
                    num_satellites, orbital_parameters, epoch, 
                    self.pattern_type
                )
            
            try:
                # Propagate constellation
                result = const.generate_positions_with_time(
                    self.start_time, self.end_time, self.step
                )
                
                positions = result["positions"]  # Shape: (num_satellites, num_time_steps, 3)
                time_stamps = result["time_stamps"]
            except Exception as e:
                print(f"Error in propagation: {str(e)}")
                print(f"Orbital parameters: {orbital_parameters}")
                print(f"Pattern parameters: {pattern_params}")
                return 1e10  # Return a high value for invalid orbits

            # Calculate orbital period
            a = orbital_parameters['semi_major_axis']  # km from Earth's center
            orbital_period = 2 * np.pi * np.sqrt(a**3 / constants.GRAVITATIONAL_CONSTANT)
            
            #-------------------------------------------------------
            # Calculate all component scores using modular functions
            #-------------------------------------------------------
            
            # 1. Coverage scores
            # Local coverage (ground station visibility)
            local_score, overlap_penalty = self._calculate_local_coverage(
                positions, time_stamps, num_satellites, orbital_period
            )
            
            # Regional coverage (specified area)
            regional_score = self._calculate_regional_coverage(
                const, orbital_parameters, num_satellites, epoch, orbital_period, positions=positions, time_stamps=time_stamps
            )
            
            # Global coverage
            global_score = self._calculate_global_coverage(
                num_satellites, orbital_parameters, positions=positions, time_stamps=time_stamps
            )
            
            # 2. Cost scores
            # Satellite cost (more satellites = higher cost) in M USD
            satellite_cost_score = self._calculate_satellite_cost(num_satellites)
            
            # Operational cost is currently deactivated; use neutral score
            operational_cost_score = 1.0
            
            # Launch cost (higher altitude/inclination = higher cost) in M USD
            # launch_cost_score = self._calculate_launch_cost(orbital_parameters, num_satellites, self.sat_mass_kg, pattern_params) # OLD
            launch_cost_result = self._calculate_launch_cost(orbital_parameters, num_satellites, self.sat_mass_kg, pattern_params, dv_deployment_m_s=self.dv_deployment_m_s, T_deployment_days=self.T_deployment_days)
            launch_cost_musd = float(launch_cost_result['total_cost_musd']) if isinstance(launch_cost_result, dict) else float(launch_cost_result)
            launch_cost_score = 1.0  # neutral, excluded in Pareto; set self.launch_cost_weight=0 to fully ignore

            # # Then normalize it to a score (0-1, lower cost = higher score)
            # # Assume max reasonable cost is ~500 M$ for normalization
            # max_reasonable_cost = 500.0
            # launch_cost_score = 1.0 - min(launch_cost_musd / max_reasonable_cost, 1.0)
            
            # 3. Science requirement score
            science_score = self._calculate_science_requirement_score(orbital_parameters)
            
            #-------------------------------------------------------
            # Combine scores with weights
            #-------------------------------------------------------
            
            # Coverage scores (weighted by user preferences)
            coverage_score = (
                self.local_weight * local_score * overlap_penalty +
                self.regional_weight * regional_score +
                self.global_weight * global_score
            ) / (self.local_weight + self.regional_weight + self.global_weight)
            
            # Cost scores (weighted by user preferences)
            cost_weight_sum = (
                self.satellite_cost_weight + 
                self.operational_cost_weight + 
                self.launch_cost_weight
            )
            
            if cost_weight_sum > 0:
                cost_score = (
                    self.satellite_cost_weight * satellite_cost_score +
                    self.operational_cost_weight * operational_cost_score +
                    self.launch_cost_weight * launch_cost_score
                ) / cost_weight_sum
            else:
                cost_score = 1.0  # No cost consideration
            
            # Final combined score - properly weighted average to ensure score between 0 and 1
            coverage_weight = self.local_weight + self.regional_weight + self.global_weight
            cost_weight = self.satellite_cost_weight + self.operational_cost_weight + self.launch_cost_weight
            
            # Calculate total weight to normalize the final score
            total_weight = coverage_weight + cost_weight + (self.science_weight if self.science_weight > 0 else 0)
            
            # Weighted average of all component scores
            combined_score = (
                coverage_weight * coverage_score + 
                cost_weight * cost_score + 
                (self.science_weight * science_score if self.science_weight > 0 else 0)
            ) / total_weight
            
            # Store this evaluation with component scores for analysis
            params_copy = params.copy()
            component_scores = {
                'local_score': local_score,
                'overlap_penalty': overlap_penalty,
                'regional_score': regional_score,
                'global_score': global_score,
                'satellite_cost_score': satellite_cost_score,
                'operational_cost_score': operational_cost_score,
                'launch_cost_score': launch_cost_score,
                'launch_cost_musd': launch_cost_musd,
                'launch_details': launch_cost_result,
                'science_score': science_score,
                'combined_score': combined_score
            }
            self.optimization_history.append((params_copy, combined_score, component_scores))
            
            # Update best parameters if this is better
            if combined_score > self.best_score:
                self.best_score = combined_score
                self.best_params = params_copy
            
            # Return negative score because optimizers minimize
            return -combined_score
            
        except Exception as e:
            # print(f"Error in objective function: {str(e)}")
            return 1e10  # Return a high value for errors
    
    def _eci_to_ecef(self, eci_position, timestamp):
        """Convert ECI position to ECEF."""
        from utilities import CoordinateConversion
        # return CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(eci_position, timestamp)
        """
        Convert ECI position to ECEF.
        
        Parameters:
        -----------
        eci_position : array
            Position in ECI frame (meters)
        timestamp : float or datetime
            Time (seconds offset or datetime object)
        """
        # If timestamp is numeric (seconds offset), pass reference_time
        if isinstance(timestamp, (int, float)):
            return CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(
                eci_position, timestamp, reference_time=self.start_time
            )
        else:
            # Already a datetime
            return CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(
                eci_position, timestamp
            )
                
    def optimize(self, method='differential_evolution', max_iter=50, popsize=15, strategy='rand2bin', 
                 mutation=(0.5, 0.7), recombination=0.7, seed=None, early_stopping=True, patience=20, 
                 improvement_threshold=0.001, workers=1):
        """
        Run the optimization process.
        
        Parameters:
        -----------
        method : str
            Optimization method to use ('differential_evolution', 'dual_annealing', etc.)
        max_iter : int
            Maximum number of iterations
        popsize : int
            Population size for evolutionary algorithms
        strategy : str
            Differential evolution strategy. Options:
            - 'best1bin': Uses best solution as base with one difference vector (faster convergence, less exploration)
            - 'rand1bin': Uses random solution as base with one difference vector (more exploration)
            - 'rand2bin': Uses random solution as base with two difference vectors (most exploration)
            - 'randtobest1bin': Combination of rand1bin and best1bin (balanced)
            - 'best1exp'
            - 'rand1exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'randtobest1exp'
            - 'currenttobest1bin'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'best2bin'           
        mutation : float or tuple(float, float)
            Mutation constant. If specified as a tuple (min, max), adaptive mutation is used.
            Larger values (0.8-1.0) give more exploration, smaller values more exploitation. 
            It is also known as differential weight (F).
            If specified as a float it should be in the range [0, 2).
            If specified as a tuple ``(min, max)`` dithering is employed. Dithering
            randomly changes the mutation constant on a generation by generation
            basis. The mutation constant for that generation is taken from
            ``U[min, max)``. Dithering can help speed convergence significantly.
            Increasing the mutation constant increases the search radius, but will
            slow down convergence.
        recombination : float
            Recombination probability, between 0 and 1. Higher values mean more mixing. Also known as crossover proablility. Increasing this value allows a larger number of mutants to progress into the next generation, but at the risk of population stability.
        seed : int, optional
            Random seed for reproducibility. Set to a specific value for repeatable results,
            or leave as None for different results each run.
        early_stopping : bool, optional
            Whether to use early stopping to terminate optimization when no improvement is detected.
        patience : int, optional
            Number of iterations to wait for improvement before early stopping.
        improvement_threshold : float, optional
            Minimum improvement required to reset early stopping counter.
        
        Returns:
        --------
        orbital_parameters : dict
            Optimized orbital parameters
        pattern_params : dict
            Optimized pattern parameters
        num_satellites : int
            Optimized number of satellites
        best_score : float
            Best score achieved
        """
        # Define parameter bounds, specific to the pattern type
        if self.pattern_type == "Walker-Delta":
            bounds = [
            (1, 100),             # num_satellites (integer between 1 and 100)
            # (6500, 8500),         # semi_major_axis (km). Earth radius is 6378 km
            (250, 1000),          # perigee_height (km)
            (0, 0.01),            # eccentricity (close to circular)
            (0, 90),              # inclination (deg)
            (0, 360),             # RAAN (deg)
            (0, 360),             # argument_of_periapsis (deg)
            ]
            bounds.append((1, 20))  # P (planes)
            bounds.append((1, 20))  # F (phase)
        elif self.pattern_type == "Flower":
            bounds = [
            (1, 100),             # num_satellites (integer between 1 and 100)
            # (6500, 20000),         # semi_major_axis (km)
            (250, 3000),          # perigee_height (km). More range for higher eccentricity
            (0, 0.65),            # eccentricity. Flower can have high e
            (0, 180),              # inclination (deg). They can be retrograde
            (0, 360),             # RAAN (deg)
            (0, 360),             # argument_of_periapsis (deg). Usually is 270 deg
            ]
            bounds.append((1, 99))   # Np (petals). Varies largely, but can be equal to num sat
            bounds.append((1, 49))   # Nd (days). Varies, usually equal to 1 or in combination with Fd to get Nd * Fd = num sat
            bounds.append((1, 99))   # Fn (satellite phasing). Varies largely, unpredictable
            bounds.append((1, 100))   # Fd (day phasing). Varies largely in combination with Nd and num sat
            # Allow Fh to vary between 0 and Nd-1 (upper bound will be clamped after sampling)
            bounds.append((0, 48))    # Fh (step phasing). Usually in range [0, Nd-1]
        
        # Select optimization algorithm
        if method == 'differential_evolution':
            # If early stopping is enabled, create a callback function to monitor progress
            if early_stopping:
                # Create local variables for tracking progress
                best_score_so_far = float('-inf')
                no_improvement_count = 0
                
                def callback(xk, convergence=0):
                    nonlocal best_score_so_far, no_improvement_count
                    # Get current best score (convert to positive since our objective returns negative)
                    current_score = -self.objective_function(xk)
                    
                    # Check if we have an improvement
                    if current_score > best_score_so_far + improvement_threshold:
                        best_score_so_far = current_score
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    # Print progress update every 10 iterations
                    if len(self.optimization_history) % 10 == 0:
                        print(f"Iteration: {len(self.optimization_history)}, Score: {current_score:.4f}")
                    
                    # Stop if no improvement for 'patience' iterations
                    if no_improvement_count >= patience:
                        print(f"\nEarly stopping triggered after {no_improvement_count} iterations without improvement.")
                        print(f"Best score found: {best_score_so_far:.4f}")
                        return True
                    return False
            else:
                callback = None
                
            # Configure and run differential evolution
            result = optimize.differential_evolution(
                self.objective_function,
                bounds,
                maxiter=max_iter,
                popsize=popsize,
                tol=0.01,
                polish=True,
                mutation=mutation,       # Control exploration vs exploitation
                recombination=recombination,  # Control mixing
                strategy=strategy,       # Strategy for creating new candidates
                updating='deferred',     # Update the population after a complete iteration
                init='random',          # Random initialization of population
                seed=seed,             # Optional seed for reproducibility
                callback=callback,       # Early stopping callback
                workers=workers  # Number of parallel workers (1 for single-threaded)
            )
        elif method == 'dual_annealing':
            result = optimize.dual_annealing(
                self.objective_function,
                bounds,
                maxiter=max_iter
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Process results
        optimized_params = result.x
        
        # Create parameter dictionaries from optimized parameters
        num_satellites, orbital_parameters, pattern_params = self._create_orbital_parameters(optimized_params)
        
        # Round discrete parameters to integers
        num_satellites = int(round(num_satellites))
        if self.pattern_type == "Walker-Delta":
            pattern_params['P'] = int(round(pattern_params['P']))
            pattern_params['F'] = int(round(pattern_params['F']))
            
            # Ensure F is a valid value for the Walker-Delta pattern
            # F must be between 0 and P-1, and T (number of satellites) must be divisible by F
            if pattern_params['F'] == 0:
                pattern_params['F'] = 1
            
            # Ensure P is at least 1
            pattern_params['P'] = max(1, pattern_params['P'])
            
            # Ensure F is between 1 and P-1
            pattern_params['F'] = min(max(1, pattern_params['F']), pattern_params['P'] - 1)
            
            # Adjust number of satellites to ensure it's divisible by both P and F
            # First make it divisible by P
            if num_satellites % pattern_params['P'] != 0:
                # Round to the nearest multiple of P
                num_satellites = pattern_params['P'] * round(num_satellites / pattern_params['P'])
                num_satellites = max(pattern_params['P'], num_satellites)  # At least P satellites
            
            # Now ensure it's also divisible by F
            # Find the least common multiple of P and F
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return a
            
            def lcm(a, b):
                return a * b // gcd(a, b)
            
            lcm_value = lcm(pattern_params['P'], pattern_params['F'])
            
            # Make the number of satellites a multiple of the LCM
            if num_satellites % lcm_value != 0:
                num_satellites = lcm_value * ((num_satellites + lcm_value // 2) // lcm_value)
                num_satellites = max(lcm_value, num_satellites)  # At least lcm(P,F) satellites
        
        print("\nOptimization Results:")
        print(f"Number of satellites: {num_satellites}")
        print(f"Orbital Parameters: {orbital_parameters}")
        if pattern_params:
            print(f"Pattern Parameters: {pattern_params}")
        # Convert back to positive score for reporting (the original minimization is done on -combined_score)
        best_score = -result.fun
        print(f"Best Score: {best_score}")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")

        # Store algorithm parameters as class attributes for later use
        self.mutation = mutation
        self.recombination = recombination
        self.strategy = strategy
        
        return orbital_parameters, pattern_params, num_satellites, best_score

    def optimize_multiple_patterns(self, patterns=None, **kwargs):
        """
        Run optimization for multiple constellation patterns and compare results.
        
        Parameters:
        -----------
        patterns : list of str, optional
            List of patterns to optimize. If None, uses ["Walker-Delta", "Flower"]
        **kwargs : 
            Arguments to pass to the optimize method
            
        Returns:
        --------
        results : dict
            Dictionary of optimization results for each pattern
        best_pattern : str
            The pattern that achieved the best score
        """
        if patterns is None:
            patterns = ["Walker-Delta", "Flower"]
        
        results = {}
        best_score = -np.inf
        best_pattern = None
        
        # Store pattern histories separately
        pattern_histories = {}

        for pattern in patterns:
            print(f"\n=== OPTIMIZING {pattern} PATTERN ===")
            # Save original pattern
            original_pattern = self.pattern_type
            # Set new pattern
            self.pattern_type = pattern

            # IMPORTANT: Reset the optimization history for each pattern
            self.optimization_history = []           
            
            # Run optimization
            orbital_params, pattern_params, num_satellites, score = self.optimize(**kwargs)
            
            # Add the final best solution to the optimization history if it's not already there, this ensures the best point appears in the plot
            if self.best_params is not None:
                # Check if best solution is already in history
                best_score_in_history = False
                for _, hist_score, _ in self.optimization_history:
                    if abs(hist_score - score) < 1e-6:
                        best_score_in_history = True
                        break
                        
                if not best_score_in_history:
                    print(f"Adding final best solution (score: {score:.4f}) to optimization history")
                    # Evaluate the best solution one more time to get component scores
                    final_score = -self.objective_function(self.best_params)
                    # Note: objective_function will append this to history with component scores
            
            # Store optimization history for this pattern
            pattern_histories[pattern] = self.optimization_history.copy()

            # Store results
            results[pattern] = {
                'orbital_params': orbital_params,
                'pattern_params': pattern_params,
                'num_satellites': num_satellites,
                'score': score,
                'best_params': self.best_params.copy() if self.best_params is not None else None
            }
            
            # Check if this is the best pattern
            if score > best_score:
                best_score = score
                best_pattern = pattern
            
            # Reset to original pattern
            self.pattern_type = original_pattern

        # Store the pattern histories in results for later use
        results['pattern_histories'] = pattern_histories
                
        # Compare results
        print("\n=== PATTERN COMPARISON ===")
        for pattern, result in results.items():
            if pattern != 'pattern_histories':  # Skip the histories entry
                print(f"{pattern}: {result['num_satellites']} satellites, " +
                    f"altitude: {result['orbital_params']['semi_major_axis'] - constants.EARTH_RADIUS:.1f} km, " +
                    f"inclination: {result['orbital_params']['inclination']:.1f}°, " +
                    f"score: {result['score']:.4f}")
        
        print(f"\nBest pattern: {best_pattern} with score {results[best_pattern]['score']:.4f}")
        
        return results, best_pattern
   
    def _calculate_local_coverage(self, positions, time_stamps, num_satellites, orbital_period):
        """
        Calculate coverage metrics from ground station (local coverage).
        
        Parameters:
        -----------
        positions : numpy array
            Satellite positions (num_satellites, num_time_steps, 3)
        time_stamps : numpy array
            Time stamps for each position
        num_satellites : int
            Number of satellites
        orbital_period : float
            Orbital period in seconds
            
        Returns:
        --------
        local_score : float
            Score for local coverage (0-1)
        overlap_penalty : float
            Penalty for overlap (0-1)
        """
        idx_oneperiod = np.searchsorted(time_stamps, orbital_period)
        all_visibilities = []
        
        # Calculate visibility for each satellite
        for satellite_index in range(num_satellites):
            visibility = []
            for idx in range(idx_oneperiod):
                sat_position = positions[satellite_index, idx, :]  # km 
                sat_time = time_stamps[idx]
                ECEF_cartesian_orbit = self._eci_to_ecef(sat_position*1000, sat_time)
                vis, _ = Coverage.is_visible_cartesian_ECEF(
                    ECEF_cartesian_orbit, self.gs_geodetic, 
                    self.elevation_threshold
                )
                visibility.append(int(vis))
            all_visibilities.append(visibility)
        
        # Convert to numpy array for calculations
        all_visibilities = np.array(all_visibilities)
        total_visible_seconds_all = np.sum(all_visibilities) * self.step
        any_visible_per_timestep = np.any(all_visibilities, axis=0)
        total_time_with_any_visible = np.sum(any_visible_per_timestep) * self.step
        
        # Calculate local coverage score - percentage of time with visibility
        total_time = len(time_stamps[:idx_oneperiod]) * self.step
        local_score = total_time_with_any_visible / total_time
        
        # Calculate overlap penalty (lower is better)
        if total_time_with_any_visible > 0:
            overlap_percentage = (total_visible_seconds_all - total_time_with_any_visible) / total_time_with_any_visible
            overlap_penalty = 1.0 / (1.0 + overlap_percentage)  # Higher overlap gives lower score
        else:
            overlap_penalty = 0.0  # No visibility at all
            
        return local_score, overlap_penalty
    
    def _calculate_regional_coverage(self, const, orbital_parameters, num_satellites, epoch, orbital_period, 
                                    positions=None, time_stamps=None):
        """
        Calculate coverage metrics for a regional area using Coverage utilities.
        
        Parameters:
        -----------
        const : Constellation object
            Constellation object
        orbital_parameters : dict
            Orbital parameters
        num_satellites : int
            Number of satellites
        epoch : datetime
            Epoch for constellation
        orbital_period : float
            Orbital period in seconds
        positions : numpy.ndarray, optional
            Pre-computed satellite positions from the optimizer
        time_stamps : numpy.ndarray, optional
            Pre-computed timestamps from the optimizer
                
        Returns:
        --------
        regional_score : float
            Score for regional coverage (0-1)
        """
        # If positions weren't passed, use the existing grid-based approach
        if positions is None or time_stamps is None:
            # Fall back to original grid-based approach
            print("Warning: Using simplified grid-based regional coverage calculation due to missing positions/timestamps.")
            return self._calculate_regional_coverage_simplified(const, orbital_parameters, num_satellites, 
                                                        epoch, orbital_period)
        
        try:
            # Ensure timestamps are datetime objects for Coverage utilities
            if isinstance(time_stamps[0], (int, float, np.integer, np.floating)):
                ts_dt = [self.start_time + timedelta(seconds=float(t)) for t in time_stamps]
            else:
                ts_dt = time_stamps

            # Create Coverage object with existing positions and timestamps
            coverage_obj = Coverage(orbital_parameters, positions=positions, times=ts_dt)
            
            # Get region from self.regional_area
            region = self.regional_area
            
            # Compute regional coverage with appropriate parameters
            regional_result = coverage_obj.compute_regional_coverage(
                positions, ts_dt,
                region,
                orbital_periods=1,
                masking_angle=15.0,  # Default masking angle
                fov=120.0,          # Default FOV
                grid_density=5.0    # Coarser grid for optimization speed
            )
            
            # Extract the coverage percentage and normalize to 0-1
            coverage_percentage = regional_result['constellation_coverage'] / 100.0
            
            return min(coverage_percentage, 1.0)  # Ensure the score is between 0 and 1
        
        except Exception as e:
            print(f"Warning: Error in regional coverage calculation: {str(e)}")
            # Fall back to original grid-based approach
            return self._calculate_regional_coverage_simplified(const, orbital_parameters, num_satellites, 
                                                        epoch, orbital_period)

    def _calculate_regional_coverage_simplified(self, const, orbital_parameters, num_satellites, epoch, orbital_period):
        """
        Very simple regional coverage proxy: visibility of region centroid over one orbital period.
        Returns value in [0,1].
        """
        from utilities_coverage import Coverage
        # Propagate (reuse optimizer window)
        res = const.generate_positions_with_time(self.start_time, self.end_time, self.step)
        positions = res["positions"]
        time_stamps = res["time_stamps"]

        # Region centroid (avg vertices)
        lats = [pt[0] for pt in self.regional_area]
        lons = [pt[1] for pt in self.regional_area]
        centroid = (sum(lats) / len(lats), sum(lons) / len(lons), 0.0)  # lat, lon, alt(m)

        # One orbital period worth of steps
        idx_oneperiod = np.searchsorted(time_stamps, orbital_period)
        idx_oneperiod = max(1, int(idx_oneperiod))

        any_visible = []
        for k in range(idx_oneperiod):
            t = time_stamps[k]
            vis_any = False
            for s in range(num_satellites):
                sat_pos_eci_km = positions[s, k, :]
                ecef_m = self._eci_to_ecef(sat_pos_eci_km * 1000.0, t)
                vis, _ = Coverage.is_visible_cartesian_ECEF(ecef_m, centroid, self.elevation_threshold)
                if vis:
                    vis_any = True
                    break
            any_visible.append(1 if vis_any else 0)

        # Fraction of timesteps with at least one sat visible
        return float(sum(any_visible)) / float(len(any_visible))

    def _calculate_global_coverage(self, num_satellites, orbital_parameters, positions=None, time_stamps=None):
        """
        Estimate global coverage based on constellation parameters using the Coverage utilities.
        
        Parameters:
        -----------
        num_satellites : int
            Number of satellites
        orbital_parameters : dict
            Orbital parameters
        positions : numpy.ndarray, optional
            Pre-computed satellite positions from the optimizer
        time_stamps : numpy.ndarray, optional
            Pre-computed timestamps from the optimizer
            
        Returns:
        --------
        global_score : float
            Score for global coverage (0-1)
        """
        # If positions weren't passed, fall back to the simplified estimate
        if positions is None or time_stamps is None:
            return self._estimate_global_coverage(num_satellites, orbital_parameters)
        
        try:
            # Ensure timestamps are datetime objects for Coverage utilities
            if isinstance(time_stamps[0], (int, float, np.integer, np.floating)):
                ts_dt = [self.start_time + timedelta(seconds=float(t)) for t in time_stamps]
            else:
                ts_dt = time_stamps

            # Create Coverage object with existing positions and timestamps
            coverage_obj = Coverage(orbital_parameters, positions=positions, times=ts_dt)
            
            # Compute global coverage with appropriate parameters
            # Use lower grid resolution for optimization (faster)
            global_coverage_result = coverage_obj.compute_global_coverage(
                positions, ts_dt,
                orbital_periods=1,
                masking_angle=15.0,  # Default masking angle
                fov=100.0,          # Default FOV
                grid_density=15.0   # Coarser grid for optimization speed
            )
            
            # Extract the coverage percentage and normalize to 0-1
            coverage_percentage = global_coverage_result['constellation_coverage'] / 100.0
            
            return min(coverage_percentage, 1.0)  # Ensure the score is between 0 and 1
        
        except Exception as e:
            print(f"Warning: Error in global coverage calculation: {str(e)}")
            # Fallback to simplified estimate
            return self._estimate_global_coverage(num_satellites, orbital_parameters)

    def _estimate_global_coverage(self, num_satellites, orbital_parameters):
        """
        Simple geometric proxy for global coverage when grid computation fails.
        
        Returns:
        --------
        coverage_proxy : float (0-1)
        """
        inc_rad = np.radians(orbital_parameters['inclination'])
        altitude_km = orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS
        
        # Higher inclination = better global coverage (polar vs equatorial)
        inc_score = min(1.0, inc_rad / (np.pi/2))  # 0 at equator, 1 at polar
        
        # More satellites = better coverage (log scale, saturates at 100)
        sat_score = np.log1p(num_satellites) / np.log1p(100)
        
        # Altitude sweet spot around 600-800 km for LEO coverage
        if 400 <= altitude_km <= 1200:
            alt_score = 1.0
        else:
            alt_score = 0.5
        
        print('WARNING [Utilities Opt]: Using simplified global coverage')
        # Weighted combination
        return 0.5 * inc_score + 0.3 * sat_score + 0.2 * alt_score

    def compute_payload(self, payloads=None, payload_mass_kg=None, cat='EO', instrument='particle', power_w=None, design_life_months=None, design_life_years=None):
        """
        Compute payload TFU cost (M$).

        Modes:
        - Multi-instrument: payloads=[{ 'type': 'fields'|'particle', 'mass_kg': float,
                                        'power_w': float, 'cat': 'EO'|'planetary',
                                        'design_life_months'?: float }, ...]
          (legacy key 'design_life_years' is also accepted and will be converted to months)
        - Single-instrument (legacy): payload_mass_kg + power_w + cat + instrument.
          For planetary particle instruments, pass design_life_months (or legacy design_life_years).

        Defaults (if nothing is provided): EO fields 10 kg/30 W + EO particle 10 kg/30 W.
        """
        # Default two-instrument payload if nothing specified
        if payloads is None:
            payloads = [
                {'type': 'fields', 'cat': 'EO', 'mass_kg': 10.0, 'power_w': 30.0},
                {'type': 'particle', 'cat': 'EO', 'mass_kg': 10.0, 'power_w': 30.0},
            ]

        # Enforce single mission category
        cats = {str(item.get('cat', 'EO')).strip().lower() for item in payloads}
        if len(cats) > 1:
            raise ValueError("All payloads must share the same 'cat' ('EO' or 'planetary'); mixing categories is not supported.")
        mission_cat = cats.pop() if cats else 'eo'

        # Accumulate NRE and TFU separately
        total_nre = 0.0
        total_tfu = 0.0
        
        for i, item in enumerate(payloads):
            inst_type = str(item.get('type', '')).strip().lower()
            m = item.get('mass_kg')
            p = item.get('power_w')

            # Convert legacy design_life_years to design_life_months if present
            dl_months = item.get('design_life_months')
            dl_years = item.get('design_life_years')
            if dl_months is None and dl_years is not None:
                dl_months = float(dl_years) * 12.0

            if inst_type == 'fields':
                nre, tfu = self._calculate_payload_cost(cat=mission_cat, mass_field=m, power_field=p)
                total_nre += nre
                total_tfu += tfu
            elif inst_type == 'particle':
                nre, tfu = self._calculate_payload_cost(cat=mission_cat, mass_particle=m, power_particle=p, design_life_planetary=dl_months)
                total_nre += nre
                total_tfu += tfu
            else:
                raise ValueError(f"Payload {i}: unknown type '{inst_type}'. Expected 'fields' or 'particle'.")

        # Support single-instrument legacy call when payloads is provided via parameters
        if payload_mass_kg is not None and power_w is not None and instrument is not None:
            # Convert legacy design_life_years to months
            dl_months_legacy = design_life_months
            if dl_months_legacy is None and design_life_years is not None:
                dl_months_legacy = float(design_life_years) * 12.0

            if instrument.lower() == 'fields':
                nre, tfu = self._calculate_payload_cost(cat=cat, mass_field=payload_mass_kg, power_field=power_w)
                total_nre += nre
                total_tfu += tfu
            elif instrument.lower() == 'particle':
                nre, tfu = self._calculate_payload_cost(cat=cat, mass_particle=payload_mass_kg, power_particle=power_w, design_life_planetary=dl_months_legacy)
                total_nre += nre
                total_tfu += tfu
            else:
                raise ValueError(f"Unknown instrument type: {instrument}")

        return total_nre, total_tfu

    def _calculate_payload_cost(
        self,
        cat='EO',
        mass_field=None,
        power_field=None,
        mass_particle=None,
        power_particle=None,
        design_life_planetary=None
    ):
        """
        Calculate payload instrument cost using CERs and return total in 2025 M$.

        Inputs:
        - cat: 'EO' or 'planetary' (default 'EO')
        - mass_field (kg), power_field (W) for Fields instrument
        - mass_particle (kg), power_particle (W) for Particle instrument
        - design_life_planetary (months) required only for planetary particle instrument

        Notes:
        - CERs are in $K FY04; convert to M$ and July 2025 dollars (x1.70 approx).
        - Fields cost formula is the same for EO and planetary; particle differs by cat.
        - If only one instrument is provided, only that cost is computed.
        - If no inputs are provided, returns 0.0.
        """
        def to_musd_2025(cost_k_fy04: float) -> float:
            return (cost_k_fy04 / 1000.0) * constants.INFLATION_2004_TO_2025

        def check_positive(name, value):
            if value is None:
                return
            if float(value) <= 0:
                raise ValueError(f"{name} must be > 0")

        # Validate inputs that are present
        check_positive("mass_field", mass_field)
        check_positive("power_field", power_field)
        check_positive("mass_particle", mass_particle)
        check_positive("power_particle", power_particle)
        check_positive("design_life_planetary", design_life_planetary)

        cat_norm = (cat or 'EO').strip().lower()
        if cat_norm not in ('eo', 'planetary'):
            raise ValueError("cat must be 'EO' or 'planetary'")

        total_musd_2025 = 0.0

        # Fields instrument (same for EO and planetary)
        if mass_field is not None or power_field is not None:
            if mass_field is None or power_field is None:
                raise ValueError("Both mass_field and power_field are required for Fields instrument cost.")
            c_k = 1464.0 * (float(mass_field) ** 0.29) * (float(power_field) ** 0.36)
            total_musd_2025 += to_musd_2025(c_k)

        # Particle instrument
        if cat_norm == 'planetary':
            # Planetary particles requires design life (months), mass, power
            if any(v is not None for v in (design_life_planetary, mass_particle, power_particle)):
                if design_life_planetary is None or mass_particle is None or power_particle is None:
                    raise ValueError("design_life_planetary (months), mass_particle, and power_particle are required for planetary particle instrument cost.")
                c_k = 247.0 * (float(design_life_planetary) ** 0.50) * (float(mass_particle) ** 0.31) * (float(power_particle) ** 0.44)
                total_musd_2025 += to_musd_2025(c_k)
        else:
            # EO particles uses EO formula (no design life)
            if mass_particle is not None or power_particle is not None:
                if mass_particle is None or power_particle is None:
                    raise ValueError("Both mass_particle and power_particle are required for EO particle instrument cost.")
                c_k = 1353.0 * (float(mass_particle) ** 0.47) * (float(power_particle) ** 0.39)
                total_musd_2025 += to_musd_2025(c_k)

        nre_payload_musd_2025 = total_musd_2025
        tfu_payload_musd_2025 = 0.4 * nre_payload_musd_2025

        return nre_payload_musd_2025, tfu_payload_musd_2025

    def compute_satellite_cost(self, mass_kg: float | None = None):
        """
        Spacecraft cost CERs (M$), driven by total spacecraft mass (kg).
        Returns (NRE_musd, tfu_musd).

        NRE from SSCM 1996 (old) (FY10$K): 1064 + 35.5*mass_bus^1.261
        TFU (STILL NEED A REFERENCE FOR THIS) (M$): 0.151 * mass^0.7
        changed TFU to 40% NRE
        """
        m_wet = float(self.sat_mass_kg if mass_kg is None else mass_kg)
        # Sum payload mass from self.payloads; ignore missing/invalid entries
        try:
            payload_mass = float(sum(float(p.get('mass_kg', 0.0)) for p in (self.payloads or [])))
        except Exception:
            payload_mass = 0.0
        # Bus mass must be positive
        m_bus = max(1.0, m_wet - payload_mass)
        nre_k2010usd = 1064 + 35.5 * (m_bus ** 1.261)
        nre_musd = nre_k2010usd / 1000.0 * constants.INFLATION_2010_TO_2025
        # tfu_musd = 0.151 * (m_bus ** 0.7)
        tfu_musd = nre_musd*0.4
        return nre_musd, tfu_musd
        
    def _calculate_launch_cost(self, orbital_parameters, num_satellites, sat_mass_kg, pattern_params=None,
                            dv_deployment_m_s: float = 100.0,
                            T_deployment_days: float = 180.0,
                            bundle_discount_per_extra_sat: float = 0.05,  # 5% off per extra sat on same launch
                            bundle_discount_cap: float = 0.30):
        """
        Calculate total launch cost in million USD (M$) and return detailed deployment information.

        Strategy:
        - Given dv_deployment_m_s, compute max |Δa_deployment| via one-way Hohmann inversion
        - Compute ΔRAAN_deployment achievable in T_deployment_days using J2 differential precession
        - If the required plane spacing can be met with less Δa (same T), reduce Δa to save sat Δv
        - Group planes based on achievable ΔRAAN; for each group choose the cheapest vehicle option
        allowing multiple launches if required by mass capacity
        - Apply a bundle discount when multiple satellites ride the same vehicle launch:
        discount_factor = 1 - min(cap, per_extra_sat * (avg_sats_per_launch - 1))

        Parameters:
        -----------
        dv_deployment_m_s : float
            One-way deployment Δv budget (m/s) available for phasing drift
        T_deployment_days : float
            Maximum deployment time (days) allowed
        bundle_discount_per_extra_sat : float
            Fractional discount per extra satellite on the same launch (e.g., 0.05 = 5% off)
        bundle_discount_cap : float
            Maximum per-launch discount fraction (e.g., 0.30 = 30% off)

        Returns:
        --------
        result : dict
            {
                'total_cost_musd': float,
                'num_launches': int,
                'num_groups': int,
                'bundle_discount_per_extra_sat': float,
                'bundle_discount_cap': float,
                'vehicle_breakdown': [
                    {
                        'group': int,
                        'vehicle': str,
                        'num_launches': int,
                        'planes': int,
                        'mass_kg': float,
                        'cost_musd': float,
                        'avg_sats_per_launch': int,
                        'discount_factor': float
                    }, ...
                ],
                'deployment': {
                    'dv_deployment_m_s': float,
                    'dv_used_m_s': float,
                    'T_deployment_days': float,
                    'T_used_days': float,
                    'a_target_km': float,
                    'a_launch_km': float,
                    'delta_a_mag_km': float,
                    'delta_raan_deployment_deg': float,
                    'delta_raan_required_deg': float,
                    'planes_per_group': int
                }
            }
        """
        from orbital_cost_function import ocf_at
        from scipy.optimize import brentq

        mu = constants.GRAVITATIONAL_CONSTANT  # km^3/s^2
        R_e = constants.EARTH_RADIUS           # km
        J2 = constants.J2_EARTH

        # --- Helpers ---
        def hohmann_oneway_dv_m_s(a1_km: float, delta_a_km: float) -> float:
            """One-way Hohmann Δv from circular a1 to circular a2=a1+Δa (m/s)."""
            r1 = float(a1_km)
            r2 = float(a1_km + delta_a_km)
            if r1 <= 0 or r2 <= 0 or r2 <= R_e:
                return np.inf
            v1 = np.sqrt(mu / r1)
            v2 = np.sqrt(mu / r2)
            a_t = 0.5 * (r1 + r2)
            v_peri_t = np.sqrt(mu * (2.0 / r1 - 1.0 / a_t))
            v_apo_t = np.sqrt(mu * (2.0 / r2 - 1.0 / a_t))
            dv1 = abs(v_peri_t - v1)
            dv2 = abs(v2 - v_apo_t)
            return (dv1 + dv2) * 1000.0  # m/s

        def invert_oneway_dv_to_delta_a_mag(a1_km: float, dv_target_m_s: float) -> float:
            """Find |Δa| ≥ 0 such that one-way Δv equals dv_target_m_s."""
            if dv_target_m_s <= 0:
                return 0.0
            lo, hi = 0.0, 10.0
            f = lambda da: hohmann_oneway_dv_m_s(a1_km, da) - dv_target_m_s
            max_hi = 1.0e5
            while f(hi) < 0.0 and hi < max_hi:
                hi *= 1.8
            if hi >= max_hi:
                return hi
            return float(brentq(f, lo, hi, maxiter=100, xtol=1e-6))

        def raan_rate_rad_per_s(a_km: float, inc_rad: float) -> float:
            """J2-induced RAAN drift rate (rad/s)."""
            return -(3.0/2.0) * J2 * (R_e**2) * np.sqrt(mu) * np.cos(inc_rad) / (a_km**(7.0/2.0))

        def compute_delta_raan_for_time_and_delta_a(a_target: float, delta_a: float, T_days: float, inc_rad: float) -> float:
            """ΔRAAN (deg) achieved over T_days with given Δa (a_launch = a_target + Δa)."""
            a_launch = a_target + delta_a
            omega_dot_target = raan_rate_rad_per_s(a_target, inc_rad)
            omega_dot_launch = raan_rate_rad_per_s(a_launch, inc_rad)
            T_seconds = T_days * 86400.0
            delta_raan_rad = (omega_dot_launch - omega_dot_target) * T_seconds
            return np.degrees(abs(delta_raan_rad))

        # --- Inputs and derived quantities ---
        altitude = orbital_parameters['semi_major_axis'] - R_e  # km
        inclination = orbital_parameters['inclination']         # deg
        inc_rad = np.radians(inclination)
        a_target_km = float(orbital_parameters['semi_major_axis'])
        sat_mass_kg = float(sat_mass_kg)
        T = int(num_satellites)

        # Planes and plane spacing
        P = 1
        if pattern_params and 'P' in pattern_params and int(pattern_params['P']) > 0:
            P = int(pattern_params['P'])
        sats_per_plane = max(1, T // P)
        plane_spacing_deg = 360.0 / P if P > 0 else 360.0

        # --- Deployment sizing from Δv and T_dep ---
        delta_a_max_km = invert_oneway_dv_to_delta_a_mag(a_target_km, float(dv_deployment_m_s))
        delta_raan_max_deg = compute_delta_raan_for_time_and_delta_a(
            a_target_km, delta_a_max_km, T_deployment_days, inc_rad
        )
        delta_raan_required = plane_spacing_deg

        if delta_raan_max_deg >= delta_raan_required:
            # Use the minimum Δa that achieves the required ΔRAAN in the allotted time
            def objective(da):
                return compute_delta_raan_for_time_and_delta_a(a_target_km, da, T_deployment_days, inc_rad) - delta_raan_required
            try:
                delta_a_used_km = brentq(objective, 0.0, delta_a_max_km, maxiter=100, xtol=1e-6)
                dv_used_m_s = hohmann_oneway_dv_m_s(a_target_km, delta_a_used_km)
                T_used_days = T_deployment_days
                delta_raan_deployment_deg = delta_raan_required
            except Exception:
                delta_a_used_km = delta_a_max_km
                dv_used_m_s = dv_deployment_m_s
                T_used_days = T_deployment_days
                delta_raan_deployment_deg = delta_raan_max_deg
        else:
            # Even with max Δa, can't reach requirement within T_dep; use max capability
            delta_a_used_km = delta_a_max_km
            dv_used_m_s = dv_deployment_m_s
            T_used_days = T_deployment_days
            delta_raan_deployment_deg = delta_raan_max_deg

        a_launch_km = a_target_km + delta_a_used_km

        # Group planes by achievable ΔRAAN
        planes_per_group = max(1, int(np.floor(delta_raan_deployment_deg / plane_spacing_deg)) + 1)
        num_groups = int(np.ceil(P / planes_per_group))

        # --- Vehicle database (cost M$ per launch, LEO payload kg) ---
        vehicles = {
            'Electron': {'payload_kg_leo': 300, 'cost_musd': 7.5, 'operational': True},
            'SSLV': {'payload_kg_leo': 500, 'cost_musd': 4.0, 'operational': True},
            'Alpha': {'payload_kg_leo': 1030, 'cost_musd': 15.0, 'operational': True},
            'Epsilon_S': {'payload_kg_leo': 1400, 'cost_musd': 30.0, 'operational': False},
            'Minotaur_IV': {'payload_kg_leo': 1735, 'cost_musd': 45.0, 'operational': True},
            'KSLV-II': {'payload_kg_leo': 2200, 'cost_musd': 80.0, 'operational': True},
            
            'Vega-C': {'payload_kg_leo': 2500, 'cost_musd': 50.0, 'operational': True},
            'PSLV-XL': {'payload_kg_leo': 2900, 'cost_musd': 27.0, 'operational': True},
            'GSLV_Mk_II': {'payload_kg_leo': 5500, 'cost_musd': 50.0, 'operational': True},
            'Ariane_62': {'payload_kg_leo': 8000, 'cost_musd': 90.0, 'operational': False},
            'Neutron_Reusable': {'payload_kg_leo': 13000, 'cost_musd': 55.0, 'operational': False},
            'Neutron_Expendable': {'payload_kg_leo': 15000, 'cost_musd': 55.0, 'operational': False},
            'Falcon_9_Launch_Site': {'payload_kg_leo': 15000, 'cost_musd': 69.85, 'operational': True},
            'H3': {'payload_kg_leo': 16000, 'cost_musd': 50.0, 'operational': False}, # too cheap and usually for GEO
            'Vulcan_Centaur_VC2': {'payload_kg_leo': 16300, 'cost_musd': 100.0, 'operational': True},
            'Falcon_9_Drone_Ship': {'payload_kg_leo': 16600, 'cost_musd': 69.85, 'operational': True},
            'Atlas_V_551': {'payload_kg_leo': 17500, 'cost_musd': 153.0, 'operational': True},
            'Ariane_64': {'payload_kg_leo': 17500, 'cost_musd': 130.0, 'operational': False},
            'Falcon_9_Expendable': {'payload_kg_leo': 20400, 'cost_musd': 90.0, 'operational': True},
            'Vulcan_Centaur_VC4': {'payload_kg_leo': 21600, 'cost_musd': 110.0, 'operational': True},
            'Vulcan_Centaur_VC6': {'payload_kg_leo': 25800, 'cost_musd': 130.0, 'operational': False},
            'Starship_Block_2': {'payload_kg_leo': 35000, 'cost_musd': 75.0, 'operational': False},
            'New_Glenn': {'payload_kg_leo': 45000, 'cost_musd': 110.0, 'operational': False},
            'Falcon_Heavy_Reusable': {'payload_kg_leo': 57000, 'cost_musd': 97.0, 'operational': False}, # too cheap and used for single sat
            'Falcon_Heavy_Expendable': {'payload_kg_leo': 63800, 'cost_musd': 150.0, 'operational': False}, # too cheap and used for single sat
        }

        total_cost_musd = 0.0
        total_launches = 0
        vehicle_breakdown = []

        for g in range(num_groups):
            planes_in_group = min(planes_per_group, P - g * planes_per_group)
            group_mass_kg = planes_in_group * sats_per_plane * sat_mass_kg
            total_sats_in_group = int(planes_in_group * sats_per_plane)

            best_vehicle_name = None
            best_group_cost_musd = float('inf')
            best_num_launches = 0
            best_avg_sats_per_launch = 1
            best_discount_factor = 1.0

            for vehicle_name, vehicle in vehicles.items():
                if not vehicle.get('operational', False):
                    continue

                # Single OCF for all vehicles
                try:
                    ocf = ocf_at(inclination, altitude)   # signature (inc, alt)
                except TypeError:
                    ocf = ocf_at(altitude, inclination)   # signature (alt, inc)

                effective_capacity_kg = vehicle['payload_kg_leo'] * ocf
                if effective_capacity_kg <= 0:
                    continue

                num_launches = int(np.ceil(group_mass_kg / effective_capacity_kg))

                # Bundle discount based on average satellites per launch
                if sat_mass_kg > 0:
                    max_sats_one_launch = max(1, int(np.floor(effective_capacity_kg / sat_mass_kg)))
                else:
                    max_sats_one_launch = 1

                avg_sats_per_launch = max(1, int(np.ceil(total_sats_in_group / max(1, num_launches))))
                avg_sats_per_launch = min(avg_sats_per_launch, max_sats_one_launch)

                discount_factor = 1.0 - min(float(bundle_discount_cap),
                                            float(bundle_discount_per_extra_sat) * max(0, avg_sats_per_launch - 1))

                # vehicle_cost_musd = num_launches * vehicle['cost_musd'] * discount_factor # previously using entire launches even if not full

                # Pay only for the fraction of launches actually used by mass (pro‑rata), still keep integer 'num_launches' for reporting and discount calculation.
                paid_launch_equiv = group_mass_kg / effective_capacity_kg  # can be fractional
                vehicle_cost_musd = paid_launch_equiv * vehicle['cost_musd'] * discount_factor

                if vehicle_cost_musd < best_group_cost_musd:
                    best_group_cost_musd = vehicle_cost_musd
                    best_vehicle_name = vehicle_name
                    best_num_launches = num_launches
                    best_avg_sats_per_launch = avg_sats_per_launch
                    best_discount_factor = discount_factor
                    best_paid_launch_equiv = paid_launch_equiv

            if best_group_cost_musd == float('inf'):
                # Infeasible for this configuration
                return {
                    'total_cost_musd': 1e6,
                    'num_launches': 0,
                    'num_groups': num_groups,
                    'bundle_discount_per_extra_sat': bundle_discount_per_extra_sat,
                    'bundle_discount_cap': bundle_discount_cap,
                    'vehicle_breakdown': [],
                    'deployment': {
                        'dv_deployment_m_s': dv_deployment_m_s,
                        'dv_used_m_s': 0.0,
                        'T_deployment_days': T_deployment_days,
                        'T_used_days': 0.0,
                        'a_target_km': a_target_km,
                        'a_launch_km': a_target_km,
                        'delta_a_mag_km': 0.0,
                        'delta_raan_deployment_deg': 0.0,
                        'delta_raan_required_deg': delta_raan_required,
                        'planes_per_group': 0
                    }
                }

            total_cost_musd += best_group_cost_musd
            total_launches += best_num_launches

            vehicle_breakdown.append({
                'group': g + 1,
                'vehicle': best_vehicle_name,
                'num_launches': best_num_launches,
                'planes': planes_in_group,
                'mass_kg': group_mass_kg,
                'cost_musd': best_group_cost_musd,
                'avg_sats_per_launch': best_avg_sats_per_launch,
                'discount_factor': best_discount_factor,
                'paid_launch_equiv': best_paid_launch_equiv
            })

        # Final result
        result = {
            'total_cost_musd': total_cost_musd,
            'num_launches': total_launches,
            'num_groups': num_groups,
            'bundle_discount_per_extra_sat': float(bundle_discount_per_extra_sat),
            'bundle_discount_cap': float(bundle_discount_cap),
            'vehicle_breakdown': vehicle_breakdown,
            'deployment': {
                'dv_deployment_m_s': float(dv_deployment_m_s),
                'dv_used_m_s': float(dv_used_m_s),
                'T_deployment_days': float(T_deployment_days),
                'T_used_days': float(T_used_days),
                'a_target_km': float(a_target_km),
                'a_launch_km': float(a_launch_km),
                'delta_a_mag_km': float(delta_a_used_km),
                'delta_raan_deployment_deg': float(delta_raan_deployment_deg),
                'delta_raan_required_deg': float(delta_raan_required),
                'planes_per_group': int(planes_per_group)
            }
        }

        # Cache for inspection
        self._last_deployment_sizing = result
        return result

    def economies_of_scale(self,
                           num_satellites: int,
                           orbital_parameters: dict,
                           sat_mass_kg: float | None = None,
                           pattern_params: dict = None,
                           learning_rate: float = 0.9,
                           payloads: list = None):
        """
        Total system cost (M$) = NRE + RE (learning curve) + Launch.
        Spacecraft TFU and NRE are computed via mass-based CERs.
        Payload TFU is computed from the provided payloads (or self.payloads if None).
        """
        N = int(num_satellites)
        if N <= 0:
            return {'total_musd': 0.0, 'NRE_musd': 0.0, 'RE_musd': 0.0, 'launch_musd': 0.0,
                    'learning_rate': learning_rate, 'C_TFU_musd': 0.0, 'payload_tfu_musd': 0.0}

        # Spacecraft costs from mass-based CERs
        m = self.sat_mass_kg if sat_mass_kg is None else float(sat_mass_kg)
        nre_sc_musd, tfu_sc_musd = self.compute_satellite_cost(m)

        # Payload TFU (M$)
        payload_def = payloads if payloads is not None else self.payloads
        nre_payload_musd_2025, tfu_payload_musd_2025 = self.compute_payload(payload_def)

        # Unit TFU for learning curve (spacecraft + payload)
        C_TFU = float(tfu_sc_musd) + float(tfu_payload_musd_2025)

        # Learning curve
        b = float(learning_rate)
        exp_b = np.log(b) / np.log(2.0)
        n = np.arange(1, N + 1, dtype=float)
        C_RE = float(np.sum(C_TFU * (n ** exp_b)))

        # Launch total and details
        C_L_res = self._calculate_launch_cost(orbital_parameters, N, m, pattern_params, dv_deployment_m_s=self.dv_deployment_m_s, T_deployment_days=self.T_deployment_days)
        if isinstance(C_L_res, dict):
            C_L = float(C_L_res.get('total_cost_musd', 0.0))
            launch_details = C_L_res
        else:
            C_L = float(C_L_res)
            launch_details = None

        C_NRE = float(nre_sc_musd) + float(nre_payload_musd_2025)
        total = C_NRE + C_RE + C_L

        return {
            'total_musd': total,
            'NRE_musd': C_NRE,
            'sc_nre_musd': float(nre_sc_musd),
            'payload_nre_musd': float(nre_payload_musd_2025),
            'RE_musd': C_RE,
            'launch_musd': C_L,
            'launch_details': launch_details,
            'learning_rate': b,
            'C_TFU_musd': C_TFU,
            'payload_tfu_musd': float(tfu_payload_musd_2025),
            'sc_tfu_musd': float(tfu_sc_musd)
        }

    def _calculate_science_requirement_score(self, orbital_parameters):
        """
        Calculate science requirement score based on orbital parameters.
        Closer to target altitude = better score.
        
        Parameters:
        -----------
        orbital_parameters : dict
            Orbital parameters
            
        Returns:
        --------
        science_score : float
            Score for science requirements (0-1, higher is better)
        """
        # IMPROVED: More flexible science requirement scoring
        
        # Calculate how close the altitude is to the target
        altitude = orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS
        distance_from_target = abs(altitude - self.target_altitude)
        
        # Get other orbital parameters that might affect science
        eccentricity = orbital_parameters['eccentricity']
        inclination = orbital_parameters['inclination']
        
        # Normalize distance by tolerance
        normalized_distance = distance_from_target / self.target_altitude_tolerance
        
        # Calculate altitude score: 1.0 if at target, decreasing as we move away
        if normalized_distance <= 1.0:
            # Within tolerance - high score
            altitude_score = 1.0 - (normalized_distance * 0.5)
        else:
            # Outside tolerance - lower score
            altitude_score = 0.5 / normalized_distance
        
        # For science missions, eccentricity may matter (often want circular orbits)
        # Lower eccentricity = better for most Earth observation
        eccentricity_score = 1.0 - eccentricity * 10.0  # Penalize eccentric orbits
        eccentricity_score = max(0.0, min(1.0, eccentricity_score))
        
        # For some missions, specific inclinations may be preferred
        # (e.g., sun-synchronous ~98° for Earth observation)
        # This could be customized based on specific science requirements
        inclination_preference = self._calculate_inclination_science_score(inclination)
        
        # Combine factors (can be weighted differently based on mission needs)
        # Default: altitude is most important
        science_score = 0.7 * altitude_score + 0.2 * eccentricity_score + 0.1 * inclination_preference
        
        return science_score
        
    def _calculate_inclination_science_score(self, inclination):
        """
        Calculate science score component based on inclination relative to target.
        
        Parameters:
        -----------
        inclination : float
            Orbit inclination in degrees
            
        Returns:
        --------
        inclination_score : float
            Score for inclination (0-1, higher is better)
        """
        # Calculate how close the inclination is to the target
        distance_from_target = abs(inclination - self.target_inclination)
        
        # Normalize distance by tolerance
        normalized_distance = distance_from_target / self.target_inclination_tolerance
        
        # Calculate inclination score: 1.0 if at target, decreasing as we move away
        if normalized_distance <= 1.0:
            # Within tolerance - high score
            inclination_score = 1.0 - (normalized_distance * 0.5)
        else:
            # Outside tolerance - lower score (exponential decay)
            inclination_score = 0.5 / normalized_distance
        
        return inclination_score

    def analyze_component_scores(self, params=None, plot=True, verbose=True, debug=False):
        """
        Analyze the component scores of the objective function.
        If params is None, uses the best parameters found during optimization.
        
        Parameters:
        -----------
        params : list or array, optional
            Parameters to analyze [num_satellites, semi_major_axis, eccentricity, inclination, 
                                  RAAN, argument_of_periapsis, P, F]
            If None, uses the best parameters found during optimization.
        plot : bool, optional
            Whether to create visualization plots of the scores
        verbose : bool, optional
            Whether to print detailed analysis to console
        debug : bool, optional
            Whether to print extra debug information including operational cost breakdown
            
        Returns:
        --------
        analysis : dict
            Dictionary of component scores and their analysis
        """
        if params is None:
            if self.best_params is None:
                print("No best parameters found. Run optimization first.")
                return None
            params = self.best_params
            
        # Create parameter dictionaries
        num_satellites, orbital_parameters, pattern_params = self._create_orbital_parameters(params)
        
        # Calculate orbital period
        a = orbital_parameters['semi_major_axis']  # km from Earth's center
        orbital_period = 2 * np.pi * np.sqrt(a**3 / constants.GRAVITATIONAL_CONSTANT)
        
        # Create the constellation for analysis
        epoch = datetime(2025, 5, 5, 0, 0, 0)
        
        if self.pattern_type == "Walker-Delta":
            P = pattern_params['P']
            F = pattern_params['F']
            
            # Create constellation
            const = Constellation(
                num_satellites, orbital_parameters, epoch, 
                self.pattern_type, P=P, F=F
            )
        elif self.pattern_type == "Flower":
            Np = pattern_params['Np']
            Nd = pattern_params['Nd']
            Fn = pattern_params['Fn']
            Fd = pattern_params['Fd']
            Fh = pattern_params['Fh']

            # Add Flower-specific parameters to orbital_parameters
            orbital_parameters.update({
                'Np': Np,
                'Nd': Nd,
                'Fn': Fn,
                'Fd': Fd,
                'Fh': Fh
            })
            
            # Create Flower constellation
            const = Constellation(
                num_satellites, orbital_parameters, epoch, 
                self.pattern_type, Np=Np, Nd=Nd, Fn=Fn, Fd=Fd, Fh=Fh
            )
        else:
            const = Constellation(
                num_satellites, orbital_parameters, epoch, 
                self.pattern_type
            )
            
        # Propagate constellation
        result = const.generate_positions_with_time(
            self.start_time, self.end_time, self.step
        )
        
        positions = result["positions"]
        time_stamps = result["time_stamps"]
        
        # Calculate all component scores
        local_score, overlap_penalty = self._calculate_local_coverage(
            positions, time_stamps, num_satellites, orbital_period
        )
        
        # Pass positions and time stamps to ensure grid coverage calculation is used
        regional_score = self._calculate_regional_coverage(
            const, orbital_parameters, num_satellites, epoch, orbital_period, positions=positions, time_stamps=time_stamps
        )
        
        global_score = self._calculate_global_coverage(
            num_satellites, orbital_parameters, positions=positions, time_stamps=time_stamps
        )
        
        satellite_cost_score = self._calculate_satellite_cost(num_satellites)
        # Operational cost currently deactivated; use neutral score
        operational_cost_score = 1.0
        launch_cost_score = self._calculate_launch_cost(orbital_parameters, num_satellites)
        science_score = self._calculate_science_requirement_score(orbital_parameters)
        
        # Calculate weighted scores
        weighted_local = self.local_weight * local_score * overlap_penalty
        weighted_regional = self.regional_weight * regional_score
        weighted_global = self.global_weight * global_score
        
        coverage_weight_sum = self.local_weight + self.regional_weight + self.global_weight
        weighted_coverage = (weighted_local + weighted_regional + weighted_global) / coverage_weight_sum
        
        weighted_satellite_cost = self.satellite_cost_weight * satellite_cost_score
        weighted_operational_cost = self.operational_cost_weight * operational_cost_score
        weighted_launch_cost = self.launch_cost_weight * launch_cost_score
        
        cost_weight_sum = self.satellite_cost_weight + self.operational_cost_weight + self.launch_cost_weight
        if cost_weight_sum > 0:
            weighted_cost = (weighted_satellite_cost + weighted_operational_cost + weighted_launch_cost) / cost_weight_sum
        else:
            weighted_cost = 1.0  # No cost consideration
            
        weighted_science = self.science_weight * science_score
        
        # Calculate final score using the same formula as in objective_function for consistency
        coverage_weight = self.local_weight + self.regional_weight + self.global_weight
        cost_weight = self.satellite_cost_weight + self.operational_cost_weight + self.launch_cost_weight
        total_weight = coverage_weight + cost_weight + (self.science_weight if self.science_weight > 0 else 0)
        
        # Need to recalculate the raw component groups with the right normalization
        # This ensures we get the same score as calculated in the objective_function
        coverage_score = (
            self.local_weight * local_score * overlap_penalty +
            self.regional_weight * regional_score +
            self.global_weight * global_score
        ) / coverage_weight if coverage_weight > 0 else 0
        
        cost_score = (
            self.satellite_cost_weight * satellite_cost_score +
            self.operational_cost_weight * operational_cost_score +
            self.launch_cost_weight * launch_cost_score
        ) / cost_weight if cost_weight > 0 else 1.0
        
        # Weighted average of component scores - exact same formula as in objective_function
        final_score = (
            coverage_weight * coverage_score + 
            cost_weight * cost_score + 
            (self.science_weight * science_score if self.science_weight > 0 else 0)
        ) / total_weight
        
        # Create analysis dictionary
        analysis = {
            'parameters': {
                'num_satellites': num_satellites,
                'altitude': orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS,
                'eccentricity': orbital_parameters['eccentricity'],
                'inclination': orbital_parameters['inclination'],
                'RAAN': orbital_parameters['RAAN'],
                'arg_periapsis': orbital_parameters['argument_of_periapsis'],
            },
            'raw_scores': {
                'local_score': local_score,
                'overlap_penalty': overlap_penalty,
                'regional_score': regional_score,
                'global_score': global_score,
                'satellite_cost_score': satellite_cost_score,
                'operational_cost_score': operational_cost_score,
                'launch_cost_score': launch_cost_score,
                'science_score': science_score
            },
            'weighted_scores': {
                'weighted_local': weighted_local,
                'weighted_regional': weighted_regional,
                'weighted_global': weighted_global,
                'weighted_coverage': weighted_coverage,
                'weighted_satellite_cost': weighted_satellite_cost,
                'weighted_operational_cost': weighted_operational_cost,
                'weighted_launch_cost': weighted_launch_cost,
                'weighted_cost': weighted_cost,
                'weighted_science': weighted_science,
                'final_score': final_score
            },
            'weights': {
                'local_weight': self.local_weight,
                'regional_weight': self.regional_weight,
                'global_weight': self.global_weight,
                'satellite_cost_weight': self.satellite_cost_weight,
                'operational_cost_weight': self.operational_cost_weight,
                'launch_cost_weight': self.launch_cost_weight,
                'science_weight': self.science_weight
            }
        }
        
        # Add pattern parameters to analysis
        if self.pattern_type == "Walker-Delta":
            analysis['parameters']['P'] = pattern_params['P']
            analysis['parameters']['F'] = pattern_params['F']
        elif self.pattern_type == "Flower":
            analysis['parameters']['Np'] = pattern_params['Np']
            analysis['parameters']['Nd'] = pattern_params['Nd']
            analysis['parameters']['Fn'] = pattern_params['Fn']
            analysis['parameters']['Fd'] = pattern_params['Fd']
            analysis['parameters']['Fh'] = pattern_params['Fh']
            
        # Print detailed analysis if verbose is enabled
        if verbose:
            print("\n=== CONSTELLATION ANALYSIS ===")
            print(f"Pattern Type: {self.pattern_type}")
            print(f"Number of satellites: {num_satellites}")
            if self.pattern_type == "Walker-Delta":
                print(f"Pattern: {num_satellites}/{pattern_params['P']}/{pattern_params['F']}")
            print(f"Altitude: {orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS:.1f} km")
            print(f"Inclination: {orbital_parameters['inclination']:.1f}°")
            
            print("\n--- Raw Component Scores ---")
            print(f"Local Coverage (Ground Station): {local_score:.3f}")
            print(f"Overlap Penalty: {overlap_penalty:.3f}")
            print(f"Regional Coverage: {regional_score:.3f}")
            print(f"Global Coverage: {global_score:.3f}")
            print(f"Satellite Cost: {satellite_cost_score:.3f}")
            print(f"Operational Cost: {operational_cost_score:.3f}")
            print(f"Launch Cost: {launch_cost_score:.3f}")
            print(f"Science Requirements: {science_score:.3f}")
            
            # Show detailed operational cost breakdown if debug is enabled
            if debug and hasattr(self, '_operational_cost_details'):
                print("\n--- Operational Cost Breakdown ---")
                details = self._operational_cost_details
                print(f"Station Keeping Cost: {details['station_keeping_cost']:.3f} (Weight: {details['weights']['station_keeping_weight']:.1f})")
                print(f"Contact Cost: {details['contact_cost']:.3f} (Weight: {details['weights']['contact_weight']:.1f})")
                print(f"Deorbit Cost: {details['deorbit_cost']:.3f} (Weight: {details['weights']['deorbit_weight']:.1f})")
                print(f"Satellite Lifetime: {details['lifetime_years']:.1f} years")
                print(f"Annual Operational Cost: {details['annual_cost']:.3f}")
                print(f"Total Operational Cost: {details['total_operational_cost']:.3f}")
                print(f"Final Operational Cost Score: {details['operational_cost_score']:.3f}")
            
            print("\n--- Final Weighted Score ---")
            print(f"Final Score: {final_score:.4f}") # This should now match the objective function's score
        
        # Create visualization plots
        if plot:
            self._plot_component_scores(analysis)
            
        return analysis
        
    def _generate_plot_directory(self):
        """
        Generate a unique directory name for plots based on optimizer parameters.
        """
        parts = [
            f"step{self.step}",
            f"locW{self.local_weight}",
            f"regW{self.regional_weight}",
            f"globW{self.global_weight}",
            f"satCw{self.satellite_cost_weight}",
            f"opsCw{self.operational_cost_weight}",
            f"lauCw{self.launch_cost_weight}",
            f"sciW{self.science_weight}",
            f"tAlt{self.target_altitude}",
        ]
        directory = "_".join(parts)
        return directory

    def _generate_plot_filename(self, plot_type, extra_params=None):
        """
        Generate a unique filename for plots based on optimizer parameters.
        """
        directory = self._generate_plot_directory()
        parts = [plot_type]
        if extra_params:
            for k, v in extra_params.items():
                parts.append(f"{k}_{v}")
        filename = f"plots/optimization/{directory}/" + "_".join(parts) + ".png"
        return filename

    def _plot_component_scores(self, analysis, extra_params=None):
        """
        Create visualization plots for component scores.
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        # Add space for title and subtitle
        fig.subplots_adjust(top=0.87)

        # 1. Raw component scores bar chart
        ax1 = fig.add_subplot(gs[0, :2])
        raw_scores = analysis['raw_scores']
        score_names = list(raw_scores.keys())
        score_values = list(raw_scores.values())

        colors = ['#3498db', '#9b59b6', '#2ecc71', '#f1c40f', '#e74c3c', '#1abc9c', '#34495e', '#d35400']
        ax1.bar(score_names, score_values, color=colors)
        ax1.set_title('Raw Component Scores')
        ax1.set_ylim(0, 1.0)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # 2. Weighted contribution pie chart
        ax2 = fig.add_subplot(gs[0, 2])
        weights = analysis['weights']
        weighted_scores = analysis['weighted_scores']

        # Only include non-zero weights
        labels = []
        sizes = []

        if weights['local_weight'] > 0:
            labels.append('Local')
            sizes.append(weighted_scores['weighted_local'])

        if weights['regional_weight'] > 0:
            labels.append('Regional')
            sizes.append(weighted_scores['weighted_regional'])

        if weights['global_weight'] > 0:
            labels.append('Global')
            sizes.append(weighted_scores['weighted_global'])

        if weights['satellite_cost_weight'] > 0:
            labels.append('Sat Cost')
            sizes.append(weighted_scores['weighted_satellite_cost'])

        if weights['operational_cost_weight'] > 0:
            labels.append('Op Cost')
            sizes.append(weighted_scores['weighted_operational_cost'])

        if weights['launch_cost_weight'] > 0:
            labels.append('Launch Cost')
            sizes.append(weighted_scores['weighted_launch_cost'])

        if weights['science_weight'] > 0:
            labels.append('Science')
            sizes.append(weighted_scores['weighted_science'])

        # Make sure sizes add up to 100% for pie chart
        sizes_normalized = [s / sum(sizes) for s in sizes]

        ax2.pie(sizes_normalized, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.axis('equal')
        ax2.set_title('Weighted Contribution to Final Score')

        # 3. Score breakdown - grouped bar chart
        ax3 = fig.add_subplot(gs[1, :])

        # Group scores into categories
        categories = ['Coverage', 'Cost', 'Science']
        category_scores = [
            weighted_scores['weighted_coverage'],
            weighted_scores['weighted_cost'],
            weighted_scores['weighted_science'] if weights['science_weight'] > 0 else 0
        ]

        # Main bars
        x = np.arange(len(categories))
        width = 0.5
        ax3.bar(x, category_scores, width=width, label='Category Total')

        # Add text labels on top of bars
        for i, v in enumerate(category_scores):
            ax3.text(i, v + 0.01, f'{v:.3f}', ha='center')

        # Add a line for final score
        ax3.axhline(y=weighted_scores['final_score'], color='r', linestyle='-', label=f'Final Score: {weighted_scores["final_score"]:.3f}')

        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.set_ylim(0, 1.0)
        ax3.set_title('Score Breakdown by Category')
        ax3.legend()
        ax3.grid(axis='y', linestyle='--', alpha=0.7)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.87])  # Adjust rect to match the top value

        # Add title and subtitle
        pattern_info = (
            f"P={analysis['parameters']['P']}, F={analysis['parameters']['F']}" if self.pattern_type == "Walker-Delta" else
            f"Np={analysis['parameters']['Np']}, Nd={analysis['parameters']['Nd']}, Fn={analysis['parameters']['Fn']}, "
            f"Fd={analysis['parameters']['Fd']}, Fh={analysis['parameters']['Fh']}"
        )
        fig.suptitle(
            f'Component Scores\n'
            f'Step: {self.step} s, '
            f'Weights: L={self.local_weight}, R={self.regional_weight}, G={self.global_weight}, '
            f'SatCost={self.satellite_cost_weight}, OpCost={self.operational_cost_weight}, '
            f'LaunchCost={self.launch_cost_weight}, Sci={self.science_weight}, '
            f'Target Altitude: {self.target_altitude} km\n'
            f'Pattern: {self.pattern_type}, Pattern Parameters: {pattern_info}',
            y=0.95
        )

        filename = self._generate_plot_filename("component_scores", extra_params)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        # plt.show()

    def plot_optimization_history(self, extra_params=None):
        """Plot the optimization history to see convergence."""
        if not self.optimization_history:
            print("No optimization history to plot.")
            return

        scores = [item[1] for item in self.optimization_history]
        plt.figure(figsize=(12, 10))
        plt.plot(scores, 'o-', label="Score")
        plt.xlabel('Evaluation Number')
        plt.ylabel('Combined Score (higher is better)')

        plt.title(
            f'Optimization Progress\n'
            f'Step: {self.step} s, '
            f'Weights: L={self.local_weight}, R={self.regional_weight}, G={self.global_weight}, '
            f'SatCost={self.satellite_cost_weight}, OpCost={self.operational_cost_weight}, '
            f'LaunchCost={self.launch_cost_weight}, Sci={self.science_weight}, '  
            f'Target Altitude: {self.target_altitude} km\n'
            f'Algorithm: Differential Evolution; mutation={self.mutation}, recombination={self.recombination}, strategy={self.strategy}.\n'
            f'Pattern: {self.pattern_type}\n'
            f'Best Pattern: {self.pattern_type if hasattr(self, "best_pattern") else "N/A"}'
        )
        plt.grid(True)

        filename = self._generate_plot_filename("optimization_history", extra_params)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        # plt.show()

    def plot_multiple_patterns_optimization_history(self, results, best_pattern, extra_params=None):
        """
        Plot optimization history for multiple patterns to compare their convergence.
        """
        if 'pattern_histories' not in results:
            print("No pattern histories found in results. Cannot plot comparison.")
            return
            
        plt.figure(figsize=(14, 8))

        # plt.tight_layout(rect=[0.0, 0.0, 0.99, 0.6])  # The rect parameter reserves space [left, bottom, right, top]
        plt.subplots_adjust(top=0.85, bottom=0.07, left=0.06, right=0.97)
        # plt.rcParams['figure.constrained_layout.use'] = True

        # Color map and marker styles for different patterns
        styles = {
            'Walker-Delta': {'color': '#3498db', 'marker': 'o', 'linestyle': ':', 'linewidth': 2.5, 'markersize': 6},
            'Flower': {'color': '#e74c3c', 'marker': 's', 'linestyle': '--', 'linewidth': 2.5, 'markersize': 6},
            'Other': {'color': '#2ecc71', 'marker': '^', 'linestyle': '-.', 'linewidth': 2.5, 'markersize': 6}
        }
        
        # Debug output to verify histories
        print("\nPattern Optimization Histories:")

        # Define fixed label positions for each pattern
        label_positions = {
            'Walker-Delta': {'offset': 0.01, 'va': 'bottom'},  # Walker-Delta labels go above
            'Flower': {'offset': -0.01, 'va': 'top'}           # Flower labels go below
        }

        for pattern, history in results['pattern_histories'].items():
            print(f"{pattern}: {len(history)} evaluations")
            if history:
                scores = [item[1] for item in history]
                print(f"  Scores: {scores}")
                
                # Use distinct styling for each pattern
                style = styles.get(pattern, styles['Other'])
                
                # Plot with enhanced visibility
                plt.plot(range(len(scores)), scores, 
                        label=f"{pattern} (Best: {results[pattern]['score']:.4f})",
                        **style)
                
        # Now add labels in alternating fashion
        patterns = list(results['pattern_histories'].keys())
        if len(patterns) >= 2:  # Only do alternating if we have at least two patterns
            for i in range(max(len(results['pattern_histories'][pattern]) for pattern in patterns)):
                # For each index position, only label one pattern
                pattern_to_label = patterns[i % len(patterns)]
                
                if i < len(results['pattern_histories'][pattern_to_label]):
                    history = results['pattern_histories'][pattern_to_label]
                    score = history[i][1]
                    
                    pos = label_positions.get(pattern_to_label, {'offset': 0.01, 'va': 'bottom'})

                plt.text(i, score + pos['offset'], f'{score:.3f}', ha='center', va=pos['va'], fontsize=8)
                    
        plt.xlabel('Evaluation Number', fontsize=12)
        plt.ylabel('Objective Function Score', fontsize=12)
        plt.title(
            f'Multi-Pattern Optimization Comparison, step: {self.step} s\n'
            f'Weights: L={self.local_weight}, R={self.regional_weight}, G={self.global_weight}, '
            f'SatCost={self.satellite_cost_weight}, OpCost={self.operational_cost_weight}, LaunchCost={self.launch_cost_weight}, Sci={self.science_weight}\n'
            f'Target Altitude: {self.target_altitude} km\n'
            f'Algorithm: Differential Evolution; mutation={self.mutation}, recombination={self.recombination}, strategy={self.strategy}\n'
            f'Best Pattern: {best_pattern}',
            fontsize=13
        )
        plt.grid(True, alpha=0.7)
        plt.legend(loc='best', fontsize=12)
        
        # Add a horizontal line for the best score
        best_score = results[best_pattern]['score']
        plt.axhline(y=best_score, color='k', linestyle=':', linewidth=2,
                    label=f'Best Score: {best_score:.4f}')
        
        # Ensure both patterns are visible by setting y-axis limits with padding
        scores_all = []
        for pattern, history in results['pattern_histories'].items():
            if history:
                scores_all.extend([item[1] for item in history])
        
        if scores_all:
            ymin = min(scores_all) - 0.05
            ymax = max(scores_all) + 0.05
            plt.ylim(ymin, ymax)
        
        # Save the plot
        filename = self._generate_plot_filename("multi_pattern_history", extra_params)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        
        return plt 

    def _repair_walker_solution(self, x):
        """
        Repair a Walker-Delta solution vector so it satisfies:
        - P >= 1
        - N >= P and N % P == 0
        - 0 <= F < P and, if F > 0, N % F == 0 (reduce F if needed)
        Returns (x_fixed, {'N': N, 'P': P, 'F': F})
        """
        xx = np.array(x, dtype=float).copy()
        N_raw = int(round(xx[0]))
        P = max(1, int(round(xx[6])))
        # Ensure N >= P and divisible by P
        N = N_raw
        if N < P:
            N = P
        else:
            rem = N % P
            if rem != 0:
                N = N - rem
                if N < P:
                    N = P
        # Repair F
        F_raw = int(round(xx[7]))
        F = min(max(0, F_raw), P - 1)
        if F > 0:
            while N % F != 0 and F > 0:
                F -= 1
        # Write back
        xx[0] = float(N)
        xx[6] = float(P)
        xx[7] = float(F)
        return xx, {'N': N, 'P': P, 'F': F}

if __name__ == "__main__":

    # Example of multi-pattern optimization
    print("\n=== EXAMPLE 4: MULTI-PATTERN OPTIMIZATION ===")
    multi_optimizer = ConstellationOptimizer(
        pattern_type="Walker-Delta",  # Initial pattern (will be changed during optimization)
        gs="AFB_Andersen",
        local_weight=0.2,
        regional_weight=0.3,
        global_weight=0.5,
        satellite_cost_weight=0.3,
        operational_cost_weight=0.1,
        launch_cost_weight=0.1,
        science_weight=0.6,
        target_altitude=600.0,
        step=1000
    )

    # Compare Walker-Delta and Flower patterns
    results, best_pattern = multi_optimizer.optimize_multiple_patterns(
        patterns=["Walker-Delta", "Flower"],
        method='differential_evolution',
        max_iter=5,
        popsize=15,
        strategy='randtobest1exp', # 'best1bin','best1exp','rand1bin','rand1exp','rand2bin','rand2exp','randtobest1bin','randtobest1exp','currenttobest1bin','currenttobest1exp','best2exp','best2bin'
        mutation=(0.5, 0.7),
        recombination=0.7,
        seed=100,
        early_stopping=True, # If set to false the plots will crash because history is not recorded (optimization still works). Probably because no callback is created and all obj functions calls happend in worker processes only, which do not transfer back to main process.
        patience=80,
        improvement_threshold=0.001,
        workers=4  # Use all available CPU cores
    )

    print(f"Best pattern: {best_pattern} with score {results[best_pattern]['score']:.4f}")

    best_orbital_params = results[best_pattern]['orbital_params']
    best_pattern_params = results[best_pattern]['pattern_params']
    print(f"Altitude: {best_orbital_params['semi_major_axis'] - constants.EARTH_RADIUS:.1f} km")
    print(f"Inclination: {best_orbital_params['inclination']:.1f}°")
    print(f"Number of satellites: {results[best_pattern]['num_satellites']}")
    print(f"Pattern parameters: {best_pattern_params}")

    multi_optimizer.plot_multiple_patterns_optimization_history(results, best_pattern)


    # def _calculate_operational_cost(self, orbital_parameters):
    #     """
    #     Calculate operational cost score based on orbital parameters.
    #     Lower altitude = quicker deorbiting = lower operational cost = higher score.
        
    #     Station keeping costs are weighted 3x more heavily than other operational cost components
    #     to reflect their significant impact on mission operations.
        
    #     Parameters:
    #     -----------
    #     orbital_parameters : dict
    #         Orbital parameters
            
    #     Returns:
    #     --------
    #     operational_cost_score : float
    #         Score for operational cost (0-1, higher is better)
    #     """
    #     # IMPROVED: More detailed operational cost model with weighted components
        
    #     # Main factors affecting operational costs:
    #     # 1. Station-keeping requirements (affected by altitude and eccentricity) - WEIGHTED 3X
    #     # 2. Ground station contact time (affected by altitude and inclination)
    #     # 3. Deorbiting requirements (affected by altitude)
    #     # 4. Satellite lifetime (affected by altitude)
        
    #     # Calculate satellite lifetime based on altitude
    #     altitude = orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS
    #     eccentricity = orbital_parameters['eccentricity']
    #     inclination = orbital_parameters['inclination']
        
    #     # Satellite lifetime increases with altitude
    #     # LEO (300-500km): ~2-5 years
    #     # LEO (500-800km): ~5-10 years
    #     # LEO (800-1200km): ~10-15 years
    #     # MEO (>1200km): ~15+ years
    #     if altitude < 500:
    #         lifetime_years = 2 + (altitude - 300) / 50
    #     elif altitude < 800:
    #         lifetime_years = 5 + (altitude - 500) / 60
    #     elif altitude < 1200:
    #         lifetime_years = 10 + (altitude - 800) / 80
    #     else:
    #         lifetime_years = 15 + (altitude - 1200) / 200
            
    #     # Cap at 25 years
    #     lifetime_years = min(lifetime_years, 25)
        
    #     # Calculate station-keeping costs
    #     # Higher altitude = less drag = less station-keeping
    #     # But higher eccentricity = more station-keeping
    #     station_keeping_factor = 1.0 + eccentricity * 10.0
    #     if altitude < 500:
    #         station_keeping_cost = 2.0 * station_keeping_factor  # High
    #     elif altitude < 800:
    #         station_keeping_cost = 1.0 * station_keeping_factor  # Medium
    #     else:
    #         station_keeping_cost = 0.5 * station_keeping_factor  # Low
            
    #     # Ground station contact costs
    #     # Higher altitude = longer contact times = lower contact costs
    #     if altitude < 500:
    #         contact_cost = 1.0  # High
    #     elif altitude < 1000:
    #         contact_cost = 0.7  # Medium
    #     else:
    #         contact_cost = 0.5  # Low
            
    #     # Deorbiting costs (higher altitude = more expensive to deorbit)
    #     if altitude < 500:
    #         deorbit_cost = 0.1  # Natural decay is fast
    #     elif altitude < 800:
    #         deorbit_cost = 0.3  # Some propulsion needed
    #     elif altitude < 1200:
    #         deorbit_cost = 0.6  # Significant propulsion needed
    #     else:
    #         deorbit_cost = 1.0  # Very expensive
            
    #     # Annual operational cost with weighted components
    #     # Station keeping is weighted 3x more heavily than other components to reflect its importance
    #     station_keeping_weight = 3.0
    #     contact_weight = 1.0
    #     deorbit_weight = 1.0
        
    #     annual_cost = (station_keeping_weight * station_keeping_cost + 
    #                   contact_weight * contact_cost + 
    #                   deorbit_weight * (deorbit_cost / lifetime_years)) / (station_keeping_weight + contact_weight + deorbit_weight)
        
    #     # Total cost over lifetime
    #     total_operational_cost = annual_cost * lifetime_years
        
    #     # Normalize score (higher is better = lower operational cost)
    #     # Calibrated to give ~0.8 for 500km, ~0.5 for 1000km, ~0.2 for 2000km
    #     operational_cost_score = 1.0 / (1.0 + (total_operational_cost / 10.0))
        
    #     # Return score and detailed breakdown for debug purposes
    #     operational_cost_details = {
    #         'station_keeping_cost': station_keeping_cost,
    #         'contact_cost': contact_cost,
    #         'deorbit_cost': deorbit_cost,
    #         'lifetime_years': lifetime_years,
    #         'annual_cost': annual_cost,
    #         'total_operational_cost': total_operational_cost,
    #         'operational_cost_score': operational_cost_score,
    #         'weights': {
    #             'station_keeping_weight': 3.0,
    #             'contact_weight': 1.0,
    #             'deorbit_weight': 1.0
    #         }
    #     }
        
    #     # Store the details for access in analyze_component_scores
    #     self._operational_cost_details = operational_cost_details
        
    #     return operational_cost_score
