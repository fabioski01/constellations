"""
# Satellite Constellation Optimizer

This module provides tools for optimizing satellite constellation configurations based on multiple
weighted objectives. The optimizer can balance trade-offs between:

## Coverage Metrics
- Local coverage (ground station visibility)
- Regional coverage (specific geographical area)
- Global coverage (Earth-wide coverage estimation)

## Cost Factors
- Satellite costs (including economies of scale)
- Operational costs (station-keeping, deorbiting, communication)
- Launch costs (based on altitude, inclination, launch vehicle capacity)

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
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Import from your existing files
from utilities_constellations import Constellation
from utilities_coverage import Coverage, get_ground_station_coordinates
import constants

class ConstellationOptimizer:
    """
    Optimizer for satellite constellation parameters based on multiple weighted objectives.
    
    This class provides a comprehensive framework for optimizing satellite constellation designs
    by balancing multiple competing objectives:
    
    1. Coverage metrics: local (ground station), regional (specific area), and global coverage
    2. Cost factors: satellite manufacturing, operational lifetime, and launch costs
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
                 local_weight=0.4,      # Ground station visibility weight
                 regional_weight=0.3,   # Regional coverage weight
                 global_weight=0.3,     # Global coverage weight
                 # Cost weights
                 satellite_cost_weight=0.2,      # Cost of satellites
                 operational_cost_weight=0.1,    # Operational costs (deorbiting, station keeping)
                 launch_cost_weight=0.1,         # Launch costs (higher for higher orbits)
                 # Science requirements
                 science_weight=0.8,             # Science requirements weight
                 target_altitude=500.0,          # Target altitude for science instruments (km)
                 target_altitude_tolerance=50.0, # Tolerance for target altitude (km)
                 # Regional coverage area (Western US as default)
                 regional_area=None              # List of (lat, lon) coordinates defining region
                 ):
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
            
        Science Requirements:
        -------------------
        science_weight : float (0-1)
            Weight for science requirements
        target_altitude : float
            Target altitude in km for science instruments
        target_altitude_tolerance : float
            Tolerance range for target altitude in km
            
        Regional Coverage Area:
        ---------------------
        regional_area : list of tuples
            List of (lat, lon) coordinates defining the region of interest
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
            pattern_params['P'] = int(round(params[6]))
            pattern_params['F'] = int(round(params[7]))
            
            # Ensure P and F are valid values
            pattern_params['P'] = max(1, pattern_params['P']) # Number of planes must be at least 1
            pattern_params['F'] = min(max(1, pattern_params['F']), pattern_params['P']-1) # Phasing parameter must be in [1, P-1]
        
            # Walker-Delta requires: 
            # 1. T (total satellites) divisible by P (number of planes)
            # 2. T (total satellites) divisible by F (phasing parameter)
            # these constraints are checked in the objective function
        
        elif self.pattern_type == "Flower":
            pattern_params['Np'] = int(round(params[6]))  # Number of petals (lobes in ground track).
            pattern_params['Nd'] = int(round(params[7]))  # Number of sidereal days for ground track repeat.
            pattern_params['Fn'] = int(round(params[8]))  # Phase number, phasing integer, must be relatively prime to Fd.
            pattern_params['Fd'] = int(round(params[9]))  # Phase denominator, phasing integer, must be relatively prime to Fn.
            pattern_params['Fh'] = int(round(params[10])) # Step phasing integer, must be in the range [0, Nd-1], can be non constant. In this case will be constant, and usually zero.

            # Ensure parameters are valid
            pattern_params['Np'] = max(1, pattern_params['Np']) # Number of petals must be at least 1
            pattern_params['Nd'] = max(1, pattern_params['Nd']) # Number of sidereal days must be at least 1
            pattern_params['Fn'] = max(1, pattern_params['Fn']) # Phase number must be at least 1
            pattern_params['Fd'] = max(1, pattern_params['Fd']) # Phase denominator must be at least 1
            pattern_params['Fh'] = min(max(0, pattern_params['Fh']), pattern_params['Nd']-1) # Step phasing must be in [0, Nd-1]
            
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
            # Satellite cost (more satellites = higher cost)
            satellite_cost_score = self._calculate_satellite_cost(num_satellites)
            
            # Operational cost (higher altitude = higher lifetime = higher cost)
            operational_cost_score = self._calculate_operational_cost(orbital_parameters)
            
            # Launch cost (higher altitude/inclination = higher cost)
            launch_cost_score = self._calculate_launch_cost(orbital_parameters, num_satellites)
            
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
        return CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(eci_position, timestamp)
            
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
            bounds.append((0, 0))    # Fh (step phasing). Used for combinations. Usually zero
        
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
            # Create Coverage object with existing positions and timestamps
            coverage_obj = Coverage(orbital_parameters, positions=positions, times=time_stamps)
            
            # Get region from self.regional_area
            region = self.regional_area
            
            # Compute regional coverage with appropriate parameters
            regional_result = coverage_obj.compute_regional_coverage(
                positions, time_stamps,
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
            # Create Coverage object with existing positions and timestamps
            coverage_obj = Coverage(orbital_parameters, positions=positions, times=time_stamps)
            
            # Compute global coverage with appropriate parameters
            # Use lower grid resolution for optimization (faster)
            global_coverage_result = coverage_obj.compute_global_coverage(
                positions, time_stamps,
                orbital_periods=1,
                masking_angle=15.0,  # Default masking angle
                fov=120.0,          # Default FOV
                grid_density=15.0   # Coarser grid for optimization speed
            )
            
            # Extract the coverage percentage and normalize to 0-1
            coverage_percentage = global_coverage_result['constellation_coverage'] / 100.0
            
            return min(coverage_percentage, 1.0)  # Ensure the score is between 0 and 1
        
        except Exception as e:
            print(f"Warning: Error in global coverage calculation: {str(e)}")
            # Fallback to simplified estimate
            return self._estimate_global_coverage(num_satellites, orbital_parameters)
        
    def _calculate_satellite_cost(self, num_satellites):
        """
        Calculate cost score based on number of satellites.
        More satellites = higher cost = lower score.
        
        Parameters:
        -----------
        num_satellites : int
            Number of satellites
            
        Returns:
        --------
        satellite_cost_score : float
            Score for satellite cost (0-1, higher is better)
        """
        # IMPROVED: More advanced satellite cost model
        
        # Base satellite cost (can be adjusted based on satellite complexity)
        # Small satellite (~100kg): $1-5M
        # Medium satellite (~300kg): $10-50M
        # Large satellite (>500kg): $50-300M
        base_satellite_cost = 10.0  # Million $ per satellite (default: medium smallsat)
        
        # Manufacturing economies of scale
        # First few satellites are expensive, then cost per satellite decreases
        if num_satellites <= 5:
            manufacturing_factor = 1.0  # No economies of scale for small numbers
        elif num_satellites <= 20:
            # Medium economies of scale (0.85 = 15% reduction for large orders)
            manufacturing_factor = 1.0 - 0.15 * (num_satellites - 5) / 15.0
        else:
            # Maximum economies of scale (20% reduction)
            manufacturing_factor = 0.8
            
        # Batch production costs
        batch_cost = num_satellites * base_satellite_cost * manufacturing_factor
        
        # Development and non-recurring engineering costs
        # These are high for first few satellites but amortize with larger constellations
        dev_cost = 50.0 + 10.0 * np.log10(num_satellites)  # Million $
        
        # Total cost
        total_cost = batch_cost + dev_cost  # Million $
        
        # Normalize score (higher is better = lower cost)
        # Calibrated to give ~0.5 for 50 satellites costing ~$500M
        cost_score = 1.0 / (1.0 + (total_cost / 500.0))
        
        return cost_score
        
    def _calculate_operational_cost(self, orbital_parameters):
        """
        Calculate operational cost score based on orbital parameters.
        Lower altitude = quicker deorbiting = lower operational cost = higher score.
        
        Station keeping costs are weighted 3x more heavily than other operational cost components
        to reflect their significant impact on mission operations.
        
        Parameters:
        -----------
        orbital_parameters : dict
            Orbital parameters
            
        Returns:
        --------
        operational_cost_score : float
            Score for operational cost (0-1, higher is better)
        """
        # IMPROVED: More detailed operational cost model with weighted components
        
        # Main factors affecting operational costs:
        # 1. Station-keeping requirements (affected by altitude and eccentricity) - WEIGHTED 3X
        # 2. Ground station contact time (affected by altitude and inclination)
        # 3. Deorbiting requirements (affected by altitude)
        # 4. Satellite lifetime (affected by altitude)
        
        # Calculate satellite lifetime based on altitude
        altitude = orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS
        eccentricity = orbital_parameters['eccentricity']
        inclination = orbital_parameters['inclination']
        
        # Satellite lifetime increases with altitude
        # LEO (300-500km): ~2-5 years
        # LEO (500-800km): ~5-10 years
        # LEO (800-1200km): ~10-15 years
        # MEO (>1200km): ~15+ years
        if altitude < 500:
            lifetime_years = 2 + (altitude - 300) / 50
        elif altitude < 800:
            lifetime_years = 5 + (altitude - 500) / 60
        elif altitude < 1200:
            lifetime_years = 10 + (altitude - 800) / 80
        else:
            lifetime_years = 15 + (altitude - 1200) / 200
            
        # Cap at 25 years
        lifetime_years = min(lifetime_years, 25)
        
        # Calculate station-keeping costs
        # Higher altitude = less drag = less station-keeping
        # But higher eccentricity = more station-keeping
        station_keeping_factor = 1.0 + eccentricity * 10.0
        if altitude < 500:
            station_keeping_cost = 2.0 * station_keeping_factor  # High
        elif altitude < 800:
            station_keeping_cost = 1.0 * station_keeping_factor  # Medium
        else:
            station_keeping_cost = 0.5 * station_keeping_factor  # Low
            
        # Ground station contact costs
        # Higher altitude = longer contact times = lower contact costs
        if altitude < 500:
            contact_cost = 1.0  # High
        elif altitude < 1000:
            contact_cost = 0.7  # Medium
        else:
            contact_cost = 0.5  # Low
            
        # Deorbiting costs (higher altitude = more expensive to deorbit)
        if altitude < 500:
            deorbit_cost = 0.1  # Natural decay is fast
        elif altitude < 800:
            deorbit_cost = 0.3  # Some propulsion needed
        elif altitude < 1200:
            deorbit_cost = 0.6  # Significant propulsion needed
        else:
            deorbit_cost = 1.0  # Very expensive
            
        # Annual operational cost with weighted components
        # Station keeping is weighted 3x more heavily than other components to reflect its importance
        station_keeping_weight = 3.0
        contact_weight = 1.0
        deorbit_weight = 1.0
        
        annual_cost = (station_keeping_weight * station_keeping_cost + 
                      contact_weight * contact_cost + 
                      deorbit_weight * (deorbit_cost / lifetime_years)) / (station_keeping_weight + contact_weight + deorbit_weight)
        
        # Total cost over lifetime
        total_operational_cost = annual_cost * lifetime_years
        
        # Normalize score (higher is better = lower operational cost)
        # Calibrated to give ~0.8 for 500km, ~0.5 for 1000km, ~0.2 for 2000km
        operational_cost_score = 1.0 / (1.0 + (total_operational_cost / 10.0))
        
        # Return score and detailed breakdown for debug purposes
        operational_cost_details = {
            'station_keeping_cost': station_keeping_cost,
            'contact_cost': contact_cost,
            'deorbit_cost': deorbit_cost,
            'lifetime_years': lifetime_years,
            'annual_cost': annual_cost,
            'total_operational_cost': total_operational_cost,
            'operational_cost_score': operational_cost_score,
            'weights': {
                'station_keeping_weight': 3.0,
                'contact_weight': 1.0,
                'deorbit_weight': 1.0
            }
        }
        
        # Store the details for access in analyze_component_scores
        self._operational_cost_details = operational_cost_details
        
        return operational_cost_score
        
    def _calculate_launch_cost(self, orbital_parameters, num_satellites):
        """
        Calculate launch cost score based on a realistic launch model.
        
        This function models launch costs considering multiple factors:
        - Altitude (higher orbits require more fuel/energy)
        - Inclination (polar and sun-synchronous orbits are more expensive)
        - Satellite mass (estimated based on altitude)
        - Launch vehicle selection based on payload capacity
        - Number of satellites per launch and total launches needed
        
        Higher scores (closer to 1.0) represent lower launch costs.
        
        Parameters:
        -----------
        orbital_parameters : dict
            Orbital parameters including semi_major_axis and inclination
        num_satellites : int
            Number of satellites in the constellation
            
        Returns:
        --------
        launch_cost_score : float
            Normalized score for launch cost (0-1, higher is better)
        """
        # IMPROVED: More realistic launch cost model
        
        # Main factors affecting launch costs:
        # 1. Altitude (higher = more fuel = higher cost)
        # 2. Inclination (higher = more fuel = higher cost)
        # 3. Number of satellites per launch (more satellites = more launches)
        # 4. Launch vehicle selection
        
        # Get orbital parameters
        altitude = orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS
        inclination = orbital_parameters['inclination']
        
        # Base cost per kg to orbit (approximate values in $K/kg)
        # Small: Electron ($20K/kg), Medium: Falcon 9 ($5K/kg), Heavy: Falcon Heavy ($3K/kg)
        if altitude < 600:  # LEO
            base_cost_per_kg = 5.0  # Thousand $ per kg
        elif altitude < 1500:  # MEO
            base_cost_per_kg = 8.0  # Thousand $ per kg
        else:  # HEO
            base_cost_per_kg = 12.0  # Thousand $ per kg
            
        # Inclination penalty (equatorial is cheapest, polar is expensive)
        # Delta-v costs increase with inclination change
        if inclination < 30:
            inclination_factor = 1.0
        elif inclination < 60:
            inclination_factor = 1.1
        elif inclination < 90:
            inclination_factor = 1.2
        else:
            inclination_factor = 1.3
            
        # Special cases
        if abs(inclination - 98) < 3:
            # Sun-synchronous orbit penalty
            inclination_factor *= 1.1
        
        # Estimate satellite mass based on altitude (higher orbit = more propulsion = heavier)
        # Typical smallsat masses: 100-300kg for LEO, 300-500kg for MEO
        if altitude < 600:
            sat_mass = 150  # kg
        elif altitude < 1200:
            sat_mass = 250  # kg
        else:
            sat_mass = 400  # kg
            
        # Determine satellites per launch based on launch vehicle payload capacity
        # Small: ~300kg, Medium: ~15,000kg, Heavy: ~30,000kg to LEO
        if sat_mass < 200:
            # Smallsats
            if num_satellites <= 3:
                # Small launcher (Electron, Astra)
                launch_capacity = 300  # kg
                launches_needed = max(1, np.ceil(num_satellites * sat_mass / launch_capacity))
            else:
                # Medium launcher with rideshare (Falcon 9)
                launch_capacity = 15000  # kg
                launches_needed = max(1, np.ceil(num_satellites * sat_mass / launch_capacity))
        else:
            # Larger satellites
            if num_satellites <= 10:
                # Medium launcher (Falcon 9)
                launch_capacity = 15000  # kg
                launches_needed = max(1, np.ceil(num_satellites * sat_mass / launch_capacity))
            else:
                # Heavy launcher (Falcon Heavy)
                launch_capacity = 30000  # kg
                launches_needed = max(1, np.ceil(num_satellites * sat_mass / launch_capacity))
        
        # Calculate total launch cost
        cost_per_launch = base_cost_per_kg * inclination_factor * launch_capacity
        total_launch_cost = launches_needed * cost_per_launch  # Thousand $
        
        # Calculate per-satellite launch cost
        per_sat_launch_cost = total_launch_cost / num_satellites if num_satellites > 0 else total_launch_cost
        
        # Normalize score (higher is better = lower launch cost)
        # Typical launch costs might range from $5M to $100M+ for a constellation
        launch_cost_score = 1.0 / (1.0 + (per_sat_launch_cost / 2000.0))  # 2000 K$ = $2M per satellite
        
        return launch_cost_score
        
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
        Calculate science score component based on inclination.
        Some missions have specific inclination requirements.
        
        Parameters:
        -----------
        inclination : float
            Orbit inclination in degrees
            
        Returns:
        --------
        inclination_score : float
            Score for inclination (0-1, higher is better)
        """
        # Example: Prefer sun-synchronous orbit for Earth observation (around 98°)
        # This could be customized or parameterized based on mission needs
        
        # Sun-synchronous orbit preference (peak at 98°)
        if abs(inclination - 98.0) < 3.0:
            return 1.0
        elif abs(inclination - 98.0) < 10.0:
            return 0.8
        
        # For global coverage missions, higher inclinations are better
        if inclination > 80:
            return 0.9
        elif inclination > 60:
            return 0.7
        elif inclination > 40:
            return 0.5
        else:
            return 0.3

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
        operational_cost_score = self._calculate_operational_cost(orbital_parameters)
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
            weighted_cost = 1.0
            
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
        target_altitude=610.0,
        step=1000
    )

    # Compare Walker-Delta and Flower patterns
    results, best_pattern = multi_optimizer.optimize_multiple_patterns(
        patterns=["Walker-Delta", "Flower"],
        method='differential_evolution',
        max_iter=100,
        popsize=150,
        strategy='rand1bin', # 'best1bin','best1exp','rand1bin','rand1exp','rand2bin','rand2exp','randtobest1bin','randtobest1exp','currenttobest1bin','currenttobest1exp','best2exp','best2bin'
        mutation=(0.5, 0.7),
        recombination=0.7,
        seed=100,
        early_stopping=True, # If set to false the plots will crash because history is not recorded (optimization still works). Probably because no callback is created and all obj functions calls happend in worker processes only, which do not transfer back to main process.
        patience=80,
        improvement_threshold=0.001,
        workers=-1  # Use all available CPU cores
    )

    print(f"Best pattern: {best_pattern} with score {results[best_pattern]['score']:.4f}")

    best_orbital_params = results[best_pattern]['orbital_params']
    best_pattern_params = results[best_pattern]['pattern_params']
    print(f"Altitude: {best_orbital_params['semi_major_axis'] - constants.EARTH_RADIUS:.1f} km")
    print(f"Inclination: {best_orbital_params['inclination']:.1f}°")
    print(f"Number of satellites: {results[best_pattern]['num_satellites']}")
    print(f"Pattern parameters: {best_pattern_params}")

    multi_optimizer.plot_multiple_patterns_optimization_history(results, best_pattern)
