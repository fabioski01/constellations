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
- Wertz, J.R. "Mission Geometry: Orbit and Constellation Design and Management", 2001
- Larson, W. and Wertz, J. "Space Mission Analysis and Design", 1999
- Ballard, A.H. "Rosette Constellations of Earth Satellites", 1980

## Author
Fabio Meloni - TU Delft MSc Student - NASA Jet Propulsion Laboratory Visiting Researcher

## Copyright
Copyright © 2025 Fabio Meloni, TU Delft, and NASA Jet Propulsion Laboratory
All rights reserved. This software is the property of the author and the organizations listed above.
This software may not be copied, distributed, or modified without express written permission.
For research and educational purposes only.

## Version
2.0.0 - June 2025 - Comprehensive modular optimization framework
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
                 start_time=None, end_time=None, step=105,
                 gs="Fucino", elevation_threshold=10.0,
                 # Coverage weights
                 local_weight=0.4,      # Ground station visibility weight
                 regional_weight=0.3,   # Regional coverage weight
                 global_weight=0.3,     # Global coverage weight
                 # Cost weights
                 satellite_cost_weight=0.2,      # Cost of satellites
                 operational_cost_weight=0.1,    # Operational costs (deorbiting, station keeping)
                 launch_cost_weight=0.1,         # Launch costs (higher for higher orbits)
                 # Science requirements
                 science_weight=0.2,             # Science requirements weight
                 target_altitude=600.0,          # Target altitude for science instruments (km)
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
        
        # Set default regional area (Western US) if none provided
        if regional_area is None:
            # Simple bounding box for Western US
            self.regional_area = [
                (32.0, -125.0),  # Southwest corner
                (32.0, -105.0),  # Southeast corner
                (49.0, -105.0),  # Northeast corner
                (49.0, -125.0)   # Northwest corner
            ]
        else:
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
        semi_major_axis = params[1]
        eccentricity = params[2]
        inclination = params[3]
        RAAN = params[4]
        arg_periapsis = params[5]
        
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
        
        # For Walker-Delta pattern, extract P and F
        pattern_params = {}
        if self.pattern_type == "Walker-Delta":
            pattern_params['P'] = int(round(params[6]))
            pattern_params['F'] = int(round(params[7]))
            
            # Ensure P and F are valid values
            pattern_params['P'] = max(1, pattern_params['P'])
            pattern_params['F'] = min(max(1, pattern_params['F']), pattern_params['P']-1)
            
            # Walker-Delta requires: 
            # 1. T (total satellites) divisible by P (number of planes)
            # 2. T (total satellites) divisible by F (phasing parameter)
            # We'll check these constraints in the objective function
        
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
                return 1e10  # Invalid number of satellites
                
            if orbital_parameters['eccentricity'] < 0 or orbital_parameters['eccentricity'] >= 1:
                return 1e10  # Invalid eccentricity
                
            if orbital_parameters['semi_major_axis'] < constants.EARTH_RADIUS + 200:
                return 1e10  # Too low orbit
            
            # Create the constellation
            epoch = datetime(2025, 5, 5, 0, 0, 0)
            
            # For Walker-Delta, handle P and F
            if self.pattern_type == "Walker-Delta":
                P = pattern_params['P']
                F = pattern_params['F']
                
                # Validate P and F
                if P < 1 or F < 1 or F >= P:
                    return 1e10
                
                # Check if number of satellites is compatible with P
                if num_satellites < P or num_satellites % P != 0:
                    return 1e10
                    
                # Check if number of satellites (T) is divisible by F
                if num_satellites % F != 0:
                    return 1e10  # Total number of satellites (T) must be divisible by phasing (F)
                    
                # Create constellation
                const = Constellation(
                    num_satellites, orbital_parameters, epoch, 
                    self.pattern_type, P=P, F=F
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
            
            positions = result["positions"]  # Shape: (num_satellites, num_time_steps, 3)
            time_stamps = result["time_stamps"]
            
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
                const, orbital_parameters, num_satellites, epoch, orbital_period
            )
            
            # Global coverage
            global_score = self._calculate_global_coverage(
                num_satellites, orbital_parameters
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
            print(f"Error in objective function: {str(e)}")
            return 1e10  # Return a high value for errors
    
    def _eci_to_ecef(self, eci_position, timestamp):
        """Convert ECI position to ECEF."""
        from utilities import CoordinateConversion
        return CoordinateConversion.ECI_cartesian_to_ECEF_cartesian(eci_position, timestamp)
    
    def _estimate_global_coverage(self, num_satellites, orbital_parameters):
        """
        Estimate global coverage based on constellation parameters. Very fast and simplified model.
        (more satellites → better coverage, higher inclination → better polar coverage)

        In satellite constellation literature, global coverage is typically calculated using more rigorous methods:
        -   Grid-based methods: Dividing Earth into cells and calculating the percentage of cells with visibility
            -   Papers by Wertz, Larson & Wertz "Space Mission Analysis & Design"
            -   Walker's original papers on satellite constellations
        -   Street of coverage: Calculating the minimum number of satellites needed for continuous coverage
            -   Rider's work on polar orbiting satellites
            -   Ballard's extensions to Walker's constellation theory
        -   Analytical coverage models: Using spherical geometry to determine coverage
            -   Extensive work by Luders & Ginsberg on optimal satellite constellation design

        To be implemented:    
            A grid-based Earth model (typically 1-5° resolution)
            Ray-tracing or geometric calculations for satellite visibility at each grid point
            Time-step evaluation to calculate coverage over a full orbital period
        Coverage metrics like:
            Mean coverage time
            Percentage of Earth covered
            Maximum revisit time
        """
        # Simple model: estimate coverage based on number of satellites, inclination, and altitude
        inclination = orbital_parameters['inclination']
        altitude = orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS
        
        # Higher inclination gives better polar coverage (up to 90°)
        inclination_factor = min(inclination, 180 - inclination) / 90.0
        
        # Higher altitude gives wider coverage area per satellite
        altitude_factor = np.sqrt(altitude / 2000)  # Normalized to ~1 at 2000km
        
        # Coverage estimate: more satellites and better distribution = better coverage
        coverage_estimate = (num_satellites * altitude_factor * inclination_factor) / 50
        
        # Cap at 1.0 (full coverage)
        return min(coverage_estimate, 1.0)
        
    def optimize(self, method='differential_evolution', max_iter=50, popsize=15, strategy='best1bin', 
                 mutation=(0.5, 0.7), recombination=0.7, seed=None, early_stopping=True, patience=20, 
                 improvement_threshold=0.001):
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
        mutation : float or tuple(float, float)
            Mutation constant. If specified as a tuple (min, max), adaptive mutation is used.
            Larger values (0.8-1.0) give more exploration, smaller values more exploitation.
        recombination : float
            Recombination probability, between 0 and 1. Higher values mean more mixing.
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
        # Define parameter bounds
        bounds = [
            (1, 100),             # num_satellites (integer between 1 and 100)
            (6500, 8500),         # semi_major_axis (km)
            (0, 0.01),            # eccentricity (close to circular)
            (0, 90),              # inclination (deg)
            (0, 360),             # RAAN (deg)
            (0, 360),             # argument_of_periapsis (deg)
        ]
        
        # Add bounds for P and F if using Walker-Delta
        if self.pattern_type == "Walker-Delta":
            bounds.append((1, 20))  # P (planes) - will be constrained by num_satellites
            bounds.append((1, 20))  # F (phase) - will be constrained by P
        
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
                callback=callback       # Early stopping callback
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
        
        return orbital_parameters, pattern_params, num_satellites, best_score
    
    def plot_optimization_history(self):
        """Plot the optimization history to see convergence."""
        if not self.optimization_history:
            print("No optimization history to plot.")
            return
        
        # Extract scores from history
        # Each item in history is (params, combined_score, component_scores)
        # Note: we use the original positive scores for plotting (before negation for minimization)
        scores = [item[1] for item in self.optimization_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(scores, 'o-')
        plt.xlabel('Evaluation Number')
        plt.ylabel('Combined Score (higher is better)')
        plt.title('Optimization Progress')
        plt.grid(True)
        
        # Save the plot
        filename = f"plots/optimization/optimization_history.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        # plt.show()
    
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
    
    def _calculate_regional_coverage(self, constellation, orbital_parameters, num_satellites, epoch, orbital_period):
        """
        Calculate coverage metrics for a regional area.
        
        Parameters:
        -----------
        constellation : Constellation object
            Constellation object
        orbital_parameters : dict
            Orbital parameters
        num_satellites : int
            Number of satellites
        epoch : datetime
            Epoch for constellation
        orbital_period : float
            Orbital period in seconds
            
        Returns:
        --------
        regional_score : float
            Score for regional coverage (0-1)
        """
        # IMPROVED: Grid-based regional coverage calculation
        # 1. Create a grid of points across the regional area
        # 2. Calculate visibility at each grid point
        # 3. Calculate percentage of region with coverage
        
        # Extract region boundaries
        lat_min = min(p[0] for p in self.regional_area)
        lat_max = max(p[0] for p in self.regional_area)
        lon_min = min(p[1] for p in self.regional_area)
        lon_max = max(p[1] for p in self.regional_area)
        
        # Grid resolution (higher = more accurate but slower)
        grid_resolution = 2  # degrees
        
        # Generate a grid of lat/lon points within the region
        lats = np.arange(lat_min, lat_max + grid_resolution, grid_resolution)
        lons = np.arange(lon_min, lon_max + grid_resolution, grid_resolution)
        
        # Calculate altitude and other orbital parameters
        inclination = orbital_parameters['inclination']
        altitude = orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS
        
        # Calculate swath width (simplified)
        earth_radius = constants.EARTH_RADIUS
        max_earth_central_angle = np.arccos(earth_radius / (earth_radius + altitude))
        swath_half_angle_deg = np.degrees(max_earth_central_angle)
        
        # Check if region is within satellite coverage (based on inclination)
        if lat_max < -inclination - swath_half_angle_deg or lat_min > inclination + swath_half_angle_deg:
            # Region is outside of satellite coverage
            return 0.0
        
        # Estimate coverage percentage
        covered_points = 0
        total_points = 0
        
        for lat in lats:
            # Weight points by cosine of latitude (account for smaller area at high latitudes)
            weight = np.cos(np.radians(lat))
            if weight < 0.01:  # Avoid division by very small numbers
                weight = 0.01
                
            for lon in lons:
                # Check if point is within the polygon region (if region is not rectangular)
                if len(self.regional_area) > 4:  # More complex than a rectangle
                    if not self._point_in_polygon(lat, lon, self.regional_area):
                        continue
                
                total_points += weight
                
                # Check if this point can be covered based on inclination and swath
                if abs(lat) <= inclination + swath_half_angle_deg:
                    # Calculate coverage based on satellite density
                    # First approximation: avg region area / avg satellite coverage area
                    region_area = self._estimate_region_area(self.regional_area)
                    
                    # Approximate satellite footprint area
                    swath_area = 2 * np.pi * earth_radius * swath_half_angle_deg / 180.0
                    
                    # Calculate coverage probability based on density
                    satellite_density_factor = min(num_satellites * swath_area / region_area, 1.0)
                    
                    # Adjust for latitude - better coverage near inclination
                    lat_coverage_factor = 1.0
                    if abs(abs(lat) - inclination) < 15:
                        # Areas near the inclination get better coverage
                        lat_coverage_factor = 1.3
                    
                    # Combined coverage probability for this point
                    coverage_probability = min(satellite_density_factor * lat_coverage_factor, 1.0)
                    covered_points += weight * coverage_probability
        
        # Calculate regional coverage percentage
        regional_coverage = covered_points / total_points if total_points > 0 else 0
        
        # Limit maximum score to 1.0
        regional_score = min(regional_coverage, 1.0)
        
        return regional_score
        
    def _point_in_polygon(self, lat, lon, polygon):
        """
        Determine if a point is inside a polygon using ray casting algorithm.
        
        Parameters:
        -----------
        lat, lon : float
            Latitude and longitude of the point to check
        polygon : list of tuples
            List of (lat, lon) coordinates defining the polygon vertices
            
        Returns:
        --------
        inside : bool
            True if point is inside polygon, False otherwise
        """
        n = len(polygon)
        inside = False
        
        p1_lat, p1_lon = polygon[0]
        for i in range(n + 1):
            p2_lat, p2_lon = polygon[i % n]
            
            if lon > min(p1_lon, p2_lon) and lon <= max(p1_lon, p2_lon) and lat <= max(p1_lat, p2_lat):
                if p1_lon != p2_lon:
                    xinters = (lon - p1_lon) * (p2_lat - p1_lat) / (p2_lon - p1_lon) + p1_lat
                    
                if p1_lat == p2_lat or lat <= xinters:
                    inside = not inside
                    
            p1_lat, p1_lon = p2_lat, p2_lon
            
        return inside
        
    def _estimate_region_area(self, region_points):
        """
        Estimate the area of a region defined by latitude/longitude points.
        Uses a simplified calculation assuming a spherical Earth.
        
        Parameters:
        -----------
        region_points : list
            List of (lat, lon) tuples defining the region
            
        Returns:
        --------
        area : float
            Area in square km
        """
        # Convert the region points to 3D coordinates on a unit sphere
        earth_radius = constants.EARTH_RADIUS
        
        # Simple rectangular approximation for now
        if len(region_points) < 3:
            return 0.0
            
        lat_min = min(p[0] for p in region_points)
        lat_max = max(p[0] for p in region_points)
        lon_min = min(p[1] for p in region_points)
        lon_max = max(p[1] for p in region_points)
        
        # Calculate the area (simplified)
        lat_range_rad = np.radians(lat_max - lat_min)
        lon_range_rad = np.radians(lon_max - lon_min)
        avg_lat_rad = np.radians((lat_max + lat_min) / 2.0)
        
        # Area calculation (spherical cap approximation)
        area = earth_radius**2 * lon_range_rad * np.cos(avg_lat_rad) * lat_range_rad
        
        return area
        
    def _calculate_global_coverage(self, num_satellites, orbital_parameters):
        """
        Estimate global coverage based on constellation parameters.
        This is a simplified model - for a real application you'd want to 
        calculate actual global coverage using a grid-based approach.
        
        Parameters:
        -----------
        num_satellites : int
            Number of satellites
        orbital_parameters : dict
            Orbital parameters
            
        Returns:
        --------
        global_score : float
            Score for global coverage (0-1)
        """
        # IMPROVED: Grid-based global coverage calculation
        # If computational time is a concern, use the simplified estimate
        # return self._estimate_global_coverage(num_satellites, orbital_parameters)
        
        # Create a grid of points across Earth's surface
        # Grid resolution (higher = more accurate but slower)
        grid_resolution = 10  # degrees
        
        # Generate a grid of lat/lon points
        lats = np.arange(-90, 91, grid_resolution)
        lons = np.arange(-180, 181, grid_resolution)
        
        # Count how many grid points would be covered
        inclination = orbital_parameters['inclination']
        altitude = orbital_parameters['semi_major_axis'] - constants.EARTH_RADIUS
        
        # Calculate swath width (simplified)
        earth_radius = constants.EARTH_RADIUS
        max_earth_central_angle = np.arccos(earth_radius / (earth_radius + altitude))
        swath_half_angle_deg = np.degrees(max_earth_central_angle)
        
        # Estimate coverage percentage
        covered_points = 0
        total_points = 0
        
        for lat in lats:
            # Weight by cosine of latitude (account for smaller area at high latitudes)
            weight = np.cos(np.radians(lat))
            if weight < 0.01:  # Avoid division by very small numbers
                weight = 0.01
                
            # Skip if satellite can't reach this latitude (based on inclination)
            if abs(lat) > inclination + swath_half_angle_deg:
                continue
                
            for lon in lons:
                total_points += weight
                
                # Simple coverage model: if a point is within the inclination range plus swath,
                # consider it potentially covered
                if abs(lat) <= inclination + swath_half_angle_deg:
                    # Calculate expected revisit frequency based on num_satellites
                    # More satellites = better coverage
                    satellite_density_factor = min(num_satellites / 50.0, 1.0)
                    
                    # Adjust by latitude - points near inclination get more passes
                    lat_coverage_factor = 1.0
                    if abs(abs(lat) - inclination) < 20:
                        # Areas near the inclination get better coverage
                        lat_coverage_factor = 1.2
                    
                    coverage_probability = satellite_density_factor * lat_coverage_factor
                    
                    # Limit maximum probability
                    coverage_probability = min(coverage_probability, 1.0)
                    covered_points += weight * coverage_probability
        
        # Calculate global coverage percentage
        global_coverage = covered_points / total_points if total_points > 0 else 0
        
        # Limit to maximum of 1.0
        return min(global_coverage, 1.0)
        
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
        
        regional_score = self._calculate_regional_coverage(
            const, orbital_parameters, num_satellites, epoch, orbital_period
        )
        
        global_score = self._calculate_global_coverage(
            num_satellites, orbital_parameters
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
        
        # Add pattern parameters if applicable
        if self.pattern_type == "Walker-Delta":
            analysis['parameters']['P'] = pattern_params['P']
            analysis['parameters']['F'] = pattern_params['F']
            
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
        
    def _plot_component_scores(self, analysis):
        """
        Create visualization plots for component scores.
        
        Parameters:
        -----------
        analysis : dict
            Analysis dictionary from analyze_component_scores
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
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
        
        # Components within each category
        coverage_components = [
            weighted_scores['weighted_local'], 
            weighted_scores['weighted_regional'],
            weighted_scores['weighted_global']
        ]
        
        cost_components = [
            weighted_scores['weighted_satellite_cost'],
            weighted_scores['weighted_operational_cost'],
            weighted_scores['weighted_launch_cost']
        ]
        
        science_components = [weighted_scores['weighted_science']] if weights['science_weight'] > 0 else [0]
        
        # Main bars
        x = np.arange(len(categories))
        width = 0.25
        ax3.bar(x, category_scores, width=0.5, label='Category Total')
        
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
        plt.tight_layout()
        
        # Save the plot
        filename = f"plots/optimization/component_scores.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        # plt.show()
        
# Example usage
if __name__ == "__main__":
    # Example 1: Global coverage optimization
    print("\n=== EXAMPLE 1: GLOBAL COVERAGE OPTIMIZATION ===")
    optimizer_global = ConstellationOptimizer(
        pattern_type="Walker-Delta",
        local_weight=0.1,        # Low priority on ground station coverage
        regional_weight=0.2,     # Low priority on regional coverage
        global_weight=0.7,       # High priority on global coverage
        satellite_cost_weight=0.4,
        operational_cost_weight=0.2,
        launch_cost_weight=0.2,
        science_weight=0.2,
        target_altitude=800.0,   # Target altitude for science requirements
    )
    
    # Run optimization with improved convergence settings
    orbital_params_global, pattern_params_global, num_satellites_global, best_score_global = optimizer_global.optimize(
        method='differential_evolution',
        max_iter=200,                # Maximum iterations
        popsize=15,                  # Balanced population size
        strategy='best1bin',         # Focus on exploiting the best solution for better convergence
        mutation=(0.5, 0.7),         # Lower mutation range for more controlled convergence
        recombination=0.7,           # Moderate recombination for balanced exploration/exploitation
        early_stopping=True,         # Enable early stopping
        patience=20,                 # Stop after 20 iterations without improvement
        improvement_threshold=0.001  # Minimum improvement to be considered significant
    )
    
    # Plot optimization history and analyze scores
    optimizer_global.plot_optimization_history()
    
    # Analyze component scores with detailed output to verify normalization fixes
    print("\nVerifying score calculations:")
    analysis = optimizer_global.analyze_component_scores(plot=True, verbose=True, debug=True)
    print(f"Best score from optimize(): {best_score_global:.4f}")
    print(f"Final score from analyze_component_scores(): {analysis['weighted_scores']['final_score']:.4f}")
    print(f"These two scores should match to verify the fix worked.")
    
    # # Example 2: Regional coverage optimization (Western US)
    # print("\n=== EXAMPLE 2: REGIONAL COVERAGE OPTIMIZATION ===")
    # # Define Western US region
    # western_us = [
    #     (32.0, -125.0),  # Southwest corner
    #     (32.0, -105.0),  # Southeast corner
    #     (49.0, -105.0),  # Northeast corner
    #     (49.0, -125.0)   # Northwest corner
    # ]
    
    # optimizer_regional = ConstellationOptimizer(
    #     pattern_type="Walker-Delta",
    #     local_weight=0.2,        # Some priority on ground station coverage
    #     regional_weight=0.6,     # High priority on regional coverage
    #     global_weight=0.2,       # Low priority on global coverage
    #     satellite_cost_weight=0.5,
    #     operational_cost_weight=0.2,
    #     launch_cost_weight=0.3,
    #     science_weight=0.0,      # No science requirements
    #     regional_area=western_us
    # )
    
    # # Run optimization
    # orbital_params_regional, pattern_params_regional, num_satellites_regional, best_score_regional = optimizer_regional.optimize(
    #     method='differential_evolution',
    #     max_iter=30,
    #     popsize=15
    # )
    
    # # Analyze regional optimization results
    # optimizer_regional.analyze_component_scores(plot=True)
    
    # # Example 3: Custom regional area (Europe)
    # print("\n=== EXAMPLE 3: CUSTOM REGION OPTIMIZATION (EUROPE) ===")
    # # Define Europe region
    # europe_region = [
    #     (35.0, -10.0),   # Southwest corner
    #     (35.0, 40.0),    # Southeast corner
    #     (70.0, 40.0),    # Northeast corner
    #     (70.0, -10.0)    # Northwest corner
    # ]
    
    # # Science-focused mission with specific altitude requirements
    # optimizer_custom = ConstellationOptimizer(
    #     pattern_type="Walker-Delta",
    #     local_weight=0.1,        # Low priority on ground station coverage
    #     regional_weight=0.4,     # Medium priority on regional coverage
    #     global_weight=0.2,       # Low priority on global coverage
    #     satellite_cost_weight=0.1,
    #     operational_cost_weight=0.0,  # Not concerned with operational costs
    #     launch_cost_weight=0.1,
    #     science_weight=0.5,      # High priority on science requirements
    #     target_altitude=600.0,   # Lower target altitude for specific science instruments
    #     target_altitude_tolerance=30.0,  # Tighter tolerance
    #     regional_area=europe_region
    # )
    
    # # Run optimization
    # orbital_params_custom, pattern_params_custom, num_satellites_custom, best_score_custom = optimizer_custom.optimize(
    #     method='differential_evolution',
    #     max_iter=30,
    #     popsize=15
    # )
    
    # # Analyze custom optimization results
    # optimizer_custom.analyze_component_scores(plot=True)
    
    # # Compare all three optimized constellations
    # print("\n=== COMPARISON OF OPTIMIZED CONSTELLATIONS ===")
    # print(f"Global:   {num_satellites_global} satellites, " + 
    #       f"altitude: {orbital_params_global['semi_major_axis'] - constants.EARTH_RADIUS:.1f} km, " +
    #       f"inclination: {orbital_params_global['inclination']:.1f}°, " +
    #       f"score: {best_score_global:.4f}")
          
    # print(f"Regional: {num_satellites_regional} satellites, " + 
    #       f"altitude: {orbital_params_regional['semi_major_axis'] - constants.EARTH_RADIUS:.1f} km, " +
    #       f"inclination: {orbital_params_regional['inclination']:.1f}°, " +
    #       f"score: {best_score_regional:.4f}")
          
    # print(f"Custom:   {num_satellites_custom} satellites, " + 
    #       f"altitude: {orbital_params_custom['semi_major_axis'] - constants.EARTH_RADIUS:.1f} km, " +
    #       f"inclination: {orbital_params_custom['inclination']:.1f}°, " +
    #       f"score: {best_score_custom:.4f}")