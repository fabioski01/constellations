"""
Multi-Objective Constellation Optimizer

This module provides tools to run multi-objective optimization for satellite
constellations. It adapts the domain-specific `ConstellationOptimizer` to the
`pymoo` NSGA-II implementation and exposes a convenient runner that evaluates
candidate designs (optionally in parallel), performs Pareto filtering and
exports results as CSV and PNG for downstream analysis.

Objectives
----------
- Total mission cost (M$) -- minimized
- Combined performance score (coverage + science) -- maximized

Features
--------
- Adapter classes for `pymoo` (single-threaded and parallel evaluation)
- Caching of detailed evaluation results for fast CSV export
- Dynamic filename and directory generation that encodes run parameters
- Simple CLI for quick experiments and local testing

Optimization algorithm (NSGA-II)
--------------------------------
This runner uses the NSGA-II evolutionary algorithm as implemented in
`pymoo` to search for Pareto-optimal constellation designs. Key aspects of
NSGA-II used here:

- Population-based search: a population of candidate solutions is evolved
    across generations.
- Non-dominated sorting: solutions are ranked by Pareto dominance (fronts),
    so selection favors non-dominated solutions.
- Crowding-distance diversity preservation: within the same Pareto front,
    solutions with larger crowding distance are preferred to maintain diversity
    along the front.
- Elitism: parents and offspring are combined and the best individuals are
    selected for the next generation to retain high-quality solutions.
- Genetic operators used by the runner: Simulated Binary Crossover (SBX) and
    Polynomial Mutation. These are configured when constructing the `NSGA2`
    algorithm object (see the call sites that set `SBX(prob=0.9, eta=15)` and
    `PolynomialMutation(prob=1.0/problem.n_var, eta=20)`).

The `pymoo` implementation handles the low-level operator mechanics; this
module configures the population size, sampling, crossover and mutation
probabilities to balance exploration and exploitation for the constellation
design problem.

Configurable Inputs (quick reference)
------------------------------------
High-level knobs you can (and most likely will) change when calling
`run_multi_objective_optimization(...)` or using the CLI:

Required / Core
    pattern_type          : str  ("Walker-Delta" | "Flower" | future patterns)
    gs                    : str  Ground station identifier (affects coverage models)

Mission / System Weights (all floats in [0,1], any can be zero)
    local_weight          : weight for local coverage metric
    regional_weight       : weight for regional coverage metric
    global_weight         : weight for global coverage metric
    science_weight        : weight for science requirement score

Design Seeding / Physical Parameters
    target_altitude       : km (used to seed semi-major axis for initial sampling)
    sat_mass_kg           : satellite wet mass for cost model
    payloads              : list[dict] optional payload definitions (type, mass_kg, power_w, cat)
    step                  : seconds between propagated samples (coverage fidelity vs speed)
    evaluation_hours      : simulation horizon length (trade accuracy vs runtime)

Evolutionary Algorithm (NSGA-II)
    pop_size              : population size per generation
    n_gen                 : number of generations
    seed                  : RNG seed for reproducibility

Parallel / Runtime Controls
    n_cores               : int | None  (None => auto detect / SLURM_CPUS_PER_TASK if cluster_mode)
    cluster_mode          : bool (enables parallel batch evaluation path + size adjustments)

File & Output
    output_dir            : base results directory (plots + CSV go under pattern/weight str)

Returned Results Dict Keys
    pareto_front          : ndarray shape (k,2) [cost, -performance]
    pareto_solutions      : ndarray shape (k,n_var)
    costs / performances  : convenience 1D arrays extracted from `pareto_front`
    csv_filename          : full path to written CSV file
    plot_filename         : full path to Pareto front PNG
    optimizer             : the configured MultiObjectiveConstellationOptimizer instance

Pattern-Specific Design Variables (bounds internally enforced)
    Walker-Delta : [N_sats, perigee_h(km), ecc, inc(deg), RAAN(deg), argp(deg), P, F]
    Flower       : [T (≈num_sats), perigee_h(km), ecc, inc, RAAN, argp, Np, Nd, Fn, Fd, Fh]

Cost Model Weights (currently fixed to zero in runner but exposed in optimizer kwargs)
    satellite_cost_weight, operational_cost_weight, launch_cost_weight

Tip: start with small pop_size / n_gen (e.g. 10 / 5) to validate setup, then scale.

Usage example
-------------
>>> from multi_objective_optimizer import run_multi_objective_optimization
>>> results = run_multi_objective_optimization(
...     pattern_type='Walker-Delta',
...     gs='AFB_Andersen',
...     pop_size=50, n_gen=30, seed=42,
...     local_weight=0.2, regional_weight=0.3, global_weight=0.5,
...     science_weight=0.6, target_altitude=600.0, sat_mass_kg=130.0
... )

The returned `results` dictionary contains keys such as `pareto_front`,
`pareto_solutions`, `csv_filename` and `plot_filename` for easy post-processing.

Author
------
Fabio Meloni - TU Delft MSc Student - NASA Jet Propulsion Laboratory Visiting Researcher
"""

import os
import math
import numpy as np
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from types import SimpleNamespace

# Reuse existing optimizer and utilities
from utilities_optimization import ConstellationOptimizer
from utilities_constellations import Constellation
import constants

# Import pymoo for multi-objective optimization
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import argparse

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from datetime import timedelta, timezone

class MultiObjectiveConstellationOptimizer(ConstellationOptimizer):
    """
    Multi-objective optimizer that extends ConstellationOptimizer.
    
    Objectives:
    - f1: Minimize total mission cost (M$) using economies_of_scale
    - f2: Maximize performance (coverage + science) - implemented as minimize -performance
    
    Uses NSGA-II evolutionary algorithm to find Pareto-optimal solutions.
    """
    
    def __init__(self, sat_mass_kg=None, payloads=None, **kwargs):
        """
        Initialize the multi-objective optimizer wrapper.

        This constructor forwards most configuration parameters to the base
        :class:`ConstellationOptimizer` while allowing optional overrides for
        `sat_mass_kg` and `payloads`.

        Parameters
        ----------
        sat_mass_kg : float or None
            Optional satellite mass in kilograms. If provided it overrides the
            base optimizer default.
        payloads : list or None
            Optional payload configuration list. When provided this replaces
            the optimizer's default payload list.
        **kwargs : dict
            Additional keyword arguments forwarded to
            :class:`ConstellationOptimizer`.
        """
        # Ensure sat_mass_kg is passed to base class if provided
        if sat_mass_kg is not None:
            kwargs['sat_mass_kg'] = float(sat_mass_kg)
        
        # Ensure payloads is passed to base class if provided  
        if payloads is not None:
            kwargs['payloads'] = payloads
            
        # Pass all args to the base optimizer
        super().__init__(**kwargs)

        # Store detailed results for CSV export (cache)
        self.detailed_cache = {}

    def _performance_score(self, local_score, overlap_penalty, regional_score, global_score, science_score):
        """
        Combine coverage and science into a single performance score in [0,1] (higher is better).
        Costs are not included here.
        """
        coverage_weight = self.local_weight + self.regional_weight + self.global_weight
        if coverage_weight > 0:
            coverage = (
                self.local_weight * local_score * overlap_penalty +
                self.regional_weight * regional_score +
                self.global_weight * global_score
            ) / coverage_weight
        else:
            coverage = 0.0
        
        # Combine coverage and science
        total_weight = coverage_weight + self.science_weight
        if total_weight > 0:
            performance = (coverage_weight * coverage + self.science_weight * science_score) / total_weight
        else:
            performance = 0.0
            
        return performance

    def evaluate_objectives(self, params):
        """
        Evaluate both objectives for a given parameter vector.
        
        Parameters:
        -----------
        params : array-like
            Design variables [num_satellites, perigee_height, eccentricity, inclination, RAAN, arg_periapsis, P, F]
            
        Returns:
        --------
        cost : float
            Total mission cost in M$
        neg_performance : float
            Negative performance score (for minimization)
        """
        try:
            if self.pattern_type == "Walker-Delta":
                P = max(1, int(round(params[6])))
                F_raw = int(round(params[7]))
                
                # Repair: force N to be multiple of P
                N_candidate = int(round(params[0]))
                N = N_candidate - (N_candidate % P) if N_candidate % P != 0 else N_candidate
                if N < 1: N = P
                params[0] = N
                
                # Repair: force F < P and F divides N
                F = min(F_raw, P - 1)
                if F > 0:
                    while N % F != 0 and F > 0:
                        F -= 1
                params[7] = F
                params[6] = P
                # print(f"[DEBUG Evalute Obj.] Repaired params for Walker-Delta: N={N}, P={P}, F={F}")

            # Create parameter dictionaries
            num_satellites, orbital_parameters, pattern_params = self._create_orbital_parameters(params)
            # DEBUG: print what we got
            # print(f"DEBUG eval: N={num_satellites}, P={pattern_params.get('P')}, F={pattern_params.get('F')}, types: P={type(pattern_params.get('P'))}, F={type(pattern_params.get('F'))}")
                        
            # Validate parameters
            if num_satellites < 1 or num_satellites > 200:
                return 1e6, 1e6  # Invalid
                
            if orbital_parameters['eccentricity'] < 0 or orbital_parameters['eccentricity'] >= 1:
                return 1e6, 1e6  # Invalid
                
            if orbital_parameters['semi_major_axis'] < constants.EARTH_RADIUS + 200:
                return 1e6, 1e6  # Invalid orbit
            
            # Pattern-specific validation
            if self.pattern_type == "Walker-Delta":
                P = pattern_params['P']
                F = pattern_params['F']

                # if P < 1 or F < 1 or F >= P:
                #     return 1e6, 1e6

                # if num_satellites < P or num_satellites % P != 0:
                #     return 1e6, 1e6

                # if num_satellites % F != 0:
                #     return 1e6, 1e6

                if P < 1 or not (0 <= F < P):
                    return 1e6, 1e6
                if num_satellites < P or (num_satellites % P) != 0:
                    return 1e6, 1e6    
                
                assert P >= 1 and 0 <= F < P, f"Invalid repaired P={P}, F={F}"
                assert num_satellites >= P and num_satellites % P == 0, f"Invalid N={num_satellites}, P={P}"
            
            elif self.pattern_type == "Flower":
                # Expect pattern_params to contain Np, Nd, Fn, Fd, Fh
                Np = int(round(pattern_params.get('Np', 1)))
                Nd = int(round(pattern_params.get('Nd', 1)))
                Fn = int(round(pattern_params.get('Fn', 1)))
                Fd = int(round(pattern_params.get('Fd', 1)))
                Fh = int(round(pattern_params.get('Fh', 0)))

                # Basic validation: positive integers and ranges
                if Np < 1 or Nd < 1 or Fn < 1 or Fd < 1 or not (0 <= Fh < max(1, Nd)):
                    return 1e6, 1e6

                # Fn and Fd should be distinct and reasonably bounded
                if Fn >= Nd or Fd >= Nd:
                    # keep a conservative validity check; invalid combos are treated as infeasible
                    return 1e6, 1e6
            
            # Create constellation and propagate
            epoch = datetime(2025, 5, 5, 0, 0, 0)
            
            if self.pattern_type == "Walker-Delta":
                const = Constellation(
                    num_satellites, orbital_parameters, epoch, 
                    self.pattern_type, P=pattern_params['P'], F=pattern_params['F']
                )

            elif self.pattern_type == "Flower":
                # Ensure orbital_parameters dictionary contains Flower-specific keys
                orbital_parameters = orbital_parameters.copy()
                orbital_parameters['Np'] = int(round(pattern_params.get('Np', 1)))
                orbital_parameters['Nd'] = int(round(pattern_params.get('Nd', 1)))
                orbital_parameters['Fn'] = int(round(pattern_params.get('Fn', 1)))
                orbital_parameters['Fd'] = int(round(pattern_params.get('Fd', 1)))
                orbital_parameters['Fh'] = int(round(pattern_params.get('Fh', 0)))

                const = Constellation(
                    num_satellites, orbital_parameters, epoch,
                    self.pattern_type,
                    Np=orbital_parameters['Np'],
                    Nd=orbital_parameters['Nd'],
                    Fn=orbital_parameters['Fn'],
                    Fd=orbital_parameters['Fd'],
                    Fh=orbital_parameters['Fh']
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

            # Calculate orbital period
            a = orbital_parameters['semi_major_axis']
            orbital_period = 2 * np.pi * np.sqrt(a**3 / constants.GRAVITATIONAL_CONSTANT)
            
            # Calculate performance components
            local_score, overlap_penalty = self._calculate_local_coverage(
                positions, time_stamps, num_satellites, orbital_period
            )
            
            if self.regional_weight > 0:
                regional_score = self._calculate_regional_coverage(
                    const, orbital_parameters, num_satellites, epoch, orbital_period, 
                    positions=positions, time_stamps=time_stamps
                )
            else:
                regional_score = 0.0

            if self.global_weight > 0:
                global_score = self._calculate_global_coverage(
                    num_satellites, orbital_parameters, positions=positions, time_stamps=time_stamps
                )
            else:
                global_score = 0.0

            science_score = self._calculate_science_requirement_score(orbital_parameters)
            
            # Calculate weighted components
            weighted_local = self.local_weight * local_score * overlap_penalty
            weighted_regional = self.regional_weight * regional_score
            weighted_global = self.global_weight * global_score
            weighted_science = self.science_weight * science_score
            
            # Calculate performance score (0-1, higher is better)
            performance = self._performance_score(
                local_score, overlap_penalty, regional_score, global_score, science_score
            )
            
            # Calculate total cost using economies of scale
            cost_breakdown = self.economies_of_scale(
                num_satellites, orbital_parameters, self.sat_mass_kg, pattern_params
            )
            total_cost = cost_breakdown['total_musd']
            
            # *** CACHE DETAILED RESULTS FOR CSV EXPORT ***
            # Create a hashable key from parameters (rounded for floating point consistency)
            param_key = tuple(np.round(params, 6))  # Round to avoid floating point precision issues
            
            self.detailed_cache[param_key] = {
                'local_score': local_score,
                'regional_score': regional_score, 
                'global_score': global_score,
                'science_score': science_score,
                'overlap_penalty': overlap_penalty,
                'weighted_local': weighted_local,
                'weighted_regional': weighted_regional,
                'weighted_global': weighted_global,
                'weighted_science': weighted_science,
                'cost_breakdown': cost_breakdown.copy()  # Store cost breakdown too
            }
            
            # Return objectives: minimize cost, minimize -performance
            return total_cost, -performance
            
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full stack trace
            return 1e6, 1e6  # Return high values for invalid solutions
        
    def get_cached_details(self, params):
        """
        Retrieve cached detailed results for given parameters.
        
        Parameters:
        -----------
        params : array-like
            Design variables
            
        Returns:
        --------
        details : dict or None
            Cached detailed results or None if not found
        """
        param_key = tuple(np.round(params, 6))
        return self.detailed_cache.get(param_key, None)
        
class ConstellationOptimizationProblem(ElementwiseProblem):
    """
    Pymoo problem definition for constellation optimization.
    """
    
    def __init__(self, optimizer):
        """
        Create a pymoo problem adapter for the provided optimizer.

        This adapter exposes the optimizer's design space and objective
        evaluations to pymoo. Currently only the "Walker-Delta" pattern is
        supported; other patterns should add matching bounds and decoding.

        Parameters
        ----------
        optimizer : MultiObjectiveConstellationOptimizer
            Instance responsible for evaluating design candidates and
            computing costs/performances.
        """
        self.optimizer = optimizer
        
        # Define bounds based on pattern type
        if optimizer.pattern_type == "Walker-Delta":
            bounds_lower = [1, 250, 0, 0, 0, 0, 1, 0]  # [num_sats, perigee_h, ecc, inc, RAAN, argp, P, F]
            bounds_upper = [100, 1000, 0, 90, 360, 360, 20, 20]
            n_var = 8
        elif optimizer.pattern_type == "Flower":
            # Flower variables: [T, perigee_h, ecc, inc, RAAN, argp, Np, Nd, Fn, Fd, Fh]
            bounds_lower = [1, 200, 0, 0, 0, 0, 1, 1, 1, 1, 0]
            bounds_upper = [200, 36000, 0.9, 180, 360, 360, 100, 365, 364, 364, 364]
            n_var = 11
        else:
            raise NotImplementedError(f"Pattern {optimizer.pattern_type} not implemented yet")
        
        super().__init__(
            n_var=n_var,
            n_obj=2,  # Cost and negative performance
            xl=np.array(bounds_lower),
            xu=np.array(bounds_upper)
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate objectives for a single solution.
        """
        cost, neg_performance = self.optimizer.evaluate_objectives(x)
        out["F"] = [cost, neg_performance]

class ParallelConstellationOptimizationProblem(Problem):
    """
    Parallel version of the optimization problem for cluster computing.
    """
    
    def __init__(self, optimizer, n_cores=None):
        """
        Parallel problem adapter that evaluates a batch of candidate
        solutions using a process pool.

        Parameters
        ----------
        optimizer : MultiObjectiveConstellationOptimizer
            Optimizer instance used to evaluate individual solutions.
        n_cores : int or None
            Maximum number of worker processes to use. If ``None``, the
            function will default to the machine's CPU count.
        """
        self.optimizer = optimizer
        self.n_cores = n_cores or mp.cpu_count()
        
        # Same bounds as existing problem
        if optimizer.pattern_type == "Walker-Delta":
            bounds_lower = [1, 250, 0, 0, 0, 0, 1, 0] # [N, perigee_h, ecc, inc, RAAN, argp, P, F]
            bounds_upper = [100, 1200, 0, 90, 360, 360, 50, 49] # [N, perigee_h, ecc, inc, RAAN, argp, P, F]
            n_var = 8
        elif optimizer.pattern_type == "Flower":
            bounds_lower = [1, 200, 0, 0, 0, 0, 1, 1, 1, 1, 0] # [T, perigee_h, ecc, inc, RAAN, argp, Np, Nd, Fn, Fd, Fh]
            bounds_upper = [200, 36000, 0.9, 180, 360, 360, 100, 365, 364, 364, 364] # [T, perigee_h, ecc, inc, RAAN, argp, Np, Nd, Fn, Fd, Fh]
            n_var = 11
        else:
            raise NotImplementedError(f"Pattern {optimizer.pattern_type} not implemented yet")
        
        super().__init__(
            n_var=n_var,
            n_obj=2,
            xl=np.array(bounds_lower),
            xu=np.array(bounds_upper)
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate multiple solutions in parallel.
        X is a matrix where each row is a solution.
        """
        n_solutions = X.shape[0]
        
        if n_solutions == 1:
            # Single evaluation - use existing method
            cost, neg_performance = self.optimizer.evaluate_objectives(X[0])
            out["F"] = np.array([[cost, neg_performance]])
        else:
            # Parallel evaluation
            with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
                futures = [executor.submit(self.optimizer.evaluate_objectives, X[i]) for i in range(n_solutions)]
                results = [future.result() for future in futures]
            
            out["F"] = np.array(results)

def generate_dynamic_filename(pattern_type, gs, local_weight, regional_weight, global_weight, 
                            science_weight, pop_size, n_gen, step, seed, sat_mass_kg, 
                            target_altitude, dv_deployment_m_s, T_deployment_days, file_type="csv"):
    """
    Generate dynamic filename based on optimization parameters.
    
    Parameters:
    -----------
    file_type : str
        "csv" or "png" for different file extensions
        
    Returns:
    --------
    filename : str
        Dynamic filename with parameters encoded
    """
    # Create parameter string
    param_str = (f"{pattern_type.lower()}_"
                f"w{local_weight:.1f}-{regional_weight:.1f}-{global_weight:.1f}-{science_weight:.1f}_"
                f"p{pop_size}_g{n_gen}_"
                f"sp{step}_sd{seed}_"
                f"m{sat_mass_kg:.0f}_"
                f"a{target_altitude:.0f}_"
                f"dv{dv_deployment_m_s:.0f}_"
                f"T{T_deployment_days:.0f}d")

    if file_type == "csv":
        return f"pareto_solutions_{param_str}.csv"
    elif file_type == "png":
        return f"pareto_front_{param_str}.png"
    else:
        return f"results_{param_str}.{file_type}"

def generate_dynamic_directory(pattern_type, gs, local_weight, regional_weight, global_weight, 
                              science_weight, base_dir="results/MOEA"):
    """
    Generate dynamic directory structure based on key parameters.

    NOTE: The `gs` (ground station) argument is accepted for backward
    compatibility but is intentionally NOT included in the path. This saves
    results under ``<base_dir>/<pattern_type>/<weight_str>/`` rather than
    ``<base_dir>/<pattern_type>/<gs>/<weight_str>/``.

    Returns
    -------
    directory : str
        Dynamic directory path
    """
    # Create directory structure: results/pattern/weights/ (gs omitted)
    weight_str = f"L{local_weight:.1f}_R{regional_weight:.1f}_G{global_weight:.1f}_Sci{science_weight:.1f}"
    directory = os.path.join(base_dir, pattern_type.lower(), weight_str)
    return directory

def run_multi_objective_optimization(
    pattern_type="Walker-Delta",
    gs="AFB_Andersen",
    local_weight=0.2,
    regional_weight=0.3,
    global_weight=0.5,
    science_weight=0.6,
    target_altitude=600.0,
    sat_mass_kg=130.0,
    payloads=None,
    pop_size=100,
    n_gen=50,
    seed=42,
    output_dir="results/MOEA",
    step=1000,
    n_cores=None, # for parellelism
    cluster_mode=False, # for parallelism
    evaluation_hours=24,
    dv_deployment_m_s: float = 100.0,
    T_deployment_days: float = 180.0
):
    """
    Run a multi-objective NSGA-II optimization for constellation design.

    The function sets up the optimizer, selects between a single-threaded
    or parallel pymoo problem adapter, runs the evolutionary algorithm and
    writes the resulting Pareto set to a CSV and PNG file.

    Parameters
    ----------
    pattern_type : str
        Name of the constellation pattern (defaults to ``"Walker-Delta"``).
    gs : str
        Ground station identifier used by coverage utilities.
    local_weight, regional_weight, global_weight, science_weight : float
        Weighting factors used to combine coverage and science into a single
        performance score.
    target_altitude : float
        Target altitude in kilometers used to seed the optimizer.
    sat_mass_kg : float
        Satellite mass in kilograms for cost modeling.
    payloads : list or None
        Optional payload configuration list.
    pop_size : int
        Population size for NSGA-II.
    n_gen : int
        Number of generations to evolve.
    seed : int
        Random seed for reproducible runs.
    output_dir : str
        Base directory where CSV and PNG outputs will be written.
    step : int
        Propagation step in seconds.
    n_cores : int or None
        Number of CPU cores to use for parallel evaluation.
    cluster_mode : bool
        If True, use cluster-friendly parallelization settings.
    evaluation_hours : float
        Length of the simulation window in hours used for coverage
        calculations.
    dv_deployment_m_s : float
        Delta-V budget for deployment phase in m/s. Higher means launch cost could decrease
    T_deployment_days: float
        Max amount of time for constellation deployment after launch. Higher means launch cost could decrease

    Returns
    -------
    results : dict
        Dictionary containing the Pareto front, solutions, file paths and the
        optimizer instance for further inspection.
    """
    
    # Set default payloads if not provided
    if payloads is None:
        payloads = [
            {'type': 'fields', 'cat': 'EO', 'mass_kg': 10.0, 'power_w': 20.0},
            {'type': 'particle', 'cat': 'EO', 'mass_kg': 10.0, 'power_w': 20.0}
        ]
    
    # Calculate start and end times
    start_time = datetime(2025, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    end_time = start_time + timedelta(hours=evaluation_hours)

    # Create optimizer (unchanged)
    optimizer = MultiObjectiveConstellationOptimizer(
        pattern_type=pattern_type,
        gs=gs,
        local_weight=local_weight,
        regional_weight=regional_weight,
        global_weight=global_weight,
        satellite_cost_weight=0.0,
        operational_cost_weight=0.0,
        launch_cost_weight=0.0,
        science_weight=science_weight,
        target_altitude=target_altitude,
        sat_mass_kg=sat_mass_kg,
        payloads=payloads,
        step=step,
        start_time=start_time,
        end_time=end_time,
        dv_deployment_m_s=dv_deployment_m_s,
        T_deployment_days=T_deployment_days
    )
    
    # Determine number of cores
    if n_cores is None:
        if cluster_mode:
            # On cluster, use SLURM environment variable or all cores
            n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
        else:
            n_cores = mp.cpu_count()
    
    print(f"Using {n_cores} cores for evaluation")
    
    # Choose problem type based on parallelization
    if n_cores > 1 and cluster_mode:
        problem = ParallelConstellationOptimizationProblem(optimizer, n_cores=n_cores)
        
        # Adjust population size for better load balancing
        if pop_size % n_cores != 0:
            adjusted_pop_size = ((pop_size // n_cores) + 1) * n_cores
            print(f"Adjusted population size from {pop_size} to {adjusted_pop_size} for better load balancing")
            pop_size = adjusted_pop_size
    else:
        # Use existing single-threaded problem
        problem = ConstellationOptimizationProblem(optimizer)
    
    # Create algorithm (unchanged)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(prob=1.0/problem.n_var, eta=20),
        eliminate_duplicates=True
    )
    
    print(f"Starting multi-objective optimization...")
    print(f"Pattern: {pattern_type}")
    print(f"Population size: {pop_size}, Generations: {n_gen}")
    print(f"Parallel mode: {cluster_mode}, Cores: {n_cores}")
    
    # Run optimization (unchanged)
    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        seed=seed,
        verbose=True
    )
    
    # Extract results
    pareto_front = res.F
    pareto_solutions = res.X
    
    # Convert negative performance back to positive
    costs = pareto_front[:, 0]
    performances = -pareto_front[:, 1]
    
    print(f"\nOptimization completed!")
    print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
    print(f"Cost range: {costs.min():.2f} - {costs.max():.2f} M$")
    print(f"Performance range: {performances.min():.3f} - {performances.max():.3f}")
    
    # Generate dynamic directory and filenames
    dynamic_dir = generate_dynamic_directory(pattern_type, gs, local_weight, regional_weight, 
                                           global_weight, science_weight, output_dir)
    os.makedirs(dynamic_dir, exist_ok=True)
    
    csv_filename = generate_dynamic_filename(pattern_type, gs, local_weight, regional_weight, 
                                           global_weight, science_weight, pop_size, n_gen, 
                                           step, seed, sat_mass_kg, target_altitude, dv_deployment_m_s, T_deployment_days, "csv")
    csv_filepath = os.path.join(dynamic_dir, csv_filename)
    
    plot_filename = generate_dynamic_filename(pattern_type, gs, local_weight, regional_weight, 
                                            global_weight, science_weight, pop_size, n_gen, 
                                            step, seed, sat_mass_kg, target_altitude, dv_deployment_m_s, T_deployment_days, "png")
    plot_filepath = os.path.join(dynamic_dir, plot_filename)
    
    # Save detailed results to CSV
    with open(csv_filepath, 'w', newline='') as csvfile:
        if pattern_type == "Walker-Delta":
            fieldnames = [
                # Solution identification
                'solution_id', 'cost_musd', 'performance', 
                # Design variables
                'num_satellites', 'perigee_height_km', 'eccentricity', 'inclination_deg',
                'RAAN_deg', 'arg_periapsis_deg', 'P_planes', 'F_phasing',
                'altitude_km', 'semi_major_axis_km',
                # Actual cost breakdown
                'NRE_cost_musd', 'RE_cost_musd', 'launch_cost_musd',
                'payload_tfu_musd', 'sc_tfu_musd', 'learning_rate',
                # Launch details (NEW)
                'launch_num_launches', 'launch_num_groups', 'launch_planes_per_group',
                'launch_dv_used_m_s', 'launch_T_used_days',
                'launch_a_target_km', 'launch_a_launch_km',
                'launch_delta_a_km', 'launch_delta_raan_deployment_deg',
                'launch_delta_raan_required_deg',
                'launch_vehicle_combo',
                'launch_param_dv_deployment_m_s', 'launch_param_T_deployment_days',
                # Detailed performance breakdown
                'local_coverage_score', 'regional_coverage_score', 'global_coverage_score',
                'science_score', 'overlap_penalty',
                # Weighted performance components
                'weighted_local', 'weighted_regional', 'weighted_global', 'weighted_science',
                # Optimization metadata
                'optimization_params_pop_size', 'optimization_params_n_gen', 
                'optimization_params_step', 'optimization_params_seed',
                'optimization_params_sat_mass_kg', 'optimization_params_target_altitude'
            ]

        elif pattern_type == "Flower":
            fieldnames = [
                'solution_id', 'cost_musd', 'performance',
                # Design variables
                'num_satellites', 'perigee_height_km', 'eccentricity', 'inclination_deg',
                'RAAN_deg', 'arg_periapsis_deg', 'Np', 'Nd', 'Fn', 'Fd', 'Fh',
                'altitude_km', 'semi_major_axis_km',
                # Costs
                'NRE_cost_musd', 'RE_cost_musd', 'launch_cost_musd',
                'payload_tfu_musd', 'sc_tfu_musd', 'learning_rate',
                # Launch details (add same fields as Walker-Delta)
                'launch_num_launches', 'launch_num_groups', 'launch_planes_per_group',
                'launch_dv_used_m_s', 'launch_T_used_days',
                'launch_a_target_km', 'launch_a_launch_km',
                'launch_delta_a_km', 'launch_delta_raan_deployment_deg',
                'launch_delta_raan_required_deg',
                'launch_vehicle_combo',
                'launch_param_dv_deployment_m_s', 'launch_param_T_deployment_days',
                # Performance
                'local_coverage_score', 'regional_coverage_score', 'global_coverage_score',
                'science_score', 'overlap_penalty',
                'weighted_local', 'weighted_regional', 'weighted_global', 'weighted_science',
                # Optimization metadata
                'optimization_params_pop_size', 'optimization_params_n_gen',
                'optimization_params_step', 'optimization_params_seed',
                'optimization_params_sat_mass_kg', 'optimization_params_target_altitude'
            ]
        else:
            raise NotImplementedError(f"CSV format for {pattern_type} not implemented")
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, (solution, cost, performance) in enumerate(zip(pareto_solutions, costs, performances)):
            # Decode solution parameters
            num_satellites, orbital_params, pattern_params = optimizer._create_orbital_parameters(solution)
            
            # Calculate derived parameters
            altitude = orbital_params['semi_major_axis'] - constants.EARTH_RADIUS
            
            # *** GET CACHED DETAILED RESULTS (FAST!) ***
            cached_details = optimizer.get_cached_details(solution)
            
            if cached_details is not None:
                # Use cached results (fast path)
                cost_breakdown = cached_details['cost_breakdown']
                local_score = cached_details['local_score']
                regional_score = cached_details['regional_score']
                global_score = cached_details['global_score']
                science_score = cached_details['science_score']
                overlap_penalty = cached_details['overlap_penalty']
                weighted_local = cached_details['weighted_local']
                weighted_regional = cached_details['weighted_regional']
                weighted_global = cached_details['weighted_global']
                weighted_science = cached_details['weighted_science']
                
                print(f"Using cached results for solution {i}")
            else:
                # Fallback: recompute via optimizer to populate cache in this process
                print(f"Warning: No cached results for solution {i}, recomputing to populate cache...")
                try:
                    _ = optimizer.evaluate_objectives(solution)  # fills optimizer.detailed_cache
                    rec = optimizer.get_cached_details(solution)
                    if rec is None:
                        raise RuntimeError("cache still empty after recompute")
                    cost_breakdown = rec['cost_breakdown']
                    local_score = rec['local_score']
                    regional_score = rec['regional_score']
                    global_score = rec['global_score']
                    science_score = rec['science_score']
                    overlap_penalty = rec['overlap_penalty']
                    weighted_local = rec['weighted_local']
                    weighted_regional = rec['weighted_regional']
                    weighted_global = rec['weighted_global']
                    weighted_science = rec['weighted_science']
                except Exception as e:
                    # Last-resort: try economies_of_scale if available, else write minimal fields
                    try:
                        cost_breakdown = optimizer.economies_of_scale(
                            num_satellites, orbital_params,
                            getattr(optimizer, 'sat_mass_kg', None), pattern_params
                        )
                    except AttributeError:
                        cost_breakdown = {'total_musd': float(cost)}
                    # No coverage breakdown available in this path
                    local_score = regional_score = global_score = science_score = 0.0
                    overlap_penalty = 1.0
                    weighted_local = weighted_regional = weighted_global = weighted_science = 0.0

            # Launch details unpack
            ld = cost_breakdown.get('launch_details') if isinstance(cost_breakdown, dict) else None
            if isinstance(ld, dict):
                dep = ld.get('deployment', {}) or {}
                vlist = ld.get('vehicle_breakdown', []) or []
                vehicle_combo = "; ".join(
                    f"G{item.get('group', idx+1)}:{item.get('vehicle','?')} x{item.get('num_launches',1)}"
                    for idx, item in enumerate(vlist)
                )
                launch_num_launches = ld.get('num_launches', 0)
                launch_num_groups = ld.get('num_groups', 0)
                launch_planes_per_group = dep.get('planes_per_group', None)
                launch_dv_used_m_s = dep.get('dv_used_m_s', None)
                launch_T_used_days = dep.get('T_used_days', None)
                launch_a_target_km = dep.get('a_target_km', None)
                launch_a_launch_km = dep.get('a_launch_km', None)
                launch_delta_a_km = dep.get('delta_a_mag_km', None)
                launch_delta_raan_deployment_deg = dep.get('delta_raan_deployment_deg', None)
                launch_delta_raan_required_deg = dep.get('delta_raan_required_deg', None)
            else:
                vehicle_combo = ""
                launch_num_launches = launch_num_groups = 0
                launch_planes_per_group = None
                launch_dv_used_m_s = launch_T_used_days = None
                launch_a_target_km = launch_a_launch_km = None
                launch_delta_a_km = launch_delta_raan_deployment_deg = launch_delta_raan_required_deg = None

            row = {
                # Solution identification
                'solution_id': i,
                'cost_musd': cost,
                'performance': performance,
                
                # Design variables
                'num_satellites': num_satellites,
                'perigee_height_km': solution[1],
                'eccentricity': solution[2],
                'inclination_deg': solution[3],
                'RAAN_deg': solution[4],
                'arg_periapsis_deg': solution[5],
                'altitude_km': altitude,
                'semi_major_axis_km': orbital_params['semi_major_axis'],
                
                # Cost breakdown
                'NRE_cost_musd': cost_breakdown.get('NRE_musd', 0),
                'sc_nre_musd': cost_breakdown.get('sc_nre_musd', 0),
                'payload_nre_musd': cost_breakdown.get('payload_nre_musd', 0),
                'RE_cost_musd': cost_breakdown.get('RE_musd', 0),
                'launch_cost_musd': cost_breakdown.get('launch_musd', 0),
                'payload_tfu_musd': cost_breakdown.get('payload_tfu_musd', 0),
                'sc_tfu_musd': cost_breakdown.get('sc_tfu_musd', 0),

                # Launch details (NEW)
                'launch_num_launches': launch_num_launches,
                'launch_num_groups': launch_num_groups,
                'launch_planes_per_group': launch_planes_per_group,
                'launch_dv_used_m_s': launch_dv_used_m_s,
                'launch_T_used_days': launch_T_used_days,
                'launch_a_target_km': launch_a_target_km,
                'launch_a_launch_km': launch_a_launch_km,
                'launch_delta_a_km': launch_delta_a_km,
                'launch_delta_raan_deployment_deg': launch_delta_raan_deployment_deg,
                'launch_delta_raan_required_deg': launch_delta_raan_required_deg,
                'launch_vehicle_combo': vehicle_combo,
                'launch_param_dv_deployment_m_s': dv_deployment_m_s,
                'launch_param_T_deployment_days': T_deployment_days,

                # *** DETAILED PERFORMANCE BREAKDOWN FROM CACHE ***
                'local_coverage_score': local_score,
                'regional_coverage_score': regional_score,
                'global_coverage_score': global_score,
                'science_score': science_score,
                'overlap_penalty': overlap_penalty,
                
                # Weighted performance components
                'weighted_local': weighted_local,
                'weighted_regional': weighted_regional,
                'weighted_global': weighted_global,
                'weighted_science': weighted_science,       

                # Add optimization metadata
                'optimization_params_pop_size': pop_size,
                'optimization_params_n_gen': n_gen,
                'optimization_params_step': step,
                'optimization_params_seed': seed,
                'optimization_params_sat_mass_kg': sat_mass_kg,
                'optimization_params_target_altitude': target_altitude
            }
            
            if pattern_type == "Walker-Delta":
                row.update({
                    'P_planes': pattern_params['P'],
                    'F_phasing': pattern_params['F']
                })
            elif pattern_type == "Flower":
                row.update({
                    'Np': pattern_params.get('Np', 1),
                    'Nd': pattern_params.get('Nd', 1),
                    'Fn': pattern_params.get('Fn', 1),
                    'Fd': pattern_params.get('Fd', 1),
                    'Fh': pattern_params.get('Fh', 0)
                })
            
            writer.writerow(row)
    
    print(f"Results saved to: {csv_filepath}")
    
    # Create and save Pareto front plot with enhanced title
    plt.figure(figsize=(12, 9))
    plt.scatter(costs, performances, c='red', s=50, alpha=0.7, edgecolors='black')
    plt.xlabel('Total Mission Cost (M$)', fontsize=12)
    plt.ylabel('Performance Score', fontsize=12)
    
    # NEW: compute total payload mass/power for title
    try:
        tot_payload_mass_kg = sum(float(p.get('mass_kg', 0.0)) for p in (payloads or []))
        tot_payload_power_w = sum(float(p.get('power_w', 0.0)) for p in (payloads or []))
    except Exception:
        tot_payload_mass_kg, tot_payload_power_w = 0.0, 0.0

    # Enhanced title with key parameters (GS removed)
    title = (
        f'Pareto Front - {pattern_type} Constellation\n'
        f'Weights: L={local_weight}, R={regional_weight}, G={global_weight}, Sci={science_weight}\n'
        f'Pop={pop_size}, Gen={n_gen}, Step={step}s, Seed={seed}, SatMass={sat_mass_kg}kg; '
        f'DV_dep={dv_deployment_m_s:.0f} m/s, T_dep={T_deployment_days:.0f} d; '
        f'Payload={tot_payload_mass_kg:.1f} kg / {tot_payload_power_w:.0f} W'
    )
    plt.title(title, fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add some annotations for extreme points
    min_cost_idx = np.argmin(costs)
    max_perf_idx = np.argmax(performances)
    
    plt.annotate(f'Min Cost\n{costs[min_cost_idx]:.1f}M$, {performances[min_cost_idx]:.3f}',
                xy=(costs[min_cost_idx], performances[min_cost_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.annotate(f'Max Performance\n{costs[max_perf_idx]:.1f}M$, {performances[max_perf_idx]:.3f}',
                xy=(costs[max_perf_idx], performances[max_perf_idx]),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Add text box with run parameters
    textstr = f'Solutions: {len(pareto_solutions)}\nCost range: ${costs.min():.1f}-{costs.max():.1f}M\nPerf range: {performances.min():.3f}-{performances.max():.3f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    print(f"Pareto front plot saved to: {plot_filepath}")
    
    # Return results for further analysis
    results = {
        'pareto_front': pareto_front,
        'pareto_solutions': pareto_solutions,
        'costs': costs,
        'performances': performances,
        'optimizer': optimizer,
        'csv_filename': csv_filepath,
        'plot_filename': plot_filepath,
        'dynamic_directory': dynamic_dir
    }
    
    return results

# Add additional command line arguments for more control
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-objective constellation optimization')
    parser.add_argument('--pattern', default='Walker-Delta', help='Constellation pattern type')
    parser.add_argument('--gs', default='AFB_Andersen', help='Ground station')
    parser.add_argument('--pop_size', type=int, default=100, help='Population size')
    parser.add_argument('--n_gen', type=int, default=49, help='Number of generations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', default='results/MOEA', help='Base output directory')
    parser.add_argument('--step', type=int, default=1500, help='Time step for propagation (seconds)')
    parser.add_argument('--evaluation_hours', type=float, default=96, help='Evaluation time period in hours')

    # Add weight arguments for more control
    parser.add_argument('--local_weight', type=float, default=0.0, help='Local coverage weight')
    parser.add_argument('--regional_weight', type=float, default=0.0, help='Regional coverage weight')
    parser.add_argument('--global_weight', type=float, default=0.9, help='Global coverage weight')
    parser.add_argument('--science_weight', type=float, default=0.7, help='Science requirement weight')
    # Sat system args

    parser.add_argument('--target_altitude', type=float, default=600.0, help='Target altitude (km)')
    parser.add_argument('--sat_mass_kg', type=float, default=130.0, help='Satellite mass (kg)')
    parser.add_argument('--dv_deployment_m_s', type=float, default=100.0, help='Delta-V for deployment phase (m/s)')
    parser.add_argument('--T_deployment_days', type=float, default=180.0, help='Deployment time after launch (days)')
    
    # Add cluster-specific arguments
    parser.add_argument('--n_cores', type=int, default=None, help='Number of cores to use')
    parser.add_argument('--cluster_mode', action='store_true', help='Enable cluster-specific optimizations')
    
    args = parser.parse_args()

    # args.pattern = 'Flower' # Override default. Can be Walker-Delta or Flower

    args.cluster_mode = True  # To enable cluster-specific optimizations
    args.n_cores = 4  # Optional: specify number of cores
    
    args.pop_size = 120
    args.n_gen = 82

    args.sat_mass_kg = 190.0 # 95, 190, 370, 760
    # Sat   type    Payload     System      Dry mass    Propellant  Wet mass ∆V     (hydrazine
    #               mass  (kg)  mass (kg)   (kg)        mass (kg)   (kg)            -like) (m/s)
    # Micro-class   23.8        28.5        52.3        42.8        95              1290
    # Low mini-class 53.2       41.8        95.0        95.0        190             1495
    # Up mini-class 114.7       51.8        166.5       203.5       370             1723
    # Small-class   266.0       38.0        304.0       456.0       760             1977
    args.target_altitude = 500.0 # 500, 700, 1000 km
    args.dv_deployment_m_s = 100.0
    args.t_deployment_days = 180.0

    custom_payloads = [
        {'type': 'fields', 'cat': 'EO', 'mass_kg': 11.9, 'power_w': 20.0}, 
        {'type': 'particle', 'cat': 'EO', 'mass_kg': 11.9, 'power_w': 20.0}
    ] # 23.8, 53.2, 114.7, 266.0. demeter 50 kg payload at 70 W (1.4 W/kg). swarm has 1.29 W/kg for tot sc, CSES has 1.23 W/kg

    # Run optimization with command line parameters
    results = run_multi_objective_optimization(
        pattern_type=args.pattern,
        gs=args.gs,
        local_weight=args.local_weight,
        regional_weight=args.regional_weight, 
        global_weight=args.global_weight,
        science_weight=args.science_weight,
        target_altitude=args.target_altitude,
        sat_mass_kg=args.sat_mass_kg,
        payloads=custom_payloads,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        seed=args.seed,
        output_dir=args.output_dir,
        step=args.step,
        n_cores=args.n_cores,        # Cluster
        cluster_mode=args.cluster_mode,  # Cluster
        evaluation_hours=96,
        dv_deployment_m_s=args.dv_deployment_m_s,
        T_deployment_days=args.T_deployment_days
    )
    
    print("\n=== OPTIMIZATION SUMMARY ===")
    print(f"Dynamic directory: {results['dynamic_directory']}")
    print(f"CSV file: {results['csv_filename']}")
    print(f"Plot file: {results['plot_filename']}")
    print(f"Solutions found: {len(results['pareto_solutions'])}")
    
    # Show some example solutions
    costs = results['costs']
    performances = results['performances']
    
    print(f"\nExample solutions:")
    print(f"Lowest cost: {costs.min():.2f} M$ (performance: {performances[np.argmin(costs)]:.3f})")
    print(f"Highest performance: {performances.max():.3f} (cost: {costs[np.argmax(performances)]:.2f} M$)")
    
    # Find a balanced solution (closest to knee point)
    norm_costs = (costs - costs.min()) / (costs.max() - costs.min())
    norm_perfs = 1 - (performances - performances.min()) / (performances.max() - performances.min())
    
    distances = np.sqrt(norm_costs**2 + norm_perfs**2)
    balanced_idx = np.argmin(distances)
    
    print(f"Balanced solution: {costs[balanced_idx]:.2f} M$ (performance: {performances[balanced_idx]:.3f})")