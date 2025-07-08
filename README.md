# Satellite Constellation Modeling, Analysis, and Optimization Framework
High-level constellation design and optimization

## Overview

This codebase provides comprehensive tools for designing, analyzing, and optimizing satellite constellations for various space missions. It features robust simulation capabilities for evaluating coverage, visibility patterns, and performance metrics across different constellation architectures, with particular focus on Walker-Delta patterns.

## Key Components

### 1. Constellation Generation (utilities_constellations.py)
- Creates various constellation types including Walker-Delta patterns
- Supports multiple satellite distribution strategies
- Integrates with SGP4 propagator for fast orbits

### 2. Coverage Analysis (utilities_coverage.py)
- Calculates ground station visibility and access times
- Determines regional and global coverage metrics
- Evaluates field-of-view and elevation angle constraints
- Simulates coverage footprints with configurable parameters

### 3. Constellation Optimization (utilities_optimization.py)
- Multi-objective optimization of the initial conditions of the constellation for trade-off of:
  - Coverage (local, regional, global)
  - Cost factors (satellite manufacturing, launch, operations)
  - Science requirements (altitude, inclination preferences, other mission requirements)
- User can select algorithms for parameter optimization
- Provides component score breakdowns and visualizations

### 4. Coordinate Transformations (utilities.py)
- Conversions between coordinate systems (ECI, ECEF, geodetic)
- Time transformations and orbit propagation utilities
- Keplerian to Cartesian conversions and vice versa

### 5. Main Application (main.py)
- Performs comparative analysis across multiple constellation configurations
- Visualizes ground tracks, coverage areas, and visibility metrics
- Generates comprehensive performance comparison plots

## Usage Examples

### Constellation Coverage Analysis

```python
# Create a Walker-Delta constellation
const = Constellation(60, orbital_parameters, epoch, "Walker-Delta", P=5, F=1)

# Generate positions over time
result = const.generate_positions_with_time(start_time, end_time, step)

# Analyze visibility from ground station
visibility_data = compute_visibility(result["positions"], result["time_stamps"], 
                                   gs_coordinates, elevation_threshold=10.0)
```

### Constellation Optimization

```python
# Configure optimizer for regional coverage
optimizer = ConstellationOptimizer(
    pattern_type="Walker-Delta",
    regional_weight=0.6,     # Emphasize regional coverage
    global_weight=0.2,       # Some global coverage
    local_weight=0.2,        # Ground station visibility
    satellite_cost_weight=0.4,
    operational_cost_weight=0.3,
    launch_cost_weight=0.3,
)

# Run optimization
params = optimizer.optimize(max_iter=150, popsize=15)

# Analyze component scores
optimizer.analyze_component_scores(plot=True)
```

### Comparative Pattern Analysis

```python
# Define patterns to compare
pattern_list = [
    ("Walker-Delta", {"P": 5, "F": 1}),
    ("Walker-Delta", {"P": 6, "F": 1}),
    ("Walker-Delta-SinglePlane", {})
]

# Run comparison with varying satellite counts
for pattern in pattern_list:
    for num_satellites in range(1, 101):
        # Analyze and store visibility metrics
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install numpy matplotlib scipy cartopy sgp4
   ```

## Key Features

- **Multiple Constellation Patterns**: Walker-Delta, Single Plane, Custom distributions
- **Fast/Precise Orbit Propagation**: SGP4 integration for fast medium-fidelity orbit modeling, custom propagator and disturbances for high-fidelity
- **Comprehensive Metrics**: Coverage, revisit time, visibility duration, overlap
- **Advanced Visualization**: Ground tracks, coverage maps, time-series analysis
- **Multi-Objective Optimization**: Balanced design for multiple mission requirements (work in progress)
- **Cost Modeling**: Satellite manufacturing, launch, and operational cost estimates (work in progress)

## Dependencies

- NumPy: Numerical operations and array handling
- SciPy: Optimization algorithms
- Matplotlib: Plotting and visualization
- Cartopy: Geographical plotting
- SGP4: Satellite orbit propagation

## License

This project is proprietary research code developed for satellite constellation analysis.

## Author

Fabio Meloni - TU Delft MSc Student - NASA Jet Propulsion Laboratory Visiting Researcher