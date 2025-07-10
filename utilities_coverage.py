"""
Satellite Coverage Analysis Utilities
-------------------------------------

This module provides classes and functions for computing and visualizing satellite coverage over the Earth, both globally and for specific regions. It supports grid-based coverage calculations for single satellites and constellations, and includes tools for:

- Instantaneous and time-integrated coverage analysis
- Global and regional coverage computation using area-weighted grids
- Visualization of coverage maps with Cartopy and Matplotlib
- Predefined and custom region support (rectangular and polygonal)
- Utility functions for ground station locations and region management

Key Classes and Functions:
- Coverage: Main class for coverage calculations and plotting
- get_ground_station_coordinates: Lookup for ground station geodetic coordinates
- get_available_regions: List all supported region names
- get_region_coordinates: Retrieve region boundaries by name

The code is designed for flexibility and extensibility, supporting both analytic and numerical approaches to satellite coverage assessment. It is suitable for LEO, GEO, and other orbits, and can be used for mission design, trade studies, and visualization of constellation performance.

Dependencies: numpy, matplotlib, cartopy, and project-specific utilities/constants.

References
- Wertz, J.R. "Orbit and Constellation Design and Management", 2001
- Larson, W. and Wertz, J. "Space Mission Analysis and Design", 1999

Author
Fabio Meloni - TU Delft MSc Student - NASA Jet Propulsion Laboratory Visiting Researcher

"""

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

    def __init__(self, orbital_parameters, state=None, positions=None, times=None):
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
            state (dict, optional): Satellite state containing:
                - position (numpy array): Satellite position [x, y, z] in km.
                - velocity (numpy array): Satellite velocity [vx, vy, vz] in km/s.
                - time (float): Time in seconds since epoch.
            positions (numpy.ndarray, optional): Array of satellite positions over time.
                For constellation analysis: shape (n_sats, n_steps, 3)
                For single satellite: shape (n_steps, 3)
            times (numpy.ndarray, optional): Array of time stamps corresponding to positions.
        """
        self.orbital_parameters = orbital_parameters
        self.state = state
        
        # Store positions and times for coverage calculations
        self.positions = positions
        self.times = times

    def compute_sample_coverage(self, masking_angle=15.0, fov=120.0, return_circle_params=False):
        """
        Compute the coordinates of the area covered by the satellite at a specific instant.

        Args:
            masking_angle (float): Minimum elevation angle (in degrees) for coverage, the same as the elevation threshold. Default is 15°. The larger the elevation angle, the smaller the coverage area.
            fov (float): Field of view of the satellite sensor in degrees. Default is 120°, which is slightly more than 2 rad, which gives gamma>1>rho, and does not limit the cap angle.
            return_circle_params (bool): If True, return additional circle parameters for grid filling.

        Returns:
            list of tuple: List of (latitude, longitude) tuples representing the covered area.
            float: Fraction of Earth covered by the satellite.
            float: Cap angle in degrees.
            dict (optional): Circle parameters if return_circle_params is True.
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

        # Return circle parameters if requested
        if return_circle_params:
            circle_params = {
                'center_lat': sat_lat[0],
                'center_lon': sat_lon[0],
                'cap_angle': cap_angle
            }
            return coverage_points, fraction_covered, cap_angle, circle_params
        else:
            return coverage_points, fraction_covered, cap_angle

    def compute_global_coverage(self, positions, times, orbital_periods=1, masking_angle=15.0, fov=120.0, grid_density=None):
        """
        Compute the global coverage of the satellite(s) over a given time period using a grid-based approach.
        
        Args:
            positions (numpy.ndarray): Satellite position(s) at each time step. 
                                      For single satellite: shape (n_steps, 3)
                                      For constellation: shape (n_sats, n_steps, 3)
            times (numpy.ndarray): Time stamps for each position.
            orbital_periods (int, optional): Number of orbital periods to consider. Default is 1.
            masking_angle (float, optional): Minimum elevation angle in degrees for coverage. Default is 15°.
            fov (float, optional): Field of view of the satellite sensor in degrees. Default is 120°.
            grid_density (float, optional): Grid density in degrees. If None, will be calculated based on satellite altitude.
                                          Lower values create a finer grid (more accurate but slower).
        
        Returns:
            dict: Dictionary containing:
                - 'single_sat_coverage': List of coverage percentages for each satellite
                - 'constellation_coverage': Overall constellation coverage percentage
                - 'grid': The grid used for computation (for visualization or reuse)
                - 'coverage_map': 2D boolean array representing the coverage map
        """
        # Determine if we're dealing with a single satellite or constellation
        is_constellation = len(positions.shape) == 3
        
        # Extract orbital period information from the positions and times
        if is_constellation:
            num_satellites = positions.shape[0]
            positions_reshaped = positions.reshape(-1, positions.shape[-1]) if is_constellation else positions
        else:
            num_satellites = 1
            positions_reshaped = positions
        
        # Calculate mean altitude to determine grid density if not specified
        # For constellation, use average altitude of all satellites
        mean_altitude_km = 0
        for i in range(num_satellites):
            pos = positions[i] if is_constellation else positions
            # Compute average distance from Earth center for each satellite
            distances = np.linalg.norm(pos, axis=1)
            mean_altitude_km += (np.mean(distances) - constants.EARTH_RADIUS) / num_satellites
        
        # Set grid density based on altitude if not specified
        if grid_density is None:
            # Higher altitude -> coarser grid, lower altitude -> finer grid
            # A good rule of thumb is to have the grid cell size approximately equal to
            # the satellite footprint diameter divided by 10
            beta = np.radians(masking_angle)
            gamma = np.radians(fov / 2.0)
            rho = constants.EARTH_RADIUS / (constants.EARTH_RADIUS + mean_altitude_km)
            
            # Calculate cap angle using the same method as in compute_sample_coverage
            if np.sin(gamma) > rho:
                alpha = np.arccos(rho * np.cos(beta)) - beta
            else:
                alpha = np.arcsin(np.sin(gamma) / rho) - gamma
            
            # Calculate the angular size of the footprint
            footprint_angular_size = np.degrees(alpha * 2)
            
            # Set grid size to 1/10th of the footprint size, with limits
            grid_density = max(0.5, min(5.0, footprint_angular_size / 10))
            
            print(f"Auto grid density: {grid_density:.2f}° (based on mean altitude: {mean_altitude_km:.1f} km)")
        
        # Create global grid
        lat_grid = np.arange(-90, 90, grid_density)
        lon_grid = np.arange(-180, 180, grid_density)
        
        # Initialize coverage grid (weighted by area - cells near poles cover less area)
        coverage_grid = np.zeros((len(lat_grid), len(lon_grid)), dtype=bool)
        
        # Calculate area weights for each latitude (proportional to cos(lat))
        # This accounts for the fact that grid cells near poles cover less area
        area_weights = np.cos(np.radians(lat_grid))
        
        # Calculate orbital period if needed
        # We'll use the mean altitude to estimate
        orbital_period_sec = 2 * np.pi * np.sqrt((constants.EARTH_RADIUS + mean_altitude_km)**3 / constants.GRAVITATIONAL_CONSTANT)
        
        # Determine which time steps to use based on orbital periods
        max_time = times[0] + orbital_periods * orbital_period_sec
        valid_time_indices = np.where(times <= max_time)[0]
        
        # Process each satellite
        single_sat_coverage = []
        
        for sat_idx in range(num_satellites):
            # Reset grid for this satellite
            sat_coverage_grid = np.zeros_like(coverage_grid)
            
            # Get positions for this satellite
            sat_positions = positions[sat_idx] if is_constellation else positions
            
            # For each valid time step
            for t_idx in valid_time_indices:
                # Get satellite position at this time step
                sat_position = sat_positions[t_idx]
                time = times[t_idx]
                
                # Create state dictionary for this position and time
                state = {"position": sat_position, "time": time}
                
                # Create a temporary Coverage object for this state
                temp_coverage = Coverage(self.orbital_parameters, state)
                
                # Compute coverage points and get circle parameters
                coverage_points, _, _, circle_params = temp_coverage.compute_sample_coverage(
                    masking_angle=masking_angle,
                    fov=fov,
                    return_circle_params=True
                )
                
                center_lat = circle_params['center_lat']
                center_lon = circle_params['center_lon']
                cap_angle = circle_params['cap_angle']
                
                # Fill in the entire coverage circle, not just the perimeter
                # For optimization, only check grid cells within the bounding box of the coverage circle
                # Convert cap_angle from degrees to km at the Earth's surface
                cap_angle_km = cap_angle * (constants.EARTH_RADIUS * np.pi / 180)
                
                # Calculate a bounding box around the coverage circle center
                # This helps avoid checking every grid cell
                lat_min_idx = max(0, np.searchsorted(lat_grid, center_lat - cap_angle) - 1)
                lat_max_idx = min(len(lat_grid), np.searchsorted(lat_grid, center_lat + cap_angle) + 1)
                lon_min_idx = max(0, np.searchsorted(lon_grid, center_lon - cap_angle * 1.5) - 1)  # Wider in longitude
                lon_max_idx = min(len(lon_grid), np.searchsorted(lon_grid, center_lon + cap_angle * 1.5) + 1)
                
                # Check each grid cell within the bounding box
                for lat_idx in range(lat_min_idx, lat_max_idx):
                    lat = lat_grid[lat_idx]
                    for lon_idx in range(lon_min_idx, lon_max_idx):
                        lon = lon_grid[lon_idx]
                        # Check if this grid point is inside the coverage circle
                        if Coverage.is_point_in_coverage_circle(lat, lon, center_lat, center_lon, cap_angle):
                            sat_coverage_grid[lat_idx, lon_idx] = True
            
            # Calculate coverage percentage for this satellite (weighted by area)
            # Calculate the total area of the Earth grid
            total_weighted_area = np.sum(area_weights) * len(lon_grid)
            
            # Calculate the area covered by this satellite
            weighted_sat_area = np.sum(sat_coverage_grid * area_weights[:, np.newaxis])
            
            # Coverage percentage is covered area divided by total area
            coverage_percentage = (weighted_sat_area / total_weighted_area) * 100
            
            single_sat_coverage.append(coverage_percentage)  # Already as a percentage
            
            # Update overall coverage grid (union of all satellites)
            coverage_grid = coverage_grid | sat_coverage_grid
        
        # Calculate constellation coverage percentage (weighted by area)
        # Calculate the total area of the Earth grid
        total_weighted_area = np.sum(area_weights) * len(lon_grid)
        
        # Calculate the area covered by the constellation (using the combined coverage grid)
        weighted_constellation_area = np.sum(coverage_grid * area_weights[:, np.newaxis])
        
        # Coverage percentage is covered area divided by total area
        constellation_coverage_percentage = (weighted_constellation_area / total_weighted_area) * 100
        
        # Print debug information
        print(f"Total grid cells: {coverage_grid.size}")
        print(f"Covered grid cells: {np.sum(coverage_grid)}")
        print(f"Coverage fraction: {np.sum(coverage_grid) / coverage_grid.size:.4f}")
        print(f"Raw coverage percentage: {np.sum(coverage_grid) / coverage_grid.size * 100:.2f}%")
        print(f"Area-weighted coverage percentage: {constellation_coverage_percentage:.2f}%")
        
        # Print per-satellite coverage
        for i, cov in enumerate(single_sat_coverage):
            print(f"  Satellite {i+1}: {cov:.2f}% area-weighted coverage")
        
        # Create result dictionary
        result = {
            'single_sat_coverage': single_sat_coverage,
            'constellation_coverage': constellation_coverage_percentage,  # Already as a percentage
            'grid': {'lat': lat_grid, 'lon': lon_grid, 'density': grid_density},
            'coverage_map': coverage_grid
        }
        
        return result

    def compute_regional_coverage(self, positions, times, region_bounds_or_name, orbital_periods=1, 
                            masking_angle=15.0, fov=120.0, grid_density=None):
        """
        Compute the coverage of a specific region by the satellite(s) over a given time period.
        
        Args:
            positions (numpy.ndarray): Satellite position(s) at each time step.
                                      For single satellite: shape (n_steps, 3)
                                      For constellation: shape (n_sats, n_steps, 3)
            times (numpy.ndarray): Time stamps for each position.
            region_bounds_or_name (str, tuple, or list): Either:
                                                      - A region name (str) like "California", "Italy", "Japan", etc.
                                                      - Region boundaries as (min_lat, max_lat, min_lon, max_lon)
                                                      - List of (lat, lon) points defining a polygon.
            orbital_periods (int, optional): Number of orbital periods to consider. Default is 1.
            masking_angle (float, optional): Minimum elevation angle in degrees for coverage. Default is 15°.
            fov (float, optional): Field of view of the satellite sensor in degrees. Default is 120°.
            grid_density (float, optional): Grid density in degrees. If None, will be calculated based on satellite altitude.
                                          Lower values create a finer grid (more accurate but slower).
        
        Returns:
            dict: Dictionary containing:
                - 'single_sat_coverage': List of coverage percentages for each satellite
                - 'constellation_coverage': Overall constellation coverage percentage
                - 'grid': The grid used for computation
                - 'coverage_map': 2D boolean array representing the coverage map
                - 'region_mask': 2D boolean array representing the region of interest
                - 'region_name': Name of the region (if a named region was provided)
        """
        # Check if a region name was provided instead of coordinates
        region_name = None
        if isinstance(region_bounds_or_name, str):
            region_name = region_bounds_or_name
            try:
                region_bounds = get_region_coordinates(region_name)
            except ValueError as e:
                raise ValueError(f"Unknown region name: '{region_name}'. "
                               f"Available regions: {', '.join(get_available_regions())}")
        else:
            region_bounds = region_bounds_or_name
        
        # Determine region type (rectangle or polygon)
        if isinstance(region_bounds, tuple) and len(region_bounds) == 4:
            # Rectangle region
            min_lat, max_lat, min_lon, max_lon = region_bounds
            is_polygon = False
        else:
            # Polygon region
            is_polygon = True
            region_points = np.array(region_bounds)
            min_lat, max_lat = np.min(region_points[:, 0]), np.max(region_points[:, 0])
            min_lon, max_lon = np.min(region_points[:, 1]), np.max(region_points[:, 1])
            
            # Add a small buffer to ensure we cover the entire region
            lat_buffer = (max_lat - min_lat) * 0.05
            lon_buffer = (max_lon - min_lon) * 0.05
            min_lat -= lat_buffer
            max_lat += lat_buffer
            min_lon -= lon_buffer
            max_lon += lon_buffer
        
        # First compute global coverage with the same parameters
        global_coverage_result = self.compute_global_coverage(
            positions, times, orbital_periods, masking_angle, fov, grid_density
        )
        
        # Extract grid and coverage map
        grid = global_coverage_result['grid']
        coverage_map = global_coverage_result['coverage_map']
        
        # Create region mask on the same grid
        lat_grid = grid['lat']
        lon_grid = grid['lon']
        region_mask = np.zeros((len(lat_grid), len(lon_grid)), dtype=bool)
        
        # Fill region mask
        for i, lat in enumerate(lat_grid):
            for j, lon in enumerate(lon_grid):
                if is_polygon:
                    # Check if point is inside polygon using ray casting algorithm
                    inside = Coverage._point_in_polygon(lat, lon, region_bounds)
                else:
                    # Check if point is inside rectangle
                    inside = (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)
                
                if inside:
                    region_mask[i, j] = True
        
        # Calculate area weights for each latitude (proportional to cos(lat))
        area_weights = np.cos(np.radians(lat_grid))
        
        # Calculate coverage within the region for each satellite
        is_constellation = len(positions.shape) == 3
        num_satellites = positions.shape[0] if is_constellation else 1
        
        # Compute region statistics
        single_sat_coverage = []
        single_sat_region_maps = []
        
        for sat_idx in range(num_satellites):
            # Get coverage grid for this satellite
            if num_satellites > 1:
                # Extract individual satellite coverage from the global result
                # We need to recompute this for each satellite
                sat_positions = positions[sat_idx:sat_idx+1] if is_constellation else positions
                sat_result = self.compute_global_coverage(
                    sat_positions, times, orbital_periods, masking_angle, fov, grid_density
                )
                sat_coverage_grid = sat_result['coverage_map']
            else:
                sat_coverage_grid = coverage_map
            
            # Apply region mask
            sat_region_coverage = sat_coverage_grid & region_mask
            single_sat_region_maps.append(sat_region_coverage)
            
            # Calculate coverage percentage (weighted by area)
            if np.any(region_mask):
                weighted_coverage = np.sum(sat_region_coverage * area_weights[:, np.newaxis])
                weighted_region = np.sum(region_mask * area_weights[:, np.newaxis])
                coverage_percentage = (weighted_coverage / weighted_region) * 100 if weighted_region > 0 else 0
            else:
                coverage_percentage = 0
                
            single_sat_coverage.append(coverage_percentage)
        
        # Calculate constellation coverage within the region
        constellation_region_coverage = coverage_map & region_mask
        
        if np.any(region_mask):
            weighted_coverage = np.sum(constellation_region_coverage * area_weights[:, np.newaxis])
            weighted_region = np.sum(region_mask * area_weights[:, np.newaxis])
            constellation_coverage_percentage = (weighted_coverage / weighted_region) * 100 if weighted_region > 0 else 0
            
            # Print debug information for regional coverage
            print(f"Region grid cells: {np.sum(region_mask)}")
            print(f"Covered region grid cells: {np.sum(constellation_region_coverage)}")
            print(f"Region coverage fraction: {np.sum(constellation_region_coverage) / np.sum(region_mask):.4f}")
            print(f"Area-weighted region coverage percentage: {constellation_coverage_percentage:.2f}%")
        else:
            constellation_coverage_percentage = 0
        
        # Create result dictionary
        result = {
            'single_sat_coverage': single_sat_coverage,
            'constellation_coverage': constellation_coverage_percentage,
            'grid': grid,
            'coverage_map': constellation_region_coverage,
            'region_mask': region_mask,
            'region_bounds': region_bounds
        }
        
        # Add region name if provided
        if region_name:
            result['region_name'] = region_name
        
        return result
    
    @staticmethod
    def _point_in_polygon(lat, lon, polygon):
        """
        Determine if a point is inside a polygon using ray casting algorithm.
        
        Args:
            lat, lon (float): Latitude and longitude of the point to check
            polygon (list): List of (lat, lon) points defining the polygon vertices
            
        Returns:
            bool: True if point is inside polygon, False otherwise
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
    
    @staticmethod
    def plot_coverage_map(coverage_result, title=None, show_region=True):
        """
        Plot the coverage map from compute_global_coverage or compute_regional_coverage results.
        
        Args:
            coverage_result (dict): Result from compute_global_coverage or compute_regional_coverage
            title (str, optional): Title for the plot. If None, a default title is used.
            show_region (bool, optional): Whether to highlight the region boundaries (for regional coverage)
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        grid = coverage_result['grid']
        coverage_map = coverage_result['coverage_map']
        
        # Create a figure with PlateCarree projection
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Add map features
        ax.coastlines(resolution='50m')
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        ax.add_feature(ccrs.cartopy.feature.BORDERS, linestyle=':')
        ax.stock_img()  # Add NASA Blue Marble image as background
        
        # Create a meshgrid for plotting
        lon_grid, lat_grid = np.meshgrid(grid['lon'], grid['lat'])
        
        # Check if this is regional coverage
        is_regional = 'region_mask' in coverage_result
        
        # Plot coverage map as binary (covered or not covered) with custom colormap
        # Create a binary colormap: transparent for not covered, blue for covered
        from matplotlib.colors import ListedColormap
        binary_cmap = ListedColormap(['none', 'blue'])
        
        # Use binary map with high contrast
        im = ax.pcolormesh(lon_grid, lat_grid, coverage_map.astype(float), 
                          transform=ccrs.PlateCarree(), alpha=0.5, cmap=binary_cmap, 
                          vmin=0, vmax=1)
        
        # If this is regional coverage and show_region is True, highlight the region
        if is_regional and show_region:
            region_bounds = coverage_result.get('region_bounds')
            
            # Check if region is defined as rectangle or polygon
            if isinstance(region_bounds, tuple) and len(region_bounds) == 4:
                # Rectangle region
                min_lat, max_lat, min_lon, max_lon = region_bounds
                # Draw rectangle
                rect = plt.Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                                   fill=False, edgecolor='red', linewidth=2, transform=ccrs.PlateCarree())
                ax.add_patch(rect)
            else:
                # Polygon region
                region_points = np.array(region_bounds)
                # Draw polygon
                ax.plot(region_points[:, 1], region_points[:, 0], 'r-', linewidth=2, transform=ccrs.PlateCarree())
                ax.plot([region_points[-1, 1], region_points[0, 1]], 
                       [region_points[-1, 0], region_points[0, 0]], 'r-', linewidth=2, 
                       transform=ccrs.PlateCarree())
        
        # Add a simpler legend instead of a continuous colorbar
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.5, label='Covered')
        ]
        ax.legend(handles=legend_elements, loc='lower left')
        
        # Set plot title
        if title is None:
            constellation_coverage = coverage_result['constellation_coverage']
            if is_regional:
                region_name = coverage_result.get('region_name', 'Regional')
                title = f"{region_name} Coverage Map (Total: {constellation_coverage:.2f}%)"
            else:
                title = f"Global Coverage Map (Total: {constellation_coverage:.2f}%)"
        ax.set_title(title)
        
        # Add coverage information as text box
        info_text = f"Coverage: {coverage_result['constellation_coverage']:.2f}%\n"
        info_text += f"Grid Density: {grid['density']:.2f}°"
        
        # If regional coverage, add region information
        if is_regional:
            region_name = coverage_result.get('region_name', 'Region')
            info_text = f"{region_name} Coverage: {coverage_result['constellation_coverage']:.2f}%\n"
            
            # Calculate region area (approximate)
            if 'region_mask' in coverage_result:
                region_mask = coverage_result['region_mask']
                area_weights = np.cos(np.radians(grid['lat']))
                weighted_region = np.sum(region_mask * area_weights[:, np.newaxis])
                total_weighted = np.sum(area_weights) * len(grid['lon'])
                region_percentage = (weighted_region / total_weighted) * 100
                info_text += f"Region Area: {region_percentage:.2f}% of Earth\n"
            
            info_text += f"Grid Density: {grid['density']:.2f}°"
        
        # Add individual satellite coverage if available
        if len(coverage_result['single_sat_coverage']) > 1:
            info_text += f"\nSatellites: {len(coverage_result['single_sat_coverage'])}"
            info_text += f"\nAvg Sat Coverage: {np.mean(coverage_result['single_sat_coverage']):.2f}%"
            info_text += f"\nMax Sat Coverage: {np.max(coverage_result['single_sat_coverage']):.2f}%"
        
        # Place text box in bottom right corner
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.97, 0.03, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        return fig
    
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
    
    @staticmethod
    def is_point_in_coverage_circle(point_lat, point_lon, center_lat, center_lon, cap_angle):
        """
        Check if a point is inside a coverage circle defined by center and cap angle.
        
        Args:
            point_lat, point_lon (float): Latitude and longitude of the point to check (in degrees)
            center_lat, center_lon (float): Latitude and longitude of the circle center (in degrees)
            cap_angle (float): The cap angle (radius) of the circle in degrees
            
        Returns:
            bool: True if the point is inside the circle, False otherwise
        """
        # Convert to radians
        point_lat_rad = np.radians(point_lat)
        point_lon_rad = np.radians(point_lon)
        center_lat_rad = np.radians(center_lat)
        center_lon_rad = np.radians(center_lon)
        cap_angle_rad = np.radians(cap_angle)
        
        # Calculate the great circle distance using the haversine formula
        dlon = point_lon_rad - center_lon_rad
        dlat = point_lat_rad - center_lat_rad
        a = np.sin(dlat/2)**2 + np.cos(point_lat_rad) * np.cos(center_lat_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Compare with cap angle
        return c <= cap_angle_rad
    

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


def get_available_regions():
    """
    Returns a list of all available predefined regions.
    
    Returns:
        list: List of region names
    """
    return sorted(list(_REGIONS.keys()))


def get_region_coordinates(region_name):
    """
    Returns the boundaries for predefined geographical regions.
    
    Args:
        region_name (str): Name of the region
        
    Returns:
        list or tuple: Region boundaries. For rectangular regions: (min_lat, max_lat, min_lon, max_lon)
                      For polygon regions: list of (lat, lon) coordinates
    """
    if region_name in _REGIONS:
        return _REGIONS[region_name]
    else:
        raise ValueError(f"Unknown region name: {region_name}")


# Dictionary of predefined regions
_REGIONS = {
    # Rectangular regions (min_lat, max_lat, min_lon, max_lon)
    "Western_US": (32.0, 49.0, -125.0, -105.0),
    "Eastern_US": (25.0, 47.0, -90.0, -70.0),
    "Continental_US": (25.0, 49.0, -125.0, -66.0),
    "Europe": (35.0, 60.0, -10.0, 25.0),
    "East_Asia": (20.0, 45.0, 100.0, 145.0),
    "South_America": (-55.0, 12.0, -80.0, -35.0),
    "Africa": (-35.0, 37.0, -20.0, 50.0),
    "Australia": (-45.0, -10.0, 110.0, 155.0),
    "Arctic": (66.5, 90.0, -180.0, 180.0),
    "Antarctica": (-90.0, -60.0, -180.0, 180.0),
    "Japan": (31.0, 45.5, 129.5, 145.8),
    "Italy": (36.6, 47.1, 6.6, 18.5),
    "United_Kingdom": (49.9, 58.7, -8.2, 1.8),
    "India": (8.0, 36.0, 68.0, 97.0),
    "Brazil": (-33.8, 5.3, -74.0, -34.8),
    "Turkey": (36.0, 42.0, 26.0, 45.0),
    "Central_Mexico": (17.0, 24.0, -105.0, -96.0),
    "Alaska": (51.0, 71.0, -170.0, -130.0),
    "Hawaii": (19.0, 22.0, -160.0, -154.0),
    "Iran": (25.0, 40.0, 44.0, 63.0),
    "Nepal": (26.0, 30.0, 80.0, 88.0),
    "Philippines": (5.0, 19.0, 117.0, 126.0),
    "New_Zealand": (-47.0, -34.0, 166.0, 179.0),
    "Chile": (-56.0, -17.0, -76.0, -67.0),
    "Indonesia": (-6.0, 6.0, 95.0, 141.0),
    "Greece": (35.0, 42.0, 19.0, 28.0),
    "Central_China": (30.0, 40.0, 100.0, 120.0),
    "Kamchatka": (51.0, 65.0, 156.0, 174.0),  # Kamchatka Peninsula, Russia
    "Caribbean": (10.0, 25.0, -90.0, -60.0),  # Caribbean Sea and islands
    "Eastern_Europe": (44.0, 56.0, 19.0, 40.0),  # Eastern Europe (excluding Russia)
    "Central_Asia_Eastern_China": (30.0, 50.0, 50.0, 110.0),  # Central Asia and western China, Mongolia, east of Kazakhstan, Nepal
    "Western_South_America_Coast": (-56.0, 2.0, -81.5, -66.0),  # Chile, Peru, Ecuador, seismic west coast
    "West_Coast_US": (32.5, 49.0, -125.0, -116.0),  # California, Oregon, Washington only
    
    # Polygon regions (list of lat, lon coordinates)
    "California": [
        (42.0, -124.4),  # Northwest corner
        (42.0, -120.0),  # Northeast corner
        (39.0, -114.6),  # East
        (35.0, -114.6),  # Southeast
        (32.5, -114.6),  # South border with Mexico
        (32.5, -117.0),  # Southwest
        (33.5, -118.0),  # Southern California coast
        (34.5, -120.5),  # Central California coast
        (38.0, -123.0),  # Northern California coast
        (42.0, -124.4)   # Back to start
    ],
    "Mediterranean": [
        (30.0, -5.0),    # Southwest (Morocco)
        (30.0, 36.0),    # Southeast (Middle East)
        (47.0, 36.0),    # Northeast (Ukraine)
        (47.0, -5.0)     # Northwest (France)
    ],
    "Iberian_Peninsula": [
        (43.8, -9.3),    # Northwest
        (43.8, 3.3),     # Northeast
        (36.0, 3.3),     # Southeast
        (36.0, -9.3),    # Southwest
        (43.8, -9.3)     # Close the polygon
    ],
    "Southeast_Asia": [
        (28.5, 92.0),    # North
        (28.5, 130.0),   # Northeast
        (0.0, 130.0),    # Southeast
        (0.0, 92.0),     # Southwest
        (28.5, 92.0)     # Close the polygon
    ],
    "Middle_East": [
        (39.0, 26.0),    # Northwest (Turkey)
        (39.0, 62.0),    # Northeast (Iran)
        (12.0, 62.0),    # Southeast (Arabian Sea)
        (12.0, 26.0),    # Southwest (Red Sea)
        (39.0, 26.0)     # Close the polygon
    ],
    "North_Africa": [
        (37.0, -17.0),   # Northwest (Morocco coast)
        (37.0, 36.0),    # Northeast (Turkey)
        (15.0, 36.0),    # Southeast (Sudan)
        (15.0, -17.0),   # Southwest (Mauritania)
        (37.0, -17.0)    # Close the polygon
    ],
    "Great_Lakes": [
        (49.5, -93.0),   # Northwest
        (49.5, -76.0),   # Northeast
        (41.0, -76.0),   # Southeast
        (41.0, -93.0),   # Southwest
        (49.5, -93.0)    # Close the polygon
    ]
}


