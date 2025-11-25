"""Repeat Ground Track (RGT) Phasing Utilities

Provides the `PhasingRGT` class plus convenience module functions for designing
repeat ground track (RGT) orbits under J2 perturbation.

Quick Reference
---------------
Goal: find semi-major axis a (and implied altitude) that satisfies a desired
repeat ratio No/Nd (satellite revolutions per sidereal day) given (inclination i, eccentricity e).

Core Class: PhasingRGT(J2=None, Rearth=None, mu=None, rotation_rate_earth=None)
    J2, Rearth, mu            : Earth constants (defaults pulled from `constants`).
    rotation_rate_earth       : Earth's sidereal rotation rate (rad/s), default 7.2921159e-5.

Key Methods
    compute_meanmotion(a)                     -> n (rad/s)
    compute_precession_Omega_RAAN(i,a,e)      -> Ω̇ (rad/s)
    compute_precession_omega_argperiapsis(i,a,e) -> ω̇ (rad/s)
    compute_precession_M(i,a,e)               -> Ṁ (rad/s)
    target_function(a,i,e,No,Nd)              -> residual for RGT condition
    solve_for_semi_major_axis(i,e,No,Nd, ...) -> (a_km, converged_bool, iterations)
    find_a_for_ratio_at_inclination(i_deg,e,ratio, ...) -> (altitude_km, a_km or None)
    plot_rgt_vs_inclinations(ratios, ...)     -> saves altitude vs inclination curves
    validate_solution(a,i,e,No,Nd)            -> prints diagnostic check of closure

RGT Condition Implemented
    (Ṁ + ω̇) / (Ω_E - Ω̇) ≈ No / Nd
    where Ω_E is Earth's rotation rate.

Typical Use
-----------
from reconfig_phasing_RGT import PhasingRGT
ph = PhasingRGT()
alt_km, a_km = ph.find_a_for_ratio_at_inclination(i_deg=51.6, e=0.0, ratio=15.0)
print(f"Target altitude ≈ {alt_km:.1f} km (a = {a_km:.1f} km)")

Outputs
    Altitude (km)  : a - Rearth
    Semi-major axis: solution a (km)
    Convergence meta: iterations + boolean (from solver)

Module-Level Convenience Functions
    compute_meanmotion, compute_precession_* wrappers use a default singleton instance.

Author
Fabio Meloni - TU Delft MSc Student - NASA Jet Propulsion Laboratory Visiting Researcher
"""

import numpy as np
import constants
import matplotlib.pyplot as plt
from fractions import Fraction

class PhasingRGT:
    """
    Class for computing repeat ground track (RGT) orbital parameters and solving
    for semi-major axes given inclination, eccentricity, and No/Nd ratios.
    """
    
    def __init__(self, planet='earth', J2=None, Rearth=None, mu=None, rotation_rate_earth=None):
        """
        Initialize PhasingRGT with planetary constants.
        
        Parameters:
        -----------
        planet : str, optional
            'earth' or 'mars' (default: 'earth')
        J2 : float, optional
            Planet's J2 coefficient (overrides planet default)
        Rearth : float, optional
            Planet's radius in km (overrides planet default)
        mu : float, optional
            Planet's gravitational parameter in km³/s² (overrides planet default)
        rotation_rate_earth : float, optional
            Planet's rotation rate in rad/s (overrides planet default)
        """

        if planet.lower() == 'venus':
            self.J2 = J2 if J2 is not None else constants.VENUS_J2
            self.Rearth = Rearth if Rearth is not None else constants.VENUS_RADIUS
            self.mu = mu if mu is not None else constants.VENUS_MU
            self.rotation_rate_earth = rotation_rate_earth if rotation_rate_earth is not None else constants.VENUS_ROTATION_RATE
            self.planet_name = 'Venus'
        elif planet.lower() == 'mars':
            self.J2 = J2 if J2 is not None else constants.MARS_J2
            self.Rearth = Rearth if Rearth is not None else constants.MARS_RADIUS
            self.mu = mu if mu is not None else constants.MARS_MU
            self.rotation_rate_earth = rotation_rate_earth if rotation_rate_earth is not None else constants.MARS_ROTATION_RATE
        else:  # default to Earth
            self.J2 = J2 if J2 is not None else constants.J2_EARTH
            self.Rearth = Rearth if Rearth is not None else constants.EARTH_RADIUS
            self.mu = mu if mu is not None else constants.GRAVITATIONAL_CONSTANT
            self.rotation_rate_earth = rotation_rate_earth if rotation_rate_earth is not None else 7.2921159e-5
    
    def compute_transfer_time(self, aTRANSF):
        """Compute Hohmann transfer time for given semi-major axis."""
        return np.pi * np.sqrt(aTRANSF**3 / self.mu)
    
    def compute_meanmotion(self, a):
        """Compute mean motion for given semi-major axis."""
        return np.sqrt(self.mu / a**3)
    
    def compute_semilatus_rectum(self, a, e):
        """Compute semi-latus rectum for given semi-major axis and eccentricity."""
        return a * (1 - e**2)
    
    def compute_precession_Omega_RAAN(self, i, a, e):
        """Compute RAAN precession rate due to J2."""
        n = self.compute_meanmotion(a)
        p = self.compute_semilatus_rectum(a, e)
        Omega_dot = -(3 * n * (self.Rearth**2) * self.J2 / (2 * p**2)) * np.cos(np.radians(i))
        return Omega_dot  # in rad/s
    
    def compute_precession_M(self, i, a, e):
        """Compute mean anomaly precession rate due to J2."""
        n = self.compute_meanmotion(a)
        p = self.compute_semilatus_rectum(a, e)
        M_dot = ((3 * n * (self.Rearth**2) * self.J2 / (4 * p**2)) *
                 (2 - 3 * (np.sin(np.radians(i)))**2) * np.sqrt(1 - e**2)) + n
        return M_dot  # in rad/s
    
    def compute_precession_omega_argperiapsis(self, i, a, e):
        """Compute argument of periapsis precession rate due to J2."""
        n = self.compute_meanmotion(a)
        p = self.compute_semilatus_rectum(a, e)
        omega_dot = (3 * n * (self.Rearth**2) * self.J2 / (4 * p**2)) * (4 - 5 * (np.sin(np.radians(i)))**2)
        return omega_dot  # in rad/s
    
    def compute_longitude_of_groundtrack(self, No, Omega, Nd, M):
        """Compute longitude of ground track."""
        return No * Omega + Nd * M
    
    def target_function(self, a, i, e, No, Nd):
        """
        Target function for iterative solver.
        Returns: No/Nd - (M_dot + omega_dot) / (rotation_rate_earth - Omega_dot)
        This should equal zero when the correct semi-major axis 'a' is found.
        
        For retrograde planets (Venus), uses absolute value of rotation rate
        and adjusts sign conventions.
        """
        M_dot = self.compute_precession_M(i, a, e)
        omega_dot = self.compute_precession_omega_argperiapsis(i, a, e)
        Omega_dot = self.compute_precession_Omega_RAAN(i, a, e)
        
        target_ratio = No / Nd
        
        # Handle retrograde rotation (Venus)
        if self.rotation_rate_earth < 0:
            # For retrograde planets, use absolute value and flip sign convention
            denominator = abs(self.rotation_rate_earth) + Omega_dot
        else:
            # Normal prograde case (Earth, Mars)
            denominator = self.rotation_rate_earth - Omega_dot
            
        computed_ratio = (M_dot + omega_dot) / denominator
        
        return target_ratio - computed_ratio
    
    def debug_calculations(self, a, i, e, No, Nd):
        """Debug function to check intermediate values."""
        print(f"\nDebug for a = {a:.3f} km:")
        n = self.compute_meanmotion(a)
        p = self.compute_semilatus_rectum(a, e)
        print(f"Mean motion n: {n:.6e} rad/s")
        print(f"Semi-latus rectum p: {p:.3f} km")
        
        M_dot = self.compute_precession_M(i, a, e)
        omega_dot = self.compute_precession_omega_argperiapsis(i, a, e)
        Omega_dot = self.compute_precession_Omega_RAAN(i, a, e)
        
        print(f"M_dot: {M_dot:.6e} rad/s")
        print(f"omega_dot: {omega_dot:.6e} rad/s")
        print(f"Omega_dot: {Omega_dot:.6e} rad/s")
        
        target_ratio = No / Nd
        computed_ratio = (M_dot + omega_dot) / (self.rotation_rate_earth - Omega_dot)
        
        print(f"Target ratio (No/Nd): {target_ratio:.6f}")
        print(f"Computed ratio: {computed_ratio:.6f}")
        print(f"Function value: {target_ratio - computed_ratio:.6e}")
        
        return M_dot, omega_dot, Omega_dot
    
    def solve_for_semi_major_axis(self, i, e, No, Nd, a_initial_guess=None, 
                                  tolerance=1e-9, max_iterations=1000):
        """
        Robust solver for semi-major axis 'a'.
        Strategy:
        1. Try to bracket a root by expanding around the initial guess.
        2. If bracket found, use bisection (robust, derivative-free).
        3. If no bracket found, fall back to a guarded secant method.
        
        Parameters:
        -----------
        i : float
            Inclination in degrees
        e : float
            Eccentricity
        No : float
            Number of revolutions
        Nd : float
            Number of days
        a_initial_guess : float, optional
            Initial guess for semi-major axis in km (default: 6371+450)
        tolerance : float, optional
            Convergence tolerance (default: 1e-9)
        max_iterations : int, optional
            Maximum number of iterations (default: 1000)
            
        Returns:
        --------
        tuple
            (a, converged, iterations) where:
            - a: semi-major axis in km
            - converged: boolean indicating convergence
            - iterations: number of iterations used
        """
        if a_initial_guess is None:
            a_initial_guess = self.Rearth + 450
            
        a0 = float(a_initial_guess)
        f0 = self.target_function(a0, i, e, No, Nd)
        if abs(f0) < tolerance:
            return a0, True, 0

        # Initial bracket around a0
        R_min = 3000.0
        R_max = 20000.0
        a_low = max(R_min, a0 - 500.0)
        a_high = min(R_max, a0 + 500.0)
        f_low = self.target_function(a_low, i, e, No, Nd)
        f_high = self.target_function(a_high, i, e, No, Nd)

        # Expand bracket until sign change or until limits reached
        expand_step = 500.0
        expand_iters = 0
        while f_low * f_high > 0 and expand_iters < 40:
            expand_iters += 1
            a_low = max(R_min, a_low - expand_step)
            a_high = min(R_max, a_high + expand_step)
            f_low = self.target_function(a_low, i, e, No, Nd)
            f_high = self.target_function(a_high, i, e, No, Nd)
            if a_low == R_min and a_high == R_max:
                break

        # If we have a valid bracket, do bisection
        if f_low * f_high <= 0:
            for iteration in range(max_iterations):
                a_mid = 0.5 * (a_low + a_high)
                f_mid = self.target_function(a_mid, i, e, No, Nd)

                if not np.isfinite(f_mid):
                    a_low += 1e-3
                    a_high -= 1e-3
                    continue

                if abs(f_mid) < tolerance or (a_high - a_low) * 0.5 < 1e-6:
                    return a_mid, True, iteration + 1

                if f_low * f_mid <= 0:
                    a_high = a_mid
                    f_high = f_mid
                else:
                    a_low = a_mid
                    f_low = f_mid

            return a_mid, False, max_iterations

        # If no bracket found, fallback to guarded secant method
        a_prev = a0
        a_curr = a0 + 100.0
        f_prev = f0
        f_curr = self.target_function(a_curr, i, e, No, Nd)

        for iteration in range(max_iterations):
            denom = (f_curr - f_prev)
            if abs(denom) < 1e-16:
                a_curr += 50.0
                f_curr = self.target_function(a_curr, i, e, No, Nd)
                denom = (f_curr - f_prev)
                if abs(denom) < 1e-16:
                    break

            a_next = a_curr - f_curr * (a_curr - a_prev) / denom
            max_step = 200.0
            step = a_next - a_curr
            if abs(step) > max_step:
                a_next = a_curr + np.sign(step) * max_step

            if a_next < R_min:
                a_next = R_min + 1.0
            if a_next > R_max:
                a_next = R_max - 1.0

            f_next = self.target_function(a_next, i, e, No, Nd)

            if abs(f_next) < tolerance or abs(a_next - a_curr) < 1e-6:
                return a_next, True, iteration + 1

            a_prev, f_prev = a_curr, f_curr
            a_curr, f_curr = a_next, f_next

        return a_curr, False, max_iterations
    
    def validate_solution(self, a, i, e, No, Nd):
        """Validate the solution by computing all components and checking the relationship."""
        M_dot = self.compute_precession_M(i, a, e)
        omega_dot = self.compute_precession_omega_argperiapsis(i, a, e)
        Omega_dot = self.compute_precession_Omega_RAAN(i, a, e)
        
        target_ratio = No / Nd
        computed_ratio = (M_dot + omega_dot) / (self.rotation_rate_earth - Omega_dot)
        
        print(f"\nValidation Results:")
        print(f"Semi-major axis: {a:.8f} km")
        print(f"Altitude: {a - self.Rearth:.8f} km")
        print(f"Target ratio (No/Nd): {target_ratio:.8f}")
        print(f"Computed ratio: {computed_ratio:.8f}")
        print(f"Difference: {abs(target_ratio - computed_ratio):.2e}")
        print(f"M_dot: {M_dot:.6e} rad/s")
        print(f"omega_dot: {omega_dot:.6e} rad/s")
        print(f"Omega_dot: {Omega_dot:.6e} rad/s")
        print(f"rotation_rate_earth - Omega_dot: {self.rotation_rate_earth - Omega_dot:.6e}")
    
    def find_a_for_ratio_at_inclination(self, i_deg, e, ratio, a_guess=None):
        """
        Find semi-major axis for given inclination and No/Nd ratio.
        
        Parameters:
        -----------
        i_deg : float
            Inclination in degrees
        e : float
            Eccentricity
        ratio : float
            No/Nd ratio
        a_guess : float, optional
            Initial guess for semi-major axis
            
        Returns:
        --------
        tuple
            (altitude_km, semi_major_axis_km) or (np.nan, None) if not found
        """
        No_local = float(ratio)
        Nd_local = 1.0
        if a_guess is None:
            a_guess = self.Rearth + 500.0
            
        try:
            a_sol, converged, iters = self.solve_for_semi_major_axis(
                i_deg, e, No_local, Nd_local, a_initial_guess=a_guess
            )
            if converged:
                return a_sol - self.Rearth, a_sol  # altitude, semi-major axis
            else:
                return np.nan, None
        except Exception:
            return np.nan, None
    
    def _ratio_to_fraction_label(self, ratio, max_denominator=100):
        """Convert a float ratio to a 'num/den' string."""
        frac = Fraction(ratio).limit_denominator(max_denominator)
        return f"{frac.numerator}/{frac.denominator}"
    
    def plot_rgt_vs_inclinations(self, ratios=None, incl_step=5, a_initial_guess_map=None, 
                                e=0.0, savefile="results/rgt_vs_inclination.png"):
        """
        Plot RGT altitude vs inclination for different No/Nd ratios.
        
        Parameters:
        -----------
        ratios : list, optional
            List of No/Nd ratios to plot (default: [14.0, 14.5, 15.0, 15.5, 16.0])
        incl_step : float, optional
            Inclination step size in degrees (default: 5)
        a_initial_guess_map : dict, optional
            Map of ratio to initial guess for semi-major axis
        e : float, optional
            Eccentricity (default: 0.0)
        savefile : str, optional
            Path to save the plot
        """
        if ratios is None:
            ratios = [14.0, 14.5, 15.0, 15.5, 16.0]

        inclinations = np.arange(0, 361, incl_step)
        plt.figure(figsize=(10, 6))

        for ratio in ratios:
            altitudes = []
            a_guess = a_initial_guess_map.get(ratio, self.Rearth + 500) if a_initial_guess_map else self.Rearth + 500
            
            for idx, inc in enumerate(inclinations):
                alt, a_sol = self.find_a_for_ratio_at_inclination(inc, e, ratio, a_guess=a_guess)
                altitudes.append(alt)
                if a_sol is not None:
                    a_guess = a_sol
                    
            label = f"No/Nd = {self._ratio_to_fraction_label(ratio)}"
            plt.plot(inclinations, altitudes, marker='o', markersize=3, label=label)

        plt.xlabel("Inclination (deg)")
        plt.ylabel("RGT altitude (km)")
        plt.title("RGT altitude vs Inclination for different No/Nd ratios")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        import os
        os.makedirs(os.path.dirname(savefile) or ".", exist_ok=True)
        plt.savefig(savefile, dpi=200)
        print(f"Plot saved to: {savefile}")
        plt.show()

    # ---------------------------------------------------------
    # NEW METHODS FOR TIME-BASED REVISIT (e.g., 2h, 12h, 24h)
    # ---------------------------------------------------------

    def _revisit_time_error(self, a, i, e, target_time_sec):
        """
        Computes error between actual revisit time and target time.
        Revisit happens when relative longitude covers 2pi.
        Rate_rel = (M_dot + omega_dot) + (|rot_planet| + Omega_dot)
        """
        M_dot = self.compute_precession_M(i, a, e)
        omega_dot = self.compute_precession_omega_argperiapsis(i, a, e)
        Omega_dot = self.compute_precession_Omega_RAAN(i, a, e)
        
        # Denominator logic handles retrograde (Venus) vs prograde
        # Logic consistent with your target_function
        if self.rotation_rate_earth < 0:
            denom_planet = abs(self.rotation_rate_earth) + Omega_dot
        else:
            denom_planet = self.rotation_rate_earth - Omega_dot
            
        # Total relative angular velocity
        w_relative = (M_dot + omega_dot) + denom_planet
        
        T_actual = 2 * np.pi / w_relative
        return T_actual - target_time_sec

    def solve_for_revisit_time(self, i, e, hours, a_guess=None):
        """
        Solves for 'a' that gives a revisit every N hours.
        Uses a simple Secant method similar to your existing solver.
        """
        target_sec = hours * 3600.0
        
        if a_guess is None:
            # Rough Keplerian guess to start
            n_target = 2 * np.pi / target_sec
            a_guess = (self.mu / n_target**2)**(1/3)

        # Simple solver loop
        a0 = a_guess
        a1 = a_guess + 10.0
        tol = 1e-4 # seconds
        
        for _ in range(50):
            err0 = self._revisit_time_error(a0, i, e, target_sec)
            err1 = self._revisit_time_error(a1, i, e, target_sec)
            
            if abs(err1) < tol:
                return a1, True
            
            # Secant step
            if abs(err1 - err0) < 1e-12: break
            a_next = a1 - err1 * (a1 - a0) / (err1 - err0)
            
            # Safety bounds (Venus surface to 100k km)
            a_next = max(self.Rearth + 100, min(a_next, self.Rearth + 100000))
            
            a0, a1 = a1, a_next
            
        return a1, False

    def plot_revisit_intervals(self, hours_list=None, incl_step=2, e=0.0, savefile="results/venus_hours_revisit.png"):
        """
        Plots Altitude vs Inclination for fixed time intervals (e.g. 2h, 4h).
        """
        if hours_list is None:
            hours_list = [2, 4, 6, 8, 12, 24]

        inclinations = np.arange(0, 181, incl_step) # 0 to 180 for Venus
        plt.figure(figsize=(10, 6))

        for h in hours_list:
            altitudes = []
            valid_incs = []
            
            # Smart guess update: use previous result as guess for next inclination
            a_guess = None 
            
            for inc in inclinations:
                a_sol, converged = self.solve_for_revisit_time(inc, e, h, a_guess)
                
                # Filter out subsurface results
                alt = a_sol - self.Rearth
                if alt > 0: 
                    altitudes.append(alt)
                    valid_incs.append(inc)
                    a_guess = a_sol # Update guess for speed
            
            if valid_incs:
                plt.plot(valid_incs, altitudes, linewidth=2, label=f"{h} Hours")

        plt.xlabel("Inclination (deg)")
        plt.ylabel("Altitude (km)")
        plt.title(f"{self.planet_name} Altitude vs Inclination for Fixed Revisit Times")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        import os
        os.makedirs(os.path.dirname(savefile) or ".", exist_ok=True)
        plt.savefig(savefile, dpi=200)
        print(f"Plot saved to: {savefile}")
        plt.show()

    def plot_altitude_vs_revisit_time(self, inclination, min_hours=1.5, max_hours=24, 
                                     step_hours=0.5, e=0.0, 
                                     savefile="results/venus_altitude_vs_revisit.png"):
        """
        Plot Altitude vs Revisit Time for a fixed inclination.
        
        Parameters:
        -----------
        inclination : float
            Fixed inclination in degrees (e.g., 0 or 50)
        min_hours : float, optional
            Minimum revisit time in hours (default: 1.5)
        max_hours : float, optional
            Maximum revisit time in hours (default: 24)
        step_hours : float, optional
            Step size for revisit times (default: 0.5)
        e : float, optional
            Eccentricity (default: 0.0)
        savefile : str, optional
            Path to save the plot
        """
        hours_range = np.arange(min_hours, max_hours + step_hours, step_hours)
        altitudes = []
        valid_hours = []
        
        print(f"\nSolving for altitudes at i={inclination}° for revisit times from {min_hours} to {max_hours} hours...")
        
        # Use previous solution as guess for next iteration
        a_guess = None
        
        for h in hours_range:
            a_sol, converged = self.solve_for_revisit_time(inclination, e, h, a_guess)
            
            if converged:
                alt = a_sol - self.Rearth
                if alt > 0:  # Valid altitude above surface
                    altitudes.append(alt)
                    valid_hours.append(h)
                    a_guess = a_sol  # Update guess
                    print(f"  {h:.1f} hours → {alt:.1f} km altitude")
                else:
                    print(f"  {h:.1f} hours → subsurface (skipped)")
            else:
                print(f"  {h:.1f} hours → no convergence")
        
        if not valid_hours:
            print("No valid solutions found!")
            return
        
        # Create the plot with enhanced formatting
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(valid_hours, altitudes, 'b-', linewidth=2.5, marker='o', markersize=5)
        
        # Enhanced axis labels with larger font
        ax.set_xlabel("Revisit Time (hours)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Altitude (km)", fontsize=14, fontweight='bold')
        
        # X-axis: Integer labels only
        x_min_int = int(np.ceil(min_hours))
        x_max_int = int(np.floor(max_hours))
        ax.set_xticks(range(x_min_int, x_max_int + 1))  # Integer ticks only
        ax.set_xlim(min_hours, max_hours)
        
        # Y-axis: Logarithmic scale with better formatting
        ax.set_yscale('log')
        from matplotlib.ticker import LogLocator, LogFormatter
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        
        # Enhanced grid
        ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
        
        # Title
        ax.set_title(f"{self.planet_name} Altitude vs Revisit Time at {inclination}° Inclination", 
                     fontsize=15, fontweight='bold', pad=15)
        
        # Tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        
        plt.tight_layout()
        
        import os
        os.makedirs(os.path.dirname(savefile) or ".", exist_ok=True)
        plt.savefig(savefile, dpi=300)
        print(f"\nPlot saved to: {savefile}")
        plt.show()
        
# Backward compatibility: provide module-level functions that use a default instance
_default_phasing_rgt = PhasingRGT()

def compute_transfer_time(aTRANSF):
    return _default_phasing_rgt.compute_transfer_time(aTRANSF)

def compute_meanmotion(a):
    return _default_phasing_rgt.compute_meanmotion(a)

def compute_semilatus_rectum(a, e):
    return _default_phasing_rgt.compute_semilatus_rectum(a, e)

def compute_precession_Omega_RAAN(i, a, e):
    return _default_phasing_rgt.compute_precession_Omega_RAAN(i, a, e)

def compute_precession_M(i, a, e):
    return _default_phasing_rgt.compute_precession_M(i, a, e)

def compute_precession_omega_argperiapsis(i, a, e):
    return _default_phasing_rgt.compute_precession_omega_argperiapsis(i, a, e)

def find_a_for_ratio_at_inclination(i_deg, e, ratio, a_guess=None):
    return _default_phasing_rgt.find_a_for_ratio_at_inclination(i_deg, e, ratio, a_guess)

def solve_for_semi_major_axis(i, e, No, Nd, a_initial_guess=None, tolerance=1e-9, max_iterations=1000):
    return _default_phasing_rgt.solve_for_semi_major_axis(i, e, No, Nd, a_initial_guess, tolerance, max_iterations)


if __name__ == "__main__":
    # # Example usage
    # phasing = PhasingRGT()
    
    # # Test parameters
    # i = 51.447684794794995  # degrees
    # e = 0.0
    # No = 15
    # Nd = 1
    
    # print(f"Solving for semi-major axis with inputs:")
    # print(f"Inclination (i): {i}°")
    # print(f"Eccentricity (e): {e}")
    # print(f"No (revolutions): {No}")
    # print(f"Nd (days): {Nd}")
    
    # a_solution, converged, iterations = phasing.solve_for_semi_major_axis(i, e, No, Nd)
    
    # if converged:
    #     print(f"\nSolution converged in {iterations} iterations!")
    #     phasing.validate_solution(a_solution, i, e, No, Nd)
    # else:
    #     print(f"\nSolution did not converge after {iterations} iterations.")
    #     print(f"Final semi-major axis estimate: {a_solution:.3f} km")

    # # For Mars RGT analysis
    # phasing_mars = PhasingRGT(planet='mars')

    # # Find RGT altitude for Mars orbitS
    # i_deg = 93.0  # Sun-synchronous-like inclination for Mars
    # e = 0.0
    # # ratio = 15.0  # 15 revolutions per Martian day
    # No = 13
    # Nd = 1

    # alt_km, a_km = phasing_mars.find_a_for_ratio_at_inclination(i_deg, e, ratio=No/Nd)
    # print(f"Mars RGT altitude: {alt_km:.1f} km")

    # # Plot RGT altitude vs inclination for Mars
    # phasing_mars.plot_rgt_vs_inclinations(
    #     ratios=[10.0, 11.0, 12.0, 13.0, 14.0],
    #     incl_step=5,
    #     a_initial_guess_map={
    #         10.0: constants.MARS_RADIUS + 1000.0,
    #         11.0: constants.MARS_RADIUS + 700.0,
    #         12.0: constants.MARS_RADIUS + 500.0,
    #         13.0: constants.MARS_RADIUS + 300.0,
    #         14.0: constants.MARS_RADIUS + 100.0,
    #     },
    #     e=0.0,
    #     savefile="results/mars_rgt_vs_inclination.png"
    # )

    # For Venus RGT analysis
    phasing_venus = PhasingRGT(planet='venus')

    # # Plot RGT altitude vs inclination for Venus
    # # Ratios are typically 3400:1 to 3900:1 for practical altitudes
    # phasing_venus.plot_rgt_vs_inclinations(
    #     ratios=[3400.0, 3500.0, 3600.0, 3700.0, 3800.0, 3900.0],
    #     incl_step=10,
    #     a_initial_guess_map={
    #         3400.0: constants.VENUS_RADIUS + 1230.0,  # ~1230 km altitude
    #         3500.0: constants.VENUS_RADIUS + 1067.0,  # ~1067 km altitude
    #         3600.0: constants.VENUS_RADIUS + 910.0,   # ~910 km altitude
    #         3700.0: constants.VENUS_RADIUS + 760.0,   # ~760 km altitude (typical)
    #         3800.0: constants.VENUS_RADIUS + 615.0,   # ~615 km altitude
    #         3900.0: constants.VENUS_RADIUS + 475.0,   # ~475 km altitude
    #     },
    #     e=0.0,
    # savefile="results/venus_rgt_vs_inclination.png"
    # )

    # # ---------------------------------------------------------
    # # RUN THE VENUS HOURS PLOT
    # # ---------------------------------------------------------
    # phasing_venus = PhasingRGT(planet='venus')
    
    # print("Generating Venus Revisit Altitude Plot...")
    # phasing_venus.plot_revisit_intervals(
    #     hours_list=[2, 4, 6, 8, 12, 24], 
    #     incl_step=5,
    #     savefile="results/venus_revisit_intervals.png"
    # )

    # print("Generating Venus Revisit Altitude Plot...")
    # phasing_venus.plot_revisit_intervals(
    #     hours_list=[1.5, 1.6, 1.7, 1.8, 1.9, 2], 
    #     incl_step=5,
    #     savefile="results/venus_revisit_intervals_low.png"
    # )

    phasing_venus.plot_altitude_vs_revisit_time(
        inclination=50.0,  # Choose 0 or 50 deg
        min_hours=1.3,
        max_hours=24.0,
        step_hours=0.1,
        savefile="results/venus_altitude_vs_revisit_50deg.png"
    )