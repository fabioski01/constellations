"""
Reconfiguration Phasing RGT Module

This module provides the PhasingRGT class for computing repeat ground track orbits
and solving for semi-major axes given orbital parameters and repeat ground track ratios.
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
    
    def __init__(self, J2=None, Rearth=None, mu=None, rotation_rate_earth=None):
        """
        Initialize PhasingRGT with Earth constants.
        
        Parameters:
        -----------
        J2 : float, optional
            Earth's J2 coefficient (default: from constants)
        Rearth : float, optional
            Earth's radius in km (default: from constants)
        mu : float, optional
            Earth's gravitational parameter in km³/s² (default: from constants)
        rotation_rate_earth : float, optional
            Earth's rotation rate in rad/s (default: 7.2921159e-5)
        """
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
        """
        M_dot = self.compute_precession_M(i, a, e)
        omega_dot = self.compute_precession_omega_argperiapsis(i, a, e)
        Omega_dot = self.compute_precession_Omega_RAAN(i, a, e)
        
        target_ratio = No / Nd
        computed_ratio = (M_dot + omega_dot) / (self.rotation_rate_earth - Omega_dot)
        
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
        R_min = 6200.0
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
    # Example usage
    phasing = PhasingRGT()
    
    # Test parameters
    i = 51.447684794794995  # degrees
    e = 0.0
    No = 15
    Nd = 1
    
    print(f"Solving for semi-major axis with inputs:")
    print(f"Inclination (i): {i}°")
    print(f"Eccentricity (e): {e}")
    print(f"No (revolutions): {No}")
    print(f"Nd (days): {Nd}")
    
    a_solution, converged, iterations = phasing.solve_for_semi_major_axis(i, e, No, Nd)
    
    if converged:
        print(f"\nSolution converged in {iterations} iterations!")
        phasing.validate_solution(a_solution, i, e, No, Nd)
    else:
        print(f"\nSolution did not converge after {iterations} iterations.")
        print(f"Final semi-major axis estimate: {a_solution:.3f} km")

