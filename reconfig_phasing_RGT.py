"""
Reconfiguration Phasing Script

This script handles satellite constellation reconfiguration with hardcoded orbital parameters.
Parameters:
- i: Inclination (degrees)
- e: Eccentricity (dimensionless)
- aGOM: Ground Object Mapping altitude or semi-major axis parameter
"""

import numpy as np
import constants
import matplotlib.pyplot as plt

J2 = constants.J2_EARTH
Rearth = constants.EARTH_RADIUS
mu = constants.GRAVITATIONAL_CONSTANT  # Gravitational parameter in km³/s²

No = 15
Nd = 1
altitudeGOM = 500.0 # km
# altitudeRGT = 553 # km for a 15/1
i = 51.447684794794995  # Inclination in degrees
e = 0.0
Omega0 = 0.0  # Initial right ascension of the ascending node (degrees)
M0 = 0.0  # Initial mean anomaly (degrees)

aGOM = altitudeGOM+Rearth  # Ground Object Mapping altitude or semi-major axis parameter in km
# aROM = altitudeRGT+Rearth  # Reconfiguration Orbit Mapping altitude or semi-major axis parameter in km
# Delta_a = 50.0
# aD = Delta_a + aGOM
# aT1 = 0.5*(aGOM+aD) # equal to 0.5*(aGOM + Delta_a)
# aT2 = 0.5*(aROM+aD) # equal to 0.5*(aROM + aGOM + Delta_a)


def compute_transfer_time(aTRANSF):
    t = np.pi * np.sqrt(aTRANSF**3 / mu)
    return t

def compute_meanmotion(a):
    n = np.sqrt(mu / a**3)
    return n

def compute_semilatus_rectum(a,e):
    p = a * (1 - e**2)
    return p

def compute_precession_Omega_RAAN(i,a,e):
    n = compute_meanmotion(a)
    p = compute_semilatus_rectum(a,e)
    # Omega_dot = (3*n*Rearth**2*J2 / 2*p**2) * np.cos(np.radians(i))
    Omega_dot = -(3*n*(Rearth**2)*J2 / (2*p**2)) * np.cos(np.radians(i))
    return Omega_dot # in rad/s

def compute_precession_M(i,a,e):
    n = compute_meanmotion(a)
    p = compute_semilatus_rectum(a,e)
    # M_dot = (-3*n*Rearth**2*J2 / 4*p**2) * (3*(np.cos(np.radians(i)))**2 - 1) * np.sqrt(1-e**2)
    M_dot = ((3*n*(Rearth**2)*J2 / (4*p**2)) * (2 - 3*(np.sin(np.radians(i)))**2) * np.sqrt(1-e**2)) + n# its delta Mdot = n + deltan

    return M_dot  # in rad/s

def compute_precession_omega_argperiapsis(i,a,e):
    n = compute_meanmotion(a)
    p = compute_semilatus_rectum(a,e)
    # omega_dot = (-3*n*Rearth**2*J2 / 4*p**2) * (5*(np.cos(np.radians(i)))**2 - 1)
    omega_dot = (3*n*(Rearth**2)*J2 / (4*p**2)) * (4 - 5*(np.sin(np.radians(i)))**2)
    return omega_dot  # in rad/s

def compute_longitude_of_groundtrack(No, Omega, Nd, M):
    Gamma = No*Omega + Nd*M
    return Gamma

# Earth's rotation rate in rad/s
rotation_rate_earth = 7.2921159e-5  # rad/s

def target_function(a, i, e, No, Nd):
    """
    Target function for the iterative solver.
    Returns: No/Nd - (M_dot + omega_dot) / (rotation_rate_earth + Omega_dot)
    This should equal zero when the correct semi-major axis 'a' is found.
    """
    M_dot = compute_precession_M(i, a, e)
    omega_dot = compute_precession_omega_argperiapsis(i, a, e)
    Omega_dot = compute_precession_Omega_RAAN(i, a, e)
    
    target_ratio = No / Nd
    computed_ratio = (M_dot + omega_dot) / (rotation_rate_earth - Omega_dot)
    
    return target_ratio - computed_ratio

def debug_calculations(a, i, e, No, Nd):
    """Debug function to check intermediate values"""
    print(f"\nDebug for a = {a:.3f} km:")
    n = compute_meanmotion(a)
    p = compute_semilatus_rectum(a, e)
    print(f"Mean motion n: {n:.6e} rad/s")
    print(f"Semi-latus rectum p: {p:.3f} km")
    
    M_dot = compute_precession_M(i, a, e)
    omega_dot = compute_precession_omega_argperiapsis(i, a, e)
    Omega_dot = compute_precession_Omega_RAAN(i, a, e)
    
    print(f"M_dot: {M_dot:.6e} rad/s")
    print(f"omega_dot: {omega_dot:.6e} rad/s")
    print(f"Omega_dot: {Omega_dot:.6e} rad/s")
    
    target_ratio = No / Nd
    computed_ratio = (M_dot + omega_dot) / (rotation_rate_earth - Omega_dot)
    
    print(f"Target ratio (No/Nd): {target_ratio:.6f}")
    print(f"Computed ratio: {computed_ratio:.6f}")
    print(f"Function value: {target_ratio - computed_ratio:.6e}")
    
    return M_dot, omega_dot, Omega_dot

def solve_for_semi_major_axis(i, e, No, Nd, a_initial_guess=6371+450, tolerance=1e-9, max_iterations=1000):
    """
    Robust solver for semi-major axis 'a'.
    Strategy:
    1. Try to bracket a root by expanding around the initial guess.
    2. If bracket found, use bisection (robust, derivative-free).
    3. If no bracket found, fall back to a guarded secant method.
    Returns (a, converged, iterations)
    """
    a0 = float(a_initial_guess)
    f0 = target_function(a0, i, e, No, Nd)
    if abs(f0) < tolerance:
        return a0, True, 0

    # initial bracket around a0
    R_min = 6200.0
    R_max = 20000.0
    a_low = max(R_min, a0 - 500.0)
    a_high = min(R_max, a0 + 500.0)
    f_low = target_function(a_low, i, e, No, Nd)
    f_high = target_function(a_high, i, e, No, Nd)

    # expand bracket until sign change or until limits reached
    expand_step = 500.0
    expand_iters = 0
    while f_low * f_high > 0 and expand_iters < 40:
        expand_iters += 1
        # expand outwards
        a_low = max(R_min, a_low - expand_step)
        a_high = min(R_max, a_high + expand_step)
        f_low = target_function(a_low, i, e, No, Nd)
        f_high = target_function(a_high, i, e, No, Nd)
        if a_low == R_min and a_high == R_max:
            break

    # If we have a valid bracket, do bisection
    if f_low * f_high <= 0:
        for iteration in range(max_iterations):
            a_mid = 0.5 * (a_low + a_high)
            f_mid = target_function(a_mid, i, e, No, Nd)

            # if iteration < 10:
                # print(f"bisection iter {iteration}: a_low={a_low:.3f}, a_high={a_high:.3f}, a_mid={a_mid:.3f}, f_mid={f_mid:.6e}")

            if not np.isfinite(f_mid):
                # numerical trouble, shrink bracket slightly and continue
                a_low += 1e-3
                a_high -= 1e-3
                continue

            if abs(f_mid) < tolerance or (a_high - a_low) * 0.5 < 1e-6:
                return a_mid, True, iteration + 1

            # choose subinterval containing root
            if f_low * f_mid <= 0:
                a_high = a_mid
                f_high = f_mid
            else:
                a_low = a_mid
                f_low = f_mid

        return a_mid, False, max_iterations

    # If no bracket found, fallback to a guarded secant method
    a_prev = a0
    a_curr = a0 + 100.0  # small step
    f_prev = f0
    f_curr = target_function(a_curr, i, e, No, Nd)

    for iteration in range(max_iterations):
        if iteration < 10:
            print(f"secant iter {iteration}: a_prev={a_prev:.3f}, a_curr={a_curr:.3f}, f_curr={f_curr:.6e}")

        # safeguard: if function values equal or nearly equal, perturb
        denom = (f_curr - f_prev)
        if abs(denom) < 1e-16:
            a_curr += 50.0
            f_curr = target_function(a_curr, i, e, No, Nd)
            denom = (f_curr - f_prev)
            if abs(denom) < 1e-16:
                break

        a_next = a_curr - f_curr * (a_curr - a_prev) / denom
        # guard step size
        max_step = 200.0
        step = a_next - a_curr
        if abs(step) > max_step:
            a_next = a_curr + np.sign(step) * max_step

        # bounds
        if a_next < R_min:
            a_next = R_min + 1.0
        if a_next > R_max:
            a_next = R_max - 1.0

        f_next = target_function(a_next, i, e, No, Nd)

        if abs(f_next) < tolerance or abs(a_next - a_curr) < 1e-6:
            return a_next, True, iteration + 1

        a_prev, f_prev = a_curr, f_curr
        a_curr, f_curr = a_next, f_next

    return a_curr, False, max_iterations

def validate_solution(a, i, e, No, Nd):
    """
    Validate the solution by computing all components and checking the relationship.
    """
    M_dot = compute_precession_M(i, a, e)
    omega_dot = compute_precession_omega_argperiapsis(i, a, e)
    Omega_dot = compute_precession_Omega_RAAN(i, a, e)
    
    target_ratio = No / Nd
    computed_ratio = (M_dot + omega_dot) / (rotation_rate_earth - Omega_dot)
    
    print(f"\nValidation Results:")
    print(f"Semi-major axis: {a:.8f} km")
    print(f"Altitude: {a - Rearth:.8f} km")
    print(f"Target ratio (No/Nd): {target_ratio:.8f}")
    print(f"Computed ratio: {computed_ratio:.8f}")
    print(f"Difference: {abs(target_ratio - computed_ratio):.2e}")
    print(f"M_dot: {M_dot:.6e} rad/s, {np.rad2deg(M_dot):.6f} deg/s, {np.rad2deg(M_dot) * 86400:.6f} deg/day")
    print(f"omega_dot: {omega_dot:.6e} rad/s, {np.rad2deg(omega_dot):.6f} deg/s, {np.rad2deg(omega_dot) * 86400:.6f} deg/day")
    print(f"Omega_dot: {Omega_dot:.6e} rad/s, {np.rad2deg(Omega_dot):.6f} deg/s, {np.rad2deg(Omega_dot) * 86400:.6f} deg/day")
    print(f"rotation_rate_earth + Omega_dot: {rotation_rate_earth + Omega_dot:.6e} rad/s")
    print(f"(M_dot + omega_dot) / (rotation_rate_earth + Omega_dot): {(M_dot + omega_dot) / (rotation_rate_earth + Omega_dot):.6e}")

def find_a_for_ratio_at_inclination(i_deg, e, ratio, a_guess=None):
    """
    Use existing solver to find semi-major axis for given inclination and No/Nd ratio.
    ratio is No/Nd (can be float). Returns altitude (a - Rearth) in km or np.nan if not found.
    """
    No_local = float(ratio)
    Nd_local = 1.0
    if a_guess is None:
        a_guess = aGOM  # use existing aGOM as sensible initial guess
    try:
        a_sol, converged, iters = solve_for_semi_major_axis(i_deg, e, No_local, Nd_local, a_initial_guess=a_guess)
        if converged:
            return a_sol - Rearth, a_sol  # altitude, semi-major axis
        else:
            return np.nan, None
    except Exception:
        return np.nan, None

def _ratio_to_fraction_label(ratio, max_denominator=100):
    """Convert a float ratio to a 'num/den' string (always show denominator)."""
    from fractions import Fraction
    frac = Fraction(ratio).limit_denominator(max_denominator)
    return f"{frac.numerator}/{frac.denominator}"

def plot_rgt_vs_inclinations_fraction_legend(ratios=None, incl_step=5, a_initial_guess_map=None, savefile="results/rgt_vs_inclination_fraclegend.png"):
    """
    Same as plot_rgt_vs_inclinations but uses fraction labels in the legend (e.g. '29/2', '15/1').
    Uses existing functions (find_a_for_ratio_at_inclination, etc.).
    """
    if ratios is None:
        ratios = [14.0, 14.5, 15.0, 15.5, 16.0]

    inclinations = np.arange(0, 361, incl_step)
    plt.figure(figsize=(10,6))

    for ratio in ratios:
        altitudes = []
        a_guess = a_initial_guess_map.get(ratio, aGOM) if a_initial_guess_map else aGOM
        for idx, inc in enumerate(inclinations):
            alt, a_sol = find_a_for_ratio_at_inclination(inc, e, ratio, a_guess=a_guess)
            altitudes.append(alt)
            if a_sol is not None:
                a_guess = a_sol
        label = f"No/Nd = {_ratio_to_fraction_label(ratio)}"
        plt.plot(inclinations, altitudes, marker='o', markersize=3, label=label)

    plt.xlabel("Inclination (deg)")
    plt.ylabel("RGT altitude (km)")
    plt.title("RGT altitude vs Inclination for different No/Nd ratios")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savefile, dpi=200)
    print(f"Plot saved to: {savefile}")
    plt.show()

# If script is run, produce and save the plot.
if __name__ == "__main__":
    # # Execute the main function
    # print(compute_longitude_of_groundtrack(No, Omega0, Nd, M0))
    
    # # Debug calculations first
    # print(f"\nDebugging with initial guess:")
    # debug_calculations(6821.0, i, e, No, Nd)
    
    # # Also try a different initial guess
    # print(f"\nDebugging with altitude ~550 km (like your aROM):")
    # debug_calculations(6371.0 + 550, i, e, No, Nd)
    
    # Solve for semi-major axis iteratively
    print(f"\nSolving for semi-major axis with inputs:")
    print(f"Inclination (i): {i}°")
    print(f"Eccentricity (e): {e}")
    print(f"No (revolutions): {No}")
    print(f"Nd (days): {Nd}")
    
    a_solution, converged, iterations = solve_for_semi_major_axis(i, e, No, Nd, a_initial_guess=6371+550)
    
    if converged:
        print(f"\nSolution converged in {iterations} iterations!")
        validate_solution(a_solution, i, e, No, Nd)
    else:
        print(f"\nSolution did not converge after {iterations} iterations.")
        print(f"Final semi-major axis estimate: {a_solution:.3f} km")
        # Final debug
        debug_calculations(a_solution, i, e, No, Nd)

    # # choose ratios to plot (adjust as needed)
    # ratios_to_plot = [14/1, 29/2, 15/1, 31/2, 16/1]
    # # optional map of initial guesses per ratio (start with aGOM)
    # initial_guess_map = {r: aGOM for r in ratios_to_plot}
    # plot_rgt_vs_inclinations_fraction_legend(ratios=ratios_to_plot, incl_step=5, a_initial_guess_map=initial_guess_map)