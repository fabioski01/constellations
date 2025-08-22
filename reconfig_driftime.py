import numpy as np
import constants
import matplotlib.pyplot as plt
from reconfig_phasing_RGT import *
from reconfig_phasing_RAAN import raan_and_true_anomaly_over_target
import os
from datetime import datetime, timezone

J2 = constants.J2_EARTH
Rearth = constants.EARTH_RADIUS
mu_earth = constants.GRAVITATIONAL_CONSTANT  # Gravitational parameter in km³/s²

No = 15
Nd = 1
ratio = No/Nd
altitudeGOM = 500.0 # km
i = 51.447684794794995  # Inclination in degrees
e = 0.0
Omega0 = 0.0  # Initial right ascension of the ascending node (degrees)
M0 = 0.0  # Initial mean anomaly (degrees)
target_lat_gd_deg = 3.295   # geodetic
target_lon_deg = 95.982     # east-positive, west negative
epoch = datetime(2025, 5, 5, 12, 0, 0, tzinfo=timezone.utc)

aGOM = altitudeGOM+Rearth  # Ground Object Mapping altitude or semi-major axis parameter in km
altitudeRGT, aROM = alt, a_sol = find_a_for_ratio_at_inclination(i, e, ratio, a_guess=aGOM)

Delta_a = 50.0
aD = Delta_a + aGOM
aT1 = 0.5*(aGOM+aD) # equal to 0.5*(aGOM + Delta_a), first Hohmann transfer
aT2 = 0.5*(aROM+aD) # equal to 0.5*(aROM + aGOM + Delta_a), second Hohmann transfer
aT3 = 0.5*(aROM+aGOM) # final Hohmann transfer from ROM-RGT back to GOM

# print(altitudeRGT, aROM)

def compute_DV_Hohmann(a1, a2, aT, mu):
    '''Compute the total delta-v for a Hohmann transfer between two orbits.
    in km'''
    v_a = np.abs(np.sqrt((2*mu/a1) - (mu/aT)) - np.sqrt(mu/a1)) # first impulse to a1 to aT to exit a1 and get into transfer
    v_b = np.abs(np.sqrt((2*mu/a2) - (mu/aT)) - np.sqrt(mu/a2)) # second impulse to exit transfer and circularize into a2
    v_tot = v_a + v_b
    return v_tot, v_a, v_b

v1, v1a, v1b = compute_DV_Hohmann(aGOM, aD, aT1, mu_earth)
v2, v2a, v2b = compute_DV_Hohmann(aD, aROM, aT2, mu_earth)
total_reconfig_DV = v1 + v2 # km /s
v3, v3a, v3b = compute_DV_Hohmann(aROM, aGOM, aT3, mu_earth) # km /s
total_DV = v1 + v2 + v3 # km /s

# print(v1, v1a, v1b)
# print(v2, v2a, v2b)
# print(v3, v3a, v3b)
print(f'+++++ DeltaV cost for reconfiguration from {aGOM-Rearth} km to {aROM-Rearth} km with Delta_a = {Delta_a} km:')
print(f'v1+v2: {total_reconfig_DV*1000} m/s, v1+v2+v3: {total_DV*1000} m/s')

def compute_transfer_time_Hohmann(aT, mu):
    t_transfer = np.pi*np.sqrt(aT**3/(mu))  # in seconds
    return t_transfer

t1 = compute_transfer_time_Hohmann(aT1, mu_earth)  # time for first transfer
t2 = compute_transfer_time_Hohmann(aT2, mu_earth)  # time for second transfer
total_reconfig_t = t1 + t2
t3 = compute_transfer_time_Hohmann(aT3, mu_earth)  # time for third transfer
total_t = t1 + t2 + t3

print(f'+++++ Transfer time for reconfiguration from {aGOM-Rearth} km to {aROM-Rearth} km with Delta_a = {Delta_a} km:')
print(f't1: {t1} s, t2: {t2} s, t3: {t3} s')
print(f"t1: {t1/3600:.2f} hours, t2: {t2/3600:.2f} hours, t3: {t3/3600:.2f} hours")

DeltaOmega_T1 = compute_precession_Omega_RAAN(i, aT1, e) * t1 # in rad
DeltaOmega_T2 = compute_precession_Omega_RAAN(i, aT2, e) * t2 # in rad
Omega_precession_D = compute_precession_Omega_RAAN(i, aD, e) # in rad/s

DeltaM_T1 = (compute_meanmotion(aT1) + compute_precession_omega_argperiapsis(i, aT1, e) + compute_precession_M(i, aT1, e)) * t1  # in rad 
DeltaM_T2 = (compute_meanmotion(aT2) + compute_precession_omega_argperiapsis(i, aT2, e) + compute_precession_M(i, aT2, e)) * t2  # in rad
M_precession_D = (compute_meanmotion(aD) + compute_precession_omega_argperiapsis(i, aD, e) + compute_precession_M(i, aD, e)) # in rad/s

# example call (use your variables/values)
sols = raan_and_true_anomaly_over_target(i, target_lat_gd_deg, target_lon_deg, epoch)

# unpack the four final params in DEG
Omega_f_asc, M_f_asc, Omega_f_des, M_f_des = (
    sols["ascending"]["RAAN_deg"],
    sols["ascending"]["true_anomaly_deg"],   # for e=0 true anomaly == mean anomaly
    sols["descending"]["RAAN_deg"],
    sols["descending"]["true_anomaly_deg"],
)

print(f'+++++ PHASING for target lat {target_lat_gd_deg} deg, lon {target_lon_deg} deg at epoch {epoch.isoformat()}')
print(f'Omega_f_asc: {Omega_f_asc} deg, M_f_asc: {M_f_asc} deg, Omega_f_des: {Omega_f_des} deg, M_f_des: {M_f_des} deg') # IN DEG

def angle_time_from_angles_rad(target_rad, current_rad, ang_rate_rad_s):
    """
    Smallest non-negative t (s) such that current_rad + ang_rate_rad_s*t ≡ target_rad (mod 2π).
    All angles in radians, ang_rate in rad/s. Returns np.inf if ang_rate ~ 0.
    """
    eps = 1e-15
    if abs(ang_rate_rad_s) < eps:
        return np.inf
    two_pi = 2.0 * np.pi
    delta = np.mod(target_rad - current_rad, two_pi)          # in [0,2π)
    if ang_rate_rad_s > 0.0:
        t = delta / ang_rate_rad_s
    else:
        # ang_rate negative -> need negative equivalent in (-2π,0] to get positive t
        t = (delta - two_pi) / ang_rate_rad_s
    if t < 0 and t > -1e-9:
        t = 0.0
    return float(t)

# current accumulated RAAN after T1+T2 (DeltaOmega_T1/2 are already in rad)
Omega_current_rad = np.mod(np.radians(Omega0) + (DeltaOmega_T1 + DeltaOmega_T2), 2.0*np.pi)
# Omega_f_asc / des must be converted to radians
Omega_f_asc_rad = np.radians(Omega_f_asc)
Omega_f_des_rad = np.radians(Omega_f_des)

tD_fromOmega_asc = angle_time_from_angles_rad(Omega_f_asc_rad, Omega_current_rad, Omega_precession_D)
tD_fromOmega_des = angle_time_from_angles_rad(Omega_f_des_rad, Omega_current_rad, Omega_precession_D)
tD_fromM_asc = (M_f_asc - (M0 + DeltaM_T1 + DeltaM_T2))/M_precession_D
tD_fromM_des = (M_f_des - (M0 + DeltaM_T1 + DeltaM_T2))/M_precession_D

# For mean anomaly same approach
M_current_rad = np.mod(np.radians(M0) + (DeltaM_T1 + DeltaM_T2), 2.0*np.pi)
M_f_asc_rad = np.radians(M_f_asc)
M_f_des_rad = np.radians(M_f_des)

tD_fromM_asc = angle_time_from_angles_rad(M_f_asc_rad, M_current_rad, M_precession_D)
tD_fromM_des = angle_time_from_angles_rad(M_f_des_rad, M_current_rad, M_precession_D)

print(f'+++++ DRIFT times for correct phasing')
print(f'tD_fromOmega_asc: {tD_fromOmega_asc} s, tD_fromOmega_des: {tD_fromOmega_des} s, tD_fromM_asc: {tD_fromM_asc} s, tD_fromM_des: {tD_fromM_des} s')
print(f"tD_fromOmega_asc: {tD_fromOmega_asc/3600:.2f} hours, tD_fromOmega_des: {tD_fromOmega_des/3600:.2f} hours, tD_fromM_asc: {tD_fromM_asc/3600:.2f} hours, tD_fromM_des: {tD_fromM_des/3600:.2f} hours")
print(f'tD_fromOmega_asc: {tD_fromOmega_asc/86400:.2f} days, tD_fromOmega_des: {tD_fromOmega_des/86400:.2f} days, tD_fromM_asc: {tD_fromM_asc/86400:.2f} days, tD_fromM_des: {tD_fromM_des/86400:.2f} days')

min_tD = min(tD_fromOmega_asc, tD_fromOmega_des) # fix RAAN to minimize tD, then change M0
print(f'+++++ minimum DRIFT time for RAAN phasing: {min_tD} s, {min_tD/3600} hours, {min_tD/86400} days')
if min_tD == tD_fromOmega_asc:
    needed_Mf = M_f_asc_rad
elif min_tD == tD_fromOmega_des:
    needed_Mf = M_f_des_rad

print(f'+++++ needed_Mf for correct phasing: {needed_Mf} rad, {np.degrees(needed_Mf)} deg')
DeltaM_D = M_precession_D*min_tD # in rad
print(f'DeltaM_D: {DeltaM_D} rad, {np.degrees(DeltaM_D)} deg')
needed_M0 = np.mod(needed_Mf - (DeltaM_T1 + DeltaM_T2 + DeltaM_D), 2*np.pi) # rad
print(f'needed_M0: {needed_M0} rad, {np.degrees(needed_M0)} deg')

# general version when you have an angular rate M_dot (rad/s) and current M0_rad:
def time_from_angle_rate(target_rad, current_rad, rate_rad_s):
    """Smallest non-negative t so current + rate*t ≡ target (mod 2π)."""
    eps = 1e-15
    if abs(rate_rad_s) < eps:
        return np.inf
    delta = np.mod(target_rad - current_rad, 2*np.pi)
    if rate_rad_s > 0:
        return delta / rate_rad_s
    else:
        return (delta - 2*np.pi) / rate_rad_s  # yields positive t for negative rate

M_dot_GOM = compute_precession_M(i, aGOM, e) # rad/s. Normal + perturbations
maneuver_wait_time_anomaly = time_from_angle_rate(needed_M0, np.radians(M0), M_dot_GOM)
print(f'+++++ Time to wait for from start of propagagation, to then start maneuver {maneuver_wait_time_anomaly} s or {maneuver_wait_time_anomaly/3600} hr, considering M0: {np.radians(M0)} rad or {M0} deg, M_dot_GOM: {M_dot_GOM} rad/s or {np.degrees(M_dot_GOM)} deg/s or {np.degrees(M_dot_GOM)*3600} deg/hr')
#################################################################################
############################# PLOTTING ##########################################
#################################################################################

def _pair_label(aGOM_alt_km, aROM_alt_km):
    """Return compact label 'GOM/ROM = 500/550 km'."""
    return f"GOM/ROM = {int(aGOM_alt_km)}/{int(aROM_alt_km)} km"

def plot_total_and_reconfig_for_pairs(pairs, delta_min=-100, delta_max=100, step=5,
                                      savefile="results/total_and_reconfig_vs_delta_a_pairs.png"):
    """
    Produce a figure with two stacked subplots sharing the x-axis:
      - Top: Total ΔV vs Δa for each (aGOM,aROM) pair (m/s)
      - Bottom: Reconfig ΔV (v1+v2) vs Δa for same pairs (m/s)
    pairs: iterable of tuples (aGOM_alt_km, aROM_alt_km) or strings "500/550"
    """
    # normalize input pairs to tuples of floats (altitudes in km)
    normalized = []
    for p in pairs:
        if isinstance(p, str):
            try:
                g, r = p.split("/")
                normalized.append((float(g), float(r)))
            except Exception:
                continue
        elif isinstance(p, (tuple, list)) and len(p) == 2:
            normalized.append((float(p[0]), float(p[1])))
        else:
            continue

    if not normalized:
        raise ValueError("No valid aGOM/aROM pairs provided.")

    delta_vals = np.arange(delta_min, delta_max + step, step)

    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'd', '^', 'v', 'x', '*']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(10,8), gridspec_kw={'height_ratios':[1,1]})
    for idx, (aGOM_alt_km, aROM_alt_km) in enumerate(normalized):
        aGOM_local = aGOM_alt_km + Rearth
        aROM_local = aROM_alt_km + Rearth

        # sanity bounds
        if aGOM_local < 6200 or aGOM_local > 20000 or aROM_local < 6200 or aROM_local > 20000:
            print(f"Skipping pair {aGOM_alt_km}/{aROM_alt_km} km (out of bounds)")
            continue

        total_dv_vals = []
        reconfig_dv_vals = []
        for da in delta_vals:
            aD = aGOM_local + da
            aT1 = 0.5 * (aGOM_local + aD)
            aT2 = 0.5 * (aD + aROM_local)
            aT3 = 0.5 * (aROM_local + aGOM_local)

            v1, _, _ = compute_DV_Hohmann(aGOM_local, aD, aT1, mu_earth)
            v2, _, _ = compute_DV_Hohmann(aD, aROM_local, aT2, mu_earth)
            v3, _, _ = compute_DV_Hohmann(aROM_local, aGOM_local, aT3, mu_earth)

            reconfig_dv_vals.append((v1 + v2) * 1000.0)      # m/s
            total_dv_vals.append((v1 + v2 + v3) * 1000.0)   # m/s

        ls = linestyles[idx % len(linestyles)]
        mk = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        label = _pair_label(aGOM_alt_km, aROM_alt_km)

        ax_top.plot(delta_vals, total_dv_vals, linestyle=ls, marker=mk, color=color,
                    linewidth=1.4, markersize=4, label=label)
        ax_bot.plot(delta_vals, reconfig_dv_vals, linestyle=ls, marker=mk, color=color,
                    linewidth=1.4, markersize=4, label=label)

    ax_top.set_ylabel("Total ΔV (m/s)")
    ax_top.set_title("Total ΔV vs Δa for different (aGOM, aROM) pairs")
    ax_top.grid(True, linestyle=':', alpha=0.6)
    ax_top.legend(loc="best", fontsize="small")

    ax_bot.set_xlabel("Δa (km)")
    ax_bot.set_ylabel("Reconfig ΔV (m/s)")
    ax_bot.set_title("Reconfig ΔV (v1 + v2) vs Δa")
    ax_bot.grid(True, linestyle=':', alpha=0.6)
    ax_bot.legend(loc="best", fontsize="small")

    os.makedirs(os.path.dirname(savefile) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(savefile, dpi=200)
    print(f"Saved plot to: {savefile}")
    # plt.show()

def plot_time_for_pairs(pairs, delta_min=-100, delta_max=100, step=5,
                        savefile="results/total_and_reconfig_time_vs_delta_a_pairs.png"):
    """
    Produce a figure with two stacked subplots sharing the x-axis:
      - Top: Total transfer time vs Δa for each (aGOM,aROM) pair (hours)
      - Bottom: Reconfig transfer time (t1 + t2) vs Δa for same pairs (hours)
    pairs: iterable of tuples (aGOM_alt_km, aROM_alt_km) or strings "500/550"
    """
    # normalize input pairs to tuples of floats (altitudes in km)
    normalized = []
    for p in pairs:
        if isinstance(p, str):
            try:
                g, r = p.split("/")
                normalized.append((float(g), float(r)))
            except Exception:
                continue
        elif isinstance(p, (tuple, list)) and len(p) == 2:
            normalized.append((float(p[0]), float(p[1])))
        else:
            continue

    if not normalized:
        raise ValueError("No valid aGOM/aROM pairs provided.")

    delta_vals = np.arange(delta_min, delta_max + step, step)

    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'd', '^', 'v', 'x', '*']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(10,8), gridspec_kw={'height_ratios':[1,1]})
    for idx, (aGOM_alt_km, aROM_alt_km) in enumerate(normalized):
        aGOM_local = aGOM_alt_km + Rearth
        aROM_local = aROM_alt_km + Rearth

        # sanity bounds
        if aGOM_local < 6200 or aGOM_local > 20000 or aROM_local < 6200 or aROM_local > 20000:
            print(f"Skipping pair {aGOM_alt_km}/{aROM_alt_km} km (out of bounds)")
            continue

        total_t_vals = []
        reconfig_t_vals = []
        for da in delta_vals:
            aD = aGOM_local + da
            aT1 = 0.5 * (aGOM_local + aD)
            aT2 = 0.5 * (aD + aROM_local)
            aT3 = 0.5 * (aROM_local + aGOM_local)

            t1 = compute_transfer_time_Hohmann(aT1, mu_earth)
            t2 = compute_transfer_time_Hohmann(aT2, mu_earth)
            t3 = compute_transfer_time_Hohmann(aT3, mu_earth)

            # convert seconds -> hours for readability
            reconfig_t_vals.append((t1 + t2) / 3600.0)
            total_t_vals.append((t1 + t2 + t3) / 3600.0)

        ls = linestyles[idx % len(linestyles)]
        mk = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        label = _pair_label(aGOM_alt_km, aROM_alt_km)

        ax_top.plot(delta_vals, total_t_vals, linestyle=ls, marker=mk, color=color,
                    linewidth=1.4, markersize=4, label=label)
        ax_bot.plot(delta_vals, reconfig_t_vals, linestyle=ls, marker=mk, color=color,
                    linewidth=1.4, markersize=4, label=label)

    ax_top.set_ylabel("Total transfer time (hours)")
    ax_top.set_title("Total transfer time vs Δa for different (aGOM, aROM) pairs (EXCLUDING DRIFT)")
    ax_top.grid(True, linestyle=':', alpha=0.6)
    ax_top.legend(loc="best", fontsize="small")

    ax_bot.set_xlabel("Δa (km)")
    ax_bot.set_ylabel("Reconfig transfer time (t1 + t2) (hours)")
    ax_bot.set_title("Reconfig transfer time vs Δa  (EXCLUDING DRIFT)")
    ax_bot.grid(True, linestyle=':', alpha=0.6)
    ax_bot.legend(loc="best", fontsize="small")

    os.makedirs(os.path.dirname(savefile) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(savefile, dpi=200)
    print(f"Saved plot to: {savefile}")
    # plt.show()

# if __name__ == "__main__":
# #     # quick call (adjust range/step as needed)
# #     plot_dV_vs_delta_a(-100, 100, 5)
#     pairs = [(500,450), (500,500), (500,504.595), (500,550),  "600/666.3734"]
#     plot_total_and_reconfig_for_pairs(pairs, delta_min=-100, delta_max=100, step=5)
#     plot_time_for_pairs(pairs, delta_min=-100, delta_max=100, step=5)



 