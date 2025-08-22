import numpy as np
from datetime import datetime, timezone
import constants
from math import atan2, tan, radians, degrees
from utilities import TimeConversion

# Reuse your TimeConversion helper for JD
# from your_module import TimeConversion  # you already have this in your codebase

def gmst_deg(dt_utc: datetime) -> float:
    """
    GMST at datetime (UTC) in degrees [0, 360).
    Reuses your TimeConversion.datetime_to_sgp4time and constants values.
    """
    jd, fr = TimeConversion.datetime_to_sgp4time(dt_utc)
    jd_total = jd + fr
    d_ut = jd_total - 2451545.0
    gmst_hours = (constants.GMST_CONSTANT + constants.EARTH_OMEGA * d_ut) % 24.0
    return (gmst_hours * 15.0) % 360.0

def wrap_deg(x):
    return (x + 360.0) % 360.0

def geodetic_to_geocentric_lat(lat_gd_deg: float) -> float:
    """
    Convert geodetic latitude (deg) to geocentric latitude (deg) using WGS-84 flattening.
    """
    f = 1.0 / 298.257223563
    lat_gd = radians(lat_gd_deg)
    lat_geo = np.arctan(((1.0 - f)**2) * np.tan(lat_gd))
    return degrees(lat_geo)

def hhat_from_i_raan(i_deg: float, raan_deg: float) -> np.ndarray:
    """
    Unit orbit normal (h-hat) in ECI for a prograde orbit.
    """
    i = radians(i_deg)
    Om = radians(raan_deg)
    h = np.array([np.sin(i)*np.sin(Om), -np.sin(i)*np.cos(Om), np.cos(i)], dtype=float)
    return h / np.linalg.norm(h)

def eci_unit_from_geo(lat_geo_deg: float, lon_inertial_deg: float) -> np.ndarray:
    """
    Unit ECI vector toward target (using geocentric latitude and inertial longitude).
    """
    lat = radians(lat_geo_deg)
    lon = radians(wrap_deg(lon_inertial_deg))
    return np.array([np.cos(lat)*np.cos(lon),
                     np.cos(lat)*np.sin(lon),
                     np.sin(lat)], dtype=float)

def pq_basis(i_deg: float, raan_deg: float):
    """
    In-plane basis vectors:
      p_hat: along line of nodes toward ascending node
      q_hat: h_hat x p_hat (right-handed)
    """
    h_hat = hhat_from_i_raan(i_deg, raan_deg)
    Om = radians(raan_deg)
    p_hat = np.array([np.cos(Om), np.sin(Om), 0.0], dtype=float)
    p_hat /= np.linalg.norm(p_hat)
    q_hat = np.cross(h_hat, p_hat)
    q_hat /= np.linalg.norm(q_hat)
    return h_hat, p_hat, q_hat

def raan_and_true_anomaly_over_target(i_deg: float,
                                      target_lat_gd_deg: float,
                                      target_lon_deg: float,
                                      epoch_utc: datetime):
    """
    Compute both ascending and descending solutions (RAAN and true anomaly, degrees)
    so the satellite is directly over the target at epoch_utc.
    Assumes circular orbit (e = 0) and sets omega = 0 so true_anomaly = argument_of_latitude.

    Returns:
        {
          "ascending":  {"RAAN_deg": ..., "true_anomaly_deg": ...},
          "descending": {"RAAN_deg": ..., "true_anomaly_deg": ...},
          "debug": {...}
        }
    """
    # 1) Convert inputs to geocentric latitude and inertial longitude at epoch
    lat_geo_deg = geodetic_to_geocentric_lat(target_lat_gd_deg)
    theta_g = gmst_deg(epoch_utc)  # GMST at epoch
    theta_deg = wrap_deg(target_lon_deg) + theta_g  # inertial longitude of target at epoch

    # 2) Feasibility check: |lat_geo| <= i
    if abs(lat_geo_deg) > abs(i_deg) + 1e-12:
        raise ValueError(f"|target_lat_geocentric|={abs(lat_geo_deg):.6f} exceeds inclination i={abs(i_deg):.6f}. No zenith overpass possible.")

    # 3) Solve sin(raan - theta) = -cot(i) * tan(lat_geo)
    i_rad = radians(i_deg)
    lat_geo_rad = radians(lat_geo_deg)
    s = - (np.cos(i_rad) / np.sin(i_rad)) * np.tan(lat_geo_rad)  # -cot(i)*tan(phi)
    s = float(np.clip(s, -1.0, 1.0))  # numerical safety

    delta_asc = degrees(np.arcsin(s))       # one branch
    delta_des = 180.0 - delta_asc           # supplementary branch

    raan_asc = wrap_deg(theta_deg + delta_asc)
    raan_des = wrap_deg(theta_deg + delta_des)

    # 4) Compute true anomaly at epoch for each plane
    #    For e=0 and omega=0: true anomaly = argument of latitude u.
    rhat_target = eci_unit_from_geo(lat_geo_deg, theta_deg)

    def true_anomaly_for_plane(i_deg_local, raan_deg_local):
        h_hat, p_hat, q_hat = pq_basis(i_deg_local, raan_deg_local)
        # Project target direction into the plane
        r_plane = rhat_target - np.dot(h_hat, rhat_target) * h_hat
        r_plane /= np.linalg.norm(r_plane)
        u_rad = np.arctan2(np.dot(r_plane, q_hat), np.dot(r_plane, p_hat))
        return wrap_deg(degrees(u_rad))

    nu_asc = true_anomaly_for_plane(i_deg, raan_asc)
    nu_des = true_anomaly_for_plane(i_deg, raan_des)

    # 5) Optional: verify which is actually ascending at the target and label consistently
    #    We compute v_hat ~ h_hat x r_hat at epoch; ascending if v_z > 0.
    def is_ascending_at_epoch(i_deg_local, raan_deg_local, nu_deg_local):
        i = radians(i_deg_local)
        Om = radians(raan_deg_local)
        nu = radians(nu_deg_local)
        cosO, sinO = np.cos(Om), np.sin(Om)
        cosi, sini = np.cos(i), np.sin(i)
        cosv, sinv = np.cos(nu), np.sin(nu)
        # r_hat in ECI for e=0, omega=0
        rx = cosO*cosv - sinO*cosi*sinv
        ry = sinO*cosv + cosO*cosi*sinv
        rz = sini*sinv
        r_hat = np.array([rx, ry, rz], dtype=float)
        r_hat /= np.linalg.norm(r_hat)
        h_hat = hhat_from_i_raan(i_deg_local, raan_deg_local)
        v_hat = np.cross(h_hat, r_hat)
        return v_hat[2] > 0.0

    asc_is_asc = is_ascending_at_epoch(i_deg, raan_asc, nu_asc)
    des_is_asc = is_ascending_at_epoch(i_deg, raan_des, nu_des)

    # If the labels are flipped, swap them
    if not asc_is_asc and des_is_asc:
        raan_asc, raan_des = raan_des, raan_asc
        nu_asc, nu_des = nu_des, nu_asc

    return {
        "ascending":  {"RAAN_deg": float(raan_asc), "true_anomaly_deg": float(nu_asc)},
        "descending": {"RAAN_deg": float(raan_des), "true_anomaly_deg": float(nu_des)},
        "debug": {
            "target_geocentric_lat_deg": float(lat_geo_deg),
            "gmst_deg": float(theta_g),
            "target_inertial_longitude_deg": float(wrap_deg(theta_deg))
        }
    }

# ----------------- example -----------------
if __name__ == "__main__":
    # Example inputs
    i_deg = 51.447684794794995
    target_lat_gd_deg = 3.295   # geodetic
    target_lon_deg = 95.982     # east-positive, west negative
    epoch = datetime(2025, 5, 5, 12, 0, 0, tzinfo=timezone.utc)

    sols = raan_and_true_anomaly_over_target(i_deg, target_lat_gd_deg, target_lon_deg, epoch)
    print("Ascending : RAAN = {:.6f} deg, true_anomaly = {:.6f} deg".format(
        sols["ascending"]["RAAN_deg"], sols["ascending"]["true_anomaly_deg"]))
    print("Descending: RAAN = {:.6f} deg, true_anomaly = {:.6f} deg".format(
        sols["descending"]["RAAN_deg"], sols["descending"]["true_anomaly_deg"]))
    # You can now write these back into your orbital_parameters before building the constellation.
