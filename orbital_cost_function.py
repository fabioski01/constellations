"""Orbital Cost Function (OCF) for Neutron DRL Launch Vehicle

Provides bilinear-interpolated Orbital Cost Function based on Neutron DRL payload
performance across inclination (40–100°) and circular altitude (200–1000 km).

OCF Definition
    OCF = ref_payload / payload_capacity(inc, alt)
    - OCF = 1.0 at reference orbit (200 km, 40°, 14000 kg payload)
    - OCF > 1.0 indicates higher launch cost per kg (reduced payload capability)

Primary Exports
    ocf_at(inc_deg, alt_km) -> float
        Query OCF at given orbit; clamps to calibrated domain [40–100°] × [200–1000 km].
    
    scale_launch_price(price_base_per_kg, base_alt_km, base_inc_deg,
                       target_alt_km, target_inc_deg, mass_kg=1.0) -> (price_per_kg, total_cost)
        Scale a known launch price from base orbit to target orbit using OCF ratio.
    
    plot_ocf_surface(save_path='results/orbital_cost_function_3d.png')
        Generate 3D surface plot (only imports matplotlib when called).

Usage Example
    from orbital_cost_function import ocf_at, scale_launch_price
    
    # Query OCF
    ocf = ocf_at(inc_deg=60, alt_km=700)  # returns ~1.15
    
    # Scale known price to different orbit
    price_tgt, total = scale_launch_price(
        price_base_per_kg=1000, base_alt_km=500, base_inc_deg=50,
        target_alt_km=700, target_inc_deg=60, mass_kg=100
    )

Data Source
    Neutron DRL Down Range Landing (DRL) Performance curve (Rocket Lab).

Author
    Fabio Meloni - TU Delft MSc Student - NASA JPL Visiting Researcher
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Data extracted from Neutron DRL Performance graph
# Format: altitude (km) -> payload mass (kg) for each inclination
data = {
    # Inclination: [(altitude, payload_mass), ...]
    40: [
        (200, 14000),
        (400, 13200),
        (500, 13000),
        (800, 11800),
        (1000, 11000)
    ],
    60: [
        (200, 13400),
        (400, 12500),
        # (500, 12000),
        (800, 11000),
        (1000, 10200)
    ],
    80: [
        (200, 12200),
        (400, 11400),
        # (500, 11000),
        (800, 10000),
        (1000, 9200)
    ],
    100: [
        (200, 10800),
        (400, 9800),
        # (500, 9400),
        (800, 8100),
        (1000, 7500)
    ]
}

# Reference point for normalization (OCF = 1.0)
ref_altitude = 200
ref_inclination = 40
ref_payload = 14000

# Build a regular knot grid (inclinations × altitudes)
inc_knots = np.array(sorted(data.keys()), dtype=float)          # [40, 60, 80, 100]
alt_knots = np.array(sorted({a for pts in data.values() for a, _ in pts}), dtype=float)  # [200, 400, 500, 800, 1000]

# Fill payload grid by linear interpolation along altitude for each inclination
payload_grid = np.empty((len(inc_knots), len(alt_knots)), dtype=float)
for i, inc in enumerate(inc_knots):
    pts = sorted(data[inc], key=lambda t: t[0])
    alts_i = np.array([a for a, _ in pts], dtype=float)
    payl_i = np.array([p for _, p in pts], dtype=float)
    # Linear interpolation across the union of altitude knots (no extrapolation beyond provided range)
    payload_grid[i, :] = np.interp(alt_knots, alts_i, payl_i)

# Define bilinear interpolator over (inclination, altitude)
# RegularGridInterpolator expects values.shape == (len(inc_knots), len(alt_knots))
interp_payload = RegularGridInterpolator(
    (inc_knots, alt_knots),
    payload_grid,
    bounds_error=False,    # allow mild extrapolation; we'll clip later
    fill_value=None        # use linear extrapolation outside the knots
)

# Plot domain (lazy generation - only used by plot_ocf_surface)
inc_min, inc_max = 40.0, 100.0
alt_min, alt_max = 200.0, 1000.0

def _clip_to_knots(inc_deg: float, alt_km: float) -> tuple[float, float]:
    """Clamp query to the calibrated domain to avoid NaNs from far extrapolation."""
    i = float(np.clip(inc_deg, float(inc_knots.min()), float(inc_knots.max())))
    a = float(np.clip(alt_km,  float(alt_knots.min()), float(alt_knots.max())))
    return i, a

def ocf_at(inc_deg: float, alt_km: float) -> float:
    """
    Orbital Cost Function at (inclination, altitude).
    OCF = ref_payload / payload_capacity(inc, alt). 1.0 = same cost as reference; >1.0 = higher cost/kg.
    """
    i, a = _clip_to_knots(inc_deg, alt_km)
    payload = float(interp_payload([[i, a]])[0])
    payload = max(payload, 1.0)  # guard
    ocf = float(ref_payload) / payload
    return float(np.clip(ocf, 1.0, 3.0))

def scale_launch_price(price_base_per_kg: float,
                       base_alt_km: float, base_inc_deg: float,
                       target_alt_km: float, target_inc_deg: float,
                       mass_kg: float = 1.0) -> tuple[float, float]:
    """
    Scale a known launch price/kg at a base orbit to a target orbit using OCF ratios.
    Returns (target_price_per_kg, total_cost_for_mass).
    Assumption: price/kg ∝ OCF (inverse of payload capability).
    """
    ocf_base = ocf_at(base_inc_deg, base_alt_km)
    ocf_tgt  = ocf_at(target_inc_deg, target_alt_km)
    price_tgt_per_kg = float(price_base_per_kg) * (ocf_tgt / ocf_base)
    total_cost = price_tgt_per_kg * float(mass_kg)
    return price_tgt_per_kg, total_cost

# --- Plotting function (lazy imports matplotlib only when called) ---
def plot_ocf_surface(save_path='results/orbital_cost_function_3d.png') -> None:
    """Generate and save 3D OCF surface plot. Imports matplotlib only when called."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Generate mesh grid for plotting
    inc_grid = np.linspace(inc_min, inc_max, 80)
    alt_grid = np.linspace(alt_min, alt_max, 120)
    Inc_mesh, Alt_mesh = np.meshgrid(inc_grid, alt_grid, indexing='ij')
    
    # Evaluate payload on grid and compute OCF
    pts = np.column_stack([Inc_mesh.ravel(), Alt_mesh.ravel()])
    payload_mesh = interp_payload(pts).reshape(Inc_mesh.shape)
    payload_mesh = np.where(np.isfinite(payload_mesh), payload_mesh, np.nan)
    payload_mesh = np.clip(payload_mesh, 1.0, None)
    OCF_mesh = ref_payload / payload_mesh
    OCF_mesh = np.clip(OCF_mesh, 1.0, 3.0)
    
    # Compute OCF values of raw data for scatter
    raw_inc = []
    raw_alt = []
    raw_ocf = []
    for inc, pts_data in data.items():
        for a, p in pts_data:
            raw_inc.append(inc)
            raw_alt.append(a)
            raw_ocf.append(ref_payload / float(p))
    raw_inc = np.array(raw_inc)
    raw_alt = np.array(raw_alt)
    raw_ocf = np.array(raw_ocf)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Inc_mesh, Alt_mesh, OCF_mesh, cmap='viridis', alpha=0.9, edgecolor='none')
    ax.scatter(raw_inc, raw_alt, raw_ocf, c='red', s=40, label='Data Points')
    
    ax.set_xlabel('Inclination (deg)', fontsize=10)
    ax.set_ylabel('Circular Altitude (km)', fontsize=10)
    ax.set_zlabel('Orbital Cost Function (OCF)', fontsize=10)
    ax.set_title('Neutron DRL Orbital Cost Function (linear interpolation)\nNormalized to 200 km, 40°', fontsize=12)
    
    # Set axis limits (Z starts at 1.0, XY end exactly at max with small lower margin)
    ax.set_zlim(1.0, float(np.nanmax(OCF_mesh)))
    pad_frac = 0.05
    x_low = max(0.0, inc_min - pad_frac * (inc_max - inc_min))
    y_low = max(0.0, alt_min - pad_frac * (alt_max - alt_min))
    ax.set_xlim(x_low, 100.0)
    ax.set_ylim(y_low, 1000.0)
    
    # View angle and axis orientation (inclination on left, altitude on right)
    ax.view_init(elev=20, azim=-45)
    ax.invert_xaxis()
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='OCF')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved OCF surface plot to {save_path}")
    print(f"OCF range: {np.nanmin(OCF_mesh):.2f} to {np.nanmax(OCF_mesh):.2f}")


if __name__ == "__main__":
    plot_ocf_surface()
    
    # Example usage demonstration
    print("\n--- Example Usage ---")
    price_base = 1000.0  # USD/kg
    base_alt, base_inc = 500.0, 50.0
    target_alt, target_inc = 700.0, 60.0
    mass = 100.0  # kg

    ocf_base = ocf_at(base_inc, base_alt)
    ocf_target = ocf_at(target_inc, target_alt)
    price_tgt_per_kg, total_cost = scale_launch_price(
        price_base, base_alt, base_inc, target_alt, target_inc, mass
    )

    print(f"Base orbit ({base_alt} km, {base_inc}°): OCF = {ocf_base:.3f}")
    print(f"Target orbit ({target_alt} km, {target_inc}°): OCF = {ocf_target:.3f}")
    print(f"Launch price scaled: ${price_tgt_per_kg:.2f}/kg")
    print(f"Total cost for {mass} kg: ${total_cost:,.0f}")