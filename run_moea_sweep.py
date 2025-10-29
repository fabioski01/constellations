"""
Multi-objective optimization parameter sweep script.

Runs MOEA with default parameters, changing ONE parameter at a time:
- Altitudes: 500, 700, 1000 km (default 700)
- Inclinations: 50, 60, 98 deg (default 50)
- Deployment ΔV: 100, 200, 300, 400 m/s (default 200)
- Satellite mass/payload combinations (4 classes, default 190 kg)

Each run writes results to a unique subdirectory under results/MOEA/.
"""

import os
from multi_objective_optimizer import run_multi_objective_optimization

def create_payload_config(total_payload_mass_kg):
    """
    Create payload configuration for given total mass.
    
    Parameters:
    -----------
    total_payload_mass_kg : float
        Total payload mass (split equally between fields and particle)
        
    Returns:
    --------
    payloads : list[dict]
        Payload configuration with proportional power (1.25 W/kg ratio)
    """
    # Split mass equally between two instruments
    mass_per_instrument = total_payload_mass_kg / 2.0
    
    # Calculate power: 1.25 W per kg payload
    power_per_instrument = 1.25 * mass_per_instrument
    
    payloads = [
        {'type': 'fields', 'cat': 'EO', 'mass_kg': mass_per_instrument, 'power_w': power_per_instrument},
        {'type': 'particle', 'cat': 'EO', 'mass_kg': mass_per_instrument, 'power_w': power_per_instrument}
    ]
    
    return payloads

def run_parameter_sweep():
    """
    Run MOEA changing ONE parameter at a time from defaults.
    """
    # Define DEFAULT parameters
    default_altitude = 700.0  # km
    default_inclination = 50.0  # degrees
    default_deployment_dv = 200.0  # m/s
    default_sat_mass = 190.0  # kg
    default_payload_mass = 53.2  # kg (for 190 kg satellite)
    
    # Define parameter ranges (INCLUDE defaults in each sweep)
    altitudes = [500.0, 700.0, 1000.0]  # Include 700.0 (default)
    inclinations = [50.0, 60.0, 98.0]   # Include 50.0 (default)
    deployment_dvs = [100.0, 200.0, 300.0, 500.0]  # Include 200.0 (default)
    
    # Satellite mass and payload combinations (sat_total_kg, payload_total_kg)
    sat_configs = [
        (95.0, 23.8),    # Micro-class
        (190.0, 53.2),   # Low mini-class (DEFAULT) - INCLUDED
        (370.0, 114.7),  # Up mini-class
        (760.0, 266.0)   # Small-class
    ]
    
    # Fixed parameters for all runs
    fixed_params = {
        'pattern_type': 'Walker-Delta',
        'gs': 'AFB_Andersen',
        'pop_size': 480,
        'n_gen': 50,
        'seed': 12,
        'output_dir': 'results/MOEA',
        'step': 500,
        'n_cores': 4,
        'cluster_mode': True,
        'evaluation_hours': 48,
        'T_deployment_days': 180.0,
        # Weights
        'local_weight': 0.0,
        'regional_weight': 0.0,
        'global_weight': 0.9,
        'science_weight': 0.7
    }
    
    # Build list of runs (one parameter changed at a time)
    runs = []

    # 0. BASELINE RUN (all defaults) - ADD THIS FIRST
    runs.append({
        'name': 'baseline_default',
        'altitude': default_altitude,
        'inclination': default_inclination,
        'dv_deployment': default_deployment_dv,
        'sat_mass': default_sat_mass,
        'payload_mass': default_payload_mass,
        'sweep_type': None  # No sweep folder for baseline
    })
    
    # 1. Altitude sweep (vary altitude, keep others at default)
    for alt in altitudes:
        runs.append({
            'name': f'altitude_{alt:.0f}km',
            'altitude': alt,
            'inclination': default_inclination,
            'dv_deployment': default_deployment_dv,
            'sat_mass': default_sat_mass,
            'payload_mass': default_payload_mass,
            'sweep_type': 'sweep_alt'  # Force all altitude runs into sweep_alt folder
        })
    
    # 2. Inclination sweep (vary inclination, keep others at default)
    # If you want to constrain it, see notes below
    for inc in inclinations:
        runs.append({
            'name': f'inclination_{inc:.0f}deg',
            'altitude': default_altitude,
            'inclination': inc,
            'dv_deployment': default_deployment_dv,
            'sat_mass': default_sat_mass,
            'payload_mass': default_payload_mass,
            'sweep_type': 'sweep_inc'  # Force all inclination runs into sweep_inc folder
        })
    
    # # 3. Deployment ΔV sweep
    # for dv in deployment_dvs:
    #     runs.append({
    #         'name': f'dv_{dv:.0f}ms',
    #         'altitude': default_altitude,
    #         'inclination': default_inclination,
    #         'dv_deployment': dv,
    #         'sat_mass': default_sat_mass,
    #         'payload_mass': default_payload_mass,
    #         'sweep_type': 'sweep_vdep'  # Force all dv runs into sweep_vdep folder
    #     })
    
    # 4. Satellite mass sweep
    for sat_mass, payload_mass in sat_configs:
        runs.append({
            'name': f'mass_{sat_mass:.0f}kg_payload_{payload_mass:.1f}kg',
            'altitude': default_altitude,
            'inclination': default_inclination,
            'dv_deployment': default_deployment_dv,
            'sat_mass': sat_mass,
            'payload_mass': payload_mass,
            'sweep_type': 'sweep_mass'  # Force all mass runs into sweep_mass folder
        })
    
    total_runs = len(runs)
    
    print("="*80)
    print(f"MULTI-OBJECTIVE OPTIMIZATION PARAMETER SWEEP")
    print("="*80)
    print(f"Total runs: {total_runs} (one parameter varied at a time)")
    print(f"Default: alt={default_altitude}km, inc={default_inclination}°, "
          f"dv={default_deployment_dv}m/s, mass={default_sat_mass}kg")
    print(f"Altitude sweep: {len(altitudes)} runs")
    print(f"Inclination sweep: {len(inclinations)} runs")
    print(f"Deployment ΔV sweep: {len(deployment_dvs)} runs")
    print(f"Satellite mass sweep: {len(sat_configs)} runs")
    print(f"Population: {fixed_params['pop_size']}, Generations: {fixed_params['n_gen']}")
    print("="*80)
    
    # Track results
    results_summary = []
    
    # Execute runs
    for run_idx, run_config in enumerate(runs, 1):
        # Create payload configuration
        payloads = create_payload_config(run_config['payload_mass'])
        
        # --- Check if output CSV already exists ---
        # Build expected output directory and filename
        from multi_objective_optimizer import generate_dynamic_directory
        output_dir = generate_dynamic_directory(
            pattern_type=fixed_params['pattern_type'],
            gs=fixed_params['gs'],
            local_weight=fixed_params['local_weight'],
            regional_weight=fixed_params['regional_weight'],
            global_weight=fixed_params['global_weight'],
            science_weight=fixed_params['science_weight'],
            pop_size=fixed_params['pop_size'],
            n_gen=fixed_params['n_gen'],
            seed=fixed_params['seed'],
            target_altitude=run_config['altitude'],
            target_inclination=run_config['inclination'],
            sat_mass_kg=run_config['sat_mass'],
            dv_deployment_m_s=run_config['dv_deployment'],
            base_dir=fixed_params['output_dir'],
            sweep_type_override=run_config['sweep_type']
        )

        # Build expected CSV filename pattern (partial match)
        expected_csv_prefix = f"pareto_solutions_{fixed_params['pattern_type'].lower()}_{fixed_params['gs']}_L{fixed_params['local_weight']:.1f}_R{fixed_params['regional_weight']:.1f}_G{fixed_params['global_weight']:.1f}_Sci{fixed_params['science_weight']:.1f}_tgtAlt{run_config['altitude']:.0f}km_tgtInc{run_config['inclination']:.0f}deg_dv{run_config['dv_deployment']:.0f}_T{fixed_params['T_deployment_days']:.0f}d_mass{run_config['sat_mass']:.0f}kg"
        # Check if any file in output_dir matches this prefix
        skip_run = False
        if os.path.exists(output_dir):
            for fname in os.listdir(output_dir):
                if fname.startswith(expected_csv_prefix) and fname.endswith('.csv'):
                    print(f"SKIPPING RUN {run_idx}/{total_runs}: {run_config['name']}")
                    print(f"  Output already exists: {os.path.join(output_dir, fname)}")
                    skip_run = True
                    break

        if skip_run:
            continue

        print(f"\n{'='*80}")
        print(f"RUN {run_idx}/{total_runs}: {run_config['name']}")
        print(f"{'='*80}")
        print(f"Target altitude: {run_config['altitude']} km")
        print(f"Target inclination: {run_config['inclination']} deg ")
        print(f"Deployment ΔV: {run_config['dv_deployment']} m/s")
        print(f"Satellite mass: {run_config['sat_mass']} kg")
        print(f"Payload mass: {run_config['payload_mass']} kg")
        print(f"  - Fields: {payloads[0]['mass_kg']:.2f} kg @ {payloads[0]['power_w']:.2f} W")
        print(f"  - Particle: {payloads[1]['mass_kg']:.2f} kg @ {payloads[1]['power_w']:.2f} W")
        print(f"  - Total power: {payloads[0]['power_w'] + payloads[1]['power_w']:.2f} W")
        print(f"  - Power/mass ratio: 1.25 W/kg")
        print("-"*80)
        
        try:
            # Run optimization
            results = run_multi_objective_optimization(
                target_altitude=run_config['altitude'],
                target_inclination=run_config['inclination'],
                sat_mass_kg=run_config['sat_mass'],
                payloads=payloads,
                dv_deployment_m_s=run_config['dv_deployment'],
                sweep_type=run_config['sweep_type'],  # Pass sweep type to ensure correct folder
                **fixed_params
            )
            
            # Store summary
            summary = {
                'run_id': run_config['name'],
                'altitude_km': run_config['altitude'],
                'inclination_deg': run_config['inclination'],
                'dv_deployment_m_s': run_config['dv_deployment'],
                'sat_mass_kg': run_config['sat_mass'],
                'payload_mass_kg': run_config['payload_mass'],
                'num_solutions': len(results['pareto_solutions']),
                'cost_min': results['costs'].min(),
                'cost_max': results['costs'].max(),
                'perf_min': results['performances'].min(),
                'perf_max': results['performances'].max(),
                'csv_file': results['csv_filename'],
                'plot_file': results['plot_filename'],
                'status': 'SUCCESS'
            }
            
            print(f"\n✓ Run completed successfully!")
            print(f"  Solutions found: {summary['num_solutions']}")
            print(f"  Cost range: {summary['cost_min']:.2f} - {summary['cost_max']:.2f} M$")
            print(f"  Performance range: {summary['perf_min']:.3f} - {summary['perf_max']:.3f}")
            
        except Exception as e:
            print(f"\n✗ Run FAILED with error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            summary = {
                'run_id': run_config['name'],
                'altitude_km': run_config['altitude'],
                'inclination_deg': run_config['inclination'],
                'dv_deployment_m_s': run_config['dv_deployment'],
                'sat_mass_kg': run_config['sat_mass'],
                'payload_mass_kg': run_config['payload_mass'],
                'status': 'FAILED',
                'error': str(e)
            }
        
        results_summary.append(summary)
    
    # Write summary CSV
    import csv
    summary_file = os.path.join(fixed_params['output_dir'], 'parameter_sweep_summary.csv')
    
    with open(summary_file, 'w', newline='') as f:
        fieldnames = ['run_id', 'altitude_km', 'inclination_deg', 'dv_deployment_m_s', 
                     'sat_mass_kg', 'payload_mass_kg', 'num_solutions', 
                     'cost_min', 'cost_max', 'perf_min', 'perf_max',
                     'csv_file', 'plot_file', 'status']
        
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results_summary)
    
    print(f"\n{'='*80}")
    print(f"PARAMETER SWEEP COMPLETED")
    print(f"{'='*80}")
    print(f"Total runs: {total_runs}")
    print(f"Successful: {sum(1 for r in results_summary if r['status'] == 'SUCCESS')}")
    print(f"Failed: {sum(1 for r in results_summary if r['status'] == 'FAILED')}")
    print(f"Summary saved to: {summary_file}")
    print("="*80)
    
    return results_summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MOEA parameter sweep (one-at-a-time)')
    parser.add_argument('--quick_test', action='store_true', 
                       help='Run quick test with first 3 runs only')
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("Running QUICK TEST mode (first 3 runs only)...")
        # Modify run list to only include first 3
        # You can implement this by slicing runs[:3] in the sweep function
    
    results_summary = run_parameter_sweep()