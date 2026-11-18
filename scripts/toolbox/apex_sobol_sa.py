"""
2-25-2026
Sobol' Sensitivity Analysis for APEX BMP Removal Efficiency
For each BMP x region x crop combo:
  1. SALib generates Saltelli parameter samples
  2. For each sample: copy template → modify params → perturb climate → run APEX → extract removal efficiency → delete copy
  3. Compute Sobol' indices (first-order + total-order)

For climate only sobol: comment out operational params

"""

import os
os.environ['MKL_NUM_THREADS'] = '1'  # limit MKL to 1 thread to avoid oversubscription in parallel runs 
os.environ['OMP_NUM_THREADS'] = '1'  # limit OpenMP to 1 thread
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # limit OpenBLAS to 1 thread
import sys
import shutil
import glob
import time
import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol


# Import existing functions

sys.path.append(r'C:\Users\maya\Documents\cbw-ag-modeling\scripts')
from apex_config_functions import (
    generate_loading_table, 
    calc_removal_efficiency,
    avg_cropyld,
    _modify_opc_file,
    _modify_sol_file, 
    _modify_sub_file, 
    modify_single_scenario_inplace
)
from apex_climate_perturb import (perturb_dly, perturb_hly, perturb_wp1)

# configuration

base_fp = # path to historic runs
work_fp = # temp folder for Sobol runs
results_fp =  # where to save final results
box_fp = # secondary save location

n_samples = 1024  # base sample size (total runs = N * (2D+2))
sim_start = 1990

# parameter ranges (continuous)
param_ranges = {
    'pec':    [0.50, 1.00],
    'delta_t': [0.0, 7.0],      # degrees C added to baseline
    'delta_p': [0.0, 0.35],     # fractional increase 
}

phu_ranges = {
     'corn':    [1500, 2500],
     'soy':     [1200, 2500],
     'alfalfa': [750, 1200],
    'wheat':   [850, 2000],
 }

 # BMP-specific parameters
bmp_params = {
     'nt': {'residue': [3.0, 12.0]},
     'ma': {'manure_incorp': [35.0, 200.0]},
}

# removal efficiency components to extract
output_metrics = ['YNkg/ha', 'YPkg/ha', 'QNkg/ha', 'QPkg/ha', 'MUSLEt/ha', 'Qmm', 'WYLDmm']

# map crop to land use code and PHU key used in modify functions
crop_config = {
    'corn':    {'lu_code': 'rc', 'phu_key': 'phu_corn'},
    'soy':     {'lu_code': 'rc', 'phu_key': 'phu_soy'},
    'alfalfa': {'lu_code': 'ha', 'phu_key': 'phu_alfalfa'},
    'wheat':   {'lu_code': 'rc', 'phu_key': 'phu_wheat'},
}

regions = ['a', 'cp', 'p', 'vr']

# Which BMPs apply to which crops
bmp_crop_map = {
    'base': ['corn', 'soy', 'alfalfa', 'wheat'],
    'cc': ['corn', 'soy'],
    'gb': ['corn', 'soy', 'alfalfa', 'wheat'],
    'ma': ['corn', 'soy', 'alfalfa', 'wheat'],
    'nm': ['corn', 'soy', 'alfalfa', 'wheat'],
    'nt': ['corn', 'soy', 'alfalfa', 'wheat'],
}

# Region folder name patterns
region_folder_map = {
    ('corn', 'a'): 'a-rc-corn',
    ('corn', 'cp'): 'cp-rc-corn',
    ('corn', 'p'): 'p-rc-corn',
    ('corn', 'vr'): 'vr-rc-corn',
    ('soy', 'a'): 'a-soy',      
    ('soy', 'cp'): 'cp-soy',
    ('soy', 'p'): 'p-soy',
    ('soy', 'vr'): 'vr-soy',
    ('alfalfa', 'a'): 'a-hay-alfalfa',
    ('alfalfa', 'cp'): 'cp-hay-alfalfa',
    ('alfalfa', 'p'): 'p-hay-alfalfa',
    ('alfalfa', 'vr'): 'vr-hay-alfalfa',
    ('wheat', 'a'): 'a-wheat',       
    ('wheat', 'cp'): 'cp-wheat',
    ('wheat', 'p'): 'p-wheat',
    ('wheat', 'vr'): 'vr-wheat',
}


# HELPERS

def get_scenario_folder_name(region, crop, bmp):
    lu = crop_config[crop]['lu_code']
    if crop == 'soy' and region in ['a', 'cp']:
        return f"{region}-soy-{bmp}-1990-2000"
    else:
        return f"{region}-{lu}-{bmp}-1990-2000"


def get_base_folder_name(region, crop):
    lu = crop_config[crop]['lu_code']
    if crop == 'soy' and region in ['a', 'cp']:
        return f"{region}-soy-base-1990-2000"
    else:
        return f"{region}-{lu}-base-1990-2000"


def get_template_path(crop, region, bmp_or_base):
    region_folder = region_folder_map[(crop, region)]
    if bmp_or_base == 'base':
        scenario = get_base_folder_name(region, crop)
    else:
        scenario = get_scenario_folder_name(region, crop, bmp_or_base)
    return os.path.join(base_fp, crop, region_folder, scenario)


def build_sobol_problem(crop, bmp):
    # for a given x BMP combo
    names = ['pec', 'phu', 'delta_t', 'delta_p']
    bounds = [
        param_ranges['pec'],
        phu_ranges[crop],
        param_ranges['delta_t'],
        param_ranges['delta_p'],
    ]

    # # add BMP-specific params
    if bmp in bmp_params:
        for param_name, param_range in bmp_params[bmp].items():
            names.append(param_name)
            bounds.append(param_range)

    return {
        'num_vars': len(names),
        'names': names,
        'bounds': bounds,
    }


def apply_params_and_climate(scenario_path, params, crop):
    # modify in place
    phu_key = crop_config[crop]['phu_key']

    # build changes dict for modify_single_scenario_inplace
    changes = {
        'pec': params['pec'],
        phu_key: params['phu'],
    }
    if 'residue' in params:
        changes['residue'] = params['residue']
    if 'manure_incorp' in params:
        changes['manure_incorp'] = params['manure_incorp']

    # modify .sol, .opc, .sub files (but don't run APEX yet)
    for f in os.listdir(scenario_path):
        fp = os.path.join(scenario_path, f)
        if f.lower().endswith('.sol'):
            _modify_sol_file(fp, changes)
        elif f.lower().endswith('.opc'):
            _modify_opc_file(fp, changes)
        elif f.lower().endswith('.sub'):
            _modify_sub_file(fp, changes)

    # Perturb climate files in place
    delta_t = params['delta_t']
    precip_factor = 1.0 + params['delta_p']  # convert fractional to multiplier

    for f in os.listdir(scenario_path):
        fp = os.path.join(scenario_path, f)
        if f.endswith('_2su.dly'):
            perturb_dly_inplace(fp, delta_t, precip_factor)
        elif f.endswith('_2su.hly'):
            perturb_hly_inplace(fp, precip_factor)
        elif f.lower().endswith('.wp1'):
            perturb_wp1_inplace(fp, delta_t, precip_factor)


def perturb_dly_inplace(filepath, temp_delta, precip_factor):
    """Perturb .dly file in place."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if len(line) < 38:
            new_lines.append(line)
            continue
        tmax = float(line[20:26]) + temp_delta
        tmin = float(line[26:32]) + temp_delta
        prcp = float(line[32:38]) * precip_factor
        new_line = line[:20] + f"{tmax:6.2f}{tmin:6.2f}{prcp:6.2f}" + line[38:]
        new_lines.append(new_line)

    with open(filepath, 'w') as f:
        f.writelines(new_lines)


def perturb_hly_inplace(filepath, precip_factor):
    """Perturb .hly file in place."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        stripped = line.rstrip('\r\n')
        if len(stripped) < 20:
            new_lines.append(line)
            continue
        prcp = float(stripped[-10:]) * precip_factor
        new_line = stripped[:-10] + f"{prcp:10.3f}" + line[len(stripped):]
        new_lines.append(new_line)

    with open(filepath, 'w') as f:
        f.writelines(new_lines)


def perturb_wp1_inplace(filepath, temp_delta, precip_factor):
    """Perturb .wp1 file in place."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for i, line in enumerate(lines):
        line_num = i + 1
        if line_num in [3, 4]:  # TMX, TMN
            new_lines.append(_perturb_wp1_line(line, temp_delta, add=True))
        elif line_num == 7:     # PRCP
            new_lines.append(_perturb_wp1_line(line, precip_factor, add=False))
        else:
            new_lines.append(line)

    with open(filepath, 'w') as f:
        f.writelines(new_lines)


def _perturb_wp1_line(line, value, add=True):
    """Perturb 12 monthly values in a .wp1 line."""
    stripped = line.rstrip('\r\n')
    parts = stripped.split()
    label = parts[-1]
    new_vals = []
    for v in parts[:12]:
        val = float(v)
        val = val + value if add else val * value
        new_vals.append(f"{val:10.2f}")
    line_ending = line[len(stripped):]
    return ''.join(new_vals) + ' ' + label + line_ending


def run_apex(scenario_path):
    """Run APEX executable in the scenario folder."""
    import subprocess
    original_dir = os.getcwd()
    os.chdir(scenario_path)
    result = subprocess.run(['apex-mks'], shell=True, capture_output=True, text=True)
    os.chdir(original_dir)
    return result.returncode

def extract_removal_efficiency(base_path, bmp_path):
    """
    Compute removal efficiency of BMP relative to baseline.
    """
    removal_components = ['Nkg/ha', 'Pkg/ha', 'MUSLEt/ha']
    try:
        base_df = generate_loading_table(base_path)
        bmp_df = generate_loading_table(bmp_path)

        base_df = base_df[base_df['year'] >= sim_start]
        bmp_df = bmp_df[bmp_df['year'] >= sim_start]

        results = {}
        for comp in removal_components:
            results[comp] = calc_removal_efficiency(base_df, bmp_df, comp)

        try:
            yld_data = avg_cropyld(bmp_path, sim_start=sim_start)
            if yld_data:
                _, yld = next(iter(yld_data.items()))
                results['crop_yield_t/ha'] = yld
            else:
                results['crop_yield_t/ha'] = np.nan
        except:
            results['crop_yield_t/ha'] = np.nan

        return results

    except Exception as e:
        print(f"  ERROR extracting results: {e}")
        return {comp: np.nan for comp in output_metrics}

def extract_loads(scenario_path):
    '''extract loads and yield from single runs'''
    try:
        df = generate_loading_table(scenario_path)
        df = df[df['year'] >= sim_start]

        results = {
            'YNkg/ha': df['YNkg/ha'].sum(),
            'YPkg/ha': df['YPkg/ha'].sum(),
            'QNkg/ha': df['QNkg/ha'].sum(),
            'QPkg/ha': df['QPkg/ha'].sum(),
            'MUSLEt/ha': df['MUSLEt/ha'].sum(),
            'Qmm': df['Qmm'].sum(),
            'WYLDmm': df['WYLDmm'].sum(),
        }

        try:
            yld_data = avg_cropyld(scenario_path, sim_start=sim_start)
            if yld_data:
                _, yld = next(iter(yld_data.items()))
                results['crop_yield_t/ha'] = yld
            else:
                results['crop_yield_t/ha'] = np.nan

        except:
            results['crop_yield_t/ha'] = np.nan
        
        return results
    
    except Exception as e:
        print(f"  ERROR extracting loads: {e}")
        return {comp: np.nan for comp in output_metrics}
    
def run_single_sample(args):
    i, sample_vals, param_names, crop, template, combo_work_fp = args
    params = {name: val for name, val in zip(param_names, sample_vals)}

    scenario_copy = os.path.join(combo_work_fp, f"run_{i}")

    try:
        shutil.copytree(template, scenario_copy)
        apply_params_and_climate(scenario_copy, params, crop)
        run_apex(scenario_copy)

        results = extract_loads(scenario_copy)
        return i, results

    except Exception as e:
        return i, {comp: np.nan for comp in output_metrics}

    finally:
        import time as _time
        for attempt in range(3):
            try:
                if os.path.exists(scenario_copy):
                    shutil.rmtree(scenario_copy)
                break
            except PermissionError:
                _time.sleep(2)


# MAIN SOBOL LOOP

def run_sobol_for_combo(crop, region, bmp):
    # Run full Sobol' analysis for one crop x region x BMP combo

    combo_name = f"{region}_{crop}_{bmp}"
    print(f"Starting Sobol': {combo_name}")

    # build problem and generate samples
    problem = build_sobol_problem(crop, bmp)
    param_values = saltelli.sample(problem, n_samples, calc_second_order=True)
    n_runs = len(param_values)
    print(f"  Parameters: {problem['names']}")
    print(f"  Total samples: {n_runs} (each needs 2 APEX runs = {n_runs*2} total)")

    # template paths
    template = get_template_path(crop, region, bmp)

    if not os.path.exists(template):
        print(f"  SKIP — template not found: {template}")
        return None

    # working directory for combo
    combo_work_fp = os.path.join(work_fp, combo_name)
    os.makedirs(combo_work_fp, exist_ok=True)

    # --- Parallel execution ---
    from concurrent.futures import ProcessPoolExecutor, as_completed

    N_WORKERS = 4

    start_time = time.time()
    completed = 0


    # Build task list
    tasks = [
        (i, sample, problem['names'], crop, template, combo_work_fp)
        for i, sample in enumerate(param_values)
    ]

    # Run in parallel
    all_results = [None] * n_runs
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_single_sample, task): task[0] for task in tasks}

        for future in as_completed(futures):
            idx, result = future.result()
            all_results[idx] = result
            completed += 1

            if completed % 100 == 0 or completed == n_runs:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (n_runs - completed) / rate / 60 if rate > 0 else 0
                print(f"  {completed}/{n_runs} done | {rate:.1f} samples/sec | ~{remaining:.0f} min remaining")

            # Save checkpoint every 500 samples
            if completed % 500 == 0:
                checkpoint_df = pd.DataFrame([r for r in all_results if r is not None])
                checkpoint_df.to_csv(os.path.join(results_fp, f"{combo_name}_checkpoint.csv"), index=False)

    # convert to arrays and compute Sobol' indices
    # combine samples + results into one CSV
    samples_df = pd.DataFrame(param_values, columns=problem['names'])
    results_df = pd.DataFrame(all_results)
    combined_df = pd.concat([samples_df, results_df], axis=1)

    os.makedirs(results_fp, exist_ok=True)
    combined_df.to_csv(os.path.join(results_fp, f"{combo_name}_results.csv"), index=False)

    # compute Sobol' indices for each output
    sobol_results = {}
    for comp in output_metrics:
        Y = results_df[comp].values
        if np.any(np.isnan(Y)):
            n_nan = np.sum(np.isnan(Y))
            print(f"  WARNING: {n_nan} NaN values in {comp}, filling with 0")
            Y = np.nan_to_num(Y, nan=0.0)

        Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)
        sobol_results[comp] = Si

        # Print summary
        print(f"\n  Sobol' indices for {comp}:")
        print(f"  {'Param':<15} {'S1':>8} {'ST':>8}")
        for j, name in enumerate(problem['names']):
            print(f"  {name:<15} {Si['S1'][j]:8.3f} {Si['ST'][j]:8.3f}")

    # Save all Sobol' indices to one CSV
# Save all Sobol' indices to one CSV
    all_si = []
    for comp in output_metrics:
        Si = sobol_results[comp]
        names = problem['names']

        # First-order and total-order
        si_df = pd.DataFrame({
            'metric': comp,
            'parameter': names,
            'S1': Si['S1'],
            'S1_conf': Si['S1_conf'],
            'ST': Si['ST'],
            'ST_conf': Si['ST_conf'],
            'S2': np.nan,
            'S2_conf': np.nan,
        })
        all_si.append(si_df)

        # Second-order (pairwise interactions)
        S2 = Si['S2']
        S2_conf = Si['S2_conf']
        for j in range(len(names)):
            for k in range(j+1, len(names)):
                all_si.append(pd.DataFrame({
                    'metric': [comp],
                    'parameter': [f"{names[j]} x {names[k]}"],
                    'S1': [np.nan],
                    'S1_conf': [np.nan],
                    'ST': [np.nan],
                    'ST_conf': [np.nan],
                    'S2': [S2[j,k]],
                    'S2_conf': [S2_conf[j,k]],
                }))

    pd.concat(all_si).to_csv(os.path.join(results_fp, f"{combo_name}_sobol.csv"), index=False)

    elapsed_total = (time.time() - start_time) / 60
    print(f"\n  Completed {combo_name} in {elapsed_total:.1f} minutes")

    # clean up combo work dir
    if os.path.exists(combo_work_fp):
        shutil.rmtree(combo_work_fp)

    return sobol_results

def combine_all_results(results_fp):
    # Combine all individual combo files into one master CSV
    all_sobol = []
    all_results = []

    for f in os.listdir(results_fp):
        fp = os.path.join(results_fp, f)

        if f.endswith("_sobol.csv"):
            parts = f.replace("_sobol.csv", "").split('_')
            region, crop, bmp = parts[0], parts[1], parts[2]
            df = pd.read_csv(fp)
            df['region'] = region
            df['crop'] = crop
            df['bmp'] = bmp
            all_sobol.append(df)
        
        elif f.endswith("_results.csv"):
            parts = f.replace("_results.csv", "").split('_')
            region, crop, bmp = parts[0], parts[1], parts[2]
            df = pd.read_csv(fp)
            df['region'] = region
            df['crop'] = crop
            df['bmp'] = bmp
            all_results.append(df)
    
    if all_sobol:
        pd.concat(all_sobol).to_csv(os.path.join(box_fp, "all_sobol.csv"), index=False)
    
    if all_results:
        pd.concat(all_results).to_csv(os.path.join(box_fp, "all_combos_results.csv"), index=False)


# RUN 

if __name__ == "__main__":

    os.makedirs(work_fp, exist_ok=True)
    os.makedirs(results_fp, exist_ok=True)

    # # # TEST MODE: single combo with small sample size
    # # N_SAMPLES = 3  # override for testing
    # run_sobol_for_combo('corn', 'vr', 'cc')

    # # Manual append
    # all_sobol = pd.read_csv(os.path.join(box_fp, 'all_sobol_climate.csv'))
    # all_results = pd.read_csv(os.path.join(box_fp, 'all_combos_results_climate.csv'))

    # all_sobol = all_sobol[~((all_sobol['region']=='vr') & (all_sobol['crop']=='corn') & (all_sobol['bmp']=='cc'))]
    # all_results = all_results[~((all_results['region']=='vr') & (all_results['crop']=='corn') & (all_results['bmp']=='cc'))]

    # new_sobol = pd.read_csv(os.path.join(results_fp, 'vr_corn_cc_sobol.csv'))
    # new_sobol['region'] = 'vr'
    # new_sobol['crop'] = 'corn'
    # new_sobol['bmp'] = 'cc'

    # new_results = pd.read_csv(os.path.join(results_fp, 'vr_corn_cc_results.csv'))
    # new_results['region'] = 'vr'
    # new_results['crop'] = 'corn'
    # new_results['bmp'] = 'cc'

    # pd.concat([all_sobol, new_sobol]).to_csv(os.path.join(box_fp, 'all_sobol_climate.csv'), index=False)
    # pd.concat([all_results, new_results]).to_csv(os.path.join(box_fp, 'all_combos_results.csv'), index=False)
    # print("Updated summary CSVs")


    for bmp, crops in bmp_crop_map.items():
            for crop in crops:
                for region in regions:
                    combo_name = f"{region}_{crop}_{bmp}"
                    sobol_file = os.path.join(results_fp, f"{combo_name}_sobol.csv")
                    if os.path.exists(sobol_file):
                        print(f"SKIP (already done): {combo_name}")
                        continue
                    try:
                        run_sobol_for_combo(crop, region, bmp)
                    except Exception as e:
                        print(f"  FAILED: {e}")
                        continue
# 
    print(f"Looking in: {results_fp}")
    print(f"Files found: {len(os.listdir(results_fp))}")
    combine_all_results(results_fp)

    print("\n\nAll Sobol' analyses complete! Results saved to:", results_fp)