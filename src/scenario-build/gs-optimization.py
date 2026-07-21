"""
Three-Stage Grid Search Optimization
Finds optimal baseline parameters that work well with all BMPs
Test all combinations, filter by objectives
"""

import os
import sys
import pandas as pd
import itertools
import warnings
from io import StringIO
import time
from datetime import datetime, timedelta

# import diagnostic functions
sys.path.append(r'path/to/where/apex_config_functions.py/lives')
from apex_config_functions import (
    create_bmp_table, 
    modify_single_scenario_inplace,
    get_scenario_id
)

warnings.filterwarnings('ignore')


# CORE FUNCTIONS---------------------------------------------------------------------

def check_objectives(metrics, objectives):
    """Check if solution meets all objectives"""
    for metric_name, objective in objectives.items():
        if metric_name not in metrics:
            return False
        
        value = metrics[metric_name]
        if pd.isna(value):
            return False
        
        if objective['type'] == 'range':
            if not (objective['min'] <= value <= objective['max']):
                return False
        elif objective['type'] == 'min':
            if value < objective['value']:
                return False
        elif objective['type'] == 'max':
            if value > objective['value']:
                return False
    
    return True


def run_grid_search_stage(base_dir, parameters, stage_name, output_folder, fert_proportions=None, objectives=None):
    """
    Run one stage of grid search optimization
    Tests all combinations and filters by objectives
    """
    print(f"STAGE: {stage_name}")
    
    # setup
    results_path = os.path.join(os.path.dirname(base_dir), output_folder)
    os.makedirs(results_path, exist_ok=True)
    
    # generate combinations
    param_names = list(parameters.keys())
    param_values = list(parameters.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"parameters: {', '.join(param_names)}")
    print(f"runs: {len(combinations)}")

    all_results = []
    
    # run all combinations
    for i, combination in enumerate(combinations):
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{len(combinations)}")
        
        changes = dict(zip(param_names, combination))

        # add fertilizer proportions if provided
        if fert_proportions:
            changes['fert_proportions'] = fert_proportions
        
        try:
            # modify scenarios in place
            scenarios = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
            for scenario in scenarios:
                modify_single_scenario_inplace(os.path.join(base_dir, scenario), changes)
            
            # get metrics
            metrics = get_metrics(base_dir)
            
            # record parameters
            for pname, pval in changes.items():
                metrics[f'param_{pname}'] = pval
            
            metrics['test_number'] = i + 1
            
            # check if objectives are met
            if objectives is not None:
                metrics['meets_objectives'] = check_objectives(metrics, objectives)
            
            all_results.append(metrics)
            
            # save progress every 100 tests
            if (i + 1) % 100 == 0:
                pd.DataFrame(all_results).to_csv(
                    os.path.join(results_path, 'progress.csv'), index=False
                )
        
        except Exception as e:
            print(f"!!! ERROR in test {i+1}: {e}")  # CHANGE THIS
            import traceback
            traceback.print_exc()  # ADD THIS
            raise  # ADD THIS - re-raise so script stops on error
    

    # process results and save to csv
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(results_path, 'all_results.csv'), index=False)
    

def get_metrics(base_dir, debug=False):
    """Extract metrics from all scenarios"""
    if not debug:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
    
        # initialize metrics dictionary
    metrics = {}
    
    try:
        # collect metrics
        bmp_df = create_bmp_table(base_dir)
        
        # check if empty
        if bmp_df.empty:
            metrics = {'error': 'create_bmp_table returned empty dataframe'}
            if debug:
                print("ERROR: create_bmp_table returned empty dataframe")
            return metrics
        
        # extract metrics
        for scenario_name in bmp_df.index:
            scenario_id = get_scenario_id(scenario_name)
            if not scenario_id:
                if debug:
                    print(f"WARNING: Could not get scenario_id for {scenario_name}")
                continue
            
            for col in bmp_df.columns:
                safe_col = col.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
                metrics[f'{scenario_id}_{safe_col}'] = bmp_df.loc[scenario_name, col]
    
    except Exception as e:
        metrics = {'error': str(e)}
        if debug:
            print(f"ERROR in get_metrics: {str(e)}")
            import traceback
            traceback.print_exc()
    finally:
        if old_stdout is not None:
            sys.stdout = old_stdout
    
    return metrics



# MAIN EXECUTION ------------------------------------------------

if __name__ == "__main__":

    # start timer
    overall_start = time.time()
    print(f"start time: {overall_start}")

# EXAMPLE:
# # WHEAT - VR

    vrwheat_base_dir = # path to scenario files
    
    # define fertilizer proportions
    vrwheat_fert_proportions = {
        'N1': 0.2,
        'N2': 0.8,
        'P': 1,
    }

    # # STAGE 2: BASELINE + MANURE AMENDMENT
    
    vrwheat_stage2_params = {
        'n_fert_base': [96.50],
        'p_fert_base': [35.92],
        'n_fert_ma': [111.14],
        'p_fert_ma': [35.92],
        'pec_ma': [0.60, 0.75, 0.8, 0.9, 1.0],
        'phu_wheat': [1200, 1500],
        'manure_incorp': [50, 75, 100],
    }
    
    
    stage2_start = time.time()
    stage2_results = run_grid_search_stage(
        vrwheat_base_dir, vrwheat_stage2_params,
        "Baseline + Manure Amendment", r"path/to/output",
        fert_proportions=vrwheat_fert_proportions
    )
    stage2_time = time.time() - stage2_start
    print(f"\nStage 2 run time: {timedelta(seconds=int(stage2_time))}")
    
    # STAGE 3: BASELINE + NO-TILL
    
    vrwheat_stage3_params = {
        'n_fert_base': [96.50],
        'p_fert_base': [35.92],
        'n_fert_nt': [111.14],
        'p_fert_nt': [35.92],
        'pec_nt': [0.60, 0.75, 0.8, 0.9, 1.0],
        'phu_wheat': [1200, 1500],
        'residue': [5, 8, 12],
    }

    stage3_start = time.time()
    stage3_results = run_grid_search_stage(
        vrwheat_base_dir, vrwheat_stage3_params,
        "Baseline + No-Till", r"path/to/output",
        fert_proportions=vrwheat_fert_proportions
    )
    stage3_time = time.time() - stage3_start
    print(f"\nStage 3 run time: {timedelta(seconds=int(stage3_time))}")

    
    print("OPTIMIZATION COMPLETE!")
    total_time = time.time() - overall_start
    print(f"Total run time: {timedelta(seconds=int(total_time))}")

