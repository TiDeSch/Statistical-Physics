import subprocess
import re
import sys
import os
import time
import json
start_time = time.time()

def update_var(script_path, updates, raw_vars=None):
    raw_vars = raw_vars or []

    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    for var, value in updates.items():
        if var in raw_vars:
            value_repr = value
        elif isinstance(value, str):
            value_repr = f'"{value}"'
        elif isinstance(value, bool):
            value_repr = "True" if value else "False"
        else:
            value_repr = repr(value)

        pattern = rf'^(\s*){var}\s*=\s*.*'
        content = re.sub(pattern, lambda m: f'{m.group(1)}{var} = {value_repr}', content, flags=re.MULTILINE)

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)

def run_script(script_path):
    print(f"Running: {script_path}")
    result = subprocess.run(
        [sys.executable, script_path],
        check=True,
        capture_output=True,
        text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)         
    return True

if __name__ == "__main__":
    SCRIPTS_FOLDER = r" "

    updates = {
        "Type": "ModelH", # Pressure, GNS, ModelH, AMB, AMB+, Kennedy
        "Directory": ['Passive', '-0.01', '-0.05', '-0.1', '-0.2', '-0.5', '-1.0'],
        #["kappa0", "kappa1", "kappa2", "kappa2.67", "kappa4", "kappa6", "kappa8"],
        #["Control_WT_15kPa", "WT_3kPa", "EcadKO_15kPa"],
        #["setC5", "setC6", "setC7"], 
        #['ModelB', 'Passive', '-0.01', '-0.05', '-0.1', '-0.2', '-0.5', '-1.0'], ['Passive', '-0.01', '-0.1', '-0.5'],
        #["AMP5", "AMP10"],
        #['zeta0.5_alpha10.0', 'zeta1.0_alpha5.0', 'zeta1.0_alpha10.0', 'zeta5.0_alpha2.5', 'zeta5.0_alpha5.0', 'zeta5.0_alpha10.0', 'zeta5.0_alpha15.0'], ['zeta0.5_alpha10.0', 'zeta1.0_alpha10.0', 'zeta5.0_alpha10.0'],
        #['K2.0_alpha0.0', 'K2.0_alpha0.5', 'K2.0_alpha1.0', 'K2.0_alpha2.0', 'K2.0_alpha5.0', 'K2.0_alpha10.0', 'K2.0_alpha15.0', 'K2.0_alpha20.0']
        #['kappa2', 'kappa2.67', 'kappa4', 'kappa6']

        "markers": ["o", "*", "v", "x", "d", "1", "p", "h"],
        "t_norm_max": 3e6, # AMB+5e6, pressure1e4, ModelH6e3, Kennedy1e4, 
        "rescale_val": 20, #20, AMB+50, 20ModelH
        "ShowPlots": False,
    }

    scripts = {
        # Field Analysis
        "field_structure_factor_autocorrelation.py": False,
        "field_distribution.py": False,
        "field_Radius_of_Gyration.py": False,

        # Field Plots 
        "plot_structure_factor_autocorrelation.py": False,
        "plot_distribution.py": False,
        "plot_Radius_of_Gyration.py": False,

        # Data extraction
        "Extract_isolines.py": False,

        # Analysis
        "analysis_Fractal_dimension.py": False,
        "analysis_Winding_angle.py": False,
        "analysis_Left_Passage_Prob.py": False,
        "analysis_Noise_plot.py": True,
        "analysis_Correlation.py": True,

        # Plots
        "plot_yardstick.py": False,
        "plot_winding_angle.py": False,
        "plot_LPP.py": False,
        "plot_driving.py": True,
        "plot_correlation.py": True
    }

    results = {}
    for script, should_run in scripts.items():
        script_path = os.path.join(SCRIPTS_FOLDER, script)
        if should_run:
            update_var(script_path, updates, raw_vars=["custom_legend"])
            success = run_script(script_path)
            results[script] = success
        else:
            print(f"Skipping: {script}")
    
    enabled_scripts = [s for s, enabled in scripts.items() if enabled]
    successful = sum(1 for success in results.values() if success)
    failed = len(results) - successful
    
    print(f"Total scripts enabled: {len(enabled_scripts)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed scripts:")
        for script, success in results.items():
            if not success:
                print(f"- {script}")

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")