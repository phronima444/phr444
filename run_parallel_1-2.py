
import concurrent.futures
import subprocess

def run_script(script_name):
    """Function to run a Python script."""
    subprocess.run(["python", script_name])

if __name__ == "__main__":
    # Names of the scripts to be run
    scripts = ["updated_trading_script_with_improvements.py", "full_trading_script_manual_corrected.py"]

    # Using ProcessPoolExecutor to run the scripts in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(run_script, scripts)
