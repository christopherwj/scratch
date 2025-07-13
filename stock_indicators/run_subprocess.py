import subprocess
import sys

def run_analysis():
    """
    Runs the GPU-accelerated analysis script as a subprocess
    and captures its output.
    """
    print("--- Starting analysis via subprocess runner ---")
    try:
        process = subprocess.Popen(
            [sys.executable, 'run_rolling_analysis_gpu.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        print("\n--- STDOUT ---")
        print(stdout)
        
        if stderr:
            print("\n--- STDERR ---")
            print(stderr)
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    run_analysis() 