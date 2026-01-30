#!/usr/bin/env python3
# run_experiments.py
import subprocess
import sys

def main():
    print("Running Quantum Disaster Management Experiments...")
    
    # Run data cleaning
    print("1. Cleaning data...")
    subprocess.run([sys.executable, "src/cleaning.py"], check=True)
    
    # Run main pipeline
    print("2. Running main pipeline...")
    subprocess.run([sys.executable, "src/main.py"], check=True)
    
    # Run validation
    print("3. Validating results...")
    subprocess.run([sys.executable, "src/analysis_validation.py"], check=True)
    
    print("✅ All experiments completed!")

if __name__ == "__main__":
    main()
