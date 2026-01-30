import numpy as np
import random
import os

# --- REPRODUCIBILITY SETUP ---
# This ensures results match the paper's reported metrics exactly.
# See Appendix A: "Random seed: 42"
np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
# -----------------------------

from cleaning import load_and_clean_data
from ai_model import train_ai_model
from quantum_qbm import run_qbm_simulation
from quantum_qaoa import run_qaoa_allocation
from quantum_walk import simulate_quantum_walk
from quantum_qpe import quantum_phase_estimation

# Ensure you create the 'DISASTERS' folder and put the CSV file there
file_path = "DISASTERS/disasters_1970_2021.csv"
df = load_and_clean_data(file_path)

print("\n🚨 Training AI model to generate disaster risk scores...")
model, risk_scores, label_encoder = train_ai_model(df)

print("\n⚛️ Running Quantum Boltzmann Machine (QBM) Simulation...")
features = df[['Year', 'Start Month', 'Start Day', 'Total Deaths', 'Total Affected', "Total Damages ('000 US$)", 'Dis Mag Value']].values
labels = df['Disaster Type'].astype('category').cat.codes.values
qbm_model, qbm_input = run_qbm_simulation(features, labels)
print("✅ QBM simulation complete.")

allocation, score = run_qaoa_allocation()

evacuation_probs = simulate_quantum_walk()

# Added missing QPE module execution to match Figure 8 of the paper
print("\n🌊 Running Quantum Phase Estimation (QPE) for Severity Simulation...")
severity_counts = quantum_phase_estimation()

print("\n🧠 Full Pipeline Completed:")
print(f"Resource Allocation via QAOA: {allocation} (Score: {score})")
print(f"Evacuation Probabilities via Quantum Walk: {evacuation_probs}")
print(f"Severity Simulation Counts: {severity_counts}")
