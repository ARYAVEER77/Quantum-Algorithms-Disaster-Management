from cleaning import load_and_clean_data
from ai_model import train_ai_model
from quantum_qbm import run_qbm_simulation
from quantum_qaoa import run_qaoa_allocation
from quantum_walk import simulate_quantum_walk

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


print("\n🧠 Full Pipeline Completed:")
print(f"Resource Allocation via QAOA: {allocation} (Score: {score})")
print(f"Evacuation Probabilities via Quantum Walk: {evacuation_probs}")
