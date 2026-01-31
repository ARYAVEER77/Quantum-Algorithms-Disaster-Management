# Quantum Algorithms for Disaster Prediction and Management

This repository accompanies the manuscript:

**“Quantum Algorithms for Disaster Prediction and Management”**
submitted to *Neural Computing and Applications* (Springer Nature).

It provides all datasets, source code, and instructions required to **exactly reproduce** the experimental results reported in the manuscript.

---

## Abstract

This work presents a proof-of-concept hybrid framework that integrates quantum computing, artificial intelligence, and data science for disaster prediction and management. The proposed system combines quantum neural networks for disaster classification, quantum approximate optimization algorithms for resource allocation, quantum walks for evacuation route optimization, and quantum phase estimation for disaster severity modeling. Using publicly available EM-DAT disaster data, the framework demonstrates how emerging quantum algorithms can enhance decision-making in complex, large-scale disaster response scenarios.

**Key Reported Result**: 94.2% (±1.8%) accuracy in multi-class disaster type prediction using a Sampler-based Quantum Neural Network.

---Quantum-Algorithms-Disaster-Management/
│
├── src/                                   # Main source code directory
│   ├── main.py                            # End-to-end pipeline orchestrator
│   ├── cleaning.py                        # Data loading & preprocessing (load_and_clean_data)
│   ├── ai_model.py                        # Classical AI risk scoring (train_ai_model)
│   ├── quantum_qbm.py                     # Quantum Boltzmann Machine / Neural Network
│   ├── quantum_qaoa.py                    # QAOA for resource allocation
│   ├── quantum_walk.py                    # Quantum Walk for evacuation routing
│   ├── quantum_qpe.py                     # QPE for severity estimation
│   └── disasters_1970_2021.csv            # Preprocessed EM-DAT disaster dataset
│
├── results/                               # Generated outputs (created on first run)
│   └── figures/                           # All visual results (Figs. 1-9 from manuscript)
│
├── requirements.txt                       # Exact Python package dependencies
├── LICENSE                                # MIT License
├── CITATION.cff                          # Citation metadata
└── README.md                              # This file

## 1. Environment Setup & Installation

### 1.1 Clone the Repository
```bash

git clone https://github.com/ARYAVEER77/Quantum-Algorithms-Disaster-Management.git
cd Quantum-Algorithms-Disaster-Management
# For Linux/macOS
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
