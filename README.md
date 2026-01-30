Quantum Algorithms for Disaster Prediction and Management

This repository accompanies the manuscript:

“Quantum Algorithms for Disaster Prediction and Management”
submitted to Neural Computing and Applications (Springer Nature).

The repository provides all datasets, source code, and instructions required to reproduce the experimental results reported in the manuscript.

Abstract

This work presents a proof-of-concept hybrid framework that integrates quantum computing, artificial intelligence, and data science for disaster prediction and management. The proposed system combines quantum neural networks for disaster classification, quantum approximate optimization algorithms for resource allocation, quantum walks for evacuation route optimization, and quantum phase estimation for disaster severity modeling. Using publicly available EM-DAT disaster data, the framework demonstrates how emerging quantum algorithms can enhance decision-making in complex, large-scale disaster response scenarios.

Repository Structure

The repository is organized to ensure clarity, reproducibility, and compliance with Springer Nature’s data and code availability guidelines.

quantum-algorithms-disaster-management/
│
├── data/
│   ├── raw/                # Original datasets
│   ├── processed/          # Cleaned and transformed datasets
│
├── src/                    # Source code for all experiments
│
├── environment/            # Dependency specifications
│
├── results/                # Generated figures and metrics
│
├── REVIEWER_INSTRUCTIONS.md
├── CITATION.cff
├── LICENSE
└── README.md

Data Availability

The datasets generated and/or analyzed during this study are publicly available in this repository.

Raw Data

The raw disaster dataset is stored under:

📁 data/raw/

This includes:

emdat_1970_2021.csv — disaster event data derived from the EM-DAT database (1970–2021), provided in CSV format for direct accessibility and reproducibility.

The EM-DAT dataset originates from the Centre for Research on the Epidemiology of Disasters (CRED) and is used strictly under academic research terms.

Detailed dataset provenance, structure, and documentation are provided in:

➡️ data/raw/README.md

Processed Data

Processed datasets generated from the raw data are stored under:

📁 data/processed/

All preprocessing and transformation steps are explicitly documented in:

➡️ src/cleaning.py

Reproducibility

All experiments were conducted with reproducibility as a primary design goal.

Python version: 3.10

Quantum simulation framework: Qiskit

Random seed: 42 (fixed across experiments where applicable)

All scripts required to reproduce the reported results are included in the repository.
