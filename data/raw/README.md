# Raw Dataset Information

This directory contains the raw datasets used in the study:

**"Quantum Algorithms for Disaster Prediction and Management"**

Author: Aryaveer Desai  
Affiliation: MVM International School, India; Harvard Summer School  

---

## Dataset Description

The primary dataset used in this study is derived from the EM-DAT
(International Disaster Database) maintained by the Centre for Research
on the Epidemiology of Disasters (CRED).

### Files included:

- `emdat_1970_2021.csv`
  - Covers global disaster records from 1970 to 2021
  - Disaster types include floods, earthquakes, storms, droughts, and wildfires
  - Variables include event type, location, severity indicators, casualties, and economic impact

---

## Source and Licensing

Original data source:
- EM-DAT: https://www.emdat.be/

The dataset is used strictly for academic and research purposes.
Licensing and usage comply with EM-DAT’s terms for non-commercial research.

---

## Data Integrity

These files are provided **without modification**.
All preprocessing, cleaning, and feature engineering steps are documented
in the `data/processed/` directory and corresponding source scripts.

---

## Reproducibility Note

Reviewers and readers are encouraged to verify:
- File integrity
- Column consistency
- Event labeling

Any preprocessing steps applied to this data are fully reproducible
using the scripts provided in the `src/` directory.
