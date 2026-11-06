# AI Network Intrusion Detection

A machine learning project that detects malicious network traffic using the **CIC-IDS-2017** dataset, with an interactive **Streamlit** dashboard for exploring the data, comparing models and running live predictions.


**Live demo:** 

---

## Project Overview

Network intrusion detection systems (IDS) must separate normal traffic from attacks such as DoS, DDoS, port scans and brute force. This project builds a binary classifier (**BENIGN vs ATTACK**) on 3.1 million real network flows, documents every data-cleaning decision, and compares two models.

## Dataset

- **Source:** CIC-IDS-2017, Canadian Institute for Cybersecurity (University of New Brunswick)
- **Size:** 8 CSV files, 3,119,345 rows x 85 columns (flow-level features from CICFlowMeter)
- **Classes (15):** BENIGN, DoS Hulk, PortScan, DDoS, DoS GoldenEye, FTP-Patator, SSH-Patator, DoS slowloris, DoS Slowhttptest, Bot, Web Attack (Brute Force, XSS, SQL Injection), Infiltration, Heartbleed
- **Imbalance:** BENIGN makes up about 73% of raw rows, while Heartbleed has only 11 rows

## Data Cleaning Pipeline

| Step | Action | Result |
|---|---|---|
| 1 | Load and merge 8 CSV files (latin-1 encoding, float32 to save RAM) | 3,119,345 x 85 |
| 2 | Replace Inf with NaN, drop NaN and empty rows | 2,827,876 rows |
| 3 | Drop identifier columns (Flow ID, IPs, Source Port, Timestamp) to avoid leakage | 80 columns |
| 4 | Drop 329,717 duplicate flows (11.7%) | 2,498,159 rows |
| 5 | Drop 8 constant columns | 72 columns |
| 6 | Convert labels to binary (BENIGN = 0, ATTACK = 1) | 2,072,418 vs 425,741 |

**Outliers were intentionally not removed.** Attacks such as DoS and PortScan naturally produce extreme values, so removing outliers would remove the attack samples the model must learn. Tree-based models are also robust to outliers.

## Models and Results

Split: 80% train / 20% test, stratified, `random_state=42`.

| Metric | Decision Tree | Random Forest |
|---|---|---|
| Accuracy | 0.9984 | 0.9986 |
| Precision (attack) | 0.9959 | 0.9964 |
| Recall (attack) | 0.9949 | 0.9952 |
| F1 (attack) | 0.9954 | 0.9958 |
| Macro F1 | 0.9972 | 0.9974 |
| False alarms (FP) | 349 | 309 |
| Missed attacks (FN) | 433 | 412 |

## Limitations (please read)

- These scores come from a **random split**, which is an **optimistic baseline**. Flows from the same attack session can appear in both train and test, so real-world performance will likely be lower.
- The data was generated in a controlled lab in 2017 and does not represent modern or encrypted traffic.
- Classes with very few samples (Infiltration, SQL Injection, Heartbleed) are only evaluated as part of the binary "ATTACK" class.

**Planned next step:** a day-based split (train on Monday to Thursday, test on Friday) and an attack-type holdout test to measure generalisation. Results will be added here.

## Dashboard Features

- **Overview:** key project numbers at a glance
- **Data Insights:** before vs after cleaning charts, feature explorer by class
- **Model Performance:** compare models, confusion matrices, top-N feature importance
- **Live Prediction:** score sample rows or upload your own CSV, adjust the attack threshold, download results

## Project Structure

```
.
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── models/
│   ├── rf_compressed.joblib    # trained Random Forest
│   └── features.joblib         # feature names
├── data/
│   └── sample_test.csv         # 5,000 held-out rows for the demo
├── assets/
│   └── cover.png
└── notebooks/
    ├── cyber_attack_detection.ipynb          # loading and cleaning
    └── cyberattackdetection-training.ipynb   # training and evaluation
```

## Run Locally

```bash
git clone https://github.com/niyamat-ali/<your-repo-name>.git
cd <your-repo-name>
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit, Joblib

## Author

**Niyamat Ali**: [GitHub](https://github.com/niyamat-ali) 
