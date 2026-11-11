# Dataset Sheet: CSE-CIC-IDS2018 (In-Distribution Baseline)

## 1. Overview
- **Name:** Communications Security Establishment & Canadian Institute for Cybersecurity IDS 2018 (CSE-CIC-IDS2018)
- **Host Institution:** University of New Brunswick (UNB), Canada & Communications Security Establishment (CSE)
- **Source Portal:** [https://www.unb.ca/cic/datasets/ids-2018.html](https://www.unb.ca/cic/datasets/ids-2018.html)
- **AWS S3 Bucket:** `s3://cse-cic-ids2018/`
- **License:** Open Academic / Research Use
- **Study Role:** In-Distribution (ID) Training, Validation Calibration, and Baseline Test Evaluation

## 2. Telemetry & Feature Extraction
- **Traffic Capture Period:** 10 days of capture in February–March 2018.
- **Flow Extractor:** CICFlowMeter-V3.
- **Raw Features:** 76-80 bidirectional statistical flow features.
- **SEMANTICSHIELD NF Subset:** 13 NetFlow-standardized features (8 exact physical mappings, 5 approximate mappings).
- **Label Schema:** 15 distinct classes (Benign + 14 attack types: DoS-Hulk, DoS-GoldenEye, DoS-Slowloris, DoS-SlowHTTPTest, DDoS-LOIC-HTTP, DDoS-HOIC, DDoS-LOIC-UDP, Bot, BruteForce-FTP, BruteForce-SSH, BruteForce-Web, BruteForce-XSS, SQL-Injection, Infiltration).

## 3. Preprocessing & Partitioning
- **Cleaning:** Stripped column whitespace, converted IEEE-754 `inf`/`-inf` to `NaN`, dropped missing values, deduplicated records.
- **Train / Test Split:** Stratified 80% train, 20% test (random seed 42).
- **Test Partition:** $N = 138,069$ flows (stored in `experiments/X_test_nf.npy`, `experiments/y_test_nf.npy`).
- **Calibration Split:** 10,000 held-out benign samples reserved exclusively for empirical percentile threshold calibration at 5% target FPR.
- **StandardScaler:** Parameters ($\mu, \sigma$) fitted exclusively on training set and serialized to `experiments/standard_scaler_nf.joblib`.
