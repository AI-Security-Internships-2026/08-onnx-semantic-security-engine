# SEMANTICSHIELD Datasets Guide & Acquisition Protocol

This document details the acquisition, preprocessing, schema alignment, and storage policy for all datasets evaluated in the **SEMANTICSHIELD** research manuscript.

---

## 1. Storage & Git Governance Policy

> [!IMPORTANT]
> **Do NOT commit raw PCAPs, multi-gigabyte CSVs, or Parquet datasets directly to this Git repository.**
> Raw telemetry files exceed Git storage limits, slow down version control, and complicate licensing.
> All raw dataset directories are strictly ignored via [`.gitignore`](../.gitignore).
> Only schema metadata, checksums, scripts, and pre-extracted stratified test partitions are version-controlled.

---

## 2. Dataset Overview & Experimental Roles

| Dataset | Provider / Source | Extractor | Total Flows | Features | Study Role |
|---|---|---|---|---|---|
| **CSE-CIC-IDS2018** | UNB / Communications Security Establishment | CICFlowMeter-V3 | 16.2M raw (138,069 test) | 76 raw $\to$ 13 NF | **In-Distribution (ID)**: Training, validation calibration, and baseline test |
| **NF-ToN-IoT-v2** | UNSW Canberra Cyber / Sarhan et al. | nProbe (NetFlow v9/IPFIX) | 13,135,881 | 43 raw $\to$ 13 NF | **Out-of-Distribution (OOD)**: Real covariate shift & cross-extractor evaluation (Track B) |
| **NF-BoT-IoT-v2** | UNSW Canberra Cyber / Sarhan et al. | nProbe (NetFlow v9/IPFIX) | 30,420,086 | 43 raw $\to$ 13 NF | **External Standardized Shift**: Same-schema cross-dataset shift & cross-model replication (Track A) |
| **Synthetic Perturbations** | Generated on-the-fly (`scripts/fixed_fpr_evaluation.py`) | Programmatic | 5,000 / mode | 13 NF | **Deployment Failure Modes**: E3 (NaN/Inf), E4 (Zero-fill), E5 (Permutation), E6 (Unit scale), E7 (Outliers), E8 (Gaussian noise) |

---

## 3. Dataset Acquisition & Download Instructions

### 3.1 CSE-CIC-IDS2018 (In-Distribution)
- **Official Portal:** [University of New Brunswick CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html)
- **AWS Open Data Registry:** `s3://cse-cic-ids2018/`
- **License:** Open Academic / Research Use
- **Target Directory:** `datasets/CIC-IDS2018/`
- **Files Required:** 10 daily CSV or Parquet captures:
  - `Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet`
  - `Bruteforce-Wednesday-14-02-2018_TrafficForML_CICFlowMeter.parquet`
  - `DDoS1-Tuesday-20-02-2018_TrafficForML_CICFlowMeter.parquet`
  - `DDoS2-Wednesday-21-02-2018_TrafficForML_CICFlowMeter.parquet`
  - `DoS1-Thursday-15-02-2018_TrafficForML_CICFlowMeter.parquet`
  - `DoS2-Friday-16-02-2018_TrafficForML_CICFlowMeter.parquet`
  - `Infil1-Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet`
  - `Infil2-Thursday-01-03-2018_TrafficForML_CICFlowMeter.parquet`
  - `Web1-Thursday-22-02-2018_TrafficForML_CICFlowMeter.parquet`
  - `Web2-Friday-23-02-2018_TrafficForML_CICFlowMeter.parquet`

**Direct Acquisition Command (via AWS CLI):**
```bash
mkdir -p datasets/CIC-IDS2018
aws s3 sync --no-sign-request s3://cse-cic-ids2018/Processsed_Traffic_For_ML_CIC/ datasets/CIC-IDS2018/
```

### 3.2 NF-ToN-IoT-v2 (Out-of-Distribution Shift)
- **Official Source:** [NetFlow Datasets for Machine Learning (Sarhan et al., UQ/UNSW)](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
- **Alternative Mirror:** [Kaggle NF-ToN-IoT-v2 Mirror](https://www.kaggle.com/datasets/dhoogla/nftoniotv2)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Target File:** `datasets/ToN-IoT/NF-ToN-IoT-V2.parquet`
- **Format:** Apache Parquet (ZSTD / Snappy compression, ~200 MB)

**Acquisition via Kaggle CLI:**
```bash
mkdir -p datasets/ToN-IoT
kaggle datasets download -d dhoogla/nftoniotv2 -p datasets/ToN-IoT/ --unzip
```

### 3.3 NF-BoT-IoT-v2 (Standardized Cross-Dataset Shift)
- **Official Source:** [NetFlow Datasets for Machine Learning (Sarhan et al., UQ/UNSW)](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
- **Alternative Mirror:** [Kaggle NF-BoT-IoT-v2 Mirror](https://www.kaggle.com/datasets/dhoogla/nfbotiotv2)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Target File:** `datasets/NF-BoT-IoT-V2/NF-BoT-IoT-V2.parquet`
- **Format:** Apache Parquet (~508 MB)

**Acquisition via Kaggle CLI:**
```bash
mkdir -p datasets/NF-BoT-IoT-V2
kaggle datasets download -d dhoogla/nfbotiotv2 -p datasets/NF-BoT-IoT-V2/ --unzip
```

---

## 4. NetFlow Standardization (13 Canonical Features)

To bridge heterogeneous network extractors (CICFlowMeter vs nProbe NetFlow v9/IPFIX), **SEMANTICSHIELD** maps bidirectional telemetry to 13 semantically aligned flow features:

| Index | Feature Name (Canonical / CIC-IDS2018) | NetFlow v2 / IPFIX Column (nProbe) | Physical Mapping Tier | Physical Meaning |
|:---:|---|---|:---:|---|
| **0** | `Flow Duration` | `FLOW_DURATION_MILLISECONDS` | **Tier 1 (Exact)** | Total bidirectional flow duration (ms) |
| **1** | `Total Fwd Packets` | `IN_PKTS` | **Tier 1 (Exact)** | Packets transmitted client $\to$ server |
| **2** | `Total Backward Packets` | `OUT_PKTS` | **Tier 1 (Exact)** | Packets transmitted server $\to$ client |
| **3** | `Fwd Packets Length Total` | `IN_BYTES` | **Tier 1 (Exact)** | Payload/total bytes client $\to$ server |
| **4** | `Bwd Packets Length Total` | `OUT_BYTES` | **Tier 1 (Exact)** | Payload/total bytes server $\to$ client |
| **5** | `Packet Length Max` | `LONGEST_FLOW_PKT` | **Tier 1 (Exact)** | Maximum observed packet length (bytes) |
| **6** | `Packet Length Min` | `SHORTEST_FLOW_PKT` | **Tier 1 (Exact)** | Minimum observed packet length (bytes) |
| **7** | `Protocol` | `PROTOCOL` | **Tier 1 (Exact)** | Transport protocol number (e.g. 6=TCP, 17=UDP) |
| **8** | `Fwd Packet Length Max` | `MAX_IP_PKT_LEN` | **Tier 2 (Approximate)** | Max IP packet length observed forward |
| **9** | `Fwd Packet Length Min` | `MIN_IP_PKT_LEN` | **Tier 2 (Approximate)** | Min IP packet length observed forward |
| **10** | `Flow Bytes/s` | `SRC_TO_DST_SECOND_BYTES` | **Tier 2 (Approximate)** | Forward data transfer rate |
| **11** | `Init Fwd Win Bytes` | `TCP_WIN_MAX_IN` | **Tier 2 (Approximate)** | Maximum initial TCP window size (client $\to$ server) |
| **12** | `Init Bwd Win Bytes` | `TCP_WIN_MAX_OUT` | **Tier 2 (Approximate)** | Maximum initial TCP window size (server $\to$ client) |

---

## 5. Preprocessing & Partitioning Pipeline

All preprocessing transformations are deterministic and reproducible:

1. **Cleaning:**
   - Strip leading/trailing whitespaces from column names.
   - Replace IEEE-754 `inf` and `-inf` with `NaN` and drop corresponding rows.
   - Remove duplicate flow records.
2. **Standardization:**
   - Features are standardized via `StandardScaler`: $z = \frac{x - \mu}{\sigma}$.
   - **Critical Integrity Rule:** The scaler (`experiments/standard_scaler_nf.joblib`) is fitted strictly on the **training partition** of CSE-CIC-IDS2018 (seed 42).
   - Test partitions and external OOD datasets (ToN-IoT, BoT-IoT) are strictly transformed using the frozen training scaler parameters without data leakage.
3. **Partitioning:**
   - 80% Stratified Training split.
   - 20% Held-Out Testing split ($N = 138,069$ samples, preserved in `experiments/X_test_nf.npy` and `experiments/y_test_nf.npy`).
   - 10,000 held-out benign samples reserved for empirical percentile threshold calibration at 5% target FPR.

---

## 6. Pre-Packaged Test Partitions

For rapid verification and reproduction without downloading the 50+ GB raw dataset collections, the exact preprocessed test matrices are provided directly in `experiments/`:

- `experiments/X_test_nf.npy` (138,069 $\times$ 13 float32 matrix, SHA-256: `8cde16a35e...`)
- `experiments/y_test_nf.npy` (138,069 integer labels, SHA-256: `dc3c050542...`)

Verify their cryptographic hashes at any time:
```bash
python scripts/download_artifacts.py --verify-checksums
```
