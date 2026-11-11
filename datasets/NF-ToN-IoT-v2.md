# Dataset Sheet: NF-ToN-IoT-v2 (Out-of-Distribution Shift)

## 1. Overview
- **Name:** NetFlow-ToN-IoT-v2 (Standardized NetFlow representation of ToN-IoT)
- **Creators:** Sarhan et al. (University of Queensland & UNSW Canberra Cyber)
- **Source Portal:** [https://staff.itee.uq.edu.au/marius/NIDS_datasets/](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
- **Kaggle Mirror:** [https://www.kaggle.com/datasets/dhoogla/nftoniotv2](https://www.kaggle.com/datasets/dhoogla/nftoniotv2)
- **Citation:** Sarhan et al., "Towards a Standard Feature Set for Network Anomaly Detection," Computer Networks, 2021.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Study Role:** Out-of-Distribution (OOD) Real Covariate & Flow-Extractor Shift (Track B)

## 2. Telemetry & Schema Details
- **Total Records:** 13,135,881 bidirectional flow records.
- **Flow Extractor:** nProbe (NetFlow v9 / IPFIX format).
- **Format:** Apache Parquet (~200 MB compressed).
- **Features:** 43 standardized NetFlow features.
- **Label Schema:**
  - Binary (`Label`): 0 (Benign), 1 (Attack).
  - Multi-class (`Attack`): Scanning, DoS, DDoS, Ransomware, Backdoor, Injection, XSS, Password, MITM.

## 3. SEMANTICSHIELD Alignment & Evaluation
- **Feature Alignment:** 13 features aligned with the trained CSE-CIC-IDS2018 model:
  - 8 Exact Mappings: `FLOW_DURATION_MILLISECONDS`, `IN_PKTS`, `OUT_PKTS`, `IN_BYTES`, `OUT_BYTES`, `LONGEST_FLOW_PKT`, `SHORTEST_FLOW_PKT`, `PROTOCOL`.
  - 5 Approximate Mappings: `MAX_IP_PKT_LEN`, `MIN_IP_PKT_LEN`, `SRC_TO_DST_SECOND_BYTES`, `TCP_WIN_MAX_IN`, `TCP_WIN_MAX_OUT`.
- **Transformation:** Transformed using the frozen CSE-CIC-IDS2018 `StandardScaler` without refitting.
- **Assurance Detection AUROC:** 0.9713 (Mahalanobis distance), 0.9455 (Cosine distance), 0.8501 (MSP confidence).
