# Dataset Sheet: NF-BoT-IoT-v2 (Standardized Cross-Dataset Shift)

## 1. Overview
- **Name:** NetFlow-BoT-IoT-v2 (Standardized NetFlow representation of BoT-IoT)
- **Creators:** Sarhan et al. (University of Queensland & UNSW Canberra Cyber)
- **Source Portal:** [https://staff.itee.uq.edu.au/marius/NIDS_datasets/](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
- **Kaggle Mirror:** [https://www.kaggle.com/datasets/dhoogla/nfbotiotv2](https://www.kaggle.com/datasets/dhoogla/nfbotiotv2)
- **Citation:** Sarhan et al., "Towards a Standard Feature Set for Network Anomaly Detection," Computer Networks, 2021.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Study Role:** Same-Schema Cross-Dataset Domain Shift (Track A) & Cross-Model Replication

## 2. Telemetry & Schema Details
- **Total Records:** 30,420,086 bidirectional flow records.
- **Flow Extractor:** nProbe (NetFlow v9 / IPFIX format).
- **Format:** Apache Parquet (~508 MB compressed).
- **Features:** 43 standardized NetFlow features (schema identical to NF-ToN-IoT-v2).
- **Traffic Composition:**
  - DDoS: 14,280,259
  - DoS: 13,645,057
  - Reconnaissance: 2,363,017
  - Benign: 129,437
  - Theft: 2,316

## 3. SEMANTICSHIELD Alignment & Evaluation
- **Feature Alignment:** Same 13 standardized features mapped from the nProbe NetFlow schema.
- **Experimental Finding:** Evaluates Track A (pure domain shift when the flow extractor and schema are identical to NF-ToN-IoT-v2).
- **Assurance Detection AUROC:** 0.9980 (Mahalanobis distance on ThreatMLP), 0.9983 (Full assurance layer).
