# NF-BoT-IoT-V2 Dataset

## Overview
- **Dataset Name:** NF-BoT-IoT-v2 (NetFlow version of BoT-IoT)
- **Source:** University of New South Wales (UNSW) Canberra Cyber / Sarhan et al.
- **Source URL:** https://www.kaggle.com/datasets/dhoogla/nfbotiotv2
- **Format:** Apache Parquet
- **Flow Extractor:** nProbe (NetFlow v9 / IPFIX format)
- **Total Records:** 30,420,086 rows, 43 standardized NetFlow features
- **Role in Study:** External Standardized Test Dataset for Track A (Same-Schema Cross-Dataset Domain Shift across nProbe-generated datasets) and Cross-Model Replication

## Schema & Feature Alignment
- **Identical 43 NetFlow v2 Columns:** Perfectly identical to NF-ToN-IoT-v2, establishing a common standardized schema baseline.
- **13 Features Mapped for SEMANTICSHIELD:**
  - 8 Exact Mappings: `FLOW_DURATION_MILLISECONDS`, `IN_PKTS`, `OUT_PKTS`, `IN_BYTES`, `OUT_BYTES`, `LONGEST_FLOW_PKT`, `SHORTEST_FLOW_PKT`, `PROTOCOL`
  - 5 Approximate Mappings: `MAX_IP_PKT_LEN`, `MIN_IP_PKT_LEN`, `SRC_TO_DST_SECOND_BYTES`, `TCP_WIN_MAX_IN`, `TCP_WIN_MAX_OUT`
- **Traffic Composition:**
  - DDoS: 14,280,259
  - DoS: 13,645,057
  - Reconnaissance: 2,363,017
  - Benign: 129,437
  - Theft: 2,316
