# NF-ToN-IoT-V2 Dataset

## Overview
- **Dataset Name:** NF-ToN-IoT-v2 (NetFlow version of ToN-IoT)
- **Source:** University of New South Wales (UNSW) Canberra Cyber / Sarhan et al.
- **Source URL:** https://staff.itee.uq.edu.au/marius/NIDS_datasets/
- **Format:** Apache Parquet
- **Flow Extractor:** nProbe (NetFlow v9 / IPFIX format)
- **Total Records:** 13,135,881 rows, 43 standardized NetFlow features
- **Role in Study:** External Out-Of-Distribution (OOD) Test Dataset (Domain Shift + NetFlow Extractor)

## Schema & Feature Alignment
- **Identical 43 NetFlow v2 Columns:** Fully standardized schema matching NF-BoT-IoT-v2.
- **13 Features Mapped for SEMANTICSHIELD:**
  - 8 Exact Mappings: `FLOW_DURATION_MILLISECONDS`, `IN_PKTS`, `OUT_PKTS`, `IN_BYTES`, `OUT_BYTES`, `LONGEST_FLOW_PKT`, `SHORTEST_FLOW_PKT`, `PROTOCOL`
  - 5 Approximate Mappings: `MAX_IP_PKT_LEN`, `MIN_IP_PKT_LEN`, `SRC_TO_DST_SECOND_BYTES`, `TCP_WIN_MAX_IN`, `TCP_WIN_MAX_OUT`
- **Label Hierarchy:** Binary (`Label`: 0 Benign, 1 Attack) and Multi-class (`Attack`: Scanning, DoS, DDoS, Ransomware, Backdoor, Injection, XSS, Password, MITM).
