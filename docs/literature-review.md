# Literature Review: ONNX-Based Semantic Security Engine

**Student:** Muhammad Sikandar Hussain
**Updated:** 12-June-2026

---

## Instructions

For each paper or resource you read, complete one entry below.
Aim for at least **10 papers** by the end of Week 2.
Use Google Scholar, IEEE Xplore, ACM DL, arXiv, or USENIX Security.

---

## Paper Summary Template

### Paper 1 — Kitsune: Ensemble Autoencoders for Network Intrusion Detection

| Field | Content |
|---|---|
| **Full title** | Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection |
| **Authors** | Yisroel Mirsky, Tomer Doitshman, Yuval Elovici, Asaf Shabtai |
| **Year** | 2018 |
| **Venue** | NDSS Symposium 2018 |
| **URL / DOI** | https://arxiv.org/abs/1802.09089 |
| **Method** | Ensemble of lightweight autoencoders trained on normal traffic; anomaly score triggers alert when reconstruction error exceeds threshold |
| **Dataset** | 9 real network attack scenarios captured on a live network |
| **Key result** | Achieves high detection accuracy on edge-like hardware with low memory footprint; runs in real-time on a Raspberry Pi-class device |
| **Limitation** | Not ONNX-based; no standardized export pipeline; research prototype only; no REST API for integration |
| **Relevance to our project** | Closest prior work combining semantic detection with edge deployment; useful design reference and baseline to beat |

**Notes / Quotes:**
> Uses unsupervised learning — no labeled attack data needed during training.
> Anomaly detection rather than classification — does not label the attack type.
> Our project improves on this by: using ONNX for portability, supervised
> classification for labeled threat types, and a FastAPI interface for
> real-world integration.

---

### Paper 2 — Interoperability in Deep Learning: Failure Analysis of ONNX Converters

| Field | Content |
|---|---|
| **Full title** | Interoperability in Deep Learning: A User Survey and Failure Analysis of ONNX Model Converters |
| **Authors** | Purvish Jajal, Wenxin Jiang, Arav Tewari, Erik Kocinare, Joseph Woo, Anusha Sarraf, Yung-Hsiang Lu, George K. Thiruvathukal, James C. Davis |
| **Year** | 2024 |
| **Venue** | ISSTA 2024 — ACM SIGSOFT International Symposium on Software Testing and Analysis |
| **URL / DOI** | https://arxiv.org/abs/2303.17708 |
| **Method** | Survey of 92 software engineers + analysis of 200 reported issues in PyTorch and TensorFlow ONNX converters |
| **Dataset** | 200 GitHub issues from ONNX converter repositories |
| **Key result** | Node conversion stage accounts for ~75% of defects; 33% of failures produce semantically incorrect models that pass silently |
| **Limitation** | Focuses on failure analysis only — does not propose fixes or improved converter designs |
| **Relevance to our project** | Critical reading before building the ONNX export pipeline — tells us exactly where PyTorch-to-ONNX conversion breaks and what to validate |

**Notes / Quotes:**
> "33% of reported failures are related to semantically incorrect models"
> — means the exported model silently produces wrong outputs without error.
> Lesson: always validate ONNX output against original PyTorch output after export.
> Use: torch.onnx.export() + onnxruntime inference comparison on same input.

---

### Paper 3 — Performance Characterization of Quantization for DNN Inference on Edge Devices

| Field | Content |
|---|---|
| **Full title** | Performance Characterization of using Quantization for DNN Inference on Edge Devices: Extended Version |
| **Authors** | Arxiv 2303.05016 |
| **Year** | 2023 |
| **Venue** | arXiv (extended version) |
| **URL / DOI** | https://arxiv.org/pdf/2303.05016 |
| **Method** | Benchmarks INT8 and FP16 quantization using ONNX, OpenVINO, TFLite, and PyTorch on Intel x86 (Cascade Lake, Skylake) and Raspberry Pi 4B (ARM) |
| **Dataset** | MobileNetV2, VGG-19, DenseNet-121 as benchmark models |
| **Key result** | INT8 quantization reduces model size by 4× with negligible accuracy loss; static quantization slightly degrades accuracy vs dynamic |
| **Limitation** | Benchmarks image classification models only — not threat classification models; results may differ for tabular/network data |
| **Relevance to our project** | Directly validates our quantization strategy for Raspberry Pi deployment; gives expected numbers for size reduction and speedup |

**Notes / Quotes:**
> Static quantization = better speed, requires calibration dataset.
> Dynamic quantization = slightly slower, no calibration needed.
> For our project: use static quantization since we have labeled training data
> available for calibration.

---

### Paper 4 — EdgeMLOps: ONNX Quantization on Raspberry Pi 4

| Field | Content |
|---|---|
| **Full title** | EdgeMLOps: Operationalizing ML Models with Cumulocity IoT and thin-edge.io for Visual Quality Inspection |
| **Authors** | arXiv 2501.17062 |
| **Year** | 2025 |
| **Venue** | arXiv |
| **URL / DOI** | https://arxiv.org/pdf/2501.17062 |
| **Method** | Deploys ONNX-quantized models on Raspberry Pi 4 (4GB); compares FP32 vs Signed-INT8-Static vs Signed-INT8-Dynamic inference time |
| **Dataset** | Visual quality inspection dataset (IoT edge deployment) |
| **Key result** | ONNX INT8 quantization achieves 2× speedup and 4× size reduction on Raspberry Pi 4 with minimal accuracy loss |
| **Limitation** | Application domain is visual inspection, not network security; edge hardware is Raspberry Pi 4 specifically |
| **Relevance to our project** | Provides concrete benchmark numbers for ONNX quantization on exactly the class of hardware we are targeting |

**Notes / Quotes:**
> "Straightforward quantization with ONNX leads to a two-time speed improvement
> on a Raspberry Pi 4."
> This is our target baseline: 2× faster, 4× smaller after quantization.
> We should reproduce similar numbers on our threat classifier and report them
> in our evaluation.

---

### Paper 5 — Edge AI Survey: Neural Networks on Embedded Systems

| Field | Content |
|---|---|
| **Full title** | Edge AI in Practice: A Survey and Deployment Framework for Neural Networks on Embedded Systems |
| **Authors** | MDPI Electronics |
| **Year** | 2025 |
| **Venue** | MDPI Electronics, Vol. 14, No. 24 |
| **URL / DOI** | https://www.mdpi.com/2079-9292/14/24/4877 |
| **Method** | Systematic literature review (PRISMA) of deep learning deployment on embedded hardware; covers pruning, quantization, lightweight architectures, hardware platforms, and software frameworks |
| **Dataset** | Survey of existing literature — no single dataset |
| **Key result** | ONNX Runtime identified as the most robust solution for cross-framework interoperability on embedded systems; quantization and pruning are the most effective compression techniques |
| **Limitation** | Survey paper — no novel experiments; findings are aggregated from existing work |
| **Relevance to our project** | Best single reference for understanding the full landscape of edge AI deployment; supports our architectural choices |

**Notes / Quotes:**
> "ONNX Runtime is a robust solution that operates on the standard ONNX format
> ensuring interoperability between different frameworks and hardware."
> Useful citation to justify our choice of ONNX Runtime over TF Lite or PyTorch Mobile
> in the architecture section of our project report.

---

## Reference Table (Quick Overview)

| # | Title (short) | Authors | Year | Method | Dataset | Relevance |
|---|---|---|---|---|---|---|
| 1 | Kitsune | Mirsky et al. | 2018 | Autoencoder ensemble | Live network captures | Closest prior work; design baseline |
| 2 | ONNX Converter Failures | Jajal et al. | 2024 | Issue analysis + survey | 200 GitHub issues | Export pipeline risk awareness |
| 3 | Quantization on Edge Devices | arXiv 2303.05016 | 2023 | INT8/FP16 benchmarks | MobileNetV2, VGG-19 | Quantization strategy validation |
| 4 | EdgeMLOps / Raspberry Pi | arXiv 2501.17062 | 2025 | ONNX on RPi 4 | IoT edge deployment | Concrete edge benchmark numbers |
| 5 | Edge AI Survey | MDPI Electronics | 2025 | PRISMA literature review | Survey | Full edge deployment landscape |


---

## Tools and Datasets Identified

| Name | Type | URL | Notes |
|---|---|---|---|
| ToN-IoT | Dataset | https://research.unsw.edu.au/projects/toniot-datasets | IoT + logs; multi-modal; Week 2 priority |
| CIC-IoT2023 | Dataset | https://www.unb.ca/cic/datasets/iotdataset-2023.html | IoT edge deployment testing |
| CSE-CIC-IDS2018 | Dataset | https://www.unb.ca/cic/datasets/ids-2018.html | for training |
| ONNX Runtime | Library / Tool | https://onnxruntime.ai | Core inference engine for this project |
| CICFlowMeter | Library / Tool | https://github.com/ahlashkari/CICFlowMeter | Feature extraction from raw network traffic |
| Zeek | Library / Tool | https://zeek.org | Network monitor; potential data source |
| Suricata | Library / Tool | https://suricata.io | Traditional IDS; baseline comparison |
| Intel OpenVINO | Library / Tool | https://github.com/openvinotoolkit/openvino | Competing edge inference runtime |

---

## Related Tools and Competitors

A brief landscape analysis of existing tools related to this project.
---

### 1. Zeek (formerly Bro)
**Type:** Open-Source Tool  
**Link:** https://zeek.org  

**Summary:**  
One of the most widely deployed open-source network security monitors.
Performs deep packet inspection, protocol analysis, and log generation.
Entirely rule and signature based — has no ML or semantic classification
component. Too heavy for constrained edge devices.

**Gap:** No semantic/ML layer, no ONNX pipeline, not edge-optimized.

---

### 2. Suricata
**Type:** Open-Source IDS/IPS  
**Link:** https://suricata.io  

**Summary:**  
High-performance open-source intrusion detection and prevention system.
Uses signature matching and protocol detection for real-time traffic
analysis. Industry standard for network-level threat detection but
relies entirely on manually written rules. Cannot generalize to
novel or obfuscated attacks.

**Gap:** No ML, no semantic understanding, not designed for edge deployment.

---

### 3. Kitsune (2018)
**Type:** Research Prototype  
**Link:** https://arxiv.org/abs/1802.09089  

**Summary:**  
Lightweight anomaly detector designed for edge-like hardware. Uses an
ensemble of autoencoders trained on normal traffic behavior to flag
anomalies semantically. The closest existing work to this project in
spirit — combines semantic detection with low-resource deployment.

**Gap:** Not ONNX-based, no standardized export pipeline, no REST API,
research prototype only with no production deployment path.

---

### 4. Intel OpenVINO
**Type:** Open-Source Edge Inference Toolkit  
**Link:** https://github.com/openvinotoolkit/openvino  

**Summary:**  
Intel's production-grade edge AI inference toolkit. Supports models
from PyTorch, TensorFlow, and ONNX. Offers strong optimization for
Intel CPUs and VPUs. General-purpose inference engine — not
security-specific. Locked to Intel hardware ecosystem.

**Gap:** Not security-specific, Intel hardware only, no threat
classification pipeline included.

---

### 5. TensorFlow Lite
**Type:** Framework  
**Link:** https://www.tensorflow.org/lite  

**Summary:**  
Google's framework for running compressed ML models on mobile and
embedded devices. Widely used for TinyML applications. Uses its own
.tflite format rather than ONNX, making it less framework-agnostic.
Not security-specific.

**Gap:** Different model format (not ONNX), not security-specific,
less flexible than ONNX Runtime for multi-framework workflows.

---

## Competitive Gap Summary

| Tool | Semantic/ML | ONNX | Edge Optimized | Security Specific | API Served |
|---|---|---|---|---|---|
| **This Project** | ✅ | ✅ | ✅ | ✅ | ✅ FastAPI |
| Zeek | ❌ | ❌ | Partial | ✅ | ❌ |
| Suricata | ❌ | ❌ | Partial | ✅ | ❌ |
| Kitsune | ✅ | ❌ | ✅ | ✅ | ❌ |
| OpenVINO | ✅ | ✅ | ✅ | ❌ | ❌ |
| TF Lite | ✅ | ❌ | ✅ | ❌ | ❌ |

**Conclusion:** No existing tool combines all five properties. This project
fills a clear gap — a semantic, ONNX-based, edge-optimized, security-specific
engine with a deployable API interface.
---

## Core Datasets for This Project

The following three datasets form the complete experimental pipeline:
**CSE-CIC-IDS2018** for training, 
**ToN-IoT** for semantic generalizationtesting, 
**IC-IoT2023** for final edge deployment evaluation.

---

### Dataset A — CSE-CIC-IDS2018
**Role in this project: Primary Training Dataset**

| Field | Details |
|---|---|
| **Full Name** | A Realistic Cyber Infrastructure Dataset — CSE-CIC-IDS2018 |
| **Source** | Communications Security Establishment (CSE) + Canadian Institute for Cybersecurity (CIC) |
| **Year** | 2018 |
| **URL** | https://www.unb.ca/cic/datasets/ids-2018.html |
| **Kaggle Mirror** | https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv |
| **Raw Size** | ~450 GB (raw pcap); ~220 GB packed |
| **Records** | ~16.8 million labeled flow records |
| **Features** | 80 features extracted via CICFlowMeter-V3 |
| **Files** | 10 CSV files — one per capture day (Feb 14 – Mar 2, 2018) |
| **Format** | CSV |
| **Label Column** | `Label` |

**Attack Categories and Record Counts:**

| Attack Type | Records | % of Total |
|---|---|---|
| Benign | 14,097,779 | 83.68% |
| DDoS attack-HOIC | 686,012 | 4.07% |
| DDoS attack-LOIC-HTTP | 576,191 | 3.42% |
| DoS attacks-Hulk | 461,912 | 2.74% |
| Bot | 286,191 | 1.70% |
| FTP-BruteForce | 193,360 | 1.15% |
| SSH-Bruteforce | 187,589 | 1.11% |
| Infiltration | 161,934 | 0.96% |
| DoS attacks-GoldenEye | 41,508 | 0.25% |
| DoS attacks-SlowHTTPTest | 139,890 | 0.83% |
| DoS attack-Slowloris | 10,990 | 0.07% |
| Brute Force-Web | 611 | 0.004% |
| Brute Force-XSS | 230 | 0.001% |
| SQL Injection | 87 | 0.0005% |
| DDoS attack-LOIC-UDP | 1,730 | 0.01% |

**Infrastructure:**
- 450 victim machines across 6 victim networks, with 50 attacking machines — significantly larger scale than CICIDS2017 which had only 14 victim machines 
- 10 days of traffic from February 14 to March 2, 2018, hosted on Amazon Web Services (AWS) 

**Key Features (sample):**

| Feature | Description |
|---|---|
| `fl_dur` | Flow duration |
| `tot_fw_pk` | Total packets in forward direction |
| `tot_bw_pk` | Total packets in backward direction |
| `fw_pkt_l_avg` | Average forward packet length |
| `bw_pkt_l_avg` | Average backward packet length |
| `fl_byt_s` | Flow bytes per second |
| `fl_pkt_s` | Flow packets per second |
| `fw_iat_avg` | Forward inter-arrival time average |
| `Dst Port` | Destination port |
| `Protocol` | Transport protocol |

**Strengths:**
- Most comprehensive modern IDS benchmark available
- Regarded as one of the most reliable sources for analyzing network anomaly-based intrusion detection methods 
- Real network infrastructure — not simulated
- Covers all major modern attack categories including Botnet, Heartbleed, DDoS variants
- Same 80-feature CICFlowMeter pipeline — directly comparable to CICIDS2017

**Weaknesses:**
- Exhibits severely imbalanced classes — rare attack types such as Brute Force-XSS and SQL Injection occur far less frequently than others, representing only 0.009% and 0.006% of total samples respectively 
- Requires SMOTE or class weighting to handle minority classes
- Large size requires significant storage and compute

### Dataset B — ToN-IoT

**Role in this project: Semantic Generalization Testing**

| Field | Details |
|---|---|
| **Full Name** | TON_IoT: Telemetry Dataset of IoT and IIoT for Data-Driven Intrusion Detection Systems |
| **Source** | Cyber Range and IoT Labs, UNSW Canberra, Australia |
| **Year** | 2020 |
| **URL** | https://research.unsw.edu.au/projects/toniot-datasets |
| **Paper** | IEEE Access, Vol. 8, 2020 — DOI: 10.1109/ACCESS.2020.3022862 |
| **Total Records** | ~22 million flow records (network subset) |
| **Features** | Network: 44–45 features; OS logs: 54 features; Telemetry: device-specific |
| **Format** | CSV + PCAP + TXT log files |

**What makes ToN-IoT unique — three data sources in one:**

| Data Source | Description | Records |
|---|---|---|
| **Network traffic** | IoT network flow data extracted via Zeek/Bro | ~22M |
| **OS logs (Windows)** | Windows 7/10 audit traces — Performance Monitor | ~1M |
| **OS logs (Linux)** | Ubuntu 14/18 system traces — atop tool | ~1M |
| **IoT Telemetry** | 10+ IoT/IIoT sensors — weather, Modbus, etc. | Varies |

**Attack Categories:**

| Attack Type | Network Records | Notes |
|---|---|---|
| Benign | 796,380 (3.56%) | Severely underrepresented |
| Scanning | 7,140,161 | Port/network reconnaissance |
| DDoS | 6,165,008 | Distributed flood attacks |
| DoS | 3,375,328 | Single-source floods |
| XSS | 2,108,944 | Cross-site scripting |
| Password | 1,365,958 | Brute-force credential attacks |
| Backdoor | 508,116 | Persistent access malware |
| Injection | 452,659 | SQL/command injection |
| Ransomware | 72,805 | File encryption attacks |
| MITM | 1,052 | Man-in-the-middle |

**Infrastructure:**
- Collected from a realistic large-scale network at the Cyber Range and IoT Labs at UNSW Canberra, deploying multiple virtual machines and hosts running Windows, Linux, and Kali operating systems across IoT, Cloud, and Edge/Fog layers 
- Includes heterogeneous data sources — a key advantage currently lacking in state-of-the-art datasets 

**Strengths:**
- **Multi-modal** — only major IDS dataset combining network traffic + OS logs + IoT telemetry
- Covers nine attack categories including scanning, DoS, DDoS, ransomware, backdoor, injection, XSS, password cracking, and Man-in-the-Middle 
- Includes Ransomware and MITM — not present in CICIDS2017 or CSE-CIC-IDS2018
- Designed specifically for AI/ML-based security evaluation
- Multiple versions available (full, reduced, NF-standardized)

**Weaknesses:**
- Severely imbalanced — 96.44% attack samples vs only 3.56% benign — opposite imbalance to CSE-CIC-IDS2018 
- Multiple separate files with different schemas require careful joining
- MITM class extremely small (1,052 records) — may need oversampling

**Why this tests semantic generalization:**
ToN-IoT uses a completely different feature extraction pipeline (Zeek/Bro
instead of CICFlowMeter), different feature names, different value ranges,
and a different attack taxonomy. If our model — trained on CSE-CIC-IDS2018
— can still correctly classify threats from ToN-IoT after feature alignment,
it demonstrates true semantic understanding of attack behavior rather than
memorization of dataset-specific patterns.

---

### Dataset C — CIC-IoT2023
**Role in this project: Final Edge Deployment Evaluation**

| Field | Details |
|---|---|
| **Full Name** | CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment |
| **Source** | Canadian Institute for Cybersecurity (CIC), University of New Brunswick |
| **Year** | 2023 |
| **URL** | https://www.unb.ca/cic/datasets/iotdataset-2023.html |
| **Paper** | MDPI Sensors, Vol. 23, Issue 13, 2023 — DOI: 10.3390/s23135941 |
| **Raw Size** | ~548 GB (raw pcap); 12.8 GB processed CSV |
| **Records** | 46,686,579 labeled flow records |
| **Features** | 47 features per record |
| **Files** | 169 files (PCAP + CSV format) |
| **Format** | CSV (processed) + PCAP (raw) |
| **Label Column** | `label` |

**Attack Categories — 33 attacks across 7 classes:**

| Class | Attack Types | Examples |
|---|---|---|
| **DDoS** | 10 sub-types | UDP Flood, TCP Flood, ICMP Flood, HTTP Flood, SSDP Amplification |
| **DoS** | 6 sub-types | SYN Flood, ACK Flood, HTTP Slow, UDP Flood |
| **Recon** | 4 sub-types | Port Scan, Ping Sweep, OS Fingerprinting, Service Discovery |
| **Web-based** | 4 sub-types | SQL Injection, XSS, Command Injection, Uploading |
| **Brute Force** | 3 sub-types | Dictionary, SSH Brute Force, FTP Brute Force |
| **Spoofing** | 2 sub-types | ARP Spoofing, DNS Spoofing |
| **Mirai** | 4 sub-types | Mirai variants targeting IoT devices |
| **Benign** | — | Normal IoT device traffic |

**Device Infrastructure:**
- Generated using a large-scale IoT topology comprising 105 devices, accurately reflecting the interconnected and dynamic nature of modern IoT environments — 67 IoT devices actively involved in attacks (63 victims and 4 attackers) and 38 Zigbee and Z-Wave devices connected to 5 hubs 
- Device types include smart home components, cameras, sensors, and microcontrollers — capturing 33 distinct attack types with approximately 548 GB of raw traffic data 

**Key Features (sample):**

| Feature | Description |
|---|---|
| `flow_duration` | Duration of the network flow |
| `Header_Length` | Packet header size |
| `Protocol Type` | Transport protocol (TCP/UDP/ICMP) |
| `Duration` | Connection duration |
| `Rate` | Packet rate |
| `Srate` | Source packet rate |
| `Drate` | Destination packet rate |
| `fin_flag_number` | FIN flag count |
| `syn_flag_number` | SYN flag count |
| `rst_flag_number` | RST flag count |
| `Packets` | Total packet count |
| `Bytes` | Total byte count |
| `Tot sum` | Sum of all packet sizes |

**Strengths:**
- Most recent and most realistic IoT IDS dataset available (2023)
- Addresses key challenges in IoT security research including class imbalance, protocol diversity, and scalability — suitable for machine learning, deep learning, and explainable AI studies 
- All attacks executed by real malicious IoT devices targeting other IoT devices — not simulated 
- Includes Mirai botnet variants — the most prevalent real-world IoT threat
- Covers Zigbee and Z-Wave protocols not present in other datasets

**Weaknesses:**
- Very large raw size (~548 GB) — use CSV subset for development
- Relatively new — fewer published baselines for direct comparison
- 47 features (different schema from CSE-CIC-IDS2018 and ToN-IoT)

**Why this is the final evaluation dataset:**
This dataset was generated from real IoT hardware in a topology that
directly mirrors the environment where our edge security engine will be
deployed. Cameras, sensors, microcontrollers, Zigbee devices — these
are exactly the devices our engine protects. Evaluating on CIC-IoT2023
gives the most realistic measure of how our engine performs in production.


---

