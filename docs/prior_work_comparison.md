# SEMANTICSHIELD: Prior-Work Comparison & Literature Gap Analysis

**Document Status:** Publication-Ready Theoretical & Architectural Framing  
**Target Venue:** IEEE Transactions on Dependable and Secure Computing (TDSC) / ACM TOPS  
**Related Documents:** [`docs/semantic_feature_audit.md`](file:///d:/Internship/08-onnx-semantic-security-engine/docs/semantic_feature_audit.md), [`docs/reference_audit_report.md`](file:///d:/Internship/08-onnx-semantic-security-engine/docs/reference_audit_report.md)

---

## 1. Executive Summary

Deploying deep-learning-based Network Intrusion Detection Systems (NIDS) to edge gateways requires bridging four distinct scientific domains that have largely evolved in isolation:
1. **Out-of-Distribution (OOD) Detection & Statistical Rigor:** Distinguishing known malicious behaviors from unfamiliar network protocols or corrupted telemetry under strict operational false positive limits (Axelsson 1999, Sommer & Paxson 2010).
2. **Network Flow Telemetry & Standardization:** Reconciling differences between packet-level extractors (e.g., CICFlowMeter) and router-level collectors (e.g., NetFlow v9/IPFIX via nProbe) governed by RFC 7012 / RFC 793.
3. **Cross-Dataset Generalization:** Mitigating catastrophic performance drops caused by syntactic vs. physical feature mismatch rather than just statistical distribution shift.
4. **Edge Compression & Runtime Interoperability:** Evaluating model quantization (FP16, INT8, INT4) and ONNX graph lowering under verified resource-constrained edge execution profiles.

This document establishes the publication-ready gap matrix comparing SEMANTICSHIELD against representative state-of-the-art systems and structures the Related Work into eight thematic pillars.

---

## 2. Systematic Prior-Work Comparison Matrix

The table below contrasts SEMANTICSHIELD with key prior art across eight critical dimensions:
- **Deployment Domain:** Operational target (Edge, Datacenter, SDN, or Offline Benchmark).
- **Flow Ingestion Format:** Input telemetry structure (Raw packets, tabular flow, switch digests).
- **Feature Reconciliation Method:** Methodology used to harmonize cross-dataset or cross-tool telemetry schemas.
- **Fixed-FPR Evaluation:** Whether evaluation enforces realistic operational false-positive-rate caps ($FPR \le 0.1\%$ or $1.0\%$) to account for the base-rate fallacy.
- **Quantization Investigated:** Precision formats analyzed (FP32, FP16, INT8, INT4) and clipping trade-offs.
- **Edge Simulation / Hardware:** Evaluation environment (Physical hardware, containerized resource quotas, or unconstrained workstations).
- **Root-Cause Error Analysis:** Granular diagnostics for failure modes (per-field physical divergence or per-tensor quantization clipping).

| Paper / System | Deployment Domain | Flow Ingestion Format | Feature Reconciliation Method | Fixed-FPR Evaluation | Quantization Investigated | Edge Simulation / Hardware | Root-Cause Error Analysis |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|
| **Kitsune**<br>*(Mirsky et al., NDSS 2018)* | Edge / IoT Gateway | Raw Packet / Incremental Stats | None (single packet stream) | ❌ No | ❌ No | Physical Hardware (RPi 3B) | ❌ No (anomaly score only) |
| **McLaughlin et al.**<br>*(arXiv 2023)* | Datacenter / Server | Tabular Flow (CIC/NetFlow) | None (uniform dataset schemas) | ⚠️ Partial (FPR@95 only) | ❌ No | ❌ None (GPU Server) | ⚠️ Partial (OOD method sensitivity) |
| **NetSight**<br>*(Handigol / Pratt et al.)* | SDN / Switch Telemetry | Packet History Digests | None (homogeneous switch logs) | ❌ No | ❌ No | Hardware Testbed (Switches) | ❌ No (path reconstruction only) |
| **Sarhan et al.**<br>*(MoNet 2022)* | Offline Benchmark | NetFlow v9 (nProbe) | Schema Projection (12/43 NF) | ❌ No | ❌ No | ❌ None (Offline Server) | ❌ No (macro dataset metrics) |
| **Cantone et al.**<br>*(Computers & Security 2024)* | Offline Benchmark | CICFlowMeter (CIC/LycoS) | Manual Column Name Subset | ❌ No | ❌ No | ❌ None (Offline Server) | ⚠️ Partial (artifact memorization) |
| **Jajal et al.**<br>*(ACM ISSTA 2024)* | General DL Deployment | N/A (General DNN Graphs) | None (operator translation) | ❌ No | ⚠️ Partial (quantization node bugs) | ❌ None (Host Workstation) | ✅ Yes (graph/operator defects) |
| **Yang et al.**<br>*(IEEE IoT-J 2022)* | IoV / Vehicle Edge | CICFlowMeter / UNSW | Per-Dataset Feature Selection | ❌ No | ❌ No | Edge Node Emulation | ❌ No (feature importance only) |
| **SEMANTICSHIELD**<br>*(This Work)* | **Edge Gateway / Embedded NIDS** | **Hybrid (NetFlow v9 $\leftrightarrow$ CICFlowMeter)** | **Physical Semantic Audit (4 Tiers; $\kappa=1.0$)** | **✅ Yes ($FPR \le 0.1\% / 1.0\%$)** | **✅ Yes (FP32/FP16/INT8/INT4)** | **✅ Yes (Simulated R0–R3 Profiles)** | **✅ Yes (Per-Field & Tensor Outliers)** |

---

## 3. The Eight Thematic Pillars of Related Work

### Pillar 1: Out-of-Distribution & Anomaly Detection Foundations in Machine Learning
- **Foundational Works:** Hendrycks & Gimpel (ICLR 2017) introduced Maximum Softmax Probability (MSP) as an OOD baseline. Lee et al. (NeurIPS 2018) formulated class-conditional Mahalanobis distance in hidden representations. Liu et al. (NeurIPS 2020) demonstrated that unnormalized Energy scores alleviate softmax overconfidence. Liang et al. (ICLR 2018) proposed ODIN (temperature scaling and input perturbation), while Sun et al. (NeurIPS 2021) introduced ReAct (rectified activation clipping) to suppress spurious activations. Sastry & Oore (ICML 2020) proposed Gram matrices to model cross-layer correlations.
- **Literature Gap:** These techniques were developed almost exclusively for vision and natural language processing tasks. When applied directly to tabular network flow telemetry, standard assumptions (such as isotropic Gaussian feature clusters or smooth input perturbations) break down due to non-smooth distributions, heavy-tailed counters, and multi-modal feature boundaries.

### Pillar 2: Deep Learning for Network Intrusion Detection Systems
- **Foundational Works:** Ahmad et al. (2021) surveyed over a decade of ML/DL approaches, documenting the shift from shallow classifiers (Random Forests, SVMs) to deep multi-layer perceptrons (MLPs), CNNs, and recurrent models. Bouidaine et al. (ETASR 2025) achieved 99.91% accuracy on CSE-CIC-IDS2018 through extensive preprocessing, class aggregation, and L2 regularization. Ferrag et al. (IEEE Access 2023) developed Edge-IIoTset to benchmark centralized and federated learning on edge IoT topologies. Yang et al. (IEEE IoT-J 2022) proposed MTH-IDS, combining tree-based feature selection with multi-tiered classifiers for connected vehicles.
- **Literature Gap:** Prior deep NIDS studies focus almost exclusively on in-distribution multi-class classification accuracy. They assume the test distribution exactly mirrors the training distribution and fail silently when exposed to unseen attack families or shifted telemetry.

### Pillar 3: Base-Rate Fallacy and Operational Constraints in High-Volume NIDS
- **Foundational Works:** In a seminal paper, Axelsson (ACM CCS 1999) proved that due to the base-rate fallacy, an intrusion detection system processing millions of events per second with an FPR of even 1% will produce an overwhelming flood of false alarms, rendering the system operationally unusable. Sommer & Paxson (IEEE S&P 2010) formalized the operational challenges of using ML in intrusion detection, noting that "the closed world" assumption of academic datasets produces misleadingly optimistic results.
- **Literature Gap:** Contemporary NIDS literature continues to rely heavily on threshold-free metrics (e.g., standard AUROC) where high true positive rates are achieved at catastrophic false positive rates (e.g., $FPR > 10\%$). Fixed-FPR thresholding ($FPR \le 0.1\%$ or $1.0\%$) and AUPRC under extreme class imbalance are rarely enforced as primary evaluation criteria.

### Pillar 4: Telemetry Standards, Packet Aggregation, and Flow Collectors
- **Foundational Works:** Hofstede et al. (IEEE Surveys 2014) surveyed flow monitoring architectures from raw packet capture to collector analysis. Claise & Trammell (RFC 7012) standardized the IP Flow Information Export (IPFIX) Information Model. Postel (RFC 793) established the Transmission Control Protocol state machine and control bit specifications. Handigol et al. (USENIX NSDI 2014) developed NetSight, demonstrating packet history digests for low-overhead network troubleshooting.
- **Literature Gap:** While network engineers adhere strictly to RFC specifications, ML researchers frequently treat flow exporter output as generic numeric tables. Differences between IPFIX Information Elements and ad-hoc software tools (like CICFlowMeter) are routinely ignored during dataset construction.

### Pillar 5: Cross-Dataset Generalization Collapse and Feature Heterogeneity
- **Foundational Works:** Pontes et al. (Computers & Security 2021) observed cross-dataset sensitivity in flow-based NIDS. Sarhan et al. (MoNet 2022) addressed dataset heterogeneity by extracting a common NetFlow v9 schema across four datasets (UNSW-NB15, BoT-IoT, ToN-IoT, CSE-CIC-IDS2018) using nProbe. Cantone et al. (Computers & Security 2024) tested cross-dataset generalization across CIC-IDS2017, CSE-CIC-IDS2018, and LycoS-IDS, demonstrating that cross-dataset accuracy drops to near-random chance.
- **Literature Gap:** Existing studies attribute generalization collapse solely to statistical distribution shift or dataset artifact memorization. None have performed a field-by-field physical quantity and unit audit to distinguish true statistical domain shift from syntactic/mathematical feature incompatibility.

### Pillar 6: Edge Machine Learning and Model Quantization on Embedded Gateways
- **Foundational Works:** Ahn et al. (arXiv 2023) characterized DNN quantization across edge hardware (x86 and Raspberry Pi 4B) using MLPerf, showing that INT8 reduces model sizes by $4\times$ with minimal degradation in CNN vision models. Liang et al. (Neurocomputing 2024) surveyed neural network pruning and quantization. Chaturvedi et al. (BTW 2025) operationalized INT8 ONNX models on Raspberry Pi 4 gateways. Nagel et al. (2021) documented white-paper best practices for post-training quantization. Lin et al. (NeurIPS 2024) demonstrated on-device training under 256 KB memory limits.
- **Literature Gap:** Quantization research focuses almost entirely on computer vision and speech architectures, where activation tensors are bounded and smoothly distributed. Tabular network flow models contain extreme feature dynamic ranges ($0$ to $>10^9$), causing standard symmetric INT8 quantization to suffer severe clipping distortion and catastrophic accuracy drops unless explicitly investigated and calibrated.

### Pillar 7: Interoperability and Runtime Failure Analysis in ONNX Deployments
- **Foundational Works:** Jajal et al. (ACM ISSTA 2024) surveyed 92 practitioners and analyzed 200 reported issues in PyTorch and TensorFlow ONNX converters, finding that 75% of defects occur during node conversion and 33% result in semantically incorrect models that pass silently without throwing runtime exceptions. Microsoft (2024) released ONNX Runtime as a cross-platform execution engine.
- **Literature Gap:** Despite the rapid adoption of ONNX Runtime on edge appliances, security literature lacks systematic testing protocols that verify mathematical equivalence between PyTorch training graphs and ONNX edge execution graphs across both normal and quantized precisions.

### Pillar 8: Realistic Adversarial Robustness and Concept Drift in Network Telemetry
- **Foundational Works:** Apruzzese et al. (ACM DTRAP 2023) established modeling frameworks for realistic adversarial evasion attacks against NIDS, demonstrating that unconstrained gradient-based perturbations produce invalid network packets. Aceto et al. (IEEE TNSM 2024) evaluated concept drift detection in network traffic classification. Depren & Ozdemir (IEEE S&P 2024) proposed MLOps lifecycles for automating NIDS retraining. The MITRE Corporation (2024) formalized adversary techniques in the MITRE ATT&CK knowledge base.
- **Literature Gap:** Adversarial defenses and drift detectors typically introduce high computational complexity that cannot execute within the tight latency envelopes ($\le 2$ ms) and memory budgets ($\le 512$ MB) of real-world edge IoT gateways.

---

## 4. Distinct Positioning of SEMANTICSHIELD

SEMANTICSHIELD directly addresses the intersections of these eight pillars:
1. **Physical Reconciliation Over Blind Alignment:** Rather than blindly intersecting feature names or projecting onto coarse schemas, SEMANTICSHIELD provides a source-verified physical audit ($N=21$, $\kappa=1.0$), identifying the exact 8 incompatible features responsible for negative transfer.
2. **Base-Rate Calibrated Operational Rigor:** Evaluates detection under strict operational envelopes ($FPR \le 0.1\%$ and $FPR \le 1.0\%$) and reports AUPRC alongside AUROC across multiple random seeds with 95% confidence intervals.
3. **Hardware-Aware Edge Precision Profiling:** Replaces theoretical hardware claims with controlled, containerized resource-constrained profiling (R0–R3 profiles: 1–4 vCPUs, 512 MB–2 GB RAM), uncovering the exact activation outlier mechanism that degrades standard INT8 quantization on tabular network telemetry.
