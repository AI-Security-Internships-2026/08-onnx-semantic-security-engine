# Research Proposal: ONNX-Based Semantic Security Engine for Edge Inference

**Student:** _[Fill in your name]_
**Supervisor:** _[Fill in supervisor name]_
**Start date:** _[Fill in]_
**Expected end date:** _[Fill in]_

---

## 1. Background

Export trained threat-classification models to ONNX format and build a lightweight semantic security engine that runs efficiently on edge hardware with minimal latency and memory footprint.

This project is carried out within the AI Security research agenda of CNIT/PNTLab Pisa (TECIP, Scuola Superiore Sant'Anna).

---

## 2. Problem Statement

_Describe in 3–5 sentences the specific gap or challenge this project addresses.
Be precise: what is broken, missing, or insufficiently studied?_

---

## 3. Research Questions

1. _RQ1: ..._
2. _RQ2: ..._
3. _RQ3: ..._

---
RQ1: Can a threat classifier trained in PyTorch be exported to ONNX and
deployed on edge hardware without introducing semantic prediction errors
during conversion?

RQ2: What is the trade-off between INT8 quantization and classification
accuracy when deploying a semantic threat classifier on a Raspberry Pi 4,
compared to full-precision inference?

RQ3: Does a classifier trained on one network intrusion dataset
(CSE-CIC-IDS2018) generalize to a structurally different dataset (ToN-IoT)
with different feature extraction, indicating it has learned semantic
attack behavior rather than dataset-specific patterns?

## 4. Proposed Methodology

### 4.1 Data Collection / Dataset

_Describe datasets or data sources you plan to use. Cite sources and licences._

### 4.2 Approach

_High-level description of the algorithm, architecture, or system design._

### 4.3 Evaluation Metrics

_How will you measure success? (accuracy, F1, latency, recall@K, etc.)_

### 4.4 Tooling

_List tools, frameworks, and libraries you plan to use._

---
Section 4 — Proposed Methodology (draft)
4.1 Data Collection / Dataset

CSE-CIC-IDS2018 (training) — answers RQ1, RQ2
ToN-IoT (generalization test) — answers RQ3
CIC-IoT2023 (edge evaluation, optional/stretch) — supports RQ2 in a real IoT context
Cite source, license (CC for CIC datasets, research/educational use), and access method (Kaggle mirror for CIC-IDS2018)

4.2 Approach

Preprocessing: clean, scale, encode labels
Model: MLP classifier in PyTorch
Export: PyTorch → ONNX (opset 17), validated against PyTorch output on held-out samples — directly answers RQ1
Optimization: INT8 static quantization via ONNX Runtime — directly answers RQ2
Generalization test: align feature schemas between CIC-IDS2018 and ToN-IoT, evaluate trained model on ToN-IoT without retraining — directly answers RQ3

4.3 Evaluation Metrics

RQ1: % of validation samples where PyTorch and ONNX predictions match (target: 100%)
RQ2: accuracy/F1 delta between FP32 and INT8 models; inference latency (ms) and model size (MB) on Raspberry Pi 4
RQ3: F1-score on ToN-IoT test set using the CIC-IDS2018-trained model, compared to F1 on CIC-IDS2018's own test set

4.4 Tooling

PyTorch, ONNX, ONNX Runtime (with quantization module), scikit-learn, pandas, FastAPI, Raspberry Pi 4 (target hardware)


## 5. Expected Outcome

_Describe the expected deliverable: a prototype, a benchmark, a dataset, a paper draft, etc._

---

## 6. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Dataset not publicly available | Medium | Use synthetic data or reach out to CNIT partners |
| Compute resources insufficient | Low | Use university HPC cluster |
| Scope too broad | High | Focus on one sub-problem; extend if time allows |

---

_Last updated: 2026-_
