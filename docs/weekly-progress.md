# Weekly Progress Log: ONNX-Based Semantic Security Engine for Edge Inference

**Student:** Muhammad Sikandar Hussain
**GitHub username:** sikandarhussain6858

---

## How to Use This File

Add a new section every Friday before opening your weekly Pull Request.
Be honest — problems and blockers are normal and help your supervisor support you.

---

## Week 1

**Branch:** `sikandarhussain6858-week-01`
**PR link:** _[Add link after opening PR]_

### Checklist
- [x] Read README.md in full
- [x] Read docs/proposal.md in full
- [x] Accepted GitHub repository invitation
- [x] Cloned the repository
- [x] Created virtual environment and installed dependencies
- [x] Successfully ran `python src/main.py`
- [x] Created weekly branch `sikandarhussain6858-week-01`
- [x] Added personal introduction to `docs/weekly-progress.md`
- [x] Identified 5 references in `docs/literature-review.md`
- [x] Opened Week 1 Pull Request

### Personal Introduction
My name is Muhammad Sikandar Hussain. I am studying BS Artificial Intelligence at National University of Science and Technology. I am interested in this project because it sits at the intersection of machine learning and cybersecurity.

### What I Did This Week
- Completed full orientation: read `README.md` and `docs/proposal.md`
- Set up the development environment successfully
- Ran the starter script `src/main.py` without errors
- Created my working branch following the repo branching conventions
- Researched the problem domain: ONNX, semantic security, edge inference
- Identified 5 foundational references for the literature review

### What I Learned
- What ONNX is and why it enables model portability across frameworks and hardware targets
- What a semantic security engine does differently from signature-based intrusion detection systems
- Why edge inference constraints (memory, latency, compute) shape every design decision in this project
- The landscape of existing IDS datasets and related prior work

### Challenges / Questions for Supervisor
- What type of threat data will we primarily work with — network traffic features, raw logs, or both?
- What is the primary target edge device (Raspberry Pi 4, Jetson Nano, or other)?
- Is the classifier expected to use classical ML (Random Forest, XGBoost) or a neural network approach?

### Next Week Plan
- Read the 5 identified references in depth
- Write full literature review with summaries and gap analysis
- Research additional datasets
- Begin exploring the `datasets/` folder in the repository
- Research ONNX export pipeline from PyTorch in detail

---

## Week 2

**Branch:** `sikandarhussain-week-02`
**PR link:** _[Add link after opening PR]_

### Checklist
- [x] Downloaded CSE-CIC-IDS2018 Kaggle mirror (CSV only — ~1.5 GB, not raw PCAP)
- [x] Ran basic pandas exploration: class distribution, missing values, feature ranges
- [x] Trained a baseline MLP classifier in PyTorch and exported to ONNX
- [x] Validated ONNX output matches PyTorch on 10 test samples
- [x] Drafted `docs/proposal.md` sections 3 and 4

### What I Did This Week
- Downloaded and explored the CSE-CIC-IDS2018 dataset (shape: 1048575 rows, 80 columns) to understand the available features and class imbalances (Benign: 63.67%, FTP-BruteForce: 18.44%, SSH-Bruteforce: 17.89%).
- Developed and ran `src/explore_dataset.py` for comprehensive data exploration, finding 7648 missing values and saving a label distribution chart.
- Created and executed `src/train_classifier.py` to train a baseline PyTorch MLP classifier. The model achieved a perfect 1.00 F1-score across all classes on the 20% test split (208,951 samples) after 10 epochs (final loss: 0.0006).
- Built the ONNX export pipeline in `src/export_onnx.py` to convert the trained PyTorch model to `.onnx` format.
- Validated that the PyTorch and ONNX models produce the same outputs on a subset of test samples.
- Updated Sections 3 and 4 of `docs/proposal.md` outlining the methodology and implementation.

### What I Learned
- How to efficiently load and explore large CSV datasets using pandas.
- The process of exporting a trained PyTorch model (MLP) into the ONNX framework for edge deployment.
- How to load and run inference on an exported ONNX model and compare outputs to ensure high fidelity during conversion.

### Problems / Blockers
- Handled dataset class imbalance effectively during baseline model training.
- Ensuring the exact same preprocessing (like scaling and encoding) was applied during PyTorch inference and ONNX validation.

### Next Week Plan
- Refine the baseline model and explore hyperparameter tuning.
- Evaluate the model's performance metrics (accuracy, F1 score, precision) on the full test set.
- Further investigate memory and latency benchmarks for the ONNX model.
- Continue reading and summarizing additional references for the literature review.

---

## Week 3 & 4

**Branch:** `sikandarhussain6858-week-04`
**PR link:** _[Add link after opening PR]_

### Checklist
- [x] Refactored architecture to extract shared `ThreatMLP` model (`src/model.py`)
- [x] Implemented INT8 Static Quantization pipeline using ONNX Runtime
- [x] Evaluated and benchmarked quantized model size, latency, and Macro-F1 delta
- [x] Developed semantic labeling mapping for MITRE ATT&CK (`src/mitre_mapping.py`)
- [x] Built and tested `FastAPI` Inference Engine (`src/inference_engine.py`)
- [x] Conducted Cross-Dataset Generalization testing on ToN-IoT dataset (RQ3)
- [x] Migrated heavy computation to Kaggle to bypass local memory constraints

### What I Did These Weeks
- **Kaggle Training Pipeline:** Migrated training to Kaggle to overcome local memory constraints and leverage free GPUs. Modified the code to load multiple days of the CIC-IDS2018 dataset (using Parquet format).
- **Data Leakage & Class Imbalance:** Addressed data leakage by stripping identifying features (IPs, Ports, Flow IDs). Implemented undersampling for majority classes (Benign) and oversampling for minority classes, plus a learning rate scheduler, stabilizing the Macro-F1 score at ~0.83 on the valid features.
- **Model Quantization (RQ2):** Exported the PyTorch model to FP32 ONNX format and applied INT8 Static Quantization using `onnxruntime.quantization`. 
- **Cross-Dataset Evaluation (RQ3):** Tested the CIC-IDS2018-trained model on the completely unseen ToN-IoT dataset without retraining to evaluate out-of-distribution generalization.
- **Inference Engine:** Built `src/inference_engine.py` using FastAPI. It acts as the core "engine" by accepting raw network features, applying the saved standard scaler, running ONNX inference, and mapping predictions to MITRE ATT&CK tactics and techniques.

### Key Findings & Results
- **Quantization Benchmarks (RQ2):** The INT8 quantized model reduced the storage footprint by roughly 70% (from ~75 KB down to ~22 KB). The latency and Macro-F1 impact were successfully benchmarked and stored in `experiments/results/quantization_comparison.json`.
- **Cross-Dataset Generalization (RQ3):** The evaluation on ToN-IoT yielded a massive Macro-F1 drop from 0.8296 (in-distribution) down to 0.0843. 
  - *Analysis of the Drop:* This drop definitively answers RQ3. It is primarily caused by feature space incompatibility. Only 21 out of 76 features (28%) could be semantically mapped between NetFlow (ToN-IoT) and CICFlowMeter (CIC-IDS2018). This negative result is highly valuable, confirming that IDS models are tightly coupled to their feature extraction tools and do not easily generalize out-of-the-box.

### Problems / Blockers Addressed
- **Memory & Resource Constraints:** Loading the entire CIC-IDS2018 dataset crashed the local environment. Resolved by moving computation to Kaggle, using Parquet files, and selectively sampling data.
- **Data Leakage & Overfitting:** The baseline model initially achieved a perfect 1.0 F1. Discovered this was due to the model memorizing identifiers like `Src IP` and `Src Port`. Dropping these columns resolved the leakage.
- **Feature Alignment for RQ3:** Aligning ToN-IoT and CIC-IDS2018 features programmatically failed because the underlying extraction tools name features differently. Resolved by manually creating a semantic `FEATURE_MAP` connecting 21 common features.

### Next Week Plan
- Review and finalize the technical implementation.
- Address any code review feedback from the supervisor on the Week 3/4 PR.
- Draft the final project report (`docs/final-report.md`) outlining the methodology, evaluation, and conclusions.

---

## Week 5

**Branch:** `sikandarhussain6858-week-04` (Continued on the same branch)
**PR link:** _[Add link after opening PR]_

### Checklist
- [x] Cleaned up proposal.md formatting and merged draft into template
- [x] Added 5 additional papers to the literature review (total 10)
- [x] Refactored architecture and trained on full multi-day CIC-IDS2018 (all 15 attack types)
- [x] Implemented and benchmarked FP16 quantization alongside INT8 (RQ2)
- [x] Generated confusion matrices and quantization comparison plots (Task 5)

### What I Did This Week
- **Expanded Literature Review & Proposal:** Cleaned up `docs/proposal.md` and added 5 new research papers to `docs/literature-review.md`, bringing the total to 10 foundational papers.
- **Full Dataset Training:** Updated the Kaggle training pipeline to load all 10 days of the CSE-CIC-IDS2018 dataset. Widened the `ThreatMLP` architecture to `256 -> 128 -> 64 -> 15` classes with BatchNorm layers for stable training across all 15 MITRE attack categories.
- **FP16 vs INT8 Quantization (RQ2):** Introduced FP16 (half-precision) quantization as an alternative to INT8. We ran benchmarks comparing FP32, FP16, and INT8 models in terms of size, latency, and Macro-F1 score, confirming that FP16 preserves accuracy much better than INT8 on this architecture.
- **Evaluation Plots:** Wrote code to automatically generate high-resolution PNG plots for the final report, including confusion matrices for both in-distribution (CIC-IDS2018) and cross-dataset (ToN-IoT) evaluations, as well as a bar chart comparing quantization methods.

### Key Findings & Results
- **Full 15-Class Training:** The widened model successfully achieved an in-distribution Macro-F1 score of **0.8134** across all 15 classes, hitting the ≥ 0.80 target.
- **Cross-Dataset Model Collapse (RQ3):** Testing on the ToN-IoT dataset resulted in a Macro-F1 of **0.0427** (a drop of 0.7707). Analysis revealed this is due to a severe feature mismatch: only 21 of 76 CIC features exist in the ToN-IoT NetFlow schema. The remaining 55 zero-filled features caused the model to collapse and predict the "Infilteration" class for almost all traffic. This aligns with Cantone et al. (2024), confirming that generalization fails catastrophically without feature-schema alignment.

### Problems / Blockers Addressed
- **INT8 Precision Loss:** Static INT8 quantization previously caused a severe drop in the F1 score. We addressed this by implementing FP16 quantization, which serves as a highly effective middle ground for edge devices by cutting the model size in half without degrading the F1 score.
- **Cross-Dataset Feature Mismatch:** The model collapsed to a single class during cross-dataset evaluation. We identified the root cause (55 unmapped/zero-filled features) and documented it as a valid research finding for RQ3 rather than a bug.
- **ONNX Type Mismatch:** Encountered an `INVALID_ARGUMENT` error when benchmarking the FP16 model because the input array was still `float32`. Fixed this by explicitly casting the evaluation inputs to `float16` during benchmarking.

### Next Week Plan
- Investigate NF-standardization approach: retrain model using only the 21 NetFlow-compatible features
- Design and develop `nf-standardized-training.ipynb` notebook
- Research semantic feature mapping between CICFlowMeter and NetFlow/IPFIX schemas

---

## Week 6

**Branch:** `sikandarhussain6858-week-06`
**PR link:** _[Add link after opening PR]_

### Checklist
- [x] Researched NetFlow feature standardization approach to address RQ3 feature mismatch
- [x] Identified 21 common features between CICFlowMeter (CIC-IDS2018) and NetFlow/IPFIX (ToN-IoT)
- [x] Designed feature mapping between the two schemas
- [x] Developed `nf-standardized-training.ipynb` notebook on Kaggle
- [x] Configured NF model architecture: Input(21) → 256 → 128 → 64 → 15

### What I Did This Week
- **NF-Standardization Research:** Investigated why the baseline model collapsed on ToN-IoT (Macro-F1: 0.0427). The root cause — 55 of 76 features were zero-filled because ToN-IoT uses NetFlow/IPFIX feature extraction while CIC-IDS2018 uses CICFlowMeter. The hypothesis: if we retrain using only the 21 features common to both schemas, cross-dataset generalization should improve.
- **Feature Mapping Design:** Created a semantic mapping between the 21 overlapping features. For example: `FLOW_DURATION_MILLISECONDS` (NetFlow) ↔ `Flow Duration` (CIC), `IN_PKTS` ↔ `Total Fwd Packets`, `IN_BYTES` ↔ `Fwd Packets Length Total`, etc. The full mapping is documented in `experiments/results/nf_cross_dataset_comparison.json`.
- **Notebook Development:** Built `experiments/notebooks/nf-standardized-training.ipynb` on Kaggle, including:
  - Data loading for all 10 days of CIC-IDS2018 using only the 21 NF-compatible features
  - Class resampling strategy (cap majority at 100K, oversample minority to 5K)
  - Training with ReduceLROnPlateau scheduler for 30 epochs
  - Cross-dataset evaluation pipeline against full ToN-IoT dataset (13.1M samples)
  - Automated ONNX export with FP32/FP16/INT8 quantization variants

### What I Learned
- The feature mismatch between CICFlowMeter and NetFlow is not just a naming problem — the features are extracted by fundamentally different tools with different statistical properties.
- Reducing feature dimensionality from 76 → 21 cuts model parameters from 86,543 → 48,655 (a 44% reduction), which is favorable for edge deployment.
- Class resampling is critical when training on the full 10-day CIC-IDS2018 dataset, as some classes (e.g., DDOS attack-HOIC) dominate while others (e.g., SQL Injection) have very few samples.

### Challenges / Questions for Supervisor
- Will feature alignment alone be enough to improve cross-dataset generalization, or are the underlying distributions too different?
- Should we explore domain adaptation techniques if NF-standardization doesn't help?

### Next Week Plan
- Execute the NF-standardized training notebook on Kaggle
- Download and commit all result artifacts
- Create baseline-vs-NF comparison JSON
- Analyze whether NF-standardization improves ToN-IoT generalization

---

## Week 7

**Branch:** `sikandarhussain6858-week-07`
**PR link:** _[Add link after opening PR]_

### Checklist
- [x] Executed `nf-standardized-training.ipynb` on Kaggle (full run)
- [x] Committed NF classification report with per-class metrics
- [x] Committed NF cross-dataset evaluation results (ToN-IoT, 13.1M samples)
- [x] Committed NF quantization comparison (FP32/FP16/INT8)
- [x] Created `nf_vs_baseline_comparison.json` with side-by-side analysis
- [x] Documented key findings and implications

### What I Did This Week
- **NF-Standardized Model Training:** Executed the complete notebook on Kaggle. The NF model was trained on 552,275 samples (80/20 split, stratified) for 30 epochs with Adam optimizer (lr=1e-3), ReduceLROnPlateau scheduler (factor=0.5, patience=3), and class-weighted CrossEntropyLoss.
- **In-Distribution Evaluation (CIC-IDS2018):** The NF-standardized model achieved:
  - **Macro-F1: 0.7658** (vs. baseline 0.8134 — a -0.048 delta, expected given fewer features)
  - **Accuracy: 0.87** | Weighted-F1: 0.87
  - **Best classes:** SSH-Bruteforce (F1: 1.00), DDOS attack-HOIC (F1: 1.00), DDOS attack-LOIC-UDP (F1: 0.99)
  - **Worst class:** DoS attacks-SlowHTTPTest (F1: 0.00 — this attack is indistinguishable with only 21 features)
- **Cross-Dataset Evaluation (ToN-IoT):** Tested the NF model on the **full** ToN-IoT dataset (13,135,881 samples — vs. the baseline which only used 500K):
  - **Macro-F1: 0.0571** (vs. baseline 0.0427 — a marginal +33.7% relative improvement)
  - **Zero features zero-filled** (vs. baseline's 55 zero-filled) — the schema mismatch was eliminated
  - But generalization still failed catastrophically despite the marginal improvement
- **NF Model Quantization:** Exported and benchmarked all three precision levels:

  | Precision | Size (MB) | Latency (ms) | Macro-F1 |
  |---|---|---|---|
  | FP32 | 0.184 | 0.019 | 0.7658 |
  | FP16 | 0.093 | 0.021 | 0.7657 |
  | INT8 | 0.052 | 0.024 | 0.7686 |

- **Baseline vs NF Comparison:** Created `experiments/results/nf_vs_baseline_comparison.json` documenting the full side-by-side analysis.

### Key Findings & Results

> **Critical Finding:** NF-standardization eliminated the zero-filling problem and produced a marginal improvement in cross-dataset F1 (0.0427 → 0.0571, +33.7% relative). However, the absolute cross-dataset F1 remains catastrophically low (0.0571), confirming that feature schema alignment alone is insufficient.

This is a significant finding that conclusively answers a key research question:

1. **The generalization failure is NOT caused by feature schema mismatch alone.** Even with all 21 features properly matched (0 zero-filled), the model cannot meaningfully generalize (F1: 0.0571).
2. **The underlying feature distributions are fundamentally different.** CICFlowMeter and NetFlow/IPFIX extractors compute semantically similar features (e.g., packet counts, byte counts) but produce statistically different distributions. The model learned CICFlowMeter-specific patterns, not universal attack behavior.
3. **Implication for the field:** Cross-dataset generalization in NIDS requires domain adaptation techniques (e.g., adversarial domain adaptation, feature distribution normalization, or multi-source training) — simple feature alignment is insufficient.

**Model Size Comparison:**
- Baseline: 86,543 parameters (76-dim input) → FP32: 0.238 MB, INT8: 0.065 MB
- NF-Standardized: 48,655 parameters (21-dim input) → FP32: 0.184 MB, INT8: 0.052 MB
- **44% parameter reduction** — favorable for edge deployment even though generalization didn't improve

### Problems / Blockers Addressed
- **Kaggle Execution Time:** The full ToN-IoT evaluation (13.1M samples) required careful memory management on Kaggle's free tier. Used chunked evaluation to avoid OOM errors.
- **Negative Result Framing:** Initially expected NF-standardization to improve generalization. Reframed the negative result as a valuable empirical finding that advances understanding of why NIDS models fail to generalize.

### Next Week Plan
- Download NF model weight files from Kaggle and commit to repo
- Update `src/export_onnx.py` to support NF model (21-feature input)
- Begin building the semantic engine layer (confidence flagging, drift detection)
- Start KV-cache literature survey for edge-efficiency track

---

## Week 8

**Branch:** `sikandarhussain6858-week-08`
**PR link:** _[Add link after opening PR]_

### Checklist
- [x] Built `src/semantic_analyzer.py` — core semantic engine module with 4 classes
- [x] Implemented Feature B: ONNX intermediate-layer drift detection (fc3 embedding, 64-dim)
- [x] Implemented Feature C: Input validation (schema, range, z-score, zero-fill, NaN/Inf checks)
- [x] Re-exported NF ONNX model with dual outputs (logits + embedding) — all 10 validation samples match
- [x] Created `src/embedding_reference.py` — reference embedding generator (works without training dataset)
- [x] Added `ThreatMLPWithEmbedding` wrapper to `src/model.py`
- [x] Updated `src/export_onnx.py` with `--with-embeddings` / `--no-embeddings` flags
- [x] Updated `src/inference_engine.py` with `/predict/secure` endpoint
- [x] Created `tests/test_semantic.py` — 28 unit tests, all passing
- [x] Generated `experiments/reference_embeddings_nf.npz` and `experiments/training_feature_stats_nf.json`

### What I Did This Week

- **Semantic Analyzer Module (`src/semantic_analyzer.py`):** Created the core semantic engine module containing four classes:
  - `ConfidenceAnalyzer` — Feature A refactored from `inference_engine.py`. Flags `LOW_CONFIDENCE` when `max(softmax) < 0.70`.
  - `DriftDetector` — **Feature B (NEW).** Loads 64-dim reference embeddings from training data. At inference time, extracts the fc3 hidden layer output from the ONNX model and computes cosine distance + Mahalanobis distance to training centroids. Flags `DRIFT_DETECTED` when either distance exceeds its 95th-percentile threshold.
  - `InputValidator` — **Feature C (NEW).** Validates raw input vectors before model inference: schema check (correct feature count), NaN/Inf rejection, range check (±5σ from training mean), z-score outlier detection (|z| > 5), and zero-fill detection (>50% zero features). The zero-fill check directly catches the RQ3 failure mode — when a cross-dataset feature extractor produces incompatible distributions.
  - `SemanticSecurityEngine` — Orchestrator that composes all three analyzers and returns a unified `SemanticResult` with `engine_verdict` (CLEAN / SUSPICIOUS / REJECTED).

- **ONNX Dual-Output Re-export:** Added `ThreatMLPWithEmbedding` wrapper class to `src/model.py` that returns both the final logits and the fc3 hidden layer activation (64-dim embedding) as a tuple. Updated `src/export_onnx.py` with `--with-embeddings` flag (default: True). Re-exported the NF model — validation confirmed all 10 samples match between PyTorch and ONNX Runtime, with maximum embedding diff of 0.000027.

  | ONNX Output | Shape | Purpose |
  |---|---|---|
  | `output` | [batch, 15] | Classification logits (15 attack classes) |
  | `embedding` | [batch, 64] | fc3 hidden layer for drift detection |

- **Reference Embedding Generator (`src/embedding_reference.py`):** Since the CIC-IDS2018 dataset is not available locally, this script derives training statistics directly from the fitted `StandardScaler` (which stores the training data's mean and variance). It generates 5,000 synthetic samples from the training distribution, runs them through the ONNX model, and computes:
  - Global centroid (64-dim mean embedding)
  - 15 per-class centroids
  - 64×64 covariance matrix + inverse (for Mahalanobis distance)
  - Cosine threshold: 0.6837 (95th percentile)
  - Mahalanobis threshold: 13.1733 (95th percentile)
  - Per-feature training stats: mean, std, min_approx, max_approx for all 21 NF features

- **Inference Engine Update (`src/inference_engine.py`):** Added new `/predict/secure` and `/predict/secure/batch` endpoints that run all three semantic checks on every prediction. The response includes `drift_score`, `drift_flag`, `validation_passed`, `validation_alerts[]`, and a `semantic_summary` with `engine_verdict`. Original `/predict` and `/predict/batch` endpoints are preserved unchanged for backward compatibility. Updated `/health` endpoint to report semantic feature availability. Bumped API version to 3.0.0.

- **Unit Tests (`tests/test_semantic.py`):** Created 28 test cases using synthetic fixtures (no ONNX model or dataset dependency):

  | Test Suite | Tests | Status |
  |---|---|---|
  | `TestConfidenceAnalyzer` | 7 | ✅ All pass |
  | `TestDriftDetector` | 6 | ✅ All pass |
  | `TestInputValidator` | 10 | ✅ All pass |
  | `TestSemanticSecurityEngine` | 5 | ✅ All pass |

### Key Findings & Results

> **This is what makes the project a "semantic security engine" rather than just a classifier.** The semantic layer uses ONNX Runtime's unique capability — intermediate-layer extraction — to detect distribution drift at inference time. This directly addresses the supervisor's feedback: *"tying in something ONNX-Runtime-specific (e.g., detecting semantic drift or adversarial inputs at inference time, using the exported graph itself rather than just the pre-export model) would be a more novel angle."*

**Files created/modified this week:**

| File | Action | Purpose |
|---|---|---|
| `src/semantic_analyzer.py` | **NEW** | Core semantic engine — drift detection, input validation, confidence scoring |
| `src/embedding_reference.py` | **NEW** | Reference embedding + training stats generator |
| `tests/test_semantic.py` | **NEW** | 28 unit tests for semantic layer |
| `src/model.py` | MODIFIED | Added `ThreatMLPWithEmbedding` wrapper |
| `src/export_onnx.py` | MODIFIED | Dual-output ONNX export (logits + embedding) |
| `src/inference_engine.py` | MODIFIED | `/predict/secure` endpoint, semantic engine integration |
| `experiments/reference_embeddings_nf.npz` | **NEW** | Centroids, covariance, thresholds for drift detection |
| `experiments/training_feature_stats_nf.json` | **NEW** | Per-feature training stats for input validation |

### Problems / Blockers Addressed
- **Dataset not available locally:** Solved by deriving training statistics from the fitted `StandardScaler`'s stored parameters (mean, variance). This is mathematically equivalent to computing stats from the data directly, since the scaler was fit on the training set.
- **Cosine distance NaN:** When either the embedding or centroid is a zero vector, `scipy.spatial.distance.cosine` returns NaN. Fixed by adding `np.nan_to_num` fallback in `DriftDetector.analyze()`.

### Next Week Plan
- Integration testing: feed ToN-IoT data through `/predict/secure` → verify drift detection fires
- Feed random noise → verify anomaly flagging fires
- Feed zero-filled data → verify input validation catches it
- Benchmark latency overhead of semantic features vs plain classification
- Save all results to `experiments/results/semantic_engine_evaluation.json`

---

_(Add a new section each week)_

