# ONNX-Based Semantic Security Engine for Edge Inference

> **CNIT/PNTLab Pisa · TECIP · Scuola Superiore Sant'Anna — AI Security Internship 2026**

---

## Research Problem

Export trained threat-classification models to ONNX format and build a lightweight semantic security engine that runs efficiently on edge hardware with minimal latency and memory footprint.

---

## Objectives

1. Conduct a systematic literature review on the topic.
2. Design and implement a proof-of-concept prototype.
3. Evaluate the prototype on real or benchmark datasets.
4. Document findings in a final technical report.
5. Present results to the research group.

---

## Expected Deliverables

| Deliverable | Due |
|---|---|
| Literature review (`docs/literature-review.md`) | Week 2 |
| Architecture design document (`docs/proposal.md`) | Week 3 |
| Working prototype (`src/`) | Week 6 |
| Evaluation results (`experiments/results/`) | Week 7 |
| Final report (`docs/final-report.md`) | Week 8 |

---

## Recommended Technology Stack

```
Python, ONNX Runtime, PyTorch, scikit-learn, NumPy, FastAPI
```

See `requirements.txt` for pinned dependencies.

---

## Weekly Workflow

```
Monday     – Review weekly tasks in tasks/week-XX.md
Tue–Thu    – Implementation / experiments
Friday     – Document progress in docs/weekly-progress.md
Friday     – Open weekly Pull Request from your branch → dev
```

---

## Branching Policy

| Branch | Purpose |
|---|---|
| `main` | Stable, supervisor-reviewed code only |
| `dev` | Integration branch — merge weekly PRs here |
| `<your-name>-week-XX` | Your working branch for each week |

**Students must never push directly to `main`.**

---

## Pull Request Policy

- One PR per week, targeting the `dev` branch.
- PR title format: `[Week XX] Brief description`
- PR description must reference the weekly task file and summarise what was done.
- A supervisor or co-student must review before merging.

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/AI-Security-Internships-2026/08-onnx-semantic-security-engine.git
cd 08-onnx-semantic-security-engine

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your weekly branch
git checkout dev
git pull origin dev
git checkout -b your-name-week-01

# 5. Run the starter script
python src/main.py
```

---

## Roadmap to September 8, 2026

**Current state:** week-06 submission is currently thin — mostly deletions from the training notebook, not yet the standardized-data retraining described in its own PR body (issue #13). A Phase 2 assignment already exists (issue #12): KV-cache memory/eviction strategy for edge-deployed models.

**Novel contribution target:** tie the "semantic security engine" name to something ONNX-Runtime-specific — detecting semantic drift or adversarial inputs at inference time using the exported graph itself, combined with the KV-cache memory/eviction work for edge deployment.

| Date | Milestone |
|---|---|
| Aug 2 | Finish the week-6 standardized-data retraining and ONNX export properly (issue #13) |
| Aug 9 | Phase 2 start (issue #12): investigate KV-cache memory/eviction strategy for edge-deployed models |
| Aug 16 | Tie ONNX-Runtime-based semantic-drift/adversarial-input detection into the memory/eviction work |
| Aug 23 | Benchmark detection latency/accuracy on ONNX Runtime vs. the pre-export model |
| Aug 30 | Full write-up |
| Sep 6 | Paper draft |
| **Sep 8** | **Final submission** |

---

## Supervisor Note

This repository is managed by **CNIT/PNTLab Pisa, TECIP, Scuola Superiore Sant'Anna**.
Please contact your supervisor before making architectural changes.
All code must be original or properly attributed.
Do **not** commit API keys, passwords, or large datasets — see `.gitignore`.
