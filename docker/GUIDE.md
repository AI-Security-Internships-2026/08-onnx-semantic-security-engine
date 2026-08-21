# 🚀 Docker Multi-Container Simulation Guide — Semantic Security Engine

This guide provides step-by-step instructions to build, run, monitor, and troubleshoot the **16-container Docker simulation testbed** for the ONNX Semantic Security Engine.

---

## 🏛️ Simulation Topology (16 Containers)

```text
                                  ┌──────────────────────────┐
                                  │   sec-engine-net         │
                                  │  (Docker Bridge Network) │
                                  └─────────────┬────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 │                                                             │
  ┌──────────────▼──────────────┐                               ┌──────────────▼──────────────┐
  │ 10 Benign / Normal Nodes    │                               │     5 Attacker Nodes        │
  │ (In-Distribution Traffic)   │                               │ (Threats, OOD, Noise, Fault)│
  ├─────────────────────────────┤                               ├─────────────────────────────┤
  │ • benign-01 to benign-10    │                               │ • attacker-01-ddos          │
  │ • Generates legitimate      │                               │ • attacker-02-bruteforce    │
  │   HTTP/HTTPS/DNS telemetry  │                               │ • attacker-03-toniot (OOD)  │
  │ • Expected: [CLEAN] /       │                               │ • attacker-04-noise         │
  │   [SUSPICIOUS] Benign       │                               │ • attacker-05-zero-fault    │
  └──────────────┬──────────────┘                               └──────────────┬──────────────┘
                 │                                                             │
                 │ POST /predict/secure (13 NetFlow Features)                  │ POST /predict/secure
                 └──────────────────────────────┬──────────────────────────────┘
                                                │
                                ┌───────────────▼───────────────┐
                                │    Central Engine Service     │
                                │       (security-engine)       │
                                ├───────────────────────────────┤
                                │ • FastAPI + ONNX Runtime      │
                                │ • Port: 8000                  │
                                │ • 13 NF Features              │
                                │ • Confidence Analyzer (>0.70) │
                                │ • 64-D Drift Detector (fc3)   │
                                │ • Input Validator (z-score)   │
                                │ • MITRE ATT&CK Mapping        │
                                └───────────────────────────────┘
```

---

## 📋 Prerequisites
1. **Docker Desktop installed:** Ensure Docker Desktop is installed on your machine.
2. **Docker Engine running:** Open Docker Desktop and verify that the status in the bottom-left corner is **"Engine running"** (green whale icon).

---

## ⚡ Quick Start Commands

### 1. Build and Start All 16 Containers
Run this command from the project root:
```powershell
docker compose -f docker/docker-compose.yml up --build -d
```
*(Or if you are already inside the `docker/` folder, run: `docker compose up --build -d`)*

---

### 2. Verify Container Status
Check that all 16 containers are running:
```powershell
docker compose -f docker/docker-compose.yml ps
```
You should see 16 services listed with status `Up` / `running`:
- `security-engine`
- `benign-01` through `benign-10`
- `attacker-01-ddos`, `attacker-02-bruteforce`, `attacker-03-toniot`, `attacker-04-noise`, `attacker-05-zero-fault`

---

## 📡 Live Stream Monitoring & Log Inspection

### View All Traffic Streams (Combined View)
```powershell
docker compose -f docker/docker-compose.yml logs -f
```

### Monitor Only the Central Security Engine
```powershell
docker compose -f docker/docker-compose.yml logs -f security-engine
```

### Monitor Specific Attacker Scenarios

1. **DDoS Attack Intercepts:**
   ```powershell
   docker compose -f docker/docker-compose.yml logs -f attacker-01-ddos
   ```
   *Expected Output:* High confidence detections (`DDOS attack-LOIC-UDP`, `DDOS attack-HOIC`, `Bot`).

2. **Brute Force Intercepts:**
   ```powershell
   docker compose -f docker/docker-compose.yml logs -f attacker-02-bruteforce
   ```
   *Expected Output:* Authentication flood classification (`SSH-Bruteforce`, `FTP-BruteForce`).

3. **Out-of-Distribution (ToN-IoT) Anomaly Detection:**
   ```powershell
   docker compose -f docker/docker-compose.yml logs -f attacker-03-toniot
   ```
   *Expected Output:* `[HIGH_RISK]` / `[SUSPICIOUS]` due to low confidence and distributional shift.

4. **Adversarial Gaussian Noise Perturbation:**
   ```powershell
   docker compose -f docker/docker-compose.yml logs -f attacker-04-noise
   ```
   *Expected Output:* `[HIGH_RISK]` with `EXTREME_OUTLIERS` and `OUT_OF_RANGE` alerts.

5. **Zero-Filled Telemetry Tampering / Evasion:**
   ```powershell
   docker compose -f docker/docker-compose.yml logs -f attacker-05-zero-fault
   ```
   *Expected Output:* Instant **`[REJECTED]`** verdict with `ZERO_FILLED` input validation alerts.

---

## 🌐 Web Browser & REST API Access

While the testbed is running, access the live FastAPI endpoints on your host machine:

| Endpoint | URL | Description |
| :--- | :--- | :--- |
| **Interactive Swagger Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive API exploration and live testing |
| **Health Check** | [http://localhost:8000/health](http://localhost:8000/health) | Real-time status, uptime, and loaded components |
| **Model Metadata** | [http://localhost:8000/model/info](http://localhost:8000/model/info) | Model variant (13 features), size, and classes |
| **MITRE ATT&CK Mapping** | [http://localhost:8000/mitre/mappings](http://localhost:8000/mitre/mappings) | Full taxonomy mapping from threat label to MITRE technique |

---

## 🛑 Stopping & Cleaning Up

To stop all 16 containers and remove the virtual network:
```powershell
docker compose -f docker/docker-compose.yml down
```

To stop and remove all container volumes/images:
```powershell
docker compose -f docker/docker-compose.yml down --rmi local
```

---

## 🔧 Troubleshooting

- **Error: `failed to connect to the docker API`**
  - *Cause:* Docker Desktop application is closed.
  - *Fix:* Launch Docker Desktop from the Windows Start menu and wait for "Engine running".
- **Port 8000 already in use:**
  - *Fix:* Stop any local python uvicorn process running on port 8000 before running `docker compose up`.
