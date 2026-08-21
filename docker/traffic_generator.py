"""
Traffic Generator for Docker Simulation Testbed
Emulates benign IoT/edge nodes or malicious threat actors sending 13-feature telemetry
to the central ONNX Semantic Security Engine.
"""

import os
import sys
import time
import json
import random
import requests
import numpy as np
from pathlib import Path

# ── Load Benign Samples Pool if available ──
BENIGN_POOL = []
for p in [Path(__file__).parent / "benign_samples.json", Path("/app/benign_samples.json"), Path("docker/benign_samples.json")]:
    if p.exists():
        try:
            with open(p, "r") as f:
                BENIGN_POOL = json.load(f)
            break
        except Exception:
            pass

# ── Configuration from Environment Variables ──
NODE_ID = os.environ.get("NODE_ID", "node-unknown")
NODE_TYPE = os.environ.get("NODE_TYPE", "benign").lower()  # benign, ddos, bruteforce, ood_toniot, noise, zero_filled
ENGINE_URL = os.environ.get("ENGINE_URL", "http://security-engine:8000/predict/secure")
INTERVAL_SEC = float(os.environ.get("INTERVAL_SEC", "1.0"))
MAX_REQUESTS = int(os.environ.get("MAX_REQUESTS", "-1"))  # -1 for infinite loop

# Reconfigure stdout for reliable terminal printing
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Exact 13 NF Features (Order matches trained model & scaler): ──
# Index 0:  Flow Duration
# Index 1:  Total Fwd Packets
# Index 2:  Total Backward Packets
# Index 3:  Fwd Packets Length Total
# Index 4:  Bwd Packets Length Total
# Index 5:  Packet Length Max
# Index 6:  Packet Length Min
# Index 7:  Protocol (6.0=TCP, 17.0=UDP)
# Index 8:  Fwd Packet Length Max
# Index 9:  Fwd Packet Length Min
# Index 10: Flow Bytes/s
# Index 11: Init Fwd Win Bytes
# Index 12: Init Bwd Win Bytes

def generate_benign_flow() -> list:
    """Generate in-distribution legitimate HTTP/HTTPS/DNS traffic calibrated to CIC-IDS2018."""
    is_tcp = random.random() < 0.85
    if is_tcp:
        protocol = 6.0
        duration = random.uniform(50000.0, 5000000.0) # 50ms to 5s in microseconds
        fwd_pkts = float(random.randint(3, 15))
        bwd_pkts = float(random.randint(2, 20))
        fwd_len_max = float(random.randint(100, 900))
        fwd_len_min = 0.0
        bwd_len_max = float(random.randint(200, 1460))
        bwd_len_min = 0.0
        pkt_max = max(fwd_len_max, bwd_len_max)
        pkt_min = 0.0
        fwd_total_bytes = fwd_pkts * random.randint(60, 200)
        bwd_total_bytes = bwd_pkts * random.randint(100, 600)
        dur_sec = max(duration / 1000000.0, 1e-6)
        bytes_per_sec = (fwd_total_bytes + bwd_total_bytes) / dur_sec
        init_fwd_win = float(random.choice([8192, 29200, 65535]))
        init_bwd_win = float(random.choice([119, 1047, 8192, 29200]))
    else:
        protocol = 17.0
        duration = random.uniform(100.0, 50000.0)
        fwd_pkts = float(random.randint(1, 4))
        bwd_pkts = float(random.randint(1, 4))
        fwd_len_max = float(random.randint(30, 80))
        fwd_len_min = fwd_len_max
        pkt_max = float(random.randint(40, 250))
        pkt_min = 30.0
        fwd_total_bytes = fwd_pkts * fwd_len_max
        bwd_total_bytes = bwd_pkts * pkt_max
        dur_sec = max(duration / 1000000.0, 1e-6)
        bytes_per_sec = (fwd_total_bytes + bwd_total_bytes) / dur_sec
        init_fwd_win = -1.0
        init_bwd_win = -1.0

    return [
        float(duration),
        float(fwd_pkts),
        float(bwd_pkts),
        float(fwd_total_bytes),
        float(bwd_total_bytes),
        float(pkt_max),
        float(pkt_min),
        float(protocol),
        float(fwd_len_max),
        float(fwd_len_min),
        float(bytes_per_sec),
        float(init_fwd_win),
        float(init_bwd_win),
    ]

def generate_ddos_flow() -> list:
    """Generate high-rate DDoS flood traffic."""
    duration = random.uniform(100000.0, 2000000.0)
    fwd_pkts = float(random.randint(500, 3000))
    bwd_pkts = float(random.randint(0, 5))
    fwd_len_max = float(random.randint(400, 1460))
    fwd_len_min = 40.0
    pkt_max = fwd_len_max
    pkt_min = 40.0
    fwd_total_bytes = fwd_pkts * fwd_len_max
    bwd_total_bytes = bwd_pkts * 40.0
    dur_sec = max(duration / 1000000.0, 1e-6)
    bytes_per_sec = (fwd_total_bytes + bwd_total_bytes) / dur_sec

    return [
        float(duration),
        float(fwd_pkts),
        float(bwd_pkts),
        float(fwd_total_bytes),
        float(bwd_total_bytes),
        float(pkt_max),
        float(pkt_min),
        float(random.choice([6.0, 17.0])),
        float(fwd_len_max),
        float(fwd_len_min),
        float(bytes_per_sec),
        float(random.choice([8192.0, 29200.0, -1.0])),
        float(-1.0),
    ]

def generate_bruteforce_flow() -> list:
    """Generate SSH/FTP authentication brute-force traffic."""
    duration = random.uniform(2000000.0, 15000000.0)
    fwd_pkts = float(random.randint(15, 40))
    bwd_pkts = float(random.randint(15, 40))
    fwd_len_max = float(random.randint(150, 400))
    fwd_len_min = 40.0
    pkt_max = max(fwd_len_max, 450.0)
    pkt_min = 40.0
    fwd_total_bytes = fwd_pkts * 120.0
    bwd_total_bytes = bwd_pkts * 120.0
    dur_sec = max(duration / 1000000.0, 1e-6)
    bytes_per_sec = (fwd_total_bytes + bwd_total_bytes) / dur_sec

    return [
        float(duration),
        float(fwd_pkts),
        float(bwd_pkts),
        float(fwd_total_bytes),
        float(bwd_total_bytes),
        float(pkt_max),
        float(pkt_min),
        float(6.0), # TCP
        float(fwd_len_max),
        float(fwd_len_min),
        float(bytes_per_sec),
        float(65535.0),
        float(65535.0),
    ]

def generate_ood_toniot_flow() -> list:
    """Simulate Out-Of-Distribution IoT telemetry with divergent NetFlow scaling."""
    duration = random.uniform(60000000.0, 300000000.0) # Divergent scale
    fwd_pkts = float(random.randint(1, 5))
    bwd_pkts = float(random.randint(0, 2))
    fwd_len = float(random.randint(60, 200))
    fwd_total = fwd_pkts * fwd_len
    bwd_total = bwd_pkts * 40.0
    dur_sec = max(duration / 1000000.0, 1e-6)
    bytes_per_sec = (fwd_total + bwd_total) / dur_sec
    return [
        float(duration),
        float(fwd_pkts),
        float(bwd_pkts),
        float(fwd_total),
        float(bwd_total),
        float(fwd_len),
        float(40.0),
        float(random.choice([6.0, 17.0])),
        float(fwd_len),
        float(40.0),
        float(bytes_per_sec),
        float(random.choice([0.0, 1024.0])),
        float(random.choice([0.0, 1024.0])),
    ]

def generate_noise_flow() -> list:
    """Simulate adversarial evasion / corrupted gaussian noise features."""
    return list(np.random.normal(loc=500.0, scale=3000.0, size=13).astype(float))

def generate_zero_flow() -> list:
    """Simulate telemetry failure or zero-fill evasion bypass attempt."""
    return [0.0] * 13

def sample_flow(node_type: str) -> list:
    if node_type == "benign":
        if BENIGN_POOL:
            return random.choice(BENIGN_POOL)
        return generate_benign_flow()
    elif node_type == "ddos":
        return generate_ddos_flow()
    elif node_type == "bruteforce":
        return generate_bruteforce_flow()
    elif node_type == "ood_toniot":
        return generate_ood_toniot_flow()
    elif node_type == "noise":
        return generate_noise_flow()
    elif node_type == "zero_filled":
        return generate_zero_flow()
    else:
        return generate_benign_flow()

def wait_for_engine(url: str, timeout: int = 60):
    """Poll the engine's /health endpoint until it is ready."""
    health_url = url.replace("/predict/secure", "/health").replace("/predict", "/health")
    print(f"[{NODE_ID}] Waiting for security engine at {health_url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                print(f"[{NODE_ID}] [OK] Connected to Security Engine! (Model: {resp.json().get('model_variant')})")
                return True
        except Exception:
            pass
        time.sleep(1)
    print(f"[{NODE_ID}] [FAIL] Timeout waiting for engine.")
    return False

def main():
    print(f"============================================================")
    print(f"  Traffic Generator: {NODE_ID} (Role: {NODE_TYPE.upper()})")
    print(f"  Target URL       : {ENGINE_URL}")
    print(f"  Stream Interval  : {INTERVAL_SEC}s")
    print(f"============================================================")

    if not wait_for_engine(ENGINE_URL):
        sys.exit(1)

    count = 0
    while MAX_REQUESTS < 0 or count < MAX_REQUESTS:
        count += 1
        features = sample_flow(NODE_TYPE)
        payload = {"features": features}

        try:
            t0 = time.perf_counter()
            response = requests.post(ENGINE_URL, json=payload, timeout=5)
            lat_ms = (time.perf_counter() - t0) * 1000

            if response.status_code == 200:
                data = response.json()
                verdict = data.get("semantic_summary", {}).get("engine_verdict", "N/A")
                pred = data.get("predictions", [{}])[0]
                label = pred.get("label", "Unknown")
                conf = pred.get("confidence", 0.0)
                drift = pred.get("drift_score", 0.0)
                alerts = pred.get("validation_alerts", [])
                
                status_tag = f"[{verdict}]"
                alert_str = f" | Alerts: {alerts}" if alerts else ""
                
                print(f"[{NODE_ID}] Flow #{count:04d} {status_tag:<11} | Label: {label:<15} (Conf: {conf:.2f}, Drift: {drift:.2f}, Lat: {lat_ms:.1f}ms){alert_str}")
            else:
                print(f"[{NODE_ID}] Flow #{count:04d} [WARN] HTTP {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[{NODE_ID}] Flow #{count:04d} [ERROR] Request failed: {e}")

        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
