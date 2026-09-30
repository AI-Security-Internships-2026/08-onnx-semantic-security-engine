"""
Shared Data and Feature Alignment Utilities for SEMANTICSHIELD Evaluations.

Provides standardized dataset loaders, feature maps (exact, approximate, mismatched),
taxonomy alignment, and evaluation metric helpers for cross-dataset and cross-model studies.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

# ── Feature Mapping Tiers ──
# 8 Exact Mappings between NetFlow v2 (nProbe) and CIC-IDS2018 (CICFlowMeter)
TIER1_EXACT_MAP: Dict[str, str] = {
    "FLOW_DURATION_MILLISECONDS": "Flow Duration",
    "IN_PKTS": "Total Fwd Packets",
    "OUT_PKTS": "Total Backward Packets",
    "IN_BYTES": "Fwd Packets Length Total",
    "OUT_BYTES": "Bwd Packets Length Total",
    "LONGEST_FLOW_PKT": "Packet Length Max",
    "SHORTEST_FLOW_PKT": "Packet Length Min",
    "PROTOCOL": "Protocol",
}

# 5 Approximate Mappings
TIER2_APPROX_MAP: Dict[str, str] = {
    "MAX_IP_PKT_LEN": "Fwd Packet Length Max",
    "MIN_IP_PKT_LEN": "Fwd Packet Length Min",
    "SRC_TO_DST_SECOND_BYTES": "Flow Bytes/s",
    "TCP_WIN_MAX_IN": "Init Fwd Win Bytes",
    "TCP_WIN_MAX_OUT": "Init Bwd Win Bytes",
}

# Full 13-feature standardized schema used by SEMANTICSHIELD
FEATURE_MAP: Dict[str, str] = {**TIER1_EXACT_MAP, **TIER2_APPROX_MAP}
NETFLOW_FEATURES = list(FEATURE_MAP.keys())
NF_FEATURES = list(FEATURE_MAP.values())

# 8 Known Mismatches from legacy 21-feature mapping (for Track B failure analysis)
TIER3_MISMATCHED_MAP: Dict[str, str] = {
    "SRC_TO_DST_AVG_THROUGHPUT": "Fwd Header Length",
    "DST_TO_SRC_AVG_THROUGHPUT": "Bwd Header Length",
    "TCP_FLAGS": "Fwd PSH Flags",
    "RETRANSMITTED_IN_PKTS": "Fwd Avg Packets/Bulk",
    "RETRANSMITTED_OUT_PKTS": "Bwd Avg Packets/Bulk",
    "RETRANSMITTED_IN_BYTES": "Fwd Avg Bytes/Bulk",
    "RETRANSMITTED_OUT_BYTES": "Bwd Avg Bytes/Bulk",
    "NUM_PKTS_UP_TO_128_BYTES": "Subflow Fwd Packets",
}

# ── Attack Taxonomy Alignment ──
TONIOT_TO_CIC_MAP: Dict[str, str] = {
    "benign": "Benign",
    "password": "FTP-BruteForce",
    "dos": "DoS attacks-Hulk",
    "ddos": "DDoS attacks-LOIC-HTTP",
    "injection": "SQL Injection",
    "xss": "Brute Force -XSS",
    "backdoor": "Infilteration",
    "scanning": "Infilteration",
    "ransomware": "Infilteration",
    "mitm": "Infilteration",
}

BOTIOT_TO_CIC_MAP: Dict[str, str] = {
    "Benign": "Benign",
    "DoS": "DoS attacks-Hulk",
    "DDoS": "DDoS attacks-LOIC-HTTP",
    "Reconnaissance": "Infilteration",
    "Theft": "Infilteration",
}


def get_toniot_path() -> Path:
    """Find NF-ToN-IoT-V2 parquet file."""
    p = DATASETS_DIR / "ToN-IoT" / "NF-ToN-IoT-V2.parquet"
    if not p.exists():
        p = DATASETS_DIR / "NF-ToN-IoT-V2" / "NF-ToN-IoT-V2.parquet"
    return p


def get_botiot_path() -> Path:
    """Find NF-BoT-IoT-V2 parquet file."""
    for cand in [
        DATASETS_DIR / "NF-BoT-IoT-V2" / "NF-BoT-IoT-V2.parquet",
        DATASETS_DIR / "BoT-IoT" / "NF-BoT-IoT-V2.parquet",
    ]:
        if cand.exists():
            return cand
    raise FileNotFoundError("NF-BoT-IoT-V2.parquet not found in datasets/NF-BoT-IoT-V2 or datasets/BoT-IoT")


def load_standardized_netflow(
    parquet_path: Path,
    n_samples: int = 50000,
    random_state: int = 42,
    feature_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load a NetFlow v2 parquet file, rename columns to canonical CIC names,
    and return (df_mapped, y_binary, y_attack_str, attack_names).
    
    Returns:
        df_mapped: DataFrame with renamed columns matching CICFlowMeter features
        y_binary: (N,) binary array (0=Benign, 1=Attack)
        y_attack_str: (N,) string array of attack names
        attack_names: unique attack names in sample
    """
    if feature_mapping is None:
        feature_mapping = FEATURE_MAP

    req_cols = list(feature_mapping.keys())
    # Ensure Label and Attack columns are also loaded if present
    load_cols = list(req_cols)
    sample_df = pd.read_parquet(parquet_path, columns=None)
    available_cols = [c for c in load_cols if c in sample_df.columns]
    
    meta_cols = [c for c in ["Label", "Attack"] if c in sample_df.columns]
    
    df = pd.read_parquet(parquet_path, columns=available_cols + meta_cols)
    
    # Stratified or balanced subsampling if dataset is larger than requested
    if len(df) > n_samples:
        if "Label" in df.columns:
            # Sample proportionately by Label
            df = df.groupby("Label", group_keys=False).apply(
                lambda x: x.sample(int(np.rint(n_samples * len(x) / len(df))), random_state=random_state)
            )
            if len(df) > n_samples:
                df = df.sample(n=n_samples, random_state=random_state)
        else:
            df = df.sample(n=n_samples, random_state=random_state)

    # Clean inf and nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna().reset_index(drop=True)

    # Extract labels
    if "Label" in df.columns:
        y_binary = df["Label"].values.astype(int)
    else:
        # Infer binary from Attack column
        y_binary = (df["Attack"] != "Benign").astype(int).values

    y_attack = df["Attack"].values if "Attack" in df.columns else np.where(y_binary == 0, "Benign", "Attack")
    attack_names = np.unique(y_attack)

    # Rename NetFlow columns to standard CIC feature names
    features_present = {nf_col: cic_col for nf_col, cic_col in feature_mapping.items() if nf_col in df.columns}
    df_mapped = df[list(features_present.keys())].rename(columns=features_present)

    return df_mapped, y_binary, y_attack, attack_names


def load_in_dist_test(n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load held-out in-distribution test samples (already scaled or raw).
    
    Returns:
        X_test (N, 13), y_test (N,)
    """
    x_path = EXPERIMENTS_DIR / "X_test_nf.npy"
    y_path = EXPERIMENTS_DIR / "y_test_nf.npy"
    if x_path.exists() and y_path.exists():
        X = np.load(x_path)
        y = np.load(y_path)
        if n_samples and len(X) > n_samples:
            indices = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        return X, y
    else:
        raise FileNotFoundError(f"Held out test arrays not found at {x_path}")


# ── Metric Computation Helpers ──

def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC, returning NaN if single class or invalid."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def safe_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUPRC (Average Precision), returning NaN if invalid."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return float("nan")


def tpr_at_fixed_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """Compute True Positive Rate at a fixed target False Positive Rate."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fprs, tprs, _ = roc_curve(y_true, scores)
    # Find operating point where FPR <= target_fpr
    valid_idx = np.where(fprs <= target_fpr)[0]
    if len(valid_idx) == 0:
        return 0.0
    return float(tprs[valid_idx[-1]])


def fpr_at_fixed_tpr(y_true: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """Compute False Positive Rate at a fixed True Positive Rate (e.g., 95% detection)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fprs, tprs, _ = roc_curve(y_true, scores)
    valid_idx = np.where(tprs >= target_tpr)[0]
    if len(valid_idx) == 0:
        return 1.0
    return float(fprs[valid_idx[0]])
