"""
Training Time Per Feature-Schema Stage Benchmark

Systematically trains ThreatMLP under identical hardware, epoch count, batch size,
and optimizer settings across all three feature schema stages:
  1. 76-Feature Baseline (CICFlowMeter full feature set)
  2. 21-Feature NetFlow-Mapped (Legacy schema with 8 mismatches)
  3. 13-Feature NetFlow-Corrected (Standardized schema with Protocol)

Measures:
  - Total wall-clock training time (seconds)
  - Per-epoch training time (mean ± std ms)
  - Training throughput (samples/sec)
  - Train/Val loss curves
  - Final test classification Macro-F1 and Accuracy
  - Model parameter count and FLOPs reduction

Outputs:
  - experiments/results/training_time_benchmark.json
  - experiments/images/training_time_comparison.png
"""

import json
import time
import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.config_loader import load_paper_config, get_provenance_metadata

EXPERIMENTS = BASE_DIR / "experiments"
RESULTS_DIR = EXPERIMENTS / "results"
IMAGES_DIR = EXPERIMENTS / "images"
DATASETS_DIR = BASE_DIR / "datasets" / "CIC-IDS2018"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

from src.model import ThreatMLP

# ── Feature Definitions ──
SCHEMA_13_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",
    "Bwd Packets Length Total",
    "Packet Length Max",
    "Packet Length Min",
    "Protocol",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Flow Bytes/s",
    "Init Fwd Win Bytes",
    "Init Bwd Win Bytes",
]

SCHEMA_21_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",
    "Bwd Packets Length Total",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Packet Length Max",
    "Packet Length Min",
    "Flow Bytes/s",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd PSH Flags",
    "Init Fwd Win Bytes",
    "Init Bwd Win Bytes",
    "Fwd Avg Packets/Bulk",
    "Bwd Avg Packets/Bulk",
    "Fwd Avg Bytes/Bulk",
    "Bwd Avg Bytes/Bulk",
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
]

# ── Training Hyperparameters ──
N_EPOCHS = 15
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
RANDOM_SEED = 42
N_SAMPLES_TO_LOAD = 200000  # High sample count for rigorous timing

def load_cic_dataset(n_samples: int = N_SAMPLES_TO_LOAD) -> pd.DataFrame:
    """Load and sample from the local CIC-IDS2018 parquet files."""
    parquet_files = sorted(DATASETS_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {DATASETS_DIR}")
    
    print(f"Loading data from {len(parquet_files)} parquet files...")
    dfs = []
    samples_per_file = max(1000, n_samples // len(parquet_files))
    
    for f in parquet_files:
        df_chunk = pd.read_parquet(f)
        df_chunk.columns = df_chunk.columns.str.strip()
        if len(df_chunk) > samples_per_file:
            df_chunk = df_chunk.sample(n=samples_per_file, random_state=RANDOM_SEED)
        dfs.append(df_chunk)
    
    df = pd.concat(dfs, ignore_index=True)
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    if "Dst Port" in df.columns:
        df = df.drop(columns=["Dst Port"])
        
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna().drop_duplicates()
    
    print(f"Loaded {len(df):,} clean dataset rows across {df.shape[1]} total columns.")
    return df

def train_and_benchmark_schema(
    schema_name: str,
    feature_cols: list,
    df: pd.DataFrame,
    label_encoder: LabelEncoder,
    device: torch.device
) -> dict:
    """Train ThreatMLP on a specific feature schema and measure wall-clock training time."""
    print(f"\n{'='*70}")
    print(f"  Benchmarking Training: {schema_name} ({len(feature_cols)} Features)")
    print(f"{'='*70}")
    
    # 1. Prepare Feature Matrix
    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = label_encoder.transform(df["Label"].values)
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=RANDOM_SEED
    )
    
    # Feature scaling (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Compute Class Weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights_tensor = torch.FloatTensor(weights).to(device)
    
    # 3. Initialize ThreatMLP Model
    input_dim = len(feature_cols)
    num_classes = len(label_encoder.classes_)
    model = ThreatMLP(input_dim=input_dim, num_classes=num_classes).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Input Dim: {input_dim}, Classes: {num_classes}, Model Parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Timed Training Loop
    epoch_times = []
    epoch_losses = []
    
    t_start_total = time.perf_counter()
    
    for epoch in range(1, N_EPOCHS + 1):
        t_start_epoch = time.perf_counter()
        model.train()
        running_loss = 0.0
        
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_b)
            
        t_end_epoch = time.perf_counter()
        epoch_sec = t_end_epoch - t_start_epoch
        epoch_times.append(epoch_sec)
        
        avg_loss = running_loss / len(train_dataset)
        epoch_losses.append(avg_loss)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch [{epoch:02d}/{N_EPOCHS:02d}] - Loss: {avg_loss:.4f} - Epoch Time: {epoch_sec*1000:.1f} ms")
            
    t_end_total = time.perf_counter()
    total_train_sec = t_end_total - t_start_total
    
    # 5. Evaluate on Test Set
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            outputs = model(X_b)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_b.numpy())
            
    test_macro_f1 = float(f1_score(all_targets, all_preds, average='macro', zero_division=0))
    test_accuracy = float(accuracy_score(all_targets, all_preds))
    
    train_throughput = float((len(train_dataset) * N_EPOCHS) / total_train_sec)
    
    print(f"  [RESULT] Total Training Time: {total_train_sec:.2f} s")
    print(f"  [RESULT] Epoch Time: {np.mean(epoch_times)*1000:.1f} ± {np.std(epoch_times)*1000:.1f} ms")
    print(f"  [RESULT] Training Throughput: {train_throughput:,.1f} samples/sec")
    print(f"  [RESULT] Test Macro-F1: {test_macro_f1:.4f} | Accuracy: {test_accuracy*100:.2f}%")
    
    return {
        "schema_name": schema_name,
        "feature_count": input_dim,
        "parameter_count": total_params,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "total_train_time_sec": round(total_train_sec, 3),
        "mean_epoch_time_ms": round(float(np.mean(epoch_times) * 1000), 2),
        "std_epoch_time_ms": round(float(np.std(epoch_times) * 1000), 2),
        "train_throughput_samples_per_sec": round(train_throughput, 1),
        "test_macro_f1": round(test_macro_f1, 4),
        "test_accuracy": round(test_accuracy, 4),
        "epoch_losses": [round(l, 4) for l in epoch_losses]
    }

def generate_comparison_plots(results: dict, figures_dir=None):
    """Generate plots comparing training time and throughput across feature schemas."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Training Efficiency & Scalability Across Feature-Schema Stages", fontsize=14, fontweight='bold')
    
    schemas = [r["schema_name"] for r in results.values()]
    times = [r["total_train_time_sec"] for r in results.values()]
    f1s = [r["test_macro_f1"] for r in results.values()]
    throughputs = [r["train_throughput_samples_per_sec"] for r in results.values()]
    
    colors = ['#E91E63', '#FF9800', '#4CAF50']
    
    # Panel 1: Training Time vs In-Distribution F1
    ax1 = axes[0]
    bars = ax1.bar(schemas, times, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    for bar, t, f in zip(bars, times, f1s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{t:.1f} s\n(F1={f:.3f})", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    ax1.set_ylabel("Total Wall-Clock Training Time (seconds)")
    ax1.set_title(f"Training Time (15 Epochs, {N_SAMPLES_TO_LOAD:,} samples)")
    ax1.set_ylim(0, max(times) * 1.3)
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel 2: Training Throughput
    ax2 = axes[1]
    bars2 = ax2.bar(schemas, throughputs, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    for bar, tp in zip(bars2, throughputs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f"{tp:,.0f}/s", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    ax2.set_ylabel("Training Throughput (samples/second)")
    ax2.set_title("Training Throughput Comparison")
    ax2.set_ylim(0, max(throughputs) * 1.25)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if figures_dir is not None:
        canonical_plot = Path(figures_dir) / "training_time_comparison.png"
        plt.savefig(canonical_plot, dpi=150, bbox_inches='tight')
        print(f"\n[SAVED] {canonical_plot}")

    plot_path = IMAGES_DIR / "training_time_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Training time per feature-schema stage benchmark.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(EXPERIMENTS / "paper_results" / "json"),
        help="Directory to save training_time_benchmark.json",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(EXPERIMENTS / "paper_results" / "figures"),
        help="Directory to save training time comparison plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  TRAINING TIME PER FEATURE-SCHEMA STAGE BENCHMARK")
    print(f"  Epochs: {N_EPOCHS}, Batch Size: {BATCH_SIZE}, Seed: {RANDOM_SEED}")
    print("=" * 70)
    
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    # ── Load Dataset ──
    df = load_cic_dataset()
    
    # ── Encode Labels ──
    label_encoder = LabelEncoder()
    label_encoder.fit(df["Label"].values)
    print(f"Classes ({len(label_encoder.classes_)}): {label_encoder.classes_}")
    
    # ── Determine Available Feature Columns for Schema 76 ──
    all_numeric_cols = [c for c in df.columns if c != "Label" and pd.api.types.is_numeric_dtype(df[c])]
    schema_76_cols = all_numeric_cols
    
    # Validate columns for 21 and 13
    schema_21_cols = [c for c in SCHEMA_21_FEATURES if c in df.columns]
    schema_13_cols = [c for c in SCHEMA_13_FEATURES if c in df.columns]
    
    print(f"Schema 76 features available: {len(schema_76_cols)}")
    print(f"Schema 21 features available: {len(schema_21_cols)}")
    print(f"Schema 13 features available: {len(schema_13_cols)}")
    
    # ── Run Benchmarks ──
    benchmarks = {}
    
    # 1. Schema 76 (Baseline)
    benchmarks["76-Feature Baseline"] = train_and_benchmark_schema(
        "76-Feature Baseline", schema_76_cols, df, label_encoder, device
    )
    
    # 2. Schema 21 (NetFlow Legacy)
    benchmarks["21-Feature NetFlow"] = train_and_benchmark_schema(
        "21-Feature NetFlow", schema_21_cols, df, label_encoder, device
    )
    
    # 3. Schema 13 (Standardized Corrected)
    benchmarks["13-Feature Standardized"] = train_and_benchmark_schema(
        "13-Feature Standardized", schema_13_cols, df, label_encoder, device
    )
    
    # Compute relative speedup vs 76-feature baseline
    base_time = benchmarks["76-Feature Baseline"]["total_train_time_sec"]
    base_params = benchmarks["76-Feature Baseline"]["parameter_count"]
    
    for k, v in benchmarks.items():
        v["speedup_vs_baseline"] = round(base_time / v["total_train_time_sec"], 2)
        v["time_reduction_pct"] = round((1 - v["total_train_time_sec"] / base_time) * 100, 2)
        v["param_reduction_pct"] = round((1 - v["parameter_count"] / base_params) * 100, 2)
        
    # ── Save JSON Results ──
    output = {
        "provenance": get_provenance_metadata(),
        "experiment": "Training Time per Feature-Schema Stage Benchmark",
        "device": str(device),
        "hyperparameters": {
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": RANDOM_SEED,
            "samples": len(df)
        },
        "schemas": benchmarks
    }
    
    canonical_json = output_dir / "training_time_benchmark.json"
    with open(canonical_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[SAVED] {canonical_json}")

    legacy_json = RESULTS_DIR / "training_time_benchmark.json"
    if legacy_json.parent.exists() and canonical_json != legacy_json:
        with open(legacy_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[MIRRORED] {legacy_json}")
    
    # ── Generate Plots ──
    generate_comparison_plots(benchmarks, figures_dir=figures_dir)
    
    # ── Print Summary Table ──
    print(f"\n{'='*105}")
    print(f"  TRAINING TIME PER FEATURE-SCHEMA STAGE — SUMMARY TABLE")
    print(f"{'='*105}")
    header = f"{'Feature Schema':<26} {'Features':>8} {'Params':>10} {'Train Time':>12} {'Epoch Time':>16} {'Throughput':>14} {'Macro-F1':>10}"
    print(header)
    print("-" * 105)
    for k, v in benchmarks.items():
        print(f"{v['schema_name']:<26} {v['feature_count']:>8} {v['parameter_count']:>10,} {v['total_train_time_sec']:>10.2f}s {v['mean_epoch_time_ms']:>10.1f}±{v['std_epoch_time_ms']:.1f}ms {v['train_throughput_samples_per_sec']:>12.0f}/s {v['test_macro_f1']:>10.4f}")
    print(f"{'='*105}")

if __name__ == "__main__":
    main()
