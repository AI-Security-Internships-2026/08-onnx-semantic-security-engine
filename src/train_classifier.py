"""
Train ThreatMLP classifier — Supports both Baseline (76-feature) and NF-Standardized (21-feature) modes.

Usage:
    # Train baseline model (all CIC features)
    python src/train_classifier.py

    # Train NF-standardized model (21 NetFlow-compatible features only)
    python src/train_classifier.py --nf
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import argparse
from pathlib import Path

from model import ThreatMLP

# ── CLI Arguments ──
parser = argparse.ArgumentParser(description="Train ThreatMLP threat classifier")
parser.add_argument(
    "--nf", action="store_true",
    help="Train using only the 21 NF-standardized features (NetFlow-compatible)"
)
parser.add_argument(
    "--epochs", type=int, default=30,
    help="Number of training epochs (default: 30)"
)
parser.add_argument(
    "--batch-size", type=int, default=1024,
    help="Training batch size (default: 1024)"
)
args = parser.parse_args()

# ── NF-Standardized Feature Set ──
# These 21 features are semantically shared between CICFlowMeter (CIC-IDS2018) and NetFlow/IPFIX (ToN-IoT)
NF_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",    # maps to IN_BYTES
    "Bwd Packets Length Total",    # maps to OUT_BYTES
    "Fwd Packet Length Max",       # maps to MAX_IP_PKT_LEN
    "Fwd Packet Length Min",       # maps to MIN_IP_PKT_LEN
    "Packet Length Max",           # maps to LONGEST_FLOW_PKT
    "Packet Length Min",           # maps to SHORTEST_FLOW_PKT
    "Flow Bytes/s",                # maps to SRC_TO_DST_SECOND_BYTES
    "Fwd Header Length",           # maps to SRC_TO_DST_AVG_THROUGHPUT
    "Bwd Header Length",           # maps to DST_TO_SRC_AVG_THROUGHPUT
    "Fwd PSH Flags",              # maps to TCP_FLAGS
    "Init Fwd Win Bytes",         # maps to TCP_WIN_MAX_IN
    "Init Bwd Win Bytes",         # maps to TCP_WIN_MAX_OUT
    "Fwd Avg Packets/Bulk",       # maps to RETRANSMITTED_IN_PKTS
    "Bwd Avg Packets/Bulk",       # maps to RETRANSMITTED_OUT_PKTS
    "Fwd Avg Bytes/Bulk",         # maps to RETRANSMITTED_IN_BYTES
    "Bwd Avg Bytes/Bulk",         # maps to RETRANSMITTED_OUT_BYTES
    "Subflow Fwd Packets",        # maps to NUM_PKTS_UP_TO_128_BYTES
    "Subflow Bwd Packets",        # maps to NUM_PKTS_1024_TO_1514_BYTES
]

# ── Configuration ──
mode_label = "NF-Standardized (21 features)" if args.nf else "Baseline (all features)"
print(f"Training mode: {mode_label}")
print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

# ── Load and preprocess dataset — all 10 days ──
print("\nLoading all CSE-CIC-IDS2018 CSV files...")
dataset_dir = Path(__file__).parent.parent / "datasets" / "CSE-CIC-IDS2018"
csv_files = sorted(dataset_dir.glob("*.csv"))
print(f"Found {len(csv_files)} CSV files")

frames = []
for csv_file in csv_files:
    print(f"  Loading {csv_file.name}...")
    chunk = pd.read_csv(csv_file, low_memory=False)
    chunk.columns = chunk.columns.str.strip()
    frames.append(chunk)
    print(f"    → {chunk.shape[0]:,} rows, {chunk.shape[1]} cols")

df = pd.concat(frames, ignore_index=True)
del frames  # free memory
print(f"Combined dataset: {df.shape[0]:,} rows, {df.shape[1]} cols")

# Drop Timestamp column if present
if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])

# Replace inf/-inf with NaN and drop rows with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# Drop duplicate rows to prevent data leakage during train/test split
df = df.drop_duplicates()

print(f"Dataset shape after preprocessing: {df.shape}")

# ── Select features based on mode ──
if args.nf:
    # Use only the 21 NF-standardized features
    available_nf = [f for f in NF_FEATURES if f in df.columns]
    missing_nf = [f for f in NF_FEATURES if f not in df.columns]
    if missing_nf:
        print(f"WARNING: Missing NF features in dataset: {missing_nf}")
    print(f"Using {len(available_nf)} NF-standardized features")
    X = df[available_nf].values
else:
    # Use all features except Label
    X = df.drop(columns=["Label"]).values

y = df["Label"].values
print(f"Feature matrix shape: {X.shape}")

# ── Encode labels ──
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")
print(f"Classes: {label_encoder.classes_}")

# ── Split data: 80/20 with stratification ──
print("Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# ── Scale features AFTER split — fit only on training data ──
print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit only on training data
X_test = scaler.transform(X_test)         # transform test using train stats

# ── Convert to PyTorch tensors ──
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# ── Initialize model ──
input_dim = X_train.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ThreatMLP(input_dim, num_classes).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ── Compute class weights for balanced loss ──
print("Computing class weights...")
class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_train), y=y_train
)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

# ── Loss, optimizer, and scheduler ──
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, verbose=True
)

# ── DataLoader ──
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# ── Training loop ──
print(f"\nTraining MLP classifier ({args.epochs} epochs)...")
best_loss = float("inf")
output_dir = Path(__file__).parent.parent / "experiments"
output_dir.mkdir(parents=True, exist_ok=True)

# File names based on mode
suffix = "_nf" if args.nf else ""

model.train()
for epoch in range(args.epochs):
    total_loss = 0.0
    num_batches = 0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    scheduler.step(avg_loss)
    print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_path = output_dir / f"threat_mlp{suffix}_best.pth"
        torch.save(model.state_dict(), best_path)

# ── Evaluate on test set ──
print("\nEvaluating on test set...")
model.eval()
with torch.no_grad():
    X_test_device = X_test_tensor.to(device)
    y_test_device = y_test_tensor.to(device)
    
    outputs = model(X_test_device)
    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    y_test_np = y_test_device.cpu().numpy()
    
    print("\nClassification Report:")
    print(classification_report(
        y_test_np, predictions,
        target_names=label_encoder.classes_
    ))

# ── Save model artifacts ──
print("\nSaving model artifacts...")

model_path = output_dir / f"threat_mlp{suffix}.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

encoder_path = output_dir / f"label_encoder{suffix}.joblib"
scaler_path = output_dir / f"standard_scaler{suffix}.joblib"

joblib.dump(label_encoder, encoder_path)
joblib.dump(scaler, scaler_path)

print(f"LabelEncoder saved to: {encoder_path}")
print(f"StandardScaler saved to: {scaler_path}")

print(f"\nTraining complete! Mode: {mode_label}")
