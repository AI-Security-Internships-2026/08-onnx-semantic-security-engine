"""
Train ThreatMLP classifier — Supports both Baseline (76-feature) and NF-Standardized (13-feature) modes.

Usage:
    # Train baseline model (all CIC features)
    python src/train_classifier.py

    # Train NF-standardized model (13 NetFlow-compatible features only)
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

from model import ThreatMLP, ThreatCNN1D

# ── CLI Arguments ──
parser = argparse.ArgumentParser(description="Train threat classifier (MLP or CNN1D)")
parser.add_argument(
    "--nf", action="store_true",
    help="Train using only the 13 NF-standardized features (NetFlow-compatible)"
)
parser.add_argument(
    "--arch", type=str, default="mlp", choices=["mlp", "cnn1d"],
    help="Model architecture: 'mlp' or 'cnn1d' (default: 'mlp')"
)
parser.add_argument(
    "--epochs", type=int, default=30,
    help="Number of training epochs (default: 30)"
)
parser.add_argument(
    "--batch-size", type=int, default=1024,
    help="Training batch size (default: 1024)"
)
parser.add_argument(
    "--max-samples", type=int, default=None,
    help="Maximum samples to load (default: None, loads all)"
)
args = parser.parse_args()

# ── NF-Standardized Feature Set ──
# These 13 features are semantically shared between CICFlowMeter (CIC-IDS2018) and NetFlow/IPFIX (ToN-IoT)
NF_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",    # maps to IN_BYTES
    "Bwd Packets Length Total",    # maps to OUT_BYTES
    "Packet Length Max",           # maps to LONGEST_FLOW_PKT
    "Packet Length Min",           # maps to SHORTEST_FLOW_PKT
    "Protocol",                    # maps to PROTOCOL
    "Fwd Packet Length Max",       # maps to MAX_IP_PKT_LEN
    "Fwd Packet Length Min",       # maps to MIN_IP_PKT_LEN
    "Flow Bytes/s",                # maps to SRC_TO_DST_SECOND_BYTES
    "Init Fwd Win Bytes",          # maps to TCP_WIN_MAX_IN
    "Init Bwd Win Bytes",          # maps to TCP_WIN_MAX_OUT
]

# ── Configuration ──
mode_label = f"NF-Standardized (13 features) [{args.arch.upper()}]" if args.nf else f"Baseline (all features) [{args.arch.upper()}]"
print(f"Training mode: {mode_label}")
print(f"Architecture: {args.arch.upper()}")
print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

# ── Load and preprocess dataset — all 10 days ──
print("\nLoading CSE-CIC-IDS2018 data files...")
base_datasets_dir = Path(__file__).parent.parent / "datasets"
dataset_dir = base_datasets_dir / "CIC-IDS2018"
if not dataset_dir.exists() or not list(dataset_dir.glob("*.parquet")):
    dataset_dir = base_datasets_dir / "CSE-CIC-IDS2018"

parquet_files = sorted(dataset_dir.glob("*.parquet"))
csv_files = sorted(dataset_dir.glob("*.csv"))

frames = []
if parquet_files:
    print(f"Found {len(parquet_files)} Parquet files in {dataset_dir}")
    for pq_file in parquet_files:
        print(f"  Loading {pq_file.name}...")
        chunk = pd.read_parquet(pq_file)
        chunk.columns = chunk.columns.str.strip()
        frames.append(chunk)
        print(f"    -> {chunk.shape[0]:,} rows, {chunk.shape[1]} cols")
elif csv_files:
    print(f"Found {len(csv_files)} CSV files in {dataset_dir}")
    for csv_file in csv_files:
        print(f"  Loading {csv_file.name}...")
        chunk = pd.read_csv(csv_file, low_memory=False)
        chunk.columns = chunk.columns.str.strip()
        frames.append(chunk)
        print(f"    -> {chunk.shape[0]:,} rows, {chunk.shape[1]} cols")
else:
    raise FileNotFoundError(f"No parquet or csv files found in {dataset_dir}")

df = pd.concat(frames, ignore_index=True)
del frames  # free memory

if args.max_samples and len(df) > args.max_samples:
    print(f"Subsampling dataset to {args.max_samples:,} rows...")
    df = df.sample(n=args.max_samples, random_state=42)

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
    available_nf = [f for f in NF_FEATURES if f in df.columns]
    missing_nf = [f for f in NF_FEATURES if f not in df.columns]
    if missing_nf:
        print(f"WARNING: Missing NF features in dataset: {missing_nf}")
    print(f"Using {len(available_nf)} NF-standardized features")
    X = df[available_nf].values
else:
    X = df.drop(columns=["Label"]).values

y = df["Label"].values
print(f"Feature matrix shape: {X.shape}")

output_dir = Path(__file__).parent.parent / "experiments"
output_dir.mkdir(parents=True, exist_ok=True)
suffix = "_nf" if args.nf else ""
model_prefix = f"threat_{args.arch}"

encoder_path = output_dir / f"label_encoder{suffix}.joblib"
scaler_path = output_dir / f"standard_scaler{suffix}.joblib"

# ── Encode labels ──
if encoder_path.exists():
    print(f"Loading existing LabelEncoder from {encoder_path}...")
    label_encoder = joblib.load(encoder_path)
    # Filter any unexpected labels if necessary
    y_encoded = label_encoder.transform(y)
else:
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

# ── Scale features AFTER split ──
if scaler_path.exists():
    print(f"Loading existing StandardScaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
else:
    print("Scaling features (fit on train)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# ── Convert to PyTorch tensors ──
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# ── Initialize model ──
input_dim = X_train.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if args.arch == "cnn1d":
    model = ThreatCNN1D(input_dim, num_classes).to(device)
else:
    model = ThreatMLP(input_dim, num_classes).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters ({args.arch.upper()}): {total_params:,}")

# ── Compute class weights for balanced loss ──
print("Computing class weights...")
counts = np.bincount(y_train, minlength=num_classes)
total_samples = len(y_train)
weights = np.where(counts > 0, total_samples / (num_classes * np.maximum(counts, 1)), 1.0).astype(np.float32)
class_weights_tensor = torch.FloatTensor(weights).to(device)

# ── Loss, optimizer, and scheduler ──
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# ── DataLoader ──
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# ── Training loop ──
print(f"\nTraining {args.arch.upper()} classifier ({args.epochs} epochs)...")
best_loss = float("inf")

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
        best_path = output_dir / f"{model_prefix}{suffix}_best.pth"
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
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    ))

# ── Save model artifacts ──
print("\nSaving model artifacts...")

model_path = output_dir / f"{model_prefix}{suffix}.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

encoder_path = output_dir / f"label_encoder{suffix}.joblib"
scaler_path = output_dir / f"standard_scaler{suffix}.joblib"

joblib.dump(label_encoder, encoder_path)
joblib.dump(scaler, scaler_path)

print(f"LabelEncoder saved to: {encoder_path}")
print(f"StandardScaler saved to: {scaler_path}")

print(f"\nTraining complete! Mode: {mode_label}")
