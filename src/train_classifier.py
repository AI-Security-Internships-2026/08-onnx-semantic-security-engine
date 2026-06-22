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
from pathlib import Path

# Load and preprocess dataset
print("Loading dataset...")
dataset_path = Path(__file__).parent.parent / "datasets" / "CSE-CIC-IDS2018" / "02-14-2018.csv"
df = pd.read_csv(dataset_path)
df.columns = df.columns.str.strip()

# Drop Timestamp column if present
if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])

# Replace inf/-inf with NaN and drop rows with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# Drop duplicate rows to prevent data leakage during train/test split
df = df.drop_duplicates()

print(f"Dataset shape after preprocessing: {df.shape}")

# Separate features and labels
print("Preparing features and labels...")
X = df.drop(columns=["Label"]).values
y = df["Label"].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")
print(f"Classes: {label_encoder.classes_}")

# Split data FIRST: 80/20 with stratification
print("Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Scale features AFTER split — fit only on training data to prevent leakage
print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit only on training data
X_test = scaler.transform(X_test)         # transform test using train stats

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Define MLP model
class ThreatMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ThreatMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

# Initialize model and move to device
input_dim = X_train.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ThreatMLP(input_dim, num_classes).to(device)

# Compute class weights for balanced loss
print("Computing class weights...")
class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_train), y=y_train
)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Create DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

# Training loop
print("\nTraining MLP classifier...")
model.train()
for epoch in range(10):
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
    print(f"Epoch {epoch + 1}/10 - Loss: {avg_loss:.4f}")

# Evaluate on test set
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

# Save model state dict
print("\nSaving model artifacts...")
output_dir = Path(__file__).parent.parent / "experiments"
output_dir.mkdir(parents=True, exist_ok=True)

model_path = output_dir / "threat_mlp.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

# Save LabelEncoder and StandardScaler
encoder_path = output_dir / "label_encoder.joblib"
scaler_path = output_dir / "standard_scaler.joblib"

joblib.dump(label_encoder, encoder_path)
joblib.dump(scaler, scaler_path)

print(f"LabelEncoder saved to: {encoder_path}")
print(f"StandardScaler saved to: {scaler_path}")

print("\nTraining complete!")
