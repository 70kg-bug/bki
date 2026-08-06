import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import copy

# --- 1. GPU Isolation and Device Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Targeting device: {device} (mapped to physical GPU 7)")

# --- 2. Load and Preprocess Data ---
print("Loading patient waveform files...")
data_files = [f"data/waveform_data/P0{i}Waveform.xlsx" for i in range(1, 8)]
death_labels = [0, 1, 0, 0, 0, 0, 1]  # P02 and P07 had death labels = 1

dfs = []
for i, path in enumerate(data_files):
    df = pd.read_excel(path)
    df["in_hospital_death"] = death_labels[i]
    df["patient_id"] = i + 1
    dfs.append(df)

merged_df = pd.concat(dfs, axis=0, ignore_index=True)
print(f"Total dataset shape: {merged_df.shape}")

# Scale features globally
features = ['Flow [l/s]', 'Pao [cm H2O]', 'Pes [cm H2O]']
scaler = StandardScaler()
merged_df[features] = scaler.fit_transform(merged_df[features])

# --- 3. Sequence Window Generation (Prevent Data Leakage) ---
def create_windows_per_patient(df, patient_ids, window_size=60, stride=10):
    X_windows = []
    y_labels = []
    
    for pid in patient_ids:
        pdf = df[df['patient_id'] == pid]
        X = pdf[features].to_numpy()
        y = pdf['in_hospital_death'].iloc[0]
        
        for i in range(0, len(X) - window_size + 1, stride):
            X_windows.append(X[i:i+window_size])
            y_labels.append(y)
            
    return np.array(X_windows, dtype=np.float32), np.array(y_labels, dtype=np.float32)

# Group-based split (Patient-level) to avoid time-series data leakage
train_patients = [1, 3, 4, 5, 7]  # Includes positive case P07
val_patients = [6]                # Validation on P06
test_patients = [2]               # Test on positive case P02

window_size = 100
stride = 20

print("Generating time-series sliding windows...")
X_train, y_train = create_windows_per_patient(merged_df, train_patients, window_size, stride)
X_val, y_val = create_windows_per_patient(merged_df, val_patients, window_size, stride)
X_test, y_test = create_windows_per_patient(merged_df, test_patients, window_size, stride)

print(f"Train windows: {X_train.shape}, Val windows: {X_val.shape}, Test windows: {X_test.shape}")

# Convert to PyTorch tensors (N, Channels, Length) for 1D CNN
X_train_t = torch.tensor(X_train).transpose(1, 2)
y_train_t = torch.tensor(y_train).unsqueeze(1)
X_val_t = torch.tensor(X_val).transpose(1, 2)
y_val_t = torch.tensor(y_val).unsqueeze(1)
X_test_t = torch.tensor(X_test).transpose(1, 2)
y_test_t = torch.tensor(y_test).unsqueeze(1)

# DataLoaders (Increase batch size for H200)
batch_size = 512
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

# --- 4. Advanced Time-Series Architecture (1D CNN) ---
class BKI_CNN(nn.Module):
    def __init__(self):
        super(BKI_CNN, self).__init__()
        # Input: (Batch, 3 channels, 100 timesteps)
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        features = self.conv_blocks(x).squeeze(2)
        return self.fc(features)

model = BKI_CNN().to(device)

# --- 5. Class Imbalance \u0026 Optimization Setup ---
# Calculate ratio for positive weight
num_neg = np.sum(y_train == 0)
num_pos = np.sum(y_train == 1)
pos_weight = torch.tensor([num_neg / (num_pos + 1e-5)], dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# --- 6. Training Loop with Early Stopping ---
epochs = 200
patience = 15
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
epochs_no_improve = 0

print("\nStarting Training with Early Stopping...")
for epoch in range(epochs):
    # Training Phase
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
        
    train_loss = running_loss / len(X_train)
    
    # Validation Phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            
    val_loss = val_loss / len(X_val)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    # Early Stopping Logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}!")
            break

# Load best weights
model.load_state_dict(best_model_wts)

# --- 7. Evaluation on Unseen Patient ---
print("\nEvaluating on pure unseen test patient (P02)...")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits)
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(y_batch.numpy())

y_pred_nn = (np.array(all_preds) >= 0.5).astype(int)
y_true = np.array(all_labels).astype(int)

print("\n--- Advanced PyTorch CNN Results (GPU 7) ---")
if len(np.unique(y_true)) > 1:
    print(f"ROC-AUC: {roc_auc_score(y_true, all_preds):.4f}")
print(f"Accuracy: {accuracy_score(y_true, y_pred_nn):.4f}")
print("Classification Report:")
print(classification_report(y_true, y_pred_nn, zero_division=0))

print("Saving advanced time-series model...")
torch.save(model.state_dict(), "bki_classifier_advanced_cnn.pt")
print("Saved to 'bki_classifier_advanced_cnn.pt'")
