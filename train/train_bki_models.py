import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tqdm import tqdm

# --- 1. GPU Isolation and Device Configuration ---
# Setting CUDA_VISIBLE_DEVICES to 7 makes GPU 7 the ONLY visible GPU to this script
# (referred to as "cuda:0" inside torch). This is clean and avoids memory conflicts.
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
    dfs.append(df)

merged_df = pd.concat(dfs, axis=0, ignore_index=True)
print(f"Total dataset shape: {merged_df.shape}")

# Features and target split
X = merged_df[['Flow [l/s]', 'Pao [cm H2O]', 'Pes [cm H2O]']].to_numpy()
y = merged_df['in_hospital_death'].to_numpy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, train_size=0.8, random_state=42, stratify=y
)

# --- 3. Scikit-Learn: Gradient Boosting Classifier ---
print("Training GradientBoostingClassifier with optimal parameters and Early Stopping...")
best_model = GradientBoostingClassifier(
    learning_rate=0.01,
    max_depth=4,
    n_estimators=10000,
    n_iter_no_change=20,     # Early stopping patience
    validation_fraction=0.1, # Use 10% of training data for early stopping validation
    random_state=42,
    verbose=1
)
best_model.fit(X_train, y_train)

y_pred_sklearn = best_model.predict(X_test)
print("\n--- Scikit-Learn Model Results ---")
print(f"Final Estimators Used (Early Stopping): {best_model.n_estimators_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_sklearn):.4f}")
print(classification_report(y_test, y_pred_sklearn))

# --- 4. PyTorch: Deep Learning Classifier on GPU 7 ---
print("Training PyTorch Classifier on GPU 7...")
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=512, shuffle=False)

class BKIClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Sigmoid applied via BCEWithLogitsLoss
        )
    def forward(self, x):
        return self.net(x)

model = BKIClassifierModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
for epoch in range(epochs):
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
    
    epoch_loss = running_loss / len(X_train)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# Evaluation
model.eval()
all_preds = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits)
        all_preds.extend(probs.cpu().numpy())

y_pred_nn = (np.array(all_preds) >= 0.5).astype(int)
print("\n--- PyTorch Neural Network Results (GPU 7) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nn):.4f}")
print(classification_report(y_test, y_pred_nn))

# --- 5. Save Models for Git Version Control ---
print("\nSaving trained models to disk...")
# Save PyTorch model weights (tensors)
torch.save(model.state_dict(), "bki_classifier_nn.pt")
print("Saved PyTorch model to 'bki_classifier_nn.pt'")

# Save scikit-learn model using pickle
import pickle
with open("bki_classifier_sklearn.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Saved scikit-learn model to 'bki_classifier_sklearn.pkl'")

