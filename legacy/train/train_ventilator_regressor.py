import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure GPU index 7 is selected
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device targeted for Google Brain Data: {device}")

data_path = "data/train.csv"
if not os.path.exists(data_path):
    print(f"Error: Google Brain Dataset '{data_path}' not found. Place the Kaggle dataset in the data folder to run this.")
    exit(1)

print("Loading Google Brain dataset...")
df = pd.read_csv(data_path)
df_data = df.drop(columns=['pressure', 'id'])
labels = df['pressure'].to_numpy()

# Exclude breath_id from features
features = df_data.drop(columns='breath_id').to_numpy()

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, labels, train_size=0.8, random_state=42
)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=512, shuffle=False)

class BreathModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=5, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )
    def forward(self, x):
        return self.layer1(x)

model = BreathModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
print("Training BreathModel on GPU 7...")
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
    
    print(f"Epoch {epoch+1}/{epochs} - MSE Loss: {running_loss/len(X_train):.4f}")

# Evaluation
model.eval()
all_preds = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch)
        all_preds.extend(preds.cpu().numpy())

y_pred = np.array(all_preds)
print("\n--- PyTorch Regression Results (GPU 7) ---")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# --- Save Model for Git Version Control ---
print("\nSaving trained model to disk...")
torch.save(model.state_dict(), "ventilator_regressor_nn.pt")
print("Saved PyTorch model to 'ventilator_regressor_nn.pt'")

