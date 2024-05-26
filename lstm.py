import pandas as pd
import time
import numpy as np

data = pd.read_csv('./data/realestate.csv')

##----------------- Data -----------------##
# process data
prod = np.array(data['Sale Amount'].values)

train = prod[:8700]
test = prod[8700:]
print(
    f"shape of train: {train.shape} "
    f"shape of test: {test.shape}"
    )

##----------------- Model -----------------##
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Create a DataLoader
# Prepare sequences for training
def create_sequences(data, len_obs, len_pred):
    xs, ys = [], []
    for i in range(len(data) - len_obs - len_pred):
        x = data[i:i+len_obs]
        y = data[i+len_obs : i+len_obs+len_pred]
        xs.append(x)
        ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    xs = xs.reshape(-1, len_obs, 1)
    ys = ys.reshape(-1, len_pred, 1)
    print(f'xs shape: {xs.shape}, ys shape: {ys.shape}')
    return xs, ys

len_obs = 9
len_pred = 1
X, y = create_sequences(train, len_obs, len_pred)

# Check that MPS is available
device = torch.device("cpu")
# un-comment to use MPS
# if not torch.backends.mps.is_available():
#     if not torch.backends.mps.is_built():
#         print("MPS not available because the current PyTorch install was not "
#               "built with MPS enabled.")
#     else:
#         print("MPS not available because the current MacOS version is not 12.3+ "
#               "and/or you do not have an MPS-enabled device on this machine.")

# else:
#     print("MPS is available")
#     device = torch.device("mps")

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

input_size = 1
hidden_size = 50
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100

tic = time.time()

for epoch in range(num_epochs):
    for batch in dataloader:
        X_batch, y_batch = batch

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, torch.reshape(y_batch, (y_batch.shape[0], 1)))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

test = torch.tensor(test, dtype=torch.float32).reshape(-1, 10, 1)
test_x = test[:, :9, :].to(device)
test_y = test[:, 9:, :]
print(f'test_x shape: {test_x.shape}, test_y shape: {test_y.shape}')

model.eval()

from sklearn.metrics import r2_score, mean_squared_error

with torch.no_grad():
    prediction = model(test_x).cpu().numpy()
    print(f'Predicted shape: {prediction.shape}')

    test_y = test_y.reshape(-1, 1)
    prediction = prediction.reshape(-1, 1)
    print(f'test_y shape: {test_y.shape}, prediction shape: {prediction.shape}')

    r2 = r2_score(test_y, prediction)
    mse = mean_squared_error(test_y, prediction)

    print(f'R² score: {r2:.4f}')
    print(f'MSE score: {mse:.4f}')

    print(f'Time elapsed: {time.time() - tic:.2f} seconds')


