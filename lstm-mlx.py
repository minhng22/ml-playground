import pandas as pd
import time
import numpy as np
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error
import mlx.core as mx

data = pd.read_csv('./data/HMC.csv')

##----------------- Data -----------------##
# process data
prod = np.array(data['Open'].values)

train = prod[:870]
test = prod[870:]
print(
    f"shape of train: {train.shape} "
    f"shape of test: {test.shape}"
    )

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
X = mx.array(X)
y = mx.array(y)
print(f"X shape {X.shape} y shape {y.shape}")

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def __call__(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)

        out = out[:, -1, :] # get last output of the sequence
        return mx.reshape(out, (out.shape[0], 1, 1)) # convert to 3D tensor

input_size = 1
hidden_size = 50
output_size = 1
batch_size = 32
num_epochs = 100
learning_rate = 1e-1

model = SimpleLSTM(input_size, hidden_size, output_size)
mx.eval(model.parameters())

tic = time.time()

# Get a function which gives the loss and gradient of the
# loss with respect to the model's trainable parameters
def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Instantiate the optimizer
optimizer = optim.SGD(learning_rate=learning_rate)

for e in range(num_epochs):
    if e % 10 == 0:
        print(f'Epoch {e}/100')
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size, :, :]
        y_batch = y[i:i+batch_size, :, :]

        loss, grads = loss_and_grad_fn(model, X_batch, y_batch)

        # Update the optimizer state and model parameters
        # in a single call
        optimizer.update(model, grads)

        # Force a graph evaluation
        mx.eval(model.parameters(), optimizer.state)

test = torch.tensor(test, dtype=torch.float32).reshape(-1, 10, 1)
test_x = test[:, :9, :]
test_y = test[:, 9:, :]
print(f'test_x shape: {test_x.shape}, test_y shape: {test_y.shape}')

model.eval()

from sklearn.metrics import r2_score, mean_squared_error

prediction = model(mx.array(test_x))
prediction = np.array(prediction, copy=False)
print(f'Predicted shape: {prediction.shape}')

test_y = test_y.reshape(-1, 1)
prediction = prediction.reshape(-1, 1)
print(f'test_y shape: {test_y.shape}, prediction shape: {prediction.shape}')

r2 = r2_score(test_y, prediction)
mse = mean_squared_error(test_y, prediction)

print(f'R² score: {r2:.4f}')
print(f'MSE score: {mse:.4f}')

print(f'Time elapsed: {time.time() - tic:.2f} seconds')