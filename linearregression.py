import numpy as np

import pandas as pd
import time

data = pd.read_csv('./data/linearregression.csv')
data['DATE'] = pd.to_datetime(data['DATE'])

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

##----------------- Data -----------------##
# process data
y = np.array(data['IPG2211A2N'].values)
y = np.reshape(y, (-1, 1))

X = np.array(data['DATE'].dt.year)
X = np.reshape(X, (-1, 1))

print('shape of X:', X.shape, 'shape of y:', y.shape)

##----------------- PyTorch MPS -----------------##
# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    print("MPS is available")
    mps_device = torch.device("mps")

tic = time.time()
X_train = torch.from_numpy(X.astype(np.float32))
y_train = torch.from_numpy(y.astype(np.float32))

X_train = X_train.to(mps_device)
y_train = y_train.to(mps_device)

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
model.to(mps_device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

y_pred = model(X_train).detach().cpu().numpy()

print('shape of y_pred:', y_pred.shape, 'shape of y:', y.shape)
print('y_pred:', y_pred)

toc = time.time()
print(f'Elapsed time: {toc-tic:.4f} seconds')

plt.scatter(y_pred, y)
plt.show()

##----------------- PyTorch -----------------##
# tic = time.time()
# X_train = torch.from_numpy(X.astype(np.float32))
# y_train = torch.from_numpy(y.astype(np.float32))

# class LinearRegression(nn.Module):
#     def __init__(self):
#         super(LinearRegression, self).__init__()
#         self.linear = nn.Linear(1, 1)

#     def forward(self, x):
#         return self.linear(x)

# model = LinearRegression()

# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)

# num_epochs = 1000
# for epoch in range(num_epochs):
#     outputs = model(X_train)
#     loss = criterion(outputs, y_train)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if (epoch+1) % 100 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# y_pred = model(X_train).detach().numpy()

# print('shape of y_pred:', y_pred.shape, 'shape of y:', y.shape)
# print('y_pred:', y_pred)

# toc = time.time()
# print(f'Elapsed time: {toc-tic:.4f} seconds')

# plt.scatter(y_pred, y)
# plt.show()

# #----------------- Scikit-learn -----------------##
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.linear_model import LinearRegression


# model = LinearRegression()
# model.fit(X, y)

# y_pred = model.predict(X)
# print('shape of y_pred:', y_pred.shape)

# plt.scatter(X, y, label='Data')
# plt.scatter(X, y_pred, color='black', label='Prediction')

# plt.plot(X, y_pred, color='red', label='Trend Line')

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.show()
