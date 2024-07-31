import numpy as np

d = np.zeros((3, 1, 2))
e = d[:, :, 0]
print(e.shape)

# a = np.zeros((10, 2))
# b = np.zeros((10, 2))
#
# c = np.concatenate((a,b), axis=2)