from sklearn.preprocessing import MinMaxScaler
import numpy as np


a = MinMaxScaler((0, 1))
b = a.fit_transform(np.reshape([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (10, 1)))

c = a.transform(np.reshape([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], (10, 1)))

d = a.transform(np.reshape([2, 2, 2, 2, 2, 2, 2], (7, 1)))
print(d)