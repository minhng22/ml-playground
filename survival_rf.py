from sksurv.datasets import load_gbsg2

X, y = load_gbsg2()
print(X)
print(y[0] , len(y))