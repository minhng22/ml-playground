from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def test_1():
    a = MinMaxScaler((0, 1))
    b = a.fit_transform(np.zeros((100, 1)))

    c = a.fit_transform(np.zeros((100, 1)))

    d = a.transform(np.zeros((7, 1)))
    print(d.shape)


def test_2():
    scaler = MinMaxScaler((0, 1))
    b = scaler.fit_transform(np.zeros((100, 8)))

    subset_scaler = MinMaxScaler()
    subset_scaler.min_ = scaler.min_[:5]
    subset_scaler.scale_ = scaler.scale_[:5]
    subset_scaler.data_min_ = scaler.data_min_[:5]
    subset_scaler.data_max_ = scaler.data_max_[:5]
    subset_scaler.data_range_ = scaler.data_range_[:5]
    subset_scaler.feature_range = scaler.feature_range

    d = subset_scaler.transform(np.zeros((100, 5)))
    print(d.shape)

    # should fail
    c = scaler.transform(np.zeros((100, 5)))
    print(c.shape)

def test_3():
    scaler = StandardScaler()
    scaler.fit(np.random.random((2, 10)))

    b = np.random.random((2, 5))

    # use subsetscaler
    subset_scaler = StandardScaler()
    subset_scaler.mean_ = scaler.mean_[:5]
    subset_scaler.var_ = scaler.var_[:5]
    subset_scaler.scale_ = scaler.scale_[:5]

    b1 = subset_scaler.transform(b)
    print(f"b1 {b1}")

    # use zero
    b2 = scaler.transform(np.concatenate((b, np.zeros((2, 5))), axis=1))[:, :5]
    print(f"b2 {b2}")

    # use rand
    # notice how b2 stays the same
    b2 = scaler.transform(np.concatenate((b, np.random.random((2, 5))), axis=1))[:, :5]
    print(f"b2 {b2}")


test_3()
