import pandas as pd

data = {
    'id': [1, 1, 1, 2, 2, 3, 3],
    'score': [85, 90, 88, 78, 80, 95, 92],
    'timestamp': ['2024-05-01', '2024-05-01', '2024-05-02', '2024-05-02', '2024-05-01', '2024-05-01', '2024-05-02']
}
df = pd.DataFrame(data)

df.interpolate()