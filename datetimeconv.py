import pandas as pd

# Sample DataFrame
data = {
    'id': [1, 1, 1, 2, 2, 3, 3],
    'score': [85, 90, 88, 78, 80, 95, 92],
    'timestamp': ['2024-05-01', '2024-05-01', '2024-05-02', '2024-05-02', '2024-05-01', '2024-05-01', '2024-05-02']
}
df = pd.DataFrame(data)

# Convert 'timestamp' column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Group by 'id' and 'timestamp', then calculate the mean score for each group
average_scores = df.groupby(['id', 'timestamp'])['score'].mean().reset_index()

print(average_scores['timestamp'])
