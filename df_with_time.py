import pandas as pd

# Example DataFrame
data = {
    "subject_id": [10000764, 10000764, 10001034, 10001034, 10001034],
    "time": ["2132-10-14 20:15:00", "2133-01-10 10:30:00", "2132-05-15 13:00:00", "2133-07-18 08:45:00", "2134-08-01 09:00:00"],
    "egfr": [45.06, 50.00, 35.50, 30.25, 28.00],
    "has_esrd": [0, 0, 0, 1, 0]
}
df = pd.DataFrame(data)

# Convert 'time' to datetime
df['time'] = pd.to_datetime(df['time'])

# Find the first occurrence of has_esrd == 1 for each subject
first_esrd_time = df[df['has_esrd'] == 1].groupby('subject_id')['time'].min().reset_index()
first_esrd_time.columns = ['subject_id', 'first_esrd_time']

# Merge the first occurrence time back to the original DataFrame
df = df.merge(first_esrd_time, on='subject_id', how='left')

# Filter out rows where 'time' is after 'first_esrd_time'
df_filtered = df[(df['first_esrd_time'].isna()) | (df['time'] <= df['first_esrd_time'])].drop(columns=['first_esrd_time'])

print(df_filtered)
