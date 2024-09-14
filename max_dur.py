import pandas as pd

# Sample DataFrame
data = {'subject_id': [1, 1, 2, 2, 3, 10017886, 10017886],
        'duration_in_days': [10, 15, 20, 5, 12, 133.66319444444446, 133.38194444444446]}
df = pd.DataFrame(data)

# Group by subject_id and keep the row with the maximum duration_in_days
df_max = df.loc[df.groupby('subject_id')['duration_in_days'].idxmax()]

# Display the result
print(df_max)