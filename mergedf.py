import pandas as pd

# Dataframes without a common key
df1 = pd.DataFrame({
    'id': [1, 2, 3, 1, 2, 3],
    'B': [4, 5, 6, 10, 2, 1]
})

df2 = pd.DataFrame({
    'id': [1, 2, 3],
    'D': [10, 11, 12]
})

# Concatenating along columns
concatenated_df = pd.merge(df1, df2, on='id', how='outer')

print(concatenated_df)
