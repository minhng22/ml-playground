import pandas as pd

df_a = pd.DataFrame([{'id': 1, 'col_a': 1}, {'id': 2, 'col_a': 3}])
df_b = pd.DataFrame([{'id': 2, 'col_b': 1}, {'id': 3, 'col_b': 3}])
lab_df = pd.merge(df_a, df_b, on='id', how='left')

print(lab_df)