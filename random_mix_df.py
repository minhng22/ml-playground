import pandas as pd

# Sample dataframe
data = {'id': [1, 1, 2, 2, 3, 3, 4, 4],
        'dead': [1, 1, 0, 1, 1, 1, 0, 0]}
df = pd.DataFrame(data)

# Group by 'id' and filter groups where all 'dead' values are 1
result = df.groupby('id').filter(lambda x: (x['dead'] == 1).all())

# Get the unique 'id's
id_list = result['id'].unique().tolist()

print(id_list)