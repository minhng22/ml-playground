import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter

# Step 1: Generate the data with more individuals
np.random.seed(42)

n_individuals = 10
n_records_per_individual = 5

data_list = []

for i in range(n_individuals):
    start_times = np.arange(0, n_records_per_individual * 5, 5)
    stop_times = start_times + 5
    event_times = np.random.randint(0, 2, size=n_records_per_individual)
    ages = np.random.randint(20, 60, size=n_records_per_individual)
    blood_pressures = np.random.randint(110, 160, size=n_records_per_individual)

    for j in range(n_records_per_individual):
        data_list.append([i + 1, start_times[j], stop_times[j], event_times[j], ages[j], blood_pressures[j]])

data = pd.DataFrame(data_list, columns=['id', 'start', 'stop', 'event', 'age', 'blood_pressure'])

# Step 2: Fit the Cox time-varying model
ctv = CoxTimeVaryingFitter()
ctv.fit(data, id_col='id', start_col='start', stop_col='stop', event_col='event')

# Step 3: Predict the survival function for new data
new_data = pd.DataFrame({
    'id': [3, 3, 4, 4],
    'start': [0, 5, 0, 5],
    'stop': [5, 10, 5, 10],
    'age': [30, 31, 40, 41],
    'blood_pressure': [130, 135, 150, 155]
})

# Calculate partial hazard for new data
partial_hazard = ctv.predict_partial_hazard(new_data)

# Aggregate partial hazards by individual
new_data['partial_hazard'] = partial_hazard.values
aggregated_hazard = new_data.groupby('id')['partial_hazard'].sum()

# To calculate the survival function, combine the aggregated hazard with the baseline survival function
baseline_survival = ctv.baseline_survival_

# Calculate the survival function for each individual in new_data
time_points = baseline_survival.index
predicted_times = []

for idx, (ind, hazard) in enumerate(aggregated_hazard.items()):
    survival_function = baseline_survival ** hazard
    survival_function = survival_function.reset_index()
    survival_function.columns = ['timeline', 'survival']

    # Find the time where the survival probability drops below a threshold (e.g., 0.5 for median survival time)
    median_survival_time = survival_function[survival_function['survival'] <= 0.5].iloc[0]['timeline']
    predicted_times.append((ind, median_survival_time))

# Print the predicted time to event (median survival time) for each individual
for ind, tte in predicted_times:
    print(f"Predicted time to event for individual {ind}: {tte}")

