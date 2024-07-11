import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

# Generate simulated data
np.random.seed(123)

# Simulate time-to-event data (T)
T = np.random.randint(1, 20, size=100)

# Simulate censoring indicator (C)
C = np.random.choice([0, 1], size=100, p=[0.3, 0.7])

# Simulate predictor variables (X)
X1 = np.random.normal(0, 1, size=100)
X2 = np.random.normal(0, 1, size=100)

# Create a pandas DataFrame for the data
data = pd.DataFrame({
    'Time': T,
    'Event': C,
    'Predictor1': X1,
    'Predictor2': X2
})

# Fit Cox Proportional Hazards model
cox = CoxPHFitter()
cox.fit(data, duration_col='Time', event_col='Event')

# Print the coefficients
print(cox.summary)

# Predict survival probabilities for new data
# Example: Predict survival at Time=10 with Predictor1=0.5 and Predictor2=-0.3
new_data = pd.DataFrame({
    'Predictor1': [0.5],
    'Predictor2': [-0.3]
})
survival_prob = cox.predict_survival_function(new_data, times=[10])
print("Predicted Survival Probability at Time=10:")
print(survival_prob)

# Plot survival curves if desired
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
cox.plot()
plt.title('Survival Curves')
plt.show()
