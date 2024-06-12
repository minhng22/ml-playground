import numpy as np
import matplotlib.pyplot as plt

# Generate a sample numpy array
A = 1000  # number of students
B = 30    # number of days
np.random.seed(0)

# Simulate scores with mean around 70 and fluctuation
scores = 70 + 10 * np.random.randn(A, B)

# Calculate the mean and standard deviation for each day
mean_scores = scores.mean(axis=0)
std_scores = scores.std(axis=0)

# Plot the mean scores with a shaded area for the standard deviation
plt.figure(figsize=(12, 6))
plt.plot(mean_scores, label='Mean Score')
plt.fill_between(range(B), mean_scores - std_scores, mean_scores + std_scores, alpha=0.2, label='Standard Deviation')

plt.title('Mean Scores of Students Over Days')
plt.xlabel('Days')
plt.ylabel('Scores')
plt.legend()
plt.show()
