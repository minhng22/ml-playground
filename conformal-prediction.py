import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class ConformalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.base_classifier.fit(X, y)
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['base_classifier'])

        # Input validation
        X = check_array(X)

        # Predict class probabilities
        class_probs = self.base_classifier.predict_proba(X)

        # Calculate p-values for each prediction
        p_values = []
        for probs in class_probs:
            p_value = 1.0 - np.max(probs)
            p_values.append(p_value)

        # Calculate significance levels (alpha levels)
        n = len(X)
        alpha_levels = np.arange(1, n + 1) / (n + 1)

        # Determine prediction sets based on alpha levels
        predictions = []
        for p_value, alpha in zip(p_values, alpha_levels):
            if p_value <= alpha:
                predictions.append((self.classes_[np.argmax(class_probs[len(predictions)])], alpha))

        return predictions

# Example usage:
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a conformal classifier
clf = ConformalClassifier(RandomForestClassifier(random_state=42))

# Fit the conformal classifier
clf.fit(X_train, y_train)

# Predict with confidence intervals
predictions = clf.predict(X_test)

# Print predicted classes and associated confidence intervals
for i, (pred_class, alpha) in enumerate(predictions):
    print(f"Instance {i}: Predicted class = {pred_class}, Significance level = {alpha}")

# Example: Evaluate coverage of the predicted intervals (should ideally cover around 95% if alpha is set to 0.05)
coverage = len([pred for pred in predictions if pred[1] >= 0.05]) / len(predictions)
print(f"Coverage of predicted intervals: {coverage:.2f}")
