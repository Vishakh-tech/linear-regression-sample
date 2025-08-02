import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic data
X = np.random.rand(100, 1) * 10
y = 5 + 2 * X + np.random.randn(100, 1)

# Fit model
regressor = LinearRegression()
regressor.fit(X, y)

# Predict
y_pred = regressor.predict(X)

# Plot
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Prediction')
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
