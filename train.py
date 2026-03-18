import numpy as np
import matplotlib.pyplot as plt
from model import LinearRegression

# Sample dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Train model
model = LinearRegression(lr=0.01, n_iters=1000)
model.fit(X, y)

# Predictions
predictions = model.predict(X)

# Plot results
plt.scatter(X, y, label="Actual Data")
plt.plot(X, predictions, label="Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression")
plt.show()
