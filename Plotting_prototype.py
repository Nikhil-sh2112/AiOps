import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Example dataset (e.g., CPU usage or network traffic over time)
data = np.array([50, 51, 52, 53, 200, 55, 56, 57, 58, 60]).reshape(-1, 1)

# Initialize Isolation Forest model for anomaly detection
model = IsolationForest(contamination=0.1)  # 10% outliers
model.fit(data)

# Predict anomalies: -1 indicates anomaly, 1 indicates normal
predictions = model.predict(data)

# Plotting the results
plt.plot(data, label="System Metric")
plt.scatter(np.arange(len(data)), data, c=predictions, cmap="coolwarm", label="Anomalies")
plt.title("Anomaly Detection in System Metric")
plt.legend()
plt.show()
