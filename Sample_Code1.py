import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Data
data = np.array([50, 51, 52, 53, 200, 55, 56, 57, 58, 60]).reshape(-1, 1)

# Model
model = IsolationForest(contamination=0.1)
model.fit(data)
predictions = model.predict(data)

# Define colors manually: red for anomalies (-1), blue for normal (1)
colors = ['red' if pred == -1 else 'blue' for pred in predictions]

# Plot
plt.plot(data, label="System Metric")
plt.scatter(np.arange(len(data)), data.flatten(), c=colors, label="Data Points")

# Add custom legend for anomalies and normal points
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='blue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Anomaly', markerfacecolor='red', markersize=8)
]
plt.legend(handles=legend_elements, title="Data Point Type")

plt.title("Anomaly Detection in System Metric")
plt.show()
