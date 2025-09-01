import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Read log file
log_file_path = "system_logs.txt"  # Update with your file path if needed
with open(log_file_path, "r") as file:
    logs = file.readlines()

# Parse logs into a structured DataFrame
data = []
for log in logs:
    parts = log.strip().split(" ", 3)  # Ensure the message part is captured fully
    if len(parts) < 4:
        continue  # Skip malformed lines
    timestamp = parts[0] + " " + parts[1]
    level = parts[2]
    message = parts[3]
    data.append([timestamp, level, message])
#     data = []: Prepare an empty list to store structured log data.
# Loop through each line from the log:
# log.strip(): Removes any whitespace or newline characters.
# split(" ", 3): Splits the line into four parts (date, time, level, and message), ensuring the message captures all remaining spaces.
# if len(parts) < 4: If the line doesn't have enough information, skip it.
# timestamp: Combines date and time as the timestamp.
# level: Extracts the log level (e.g., INFO, ERROR).
# message: Extracts the log message.
# Appends a structured entry [timestamp, level, message] to data.

df = pd.DataFrame(data, columns=["timestamp", "level", "message"])

# Convert timestamp to datetime format for sorting
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Assign numeric scores to log levels
level_mapping = {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
df["level_score"] = df["level"].map(level_mapping)

# Add message length as a new feature
df["message_length"] = df["message"].apply(len)

# AI Model for Anomaly Detection (Isolation Forest)
model = IsolationForest(contamination=0.1, random_state=42)  # Lower contamination for better accuracy
df["anomaly"] = model.fit_predict(df[["level_score", "message_length"]])

# Mark anomalies in a readable format
df["is_anomaly"] = df["anomaly"].apply(lambda x: "❌ Anomaly" if x == -1 else "✅ Normal")

# Print only detected anomalies
anomalies = df[df["is_anomaly"] == "❌ Anomaly"]
print("\n🔍 **Detected Anomalies:**\n", anomalies)

