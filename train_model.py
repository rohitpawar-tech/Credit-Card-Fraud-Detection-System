import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("archive (1)/fraudTrain.csv")

# Select features
features = ["amt", "merch_lat", "merch_long", "city_pop", "unix_time"]
X = df[features]
y = df["is_fraud"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Save model + scaler
with open("model/fraud_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("✅ Model trained and saved successfully!")
