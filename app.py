from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# -------------------------------
# Dummy Training (for demo)
# -------------------------------
X_train = np.array([
    [10, 20, 30, 50000, 1325376000],
    [2000, 50, -20, 1000, 1325379000],
    [15, 18, 70, 300000, 1325380000],
    [3000, 60, -10, 500, 1325382000]
])

y_train = np.array([0, 1, 0, 1])  # 0 = Legit, 1 = Fraud

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        amount = float(request.form['amount'])
        lat = float(request.form['lat'])
        lon = float(request.form['lon'])
        city_pop = float(request.form['city_pop'])
        unix_time = float(request.form['unix_time'])

        data = np.array([[amount, lat, lon, city_pop, unix_time]])
        data_scaled = scaler.transform(data)

        prediction = model.predict(data_scaled)[0]

        if prediction == 1:
            result = "❌ FRAUDULENT TRANSACTION"
        else:
            result = "✅ LEGITIMATE TRANSACTION"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
