import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pre-trained Logistic Regression model
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature order must exactly match the order used during training
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            values = [float(request.form[feature]) for feature in FEATURE_ORDER]
            input_array = np.array(values).reshape(1, -1)

            result = model.predict(input_array)[0]
            prediction = "High Risk of Heart Disease" if result == 1 else "Low Risk of Heart Disease"

            # Logistic Regression supports predict_proba — show model confidence
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_array)[0]
                probability = round(max(proba) * 100, 2)

        except (ValueError, KeyError) as e:
            error = "Please fill in all fields with valid numeric values."

    return render_template("index.html", prediction=prediction, probability=probability, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)