from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model (file is in the SAME folder as app.py)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(MODEL_PATH)

FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(request.form.get(f)) for f in FEATURES]
        X = np.array(values).reshape(1, -1)
        pred = float(model.predict(X)[0])
        return render_template("result.html", prediction=round(pred, 2))
    except Exception as e:
        return f"Prediction error: {e}", 500

# NOTE: Do NOT run app.run() on Heroku (gunicorn runs it)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
