from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# --- Load model once at startup (and fail fast with a clear message) ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
model = None
model_error = None

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model_error = f"Failed to load model at {MODEL_PATH}: {e}"

# --- Feature names (must match your HTML form field names) ---
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
    "sulfates",
    "alcohol"
]

@app.route("/", methods=["GET"])
def home():
    # If the model didn't load, return immediately (prevents Heroku H12 timeouts)
    if model is None:
        return f"Model load error: {model_error}", 500
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return f"Model load error: {model_error}", 500

    try:
        values = []
        for f in FEATURES:
            v = request.form.get(f)
            if v is None or v == "":
                return f"Missing value for: {f}", 400
            values.append(float(v))

        X = np.array(values).reshape(1, -1)
        pred = float(model.predict(X)[0])

        return render_template("result.html", prediction=round(pred, 2))

    except Exception as e:
        return f"Prediction error: {e}", 500

if __name__ == "__main__":
    # Local run (Heroku uses gunicorn, but this keeps local testing correct)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True))
