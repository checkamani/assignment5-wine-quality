from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

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
    "sulfates",
    "alcohol"
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(request.form.get(f)) for f in FEATURES]
    X = np.array(values).reshape(1, -1)
    pred = model.predict(X)[0]
    return render_template("result.html", prediction=round(pred,2))

if __name__ == "__main__":
    app.run(debug=True)
