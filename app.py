from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("wine_quality_model.pkl")

FEATURES = [
    "fixed acidity","volatile acidity","citric acid","residual sugar",
    "chlorides","free sulfur dioxide","total sulfur dioxide",
    "density","pH","sulphates","alcohol","type"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(request.form[f]) for f in FEATURES]
    prediction = model.predict(np.array(values).reshape(1, -1))[0]
    return render_template("result.html", prediction=round(prediction,2))

if __name__ == "__main__":
    app.run()
