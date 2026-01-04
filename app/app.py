from flask import Flask, render_template, request
import joblib
import numpy as np
import sys
import os

# ensure project root is on sys.path so sibling packages (utils, model) are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.differential import top_diseases
from utils.recommend import recommend

app = Flask(__name__)

model = joblib.load("model/model.pkl")
encoder = joblib.load("model/label_encoder.pkl")
scaler = joblib.load("model/scaler.pkl")

features = ["age","gender","fever","cough","fatigue",
            "bp","glucose","cholesterol","bmi","smoking"]

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        print(request.form)
        data = np.array([[float(request.form.get(f, 0)) for f in features]])
        data = scaler.transform(data)

        results = top_diseases(model, data, encoder)
        tests = recommend(results[0]['disease'])

        return render_template("result.html",
                               results=results,
                               tests=tests)
    return render_template("index.html")

app.run(debug=True)
