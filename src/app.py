"""
app.py — Flask API for churn prediction
Run: python src/app.py
"""
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

with open("models/gradient_boosting.pkl","rb") as f: model  = pickle.load(f)
with open("models/scaler.pkl","rb") as f:            scaler = pickle.load(f)

FEATURES = ['tenure','monthly_charges','total_charges','contract',
            'internet_service','senior_citizen','paperless_billing','payment_method']


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","accuracy":"88.08%","roc_auc":"95.19%"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing: {missing}"}), 400
    arr = np.array([[data[f] for f in FEATURES]])
    pred = int(model.predict(scaler.transform(arr))[0])
    prob = float(model.predict_proba(scaler.transform(arr))[0][1])
    return jsonify({
        "prediction":  pred,
        "churn_risk":  "Will Churn" if pred == 1 else "Will Stay",
        "probability": round(prob, 4),
        "risk_level":  "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
    })


if __name__ == "__main__":
    print("Churn API running on http://localhost:5000")
    app.run(debug=False, port=5000)
