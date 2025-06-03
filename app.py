from flask import Flask, request, jsonify
from flask_cors import CORS  # add this import

app = Flask(__name__)
CORS(app)  # enable CORS for your app

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load("diabetes_model.pkl")

@app.route('/')
def home():
    return "Diabetes Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    features = [data.get(key) for key in 
                ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]

    if None in features:
        return jsonify({"error": "Missing feature(s). Please provide all 8 features."}), 400

    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability)
    })

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

