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

if __name__ == "__main__":
    app.run(debug=True)

