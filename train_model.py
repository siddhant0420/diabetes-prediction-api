import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

data = pd.read_csv(url, header=None, names=columns)

# Split features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the trained model to a file
joblib.dump(model, "diabetes_model.pkl")
print("Model saved as diabetes_model.pkl")

# app.py

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
    
    # Extract features from JSON input in the expected order
    features = [data.get(key) for key in 
                ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
    
    # Check if any feature is missing
    if None in features:
        return jsonify({"error": "Missing feature(s). Please provide all 8 features."}), 400
    
    # Convert features to numpy array and reshape for model input
    input_array = np.array(features).reshape(1, -1)
    
    # Predict
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]
    
    # Return result
    result = {
        "prediction": int(prediction),  # 0 or 1
        "probability": float(probability)
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)

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
    
    # Extract features in correct order
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
