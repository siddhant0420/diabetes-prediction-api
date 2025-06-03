import requests

url = "https://diabetes-prediction-api-1eji.onrender.com/predict"

data = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 90,
    "BMI": 30.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 28
}

response = requests.post(url, json=data)
print(response.json())
