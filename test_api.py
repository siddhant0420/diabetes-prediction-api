
import requests

url = "http://127.0.0.1:5000/predict"

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
print(response.status_code)
print(response.json())
