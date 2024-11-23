import requests
import json

# Define the endpoint URL
endpoint = "https://us-central1-airquality-438719.cloudfunctions.net/predict-function/predict"

# Define the input payload with the correct feature names
payload = {
    "instances": [
        {
            "pm25_boxcox": 0.5,
            "lag_1": 0.6,
            "lag_2": 0.7,
            "lag_3": 0.8,
            "lag_4": 0.9,
            "lag_5": 1.0,
            "rolling_mean_3": 0.4,
            "rolling_mean_6": 0.3,
            "rolling_mean_24": 0.2,
            "rolling_std_3": 0.1,
            "rolling_std_6": 0.2,
            "rolling_std_24": 0.3,
            "ema_3": 0.7,
            "ema_6": 0.6,
            "ema_24": 0.5,
            "diff_1": 0.4,
            "diff_2": 0.3,
            "hour": 15,
            "day_of_week": 2,
            "day_of_year": 123,
            "month": 5,
            "sin_hour": 0.707,
            "cos_hour": 0.707,
            "sin_day_of_week": 0.866,
            "cos_day_of_week": 0.5
        }
    ]
}

# Send the POST request to the endpoint
try:
    headers = {"Content-Type": "application/json"}
    response = requests.post(endpoint, headers=headers, data=json.dumps(payload))

    # Check the status code and print the response
    if response.status_code == 200:
        print("Predictions:", response.json())
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")

except Exception as e:
    print(f"An error occurred: {e}")
