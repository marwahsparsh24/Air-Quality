import streamlit as st
import requests
import json
import math
from datetime import datetime
from google.cloud import bigquery

# Initialize BigQuery client
client = bigquery.Client()

# Define the endpoint URL
endpoint = "https://us-central1-airquality-438719.cloudfunctions.net/predict-function/predict"

# Helper function to compute cyclic features
def compute_cyclic_features(value, max_value):
    sin_val = math.sin(2 * math.pi * value / max_value)
    cos_val = math.cos(2 * math.pi * value / max_value)
    return sin_val, cos_val

# Store data in BigQuery
def store_in_bigquery(input_data, predicted_value, predictions_table, datetime_obj):
    predictions_schema = [
        bigquery.SchemaField("input_data", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("predicted_value", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="NULLABLE"),
    ]

    try:
        client.get_table(predictions_table)
    except:
        table = bigquery.Table(predictions_table, schema=predictions_schema)
        client.create_table(table)

    rows_to_insert = [
        {
            "input_data": json.dumps(input_data),
            "predicted_value": predicted_value,
            "timestamp": datetime_obj.isoformat(),
        }
    ]
    errors = client.insert_rows_json(predictions_table, rows_to_insert)
    if errors:
        st.error(f"Failed to store data in BigQuery: {errors}")

def main():
    st.title("Air Quality Prediction")

    st.header("Enter Date and Time for Prediction")
    input_date = st.date_input("Select a date for prediction:")
    input_time = st.time_input("Select a time for prediction:")
    

    if st.button("Predict Air Quality"):
        try:
            datetime_obj = datetime.combine(input_date, input_time)

            day_of_week = datetime_obj.weekday()
            day_of_year = datetime_obj.timetuple().tm_yday
            month = datetime_obj.month
            hour = datetime_obj.hour

            sin_hour, cos_hour = compute_cyclic_features(hour, 24)
            sin_day_of_week, cos_day_of_week = compute_cyclic_features(day_of_week, 7)
            # given the location  

            payload = {
                "instances": [
                    {
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
                        "hour": hour,
                        "day_of_week": day_of_week,
                        "day_of_year": day_of_year,
                        "month": month,
                        "sin_hour": sin_hour,
                        "cos_hour": cos_hour,
                        "sin_day_of_week": sin_day_of_week,
                        "cos_day_of_week": cos_day_of_week,
                    }
                ]
            }

            headers = {"Content-Type": "application/json"}
            response = requests.post(endpoint, headers=headers, data=json.dumps(payload))

            if response.status_code == 200:
                prediction = response.json()
                predicted_value = prediction["predictions"][0]
                st.success("Prediction Successful!")
                st.write(f"Predicted Air Quality Value: {predicted_value}")

                predictions_table = "airquality-438719.airqualityuser.predictions"

                # Store new prediction in BigQuery
                store_in_bigquery(payload["instances"][0], predicted_value, predictions_table, datetime_obj)
            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

#change input
# do for multiple values
# add graph showing outputs
# explain it using LLM/GPT

