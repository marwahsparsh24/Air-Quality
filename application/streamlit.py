# import streamlit as st
# import requests
# import json
# import math
# from datetime import datetime

# # Define the endpoint URL
# endpoint = "https://us-central1-airquality-438719.cloudfunctions.net/predict-function/predict"

# # Helper function to compute cyclic features
# def compute_cyclic_features(value, max_value):
#     sin_val = math.sin(2 * math.pi * value / max_value)
#     cos_val = math.cos(2 * math.pi * value / max_value)
#     return sin_val, cos_val

# # Streamlit application
# def main():
#     st.title("Air Quality Prediction")

#     # Input widgets for date and time
#     st.header("Enter Date and Time for Prediction")
#     input_date = st.date_input("Select a date for prediction:")
#     input_time = st.time_input("Select a time for prediction:")

#     # Button to trigger prediction
#     if st.button("Predict Air Quality"):
#         try:
#             # Combine date and time into a datetime object
#             datetime_obj = datetime.combine(input_date, input_time)

#             # Extract features
#             day_of_week = datetime_obj.weekday()
#             day_of_year = datetime_obj.timetuple().tm_yday
#             month = datetime_obj.month
#             hour = datetime_obj.hour

#             # Compute cyclic features
#             sin_hour, cos_hour = compute_cyclic_features(hour, 24)
#             sin_day_of_week, cos_day_of_week = compute_cyclic_features(day_of_week, 7)

#             # Prepare the payload for prediction
#             payload = {
#                 "instances": [
#                     {
#                         "lag_1": 0.6,
#                         "lag_2": 0.7,
#                         "lag_3": 0.8,
#                         "lag_4": 0.9,
#                         "lag_5": 1.0,
#                         "rolling_mean_3": 0.4,
#                         "rolling_mean_6": 0.3,
#                         "rolling_mean_24": 0.2,
#                         "rolling_std_3": 0.1,
#                         "rolling_std_6": 0.2,
#                         "rolling_std_24": 0.3,
#                         "ema_3": 0.7,
#                         "ema_6": 0.6,
#                         "ema_24": 0.5,
#                         "diff_1": 0.4,
#                         "diff_2": 0.3,
#                         "hour": hour,
#                         "day_of_week": day_of_week,
#                         "day_of_year": day_of_year,
#                         "month": month,
#                         "sin_hour": sin_hour,
#                         "cos_hour": cos_hour,
#                         "sin_day_of_week": sin_day_of_week,
#                         "cos_day_of_week": cos_day_of_week
#                     }
#                 ]
#             }

#             # Send the POST request to the API endpoint
#             headers = {"Content-Type": "application/json"}
#             response = requests.post(endpoint, headers=headers, data=json.dumps(payload))

#             # Handle response and display prediction
#             if response.status_code == 200:
#                 prediction = response.json()  # Parse JSON response
#                 # Assuming the prediction value is at 'predictions[0]'
#                 predicted_value = prediction["predictions"][0]
#                 st.success("Prediction Successful!")
#                 st.write(f"Predicted Air Quality Value: {predicted_value}")
#             else:
#                 st.error(f"Error {response.status_code}: {response.text}")

#         except Exception as e:
#             st.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()


import streamlit as st
import requests
import json
import math
from datetime import datetime
from google.cloud import bigquery
import pandas as pd
from google.cloud import storage
import io
from io import BytesIO
import pickle5 as pickle
import os 

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

# # Populate and delete a temporary BigQuery table
# def populate_temp_feature_eng_table(feature_eng_file, temp_feature_eng_table):

#     feature_eng_schema = [
#         bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
#         bigquery.SchemaField("feature_data", "STRING", mode="REQUIRED"),
#     ]
#     table = bigquery.Table(temp_feature_eng_table, schema=feature_eng_schema)
#     client.create_table(table)

#     client_st = storage.Client()
#     bucket_name = 'airquality-mlops-rg'

#     # Get the bucket and the blob (file)
#     bucket = client_st.bucket(bucket_name)
#     blob_name = os.path.join(feature_eng_file)
#     blob = bucket.blob(blob_name)
#     pickle_data = blob.download_as_bytes() 
#     feature_data = pickle.load(BytesIO(pickle_data))
#     feature_data['timestamp'] = feature_data.index
#     feature_data['timestamp'] = pd.to_datetime(feature_data['timestamp'])
#     rows_to_insert = [
#         {"timestamp": row["timestamp"], "feature_data": json.dumps(row.to_dict())}
#         for _, row in feature_data.iterrows()
#     ]
#     errors = client.insert_rows_json(temp_feature_eng_table, rows_to_insert)
#     if errors:
#         st.error(f"Failed to populate the temporary table: {errors}")

# def compare_and_trigger_with_temp_table(predictions_table, temp_feature_eng_table, threshold, cloud_function_url):
#     try:
#         # Query predictions table
#         predictions_query = f"SELECT timestamp, predicted_value FROM `{predictions_table}`"
#         predictions = client.query(predictions_query).to_dataframe()

#         # Query feature engineering table
#         temp_feature_eng_query = f"SELECT timestamp, feature_data FROM `{temp_feature_eng_table}`"
#         temp_feature_eng = client.query(temp_feature_eng_query).to_dataframe()

#         # Parse feature data and extract pm25
#         temp_feature_eng["parsed_data"] = temp_feature_eng["feature_data"].apply(json.loads)
#         temp_feature_eng["pm25"] = temp_feature_eng["parsed_data"].apply(lambda x: x["pm25"])

#         # Merge predictions and feature data
#         merged = predictions.merge(temp_feature_eng[["timestamp", "pm25"]], on="timestamp", how="inner")

#         # Calculate the absolute difference between predicted and actual pm25
#         merged["difference"] = abs(merged["predicted_value"] - merged["pm25"])

#         # Check for threshold breaches
#         if not merged[merged["difference"] > threshold].empty:
#             response = requests.post(cloud_function_url)
#             if response.status_code == 200:
#                 st.success("Cloud Function triggered successfully.")
#             else:
#                 st.error(f"Failed to trigger Cloud Function: {response.text}")
#     finally:
#         # Delete the temporary table
#         client.delete_table(temp_feature_eng_table, not_found_ok=True)
#         print(f"Temporary table {temp_feature_eng_table} deleted.")

# # Streamlit application
# def main():
#     st.title("Air Quality Prediction")

#     st.header("Enter Date and Time for Prediction")
#     input_date = st.date_input("Select a date for prediction:")
#     input_time = st.time_input("Select a time for prediction:")

#     if st.button("Predict Air Quality"):
#         try:
#             datetime_obj = datetime.combine(input_date, input_time)

#             day_of_week = datetime_obj.weekday()
#             day_of_year = datetime_obj.timetuple().tm_yday
#             month = datetime_obj.month
#             hour = datetime_obj.hour

#             sin_hour, cos_hour = compute_cyclic_features(hour, 24)
#             sin_day_of_week, cos_day_of_week = compute_cyclic_features(day_of_week, 7)

#             payload = {
#                 "instances": [
#                     {
#                         "lag_1": 0.6,
#                         "lag_2": 0.7,
#                         "lag_3": 0.8,
#                         "lag_4": 0.9,
#                         "lag_5": 1.0,
#                         "rolling_mean_3": 0.4,
#                         "rolling_mean_6": 0.3,
#                         "rolling_mean_24": 0.2,
#                         "rolling_std_3": 0.1,
#                         "rolling_std_6": 0.2,
#                         "rolling_std_24": 0.3,
#                         "ema_3": 0.7,
#                         "ema_6": 0.6,
#                         "ema_24": 0.5,
#                         "diff_1": 0.4,
#                         "diff_2": 0.3,
#                         "hour": hour,
#                         "day_of_week": day_of_week,
#                         "day_of_year": day_of_year,
#                         "month": month,
#                         "sin_hour": sin_hour,
#                         "cos_hour": cos_hour,
#                         "sin_day_of_week": sin_day_of_week,
#                         "cos_day_of_week": cos_day_of_week,
#                     }
#                 ]
#             }

#             headers = {"Content-Type": "application/json"}
#             response = requests.post(endpoint, headers=headers, data=json.dumps(payload))

#             if response.status_code == 200:
#                 prediction = response.json()
#                 predicted_value = prediction["predictions"][0]
#                 st.success("Prediction Successful!")
#                 st.write(f"Predicted Air Quality Value: {predicted_value}")

#                 predictions_table = "airquality-438719.airqualityuser.predictions"
#                 temp_feature_eng_table = "airquality-438719.airqualityuser.temp_feature_eng_data"
#                 feature_eng_file = 'processed/train/feature_eng_data.pkl'
#                 cloud_function_url = "https://us-central1-your-project-id.cloudfunctions.net/trigger-function"

#                 store_in_bigquery(payload["instances"][0], predicted_value, predictions_table)

#                 populate_temp_feature_eng_table(feature_eng_file, temp_feature_eng_table)

#                 compare_and_trigger_with_temp_table(
#                     predictions_table, temp_feature_eng_table, threshold=2, cloud_function_url=cloud_function_url
#                 )
#             else:
#                 st.error(f"Error {response.status_code}: {response.text}")

#         except Exception as e:
#             st.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()
# Update predictions for all entries in the predictions table
# def update_predictions(predictions_table, endpoint):
#     try:
#         # Fetch all rows from the predictions table
#         predictions_query = f"SELECT * FROM `{predictions_table}`"
#         rows = client.query(predictions_query).to_dataframe()

#         # Update predictions
#         for _, row in rows.iterrows():
#             input_data = json.loads(row["input_data"])
#             headers = {"Content-Type": "application/json"}
#             payload = {"instances": [input_data]}

#             response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
#             if response.status_code == 200:
#                 new_prediction = response.json()["predictions"][0]

#                 # Update the prediction in BigQuery
#                 query = f"""
#                     UPDATE `{predictions_table}`
#                     SET predicted_value = @new_prediction
#                     WHERE input_data = @input_data
#                 """
#                 job_config = bigquery.QueryJobConfig(
#                     query_parameters=[
#                         bigquery.ScalarQueryParameter("new_prediction", "FLOAT", new_prediction),
#                         bigquery.ScalarQueryParameter("input_data", "STRING", json.dumps(input_data)),
#                     ]
#                 )
#                 client.query(query, job_config=job_config)
#     except Exception as e:
#         st.error(f"An error occurred while updating predictions: {e}")

# Streamlit application
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
                # temp_feature_eng_table = "airquality-438719.airqualityuser.temp_feature_eng_data"
                # feature_eng_file = 'processed/train/feature_eng_data.pkl'
                # cloud_function_url = "https://us-central1-airquality-438719.cloudfunctions.net/model-decay"

                # Store new prediction in BigQuery
                store_in_bigquery(payload["instances"][0], predicted_value, predictions_table, datetime_obj)

                # # Update all existing predictions in BigQuery
                # update_predictions(predictions_table, endpoint)

                # # Populate temporary feature engineering table and trigger logic
                # populate_temp_feature_eng_table(feature_eng_file, temp_feature_eng_table)

                # compare_and_trigger_with_temp_table(
                #     predictions_table, temp_feature_eng_table, threshold=2, cloud_function_url=cloud_function_url
                # )
            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
