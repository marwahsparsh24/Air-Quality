from google.cloud import storage, bigquery
from io import BytesIO
import pickle5 as pickle
import pandas as pd
import json
import os

# Initialize BigQuery and Storage clients
client = bigquery.Client(project="airquality-438719")
storage_client = storage.Client()

# Paths and table information
bucket_name = "airquality-mlops-rg"
feature_data_path = 'processed/test/feature_eng_data.pkl'
feature_data_path_train = 'processed/train/feature_eng_data.pkl'
full_table_id = "airquality-438719.airqualityuser.allfeatures"

# Check if BigQuery table exists, create if not
try:
    client.get_table(full_table_id)
    print(f"Table {full_table_id} exists.")
except Exception:
    schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("feature_data", "STRING", mode="NULLABLE")
    ]
    table = bigquery.Table(full_table_id, schema=schema)
    client.create_table(table)
    print(f"Table {full_table_id} created.")

# Function to load data from GCS and upload to BigQuery
def populate_temp_feature_eng_table(feature_file_path):
    # Load data from the GCS bucket
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(feature_file_path)
    pickle_data = blob.download_as_bytes()
    feature_data = pickle.load(BytesIO(pickle_data))

    # Ensure timestamp column is properly formatted
    feature_data['timestamp'] = feature_data.index
    feature_data['timestamp'] = pd.to_datetime(feature_data['timestamp'], errors='coerce')
    if feature_data['timestamp'].isnull().any():
        raise ValueError("Invalid timestamps found in the data. Please check the input data.")

    # Convert feature data to rows for BigQuery
    rows_to_insert = []
    for _, row in feature_data.iterrows():
        timestamp = row["timestamp"].isoformat() if pd.notnull(row["timestamp"]) else None
        feature_dict = {
            "pm25": row["pm25"],
            "pm25_boxcox": row["pm25_boxcox"],
            "lag_1": row["lag_1"],
            "lag_2": row["lag_2"],
            "lag_3": row["lag_3"],
            "lag_4": row["lag_4"],
            "lag_5": row["lag_5"],
            "rolling_mean_3": row["rolling_mean_3"],
            "rolling_mean_6": row["rolling_mean_6"],
            "rolling_mean_24": row["rolling_mean_24"],
            "rolling_std_3": row["rolling_std_3"],
            "rolling_std_6": row["rolling_std_6"],
            "rolling_std_24": row["rolling_std_24"],
            "ema_3": row["ema_3"],
            "ema_6": row["ema_6"],
            "ema_24": row["ema_24"],
            "diff_1": row["diff_1"],
            "diff_2": row["diff_2"],
            "hour": row["hour"],
            "day_of_week": row["day_of_week"],
            "day_of_year": row["day_of_year"],
            "month": row["month"],
            "sin_hour": row["sin_hour"],
            "cos_hour": row["cos_hour"],
            "sin_day_of_week": row["sin_day_of_week"],
            "cos_day_of_week": row["cos_day_of_week"]
        }
        rows_to_insert.append({
            "timestamp": timestamp, 
            "feature_data": json.dumps(feature_dict)
        })

    # Convert rows to DataFrame for BigQuery
    dataframe = pd.DataFrame(rows_to_insert)

    # Configure load job for BigQuery
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # Overwrites existing data
        schema=[
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("feature_data", "STRING", mode="NULLABLE")
        ]
    )

    # Load data into BigQuery
    job = client.load_table_from_dataframe(dataframe, full_table_id, job_config=job_config)
    job.result()  # Wait for the job to complete
    print(f"Data from {feature_file_path} successfully loaded into {full_table_id}.")

# Call the function for test and training data
populate_temp_feature_eng_table(feature_data_path)
populate_temp_feature_eng_table(feature_data_path_train)
