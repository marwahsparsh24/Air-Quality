# access data from bucket
# delete the existing data and add new data
# save training and testing data on bigquery
# change it according to the table
from google.cloud import storage
import io
from io import BytesIO
import pickle5 as pickle
import numpy as np
from scipy import stats
from google.cloud import bigquery
import os
import pandas as pd
import json
client = bigquery.Client(project="airquality-438719")

feature_data_path = f'processed/test/feature_eng_data.pkl'


bucket_name = "airquality-mlops-rg"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(feature_data_path)
pickle_data = blob.download_as_bytes()
feature_data = pickle.load(BytesIO(pickle_data))

def populate_temp_feature_eng_table(feature_eng_file):
 
    # Load data from the pickle file
    client_st = storage.Client()
    bucket_name = 'airquality-mlops-rg'
 
    # Get the bucket and the blob (file)
    bucket = client_st.bucket(bucket_name)
    blob_name = os.path.join(feature_eng_file)
    blob = bucket.blob(blob_name)
    pickle_data = blob.download_as_bytes()
    feature_data = pickle.load(BytesIO(pickle_data))
 
    # Ensure the timestamp column is present and formatted correctly
    feature_data['timestamp'] = feature_data.index
    feature_data['timestamp'] = pd.to_datetime(feature_data['timestamp'])
    rows_to_insert = []
    for _, row in feature_data.iterrows():
        timestamp = row["timestamp"].isoformat()
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
        # Add the row to the list to insert into BigQuery
        rows_to_insert.append({
        "timestamp": timestamp, "feature_data": json.dumps(feature_dict)})
    return rows_to_insert
 