from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
from google.cloud import storage
import io
import os

def load_data(bucket_name, input_file_path):
    """Load data from GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(input_file_path)

    # Download the file as bytes
    pickle_data = blob.download_as_bytes()
    
    # Read the pickle file into a pandas DataFrame
    data = pd.read_pickle(io.BytesIO(pickle_data))
    print(f"Data loaded from gs://{bucket_name}/{input_file_path}.")
    return data

def fill_missing_values(data):
    """Fill missing values in the DataFrame."""
    if 'pm25' in data.columns:
        original_missing_count = data['pm25'].isnull().sum()
        data['pm25'] = data['pm25'].interpolate(method='linear')
        filled_missing_count = data['pm25'].isnull().sum()
        print(f"'pm25' missing values interpolated: {original_missing_count - filled_missing_count} filled.")
    else:
        print("'pm25' column not found for interpolation.")
    return data

def save_data(data, bucket_name, output_file_path):
    """Save the processed DataFrame back to GCS as a pickle file."""
    output_pickle_data = io.BytesIO()
    data.to_pickle(output_pickle_data)
    output_pickle_data.seek(0)  # Go back to the start of the BytesIO stream

    # Upload the pickle file to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    output_blob = bucket.blob(output_file_path)
    output_blob.upload_from_file(output_pickle_data, content_type='application/octet-stream')
    
    print(f"Processed DataFrame saved as 'gs://{bucket_name}/{output_file_path}'.")

def process_data(bucket_name, input_file_path, output_file_path, **kwargs):
    """Load, check for missing values, fill them, and save the data."""
    data = load_data(bucket_name, input_file_path)
    data = fill_missing_values(data)
    save_data(data, bucket_name, output_file_path)
