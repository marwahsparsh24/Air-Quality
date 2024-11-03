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

def drop_columns(data, columns_to_drop):
    """Drop unnecessary columns from the DataFrame."""
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    print(f"Columns dropped: {columns_to_drop}")
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

def clean_data(bucket_name, input_file_path, output_file_path, **kwargs):
    """Load data, drop unnecessary columns, and save the cleaned data."""
    data = load_data(bucket_name, input_file_path)
    
    # Define columns to drop
    columns_to_drop = ['co', 'no', 'no2', 'o3', 'so2']
    data = drop_columns(data, columns_to_drop)
    
    # Set the 'date' column as index if it exists
    if 'date' in data.columns:
        data.set_index('date', inplace=True)

    # Save the cleaned data
    save_data(data, bucket_name, output_file_path)
