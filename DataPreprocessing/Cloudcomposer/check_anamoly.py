from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
from google.cloud import storage
import io
import os

class DataCleaner:
    def __init__(self, data):
        self.data = data

    def handle_outliers(self, column_name='pm25'):
        if column_name not in self.data.columns:
            raise ValueError(f"'{column_name}' column not found in the DataFrame.")
        Q1 = self.data[column_name].quantile(0.25)
        Q3 = self.data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        column_median = self.data[column_name].median()
        self.data.loc[(self.data[column_name] < lower_bound) | (self.data[column_name] > upper_bound), column_name] = column_median
        print(f"Outliers in '{column_name}' replaced with median value {column_median}.")

    def replace_negative_with_zero(self, column_name='pm25'):
        if column_name not in self.data.columns:
            raise ValueError(f"'{column_name}' column not found in the DataFrame.")
        self.data[column_name] = self.data[column_name].clip(lower=0)
        print(f"Negative values in '{column_name}' replaced with 0.")

    def save_as_pickle(self, bucket_name, output_file_path):
        if self.data is not None:
            output_pickle_data = io.BytesIO()
            self.data.to_pickle(output_pickle_data)
            output_pickle_data.seek(0)  # Go back to the start of the BytesIO stream
            
            # Upload the pickle file to GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            output_blob = bucket.blob(output_file_path)
            output_blob.upload_from_file(output_pickle_data, content_type='application/octet-stream')
            print(f"Cleaned DataFrame saved as 'gs://{bucket_name}/{output_file_path}'.")
        else:
            print("No data available to save. Please load and clean the data first.")

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

def anomaly_detection(bucket_name, input_file_path, output_file_path, **kwargs):
    """Load, clean, and save the data with anomaly detection."""
    data = load_data(bucket_name, input_file_path)
    cleaner = DataCleaner(data)
    cleaner.handle_outliers(column_name='pm25')
    cleaner.replace_negative_with_zero(column_name='pm25')
    cleaner.save_as_pickle(bucket_name, output_file_path)
