import pandas as pd
from google.cloud import storage
from io import BytesIO
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
import os

def pull_train_file():
    # Path to the train file in Google Cloud Storage
    gcs_bucket = os.getenv("airquality-mlops-rg")  # Set this environment variable in your Composer environment
    train_file_path = f'gs://{gcs_bucket}/processed_data/train_data.pkl'  # Update to your actual path

    try:
        # Read the pickle file directly from GCS
        train_df = pd.read_pickle(train_file_path, storage_options={'token': 'cloud'})
        print("Train DataFrame loaded successfully.")
        # Proceed with further processing
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {train_file_path} does not exist in GCS.")
    except Exception as e:
        print(f"An error occurred: {e}")