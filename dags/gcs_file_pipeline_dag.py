from airflow import DAG
from scipy import stats
from airflow.operators.python import PythonOperator
from datetime import datetime
import numpy as np
from google.cloud import storage
import io
import os
import pandas as pd

from google.cloud import storage
from my_operators.stacks_csv import stack_csvs_to_pickle
from my_operators.split_data import split_data
from my_operators.pull_train import pull_train_file
from my_operators.pivot import pivot_data_task
from my_operators.drop_columns import clean_data
from my_operators.check_missing import process_data
from my_operators.check_anamoly import anomaly_detection
from my_operators.feature_eng import feature_engineering
from my_operators.schema_gen import data_validation

# Helper function to check for new files in GCS
def check_for_new_files(bucket_name, folder_path, **kwargs):
    # Create a GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # List the files in the folder
    blobs = bucket.list_blobs(prefix=folder_path)
    new_files = [blob.name for blob in blobs if blob.name.endswith('.csv')]
    
    # Pass the list of files as an XCom to the next task
    if new_files:
        kwargs['ti'].xcom_push(key='file_list', value=new_files)
    else:
        raise ValueError("No new files found in the specified GCS folder")

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime.now(),
    'email_on_failure': False,
    'retries': 1
}

# Define the DAG
with DAG(
    'gcs_file_pipeline_dag',
    default_args=default_args,
    description='Check GCS, stack CSVs into a pickle, and split the data',
    schedule_interval=None,  # Triggered manually or on a schedule
) as dag:

    # Task 1: Check for new files in GCS
    check_for_files_task = PythonOperator(
        task_id='check_for_new_files',
        python_callable=check_for_new_files,
        op_kwargs={
            'bucket_name': 'airquality-mlops-rg',
            'folder_path': 'api_data/'
        },
        provide_context=True,
    )

    # Task 2: Stack CSVs from GCS and save as a pickle
    stack_csvs_task = PythonOperator(
        task_id='stack_csvs',
        python_callable=stack_csvs_to_pickle,
        op_kwargs={
            'bucket_name': 'airquality-mlops-rg',
            'folder_path': 'api_data/',
            'output_pickle_file': 'processed_data/stacked_air_pollution.pkl'
        },
    )

    # Task 3: Split the data
    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        op_kwargs={
            'bucket_name': 'airquality-mlops-rg',
            'input_pickle_file': 'processed_data/stacked_air_pollution.pkl',
            'output_train_file': 'processed_data/train_data.pkl',
            'output_test_file': 'processed_data/test_data.pkl'
        },
    )

        # Dummy task to pull train file path
    pull_train_file_task = PythonOperator(
        task_id='pull_train_file',
        python_callable=pull_train_file,
        provide_context=True,  # Ensure context is provided
    )

        # Task 4: Pivot data
    pivot_task = PythonOperator(
        task_id='pivot_gcs_data',
        python_callable=pivot_data_task,
        op_kwargs={
            'bucket_name': 'airquality-mlops-rg',
            'input_file_path': "processed_data/train_data.pkl",  # Pull train file path from the previous task
            'output_file_path': 'processed_data/pivoted_train_data.pkl',
        },
        provide_context=True,
    )


    clean_data_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        op_kwargs={
            'bucket_name': 'airquality-mlops-rg',
            'input_file_path': 'processed_data/schema_gen.pkl',
            'output_file_path': 'processed_data/cleaned_train_data.pkl',
        }
    )
    
    # Define the task using PythonOperator
    process_data_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
        op_kwargs={
            'bucket_name': 'airquality-mlops-rg',
            'input_file_path': 'processed_data/cleaned_train_data.pkl',
            'output_file_path': 'processed_data/fill_na.pkl'
        }
    )

 # Define the task using PythonOperator
    check_anamoly_task = PythonOperator(
        task_id='anomaly_detection_and_cleaning',
        python_callable=anomaly_detection,
        op_kwargs={
            'bucket_name': 'airquality-mlops-rg',
            'input_file_path': 'processed_data/fill_na.pkl',
            'output_file_path': 'processed_data/check_anamoly.pkl'
        }
    )

    feature_engineering_task = PythonOperator(
        task_id='feature_engineering',
        python_callable= feature_engineering,
        op_kwargs={
            'bucket_name': 'airquality-mlops-rg',
            'input_file_path': 'processed_data/check_anamoly.pkl',
            'output_file_path': 'processed_data/feature_engineer.pkl'
        }
    )

    data_validation_task = PythonOperator(
        task_id='data_validation',
        python_callable=data_validation,
        op_kwargs={
            'bucket_name': 'airquality-mlops-rg',
            'file_path': 'processed_data/pivoted_train_data.pkl',
            'output_file_path': 'processed_data/schema_gen.pkl'
        }
    )

# Define task dependencies
check_for_files_task >> stack_csvs_task >> split_data_task >> pull_train_file_task >> pivot_task >> data_validation_task >> clean_data_task >> process_data_task >> check_anamoly_task >> feature_engineering_task
