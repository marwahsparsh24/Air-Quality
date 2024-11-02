from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from google.cloud import storage
from datetime import datetime
import pandas as pd
import tensorflow as tf
import os
import io

# Function to load data
def load_data(bucket_name, file_path):
    """Load data from GCS."""
    from google.cloud import storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    # Download the file as bytes
    pickle_data = blob.download_as_bytes()
    data = pd.read_pickle(io.BytesIO(pickle_data))
    print(f"Data loaded from gs://{bucket_name}/{file_path}.")
    return data

# Function to generate schema using TensorFlow
def generate_schema(data):
    """Generate schema for the dataset."""
    schema = {
        "features": {}
    }
    for column in data.columns:
        if data[column].dtype == 'object':
            schema["features"][column] = {"type": "string"}
        elif data[column].dtype in ['int64', 'float64']:
            schema["features"][column] = {"type": "number"}
        else:
            schema["features"][column] = {"type": "unknown"}
    
    print("Schema generated:")
    print(schema)
    return schema

# Function to validate data against schema
def validate_data(data, schema):
    """Validate data against the generated schema."""
    valid = True
    validation_results = []

    for column, col_schema in schema["features"].items():
        if column not in data.columns:
            valid = False
            validation_results.append(f"Missing column: {column}")
            continue
        
        if col_schema["type"] == "string" and not all(isinstance(val, str) for val in data[column]):
            valid = False
            validation_results.append(f"Invalid type in column {column}: expected string")
        
        elif col_schema["type"] == "number" and not all(isinstance(val, (int, float)) for val in data[column]):
            valid = False
            validation_results.append(f"Invalid type in column {column}: expected number")
    
    if valid:
        print("Data validation passed.")
    else:
        print("Data validation failed:")
        for result in validation_results:
            print(result)

    return valid

# Main function to orchestrate loading, schema generation, and validation
def data_validation(bucket_name, input_file_path, output_file_path):
    data = load_data(bucket_name, input_file_path)
    schema = generate_schema(data)
    validation_result = validate_data(data, schema)
    schema_output_path = io.BytesIO()
    pd.DataFrame([schema]).to_pickle(schema_output_path)
    schema_output_path.seek(0)

    # Save schema to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    output_blob = bucket.blob(output_file_path)
    output_blob.upload_from_file(schema_output_path, content_type='application/octet-stream')
    print(f"Schema saved to gs://{bucket_name}/{output_file_path}")

    return validation_result
