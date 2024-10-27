import pandas as pd
from google.cloud import storage
import io

def pivot_data_task(bucket_name, input_file_path, output_file_path, **kwargs):
    """Pivot the data based on 'date', 'parameter', and 'value', and save it to GCS."""
    storage_client = storage.Client()

    try:
        # Load data from GCS
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(input_file_path)
        
        # Download the file as bytes
        pickle_data = blob.download_as_bytes()
        
        # Read the pickle file into a pandas DataFrame
        data = pd.read_pickle(io.BytesIO(pickle_data))
        print(f"Data loaded from gs://{bucket_name}/{input_file_path}")

        # Convert the 'date' column to datetime format
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            print("Date column converted to datetime.")
        else:
            raise ValueError("No 'date' column in the DataFrame.")

        # Pivot the data
        if all(col in data.columns for col in ['date', 'parameter', 'value']):
            pivoted_data = data.pivot_table(index='date', columns='parameter', values='value').reset_index()
            print("Data pivoted successfully.")
        else:
            raise ValueError("Missing one or more required columns: 'date', 'parameter', 'value'.")

        # Save the pivoted DataFrame back to GCS
        output_pickle_data = io.BytesIO()
        pivoted_data.to_pickle(output_pickle_data)
        output_pickle_data.seek(0)  # Go back to the start of the BytesIO stream

        # Upload the pickle file to GCS
        output_blob = bucket.blob(output_file_path)
        output_blob.upload_from_file(output_pickle_data, content_type='application/octet-stream')
        
        print(f"Pivoted DataFrame saved as 'gs://{bucket_name}/{output_file_path}'")

    except Exception as e:
        print(f"An error occurred during pivoting: {e}")
        raise
