import pandas as pd
from google.cloud import storage
from io import BytesIO

def split_data(bucket_name, input_pickle_file, output_train_file, output_test_file, **kwargs):
    """Split a stacked pickle file into train and test datasets, and save them back to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    try:
        # Download pickle file from GCS
        blob = bucket.blob(input_pickle_file)
        pickle_data = blob.download_as_bytes()
        stacked_df = pd.read_pickle(BytesIO(pickle_data))
        
        # Split the data into train and test sets (e.g., 80% train, 20% test)
        train_df = stacked_df.sample(frac=0.8, random_state=42)
        test_df = stacked_df.drop(train_df.index)

        # Save train and test sets back to GCS
        train_buffer = BytesIO()
        test_buffer = BytesIO()
        train_df.to_pickle(train_buffer)
        test_df.to_pickle(test_buffer)
        
        train_buffer.seek(0)
        test_buffer.seek(0)
        
        bucket.blob(output_train_file).upload_from_file(train_buffer, content_type='application/octet-stream')
        bucket.blob(output_test_file).upload_from_file(test_buffer, content_type='application/octet-stream')

        print(f"Train data saved to {output_train_file}, Test data saved to {output_test_file}.")

        # Push the output file paths to XCom for downstream tasks
        kwargs['ti'].xcom_push(key='train_file', value=output_train_file)
        kwargs['ti'].xcom_push(key='test_file', value=output_test_file)

    except Exception as e:
        print(f"An error occurred while splitting data: {e}")
        raise
