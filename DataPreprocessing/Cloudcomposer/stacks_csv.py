import pandas as pd
from google.cloud import storage
from io import BytesIO

def stack_csvs_to_pickle(bucket_name, folder_path, output_pickle_file):
    """Load CSVs from GCS, stack them into a single DataFrame, and save as a pickle."""
    storage_client = storage.Client()
    dataframes = []
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_path)
    csv_files = [blob for blob in blobs if blob.name.endswith('.csv')]

    if not csv_files:
        print("No CSV files found.")
        return

    for blob in csv_files:
        print(f"Loading file from GCS: {blob.name}")
        data = blob.download_as_bytes()
        df = pd.read_csv(BytesIO(data))
        dataframes.append(df)

    # Stack the dataframes
    if dataframes:
        stacked_df = pd.concat(dataframes, ignore_index=True)
        print("DataFrames stacked successfully.")

        # Save the stacked DataFrame as a pickle file
        pickle_buffer = BytesIO()
        stacked_df.to_pickle(pickle_buffer)
        pickle_buffer.seek(0)
        blob = bucket.blob(output_pickle_file)
        blob.upload_from_file(pickle_buffer, content_type='application/octet-stream')
        print(f"Stacked DataFrame saved to GCS at '{output_pickle_file}'.")
