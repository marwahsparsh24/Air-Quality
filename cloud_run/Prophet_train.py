import sys
import os
import time
import pickle
import pandas as pd
import mlflow
import mlflow.pyfunc
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.special import inv_boxcox
from prophet import Prophet
from google.cloud import storage
import io
from io import BytesIO
import pickle5 as pickle

def setup_mlflow_tracking():
    MLFLOW_TRACKING_DIR = os.environ.get('MLFLOW_TRACKING_DIR', '/app/mlruns')
    os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR}")

class ProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, prophet_model):
        self.prophet_model = prophet_model
    
    def predict(self, context, model_input):
        # Assuming model_input is a DataFrame with 'ds' column (date)
        return self.prophet_model.predict(model_input)['yhat']

class ProphetPM25Model:
    def __init__(self, train_file_gcs, test_file_gcs, lambda_value, model_save_path_gcs):
        self.train_file_gcs = train_file_gcs
        self.test_file_gcs = test_file_gcs
        self.lambda_value = lambda_value
        self.model_save_path_gcs = model_save_path_gcs
        self.model = Prophet()
        self.X_train = None
        self.y_train = None
        self.y_train_original = None

    def load_data(self):
        # Initialize Google Cloud Storage client        
        client = storage.Client()

        # Specify your bucket name and the path to the pickle file in the 'processed' folder
        bucket_name = 'airquality-mlops-rg'
        pickle_file_path = 'processed/train/feature_eng_data.pkl'

        # Get the bucket and the blob (file)
        bucket = client.bucket(bucket_name)
        blob_name = os.path.join(pickle_file_path)
        blob = bucket.blob(blob_name)
        pickle_data = blob.download_as_bytes() 
        train_data = pickle.load(BytesIO(pickle_data))

        
        # Extract Box-Cox transformed y and original y
        for column in train_data.columns:
            if column == 'pm25_boxcox' or column == 'pm25_log':
                self.y_train = train_data[column]
                break
        self.y_train_original = train_data['pm25']
        self.X_train = train_data.drop(columns=['pm25'])

    def preprocess_data(self):
        # Prepare training data in Prophet format
        self.df_train = pd.DataFrame({
            'ds': self.X_train.index.tz_localize(None),  # Remove timezone
            'y': self.y_train  # Use Box-Cox transformed target
        })

    def train_model(self):
        # Train the Prophet model
        self.model.fit(self.df_train)
        wrapped_model = ProphetWrapper(self.model)
        mlflow.pyfunc.log_model(artifact_path="prophet_pm25_model", python_model=wrapped_model, input_example=self.df_train.head(1))

    def save_weights(self):
        
        storage_client = storage.Client()

        # Define the bucket and the path to store the model weights
        bucket_name = "airquality-mlops-rg"
        model_blob_path = "weights/prophet_pm25_model.pth"  # Path in the bucket where the model will be saved

        # Get the bucket
        bucket = storage_client.bucket(bucket_name)

        # Create a blob object for the specified path
        model_blob = bucket.blob(model_blob_path)

        # Serialize the model weights using pickle
        buffer = BytesIO()
        pickle.dump(self.model, buffer)  # Serialize the model weights
        buffer.seek(0)  # Ensure the buffer is at the start before uploading

        # Upload the serialized model weights to GCS
        model_blob.upload_from_file(buffer, content_type='application/octet-stream')

        print(f"Model weights saved and uploaded to GCS at {model_blob_path}")
        
        #mlflow.log_artifact(local_model_path)

# Main function to orchestrate the workflow
def main():
    setup_mlflow_tracking()
    # Initialize the storage client and set GCS paths
    bucket_name = "airquality-mlops-rg"
    train_file_gcs = f'gs://{bucket_name}/processed/train/feature_eng_data.pkl'
    model_save_path_gcs = f'gs://{bucket_name}/weights/prophet_pm25_model.pth'

    # Configure MLflow
    #mlflow.set_tracking_uri("postgresql://airquality:Prediction@34.42.22.132:5432/air?connect_timeout=100")  # Use GCS for artifact storage
    mlflow.set_experiment("PM2.5 Prophet")

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run() as run:
        start_time = time.time()
        prophet_model = ProphetPM25Model(train_file_gcs, None, None, model_save_path_gcs)
        prophet_model.load_data()
        prophet_model.preprocess_data()
        prophet_model.train_model()
        train_duration = time.time() - start_time
        mlflow.log_metric("training_duration", train_duration)
        prophet_model.save_weights()
    mlflow.end_run()

if __name__ == "__main__":
    main()
