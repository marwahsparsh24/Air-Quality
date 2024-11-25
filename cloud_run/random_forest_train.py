import sys
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.special import inv_boxcox
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import mlflow
import time
import mlflow.sklearn
import shap
from sklearn.model_selection import GridSearchCV
from google.cloud import storage
import io
from io import BytesIO
import pickle5 as pickle

def setup_mlflow_tracking():
    MLFLOW_TRACKING_DIR = os.environ.get('MLFLOW_TRACKING_DIR', '/app/mlruns')
    os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR}")

class RandomForestPM25Model:
    def __init__(self, train_file, test_file, lambda_value, model_save_path):
        self.train_file = train_file
        self.test_file = test_file
        self.lambda_value = lambda_value
        self.model_save_path = model_save_path
        self.param_grid = {
            'n_estimators': [100, 200]
        }
        #self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model = RandomForestRegressor(random_state=42)
        # mlflow.log_param("n_estimators",100)
        # mlflow.log_param("random_state",42)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_train_original = None
        self.y_test_original = None
    
    def load_data(self):
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

    def grid_search_cv(self):
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(self.X_train, self.y_train)

        # Log the best parameters and best RMSE
        best_params = grid_search.best_params_
        mlflow.log_params(best_params)
        print("Best parameters:", best_params)
        
        # Set the model to the best estimator
        self.model = grid_search.best_estimator_

        # Log the best model in MLflow
        mlflow.sklearn.log_model(self.model, "XGBoost", input_example=self.X_train[:5])
    
    # def train_model(self):
    #     # Train the model
    #     self.model.fit(self.X_train, self.y_train)
    #     mlflow.sklearn.log_model(self.model,"RandomForest",input_example=self.X_train[:5])

    #     return y_pred_original
    #
    def save_weights(self):
        storage_client = storage.Client()

        # Define the bucket and the path to store the model weights
        bucket_name = "airquality-mlops-rg"
        model_blob_path = "weights/rf_model.pth"  # Path in the bucket where the model will be saved

        # Get the bucket
        bucket = storage_client.bucket(bucket_name)

        # Create a blob object for the specified path
        model_blob = bucket.blob(model_blob_path)

        buffer = BytesIO()
        pickle.dump(self.model, buffer)  # Serialize the model weights
        buffer.seek(0)  # Ensure the buffer is at the start before uploading

        # Upload the serialized model weights to GCS
        model_blob.upload_from_file(buffer, content_type='application/octet-stream')

        print(f"Model weights saved and uploaded to GCS at {model_blob_path}")

def main():
    setup_mlflow_tracking()
    mlflow.set_experiment("PM2.5 Random Forest")
    bucket_name = "airquality-mlops-rg"
    train_file_gcs = f'gs://{bucket_name}/processed/train/feature_eng_data.pkl'
    model_save_path_gcs = f'gs://{bucket_name}/weights/model/rf_model.pth'

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():
        start_time = time.time()
        rf_model = RandomForestPM25Model(train_file_gcs, None, None, model_save_path_gcs)
        rf_model.load_data()
        rf_model.grid_search_cv()
        #rf_model.train_model()
        train_duration = time.time() - start_time
        mlflow.log_metric("training_duration", train_duration)
        rf_model.save_weights()
    mlflow.end_run()

if __name__ == "__main__":
    # mlflow.set_tracking_uri("file:///opt/airflow/dags/mlruns")
    main()
