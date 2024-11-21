import os
import time
import pickle5 as pickle
from io import BytesIO
from google.cloud import storage
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Define the training function as a component
def train_pm25_model(
    train_file_gcs: str,
    model_save_path_gcs: str,
    bucket_name: str,
    param_grid: dict = {"n_estimators": [100, 200]},
):
    """
    Train a RandomForestRegressor model for PM2.5 prediction using training data in GCS.
    
    Parameters:
    train_file_gcs (str): GCS path to the training data file (pickle format).
    model_save_path_gcs (str): GCS path to save the trained model.
    bucket_name (str): Name of the GCS bucket.
    param_grid (dict): Parameter grid for GridSearchCV.
    """
    # Validate environment variable for GCP credentials
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        raise EnvironmentError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set. Ensure the service account key is correctly configured."
        )

    # Initialize GCS client
    client = storage.Client()

    # Load training data from GCS
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(train_file_gcs.replace(f"gs://{bucket_name}/", ""))
    pickle_data = blob.download_as_bytes()
    train_data = pickle.load(BytesIO(pickle_data))

    # Extract features and labels
    y_train = None
    for column in train_data.columns:
        if column == "pm25_boxcox" or column == "pm25_log":
            y_train = train_data[column]
            break
    if y_train is None:
        raise ValueError("No valid target column found in training data.")
    X_train = train_data.drop(columns=["pm25"])

    # Train the model with GridSearchCV
    print("Starting model training...")
    start_time = time.time()
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=3, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X_train, y_train)
    print("Model training completed.")
    print(f"Best parameters: {grid_search.best_params_}")

    # Save the trained model to GCS
    print("Saving model weights to GCS...")
    best_model = grid_search.best_estimator_
    buffer = BytesIO()
    pickle.dump(best_model, buffer)
    buffer.seek(0)

    model_blob = bucket.blob(model_save_path_gcs.replace(f"gs://{bucket_name}/", ""))
    model_blob.upload_from_file(buffer, content_type="application/octet-stream")
    print(f"Model weights saved at: {model_save_path_gcs}")
    print(f"Training duration: {time.time() - start_time} seconds.")

if __name__ == "__main__":
    # Define paths and parameters
    bucket_name = "airquality-mlops-rg"
    train_file_gcs = "processed/train/feature_eng_data.pkl"
    model_save_path_gcs = "weights/model/model.pkl"

    # Call the training function
    train_pm25_model(
        train_file_gcs=train_file_gcs,
        model_save_path_gcs=model_save_path_gcs,
        bucket_name=bucket_name,
    )
