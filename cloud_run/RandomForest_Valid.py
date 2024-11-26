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
from scipy import stats
from sklearn.model_selection import GridSearchCV
from google.cloud import storage
import io
from io import BytesIO
import pickle5 as pickle
import numpy as np



def setup_mlflow_tracking():
    MLFLOW_TRACKING_DIR = os.environ.get('MLFLOW_TRACKING_DIR', '/app/mlruns')
    os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR}")


class DataFeatureEngineer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.fitting_lambda = None
    
    def get_lambda(self):
        return self.fitting_lambda

    def load_data(self):

        client = storage.Client()

        bucket_name = 'airquality-mlops-rg'

        # Get the bucket and the blob (file)
        bucket = client.bucket(bucket_name)

        blob_name = os.path.join(self.file_path)
        blob = bucket.blob(blob_name)
        pickle_data = blob.download_as_bytes() 
        self.data = pickle.load(BytesIO(pickle_data))
        print(f"Data loaded from {self.file_path}.")
    
    def handle_skewness(self, column_name='pm25'):
        skewness = self.data[column_name].skew()
        print(f'Original Skewness: {skewness}')
        if np.abs(skewness)<0.5:
            return column_name
        else:
            self.data[f'{column_name}_log'] = np.log1p(self.data[column_name])
            log_skewness = self.data[f'{column_name}_log'].skew()
            print(f'Log Transformed Skewness: {log_skewness}')

            self.data[f'{column_name}_boxcox'], self.fitting_lambda = stats.boxcox(self.data[column_name] + 1)
            boxcox_skewness = self.data[f'{column_name}_boxcox'].skew()
            print(f'Box-Cox Transformed Skewness: {boxcox_skewness}')

            if abs(boxcox_skewness) < abs(log_skewness):
                self.data.drop(columns=[f'{column_name}_log'],inplace=True)
                print("Choosing Box-Cox transformed column.")
                return f'{column_name}_boxcox'
            else:
                print("Choosing Log transformed column.")
                self.data.drop(columns=[f'{column_name}_boxcox'],inplace=True)
                return f'{column_name}_log'

    def feature_engineering(self, chosen_column):
        # Create lag features
        for lag in range(1, 6):  # Creates lag_1 to lag_5
            self.data[f'lag_{lag}'] = self.data[chosen_column].shift(lag)

        self.data['rolling_mean_3'] = self.data[chosen_column].rolling(window=3).mean()
        self.data['rolling_mean_6'] = self.data[chosen_column].rolling(window=6).mean()
        self.data['rolling_mean_24'] = self.data[chosen_column].rolling(window=24).mean()
        self.data['rolling_std_3'] = self.data[chosen_column].rolling(window=3).std()
        self.data['rolling_std_6'] = self.data[chosen_column].rolling(window=6).std()
        self.data['rolling_std_24'] = self.data[chosen_column].rolling(window=24).std()
        self.data['ema_3'] = self.data[chosen_column].ewm(span=3, adjust=False).mean()
        self.data['ema_6'] = self.data[chosen_column].ewm(span=6, adjust=False).mean()
        self.data['ema_24'] = self.data[chosen_column].ewm(span=24, adjust=False).mean()
        self.data['diff_1'] = self.data[chosen_column].diff(1)
        self.data['diff_2'] = self.data[chosen_column].diff(2)
        self.data['hour'] = self.data.index.hour
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['day_of_year'] = self.data.index.dayofyear
        self.data['month'] = self.data.index.month
        self.data['sin_hour'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['cos_hour'] = np.cos(2 * np.pi * self.data['hour'] / 24)
        self.data['sin_day_of_week'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['cos_day_of_week'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
        self.data.dropna(inplace=True)
        print("Feature engineering completed and NaN values dropped.")


class RandomForestPM25Model:
    def __init__(self, train_file, test_file, lambda_value, model_save_path):
        self.train_file = train_file
        self.test_file = test_file
        self.lambda_value = lambda_value
        self.model_save_path = model_save_path
        self.param_grid = {
            'n_estimators': [100, 200]
        }
        self.model = RandomForestRegressor(random_state=42)

        bucket_name = "airquality-mlops-rg"
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(self.model_save_path)

        temp_file_name = "/tmp/temp_model.pth"
        # Download the file from the bucket to the temporary path
        blob.download_to_filename(temp_file_name)

        try:
            with open(temp_file_name, 'rb') as f:
                self.model = pd.read_pickle(f)
            print("Model loaded successfully using pandas.read_pickle.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_train_original = None
        self.y_test_original = None
    
    def load_data(self):
        # Load training and test data
        client = storage.Client()

        bucket_name = 'airquality-mlops-rg'
        pickle_file_path = 'processed/train/feature_eng_data.pkl'
        pickle_file_path_test = 'processed/test/feature_eng_data.pkl'

        # Get the bucket and the blob (file)
        bucket = client.bucket(bucket_name)

        blob_name = os.path.join(pickle_file_path)
        blob = bucket.blob(blob_name)
        pickle_data = blob.download_as_bytes() 
        train_data = pickle.load(BytesIO(pickle_data))

        blob_name_test = os.path.join(pickle_file_path_test)
        blob_test = bucket.blob(blob_name_test)
        pickle_data_test = blob_test.download_as_bytes() 
        test_data = pickle.load(BytesIO(pickle_data_test))

        # Extract Box-Cox transformed y and original y
        for column in train_data.columns:
            if column == 'pm25_boxcox' or column == 'pm25_log':
                self.y_train = train_data[column]
                break
        self.y_train_original = train_data['pm25']
        self.X_train = train_data.drop(columns=['pm25'])

        for column in test_data.columns:
            if column == 'pm25_boxcox' or column == 'pm25_log':
                self.y_test = test_data[column]
                break
        self.y_test_original = test_data['pm25']
        self.X_test = test_data.drop(columns=['pm25'])

    def hyperparameter_sensitivity(self, param_name, param_values):
        sensitivity_results = []

        for value in param_values:
            # Set the hyperparameter to the current value
            self.model.set_params(**{param_name: value})
            
            # Train and evaluate the model
            self.model.fit(self.X_train, self.y_train)
            y_pred = self.model.predict(self.X_test)
            rmse = mean_squared_error(self.y_test, y_pred, squared=False)
            
            # Log the RMSE and the hyperparameter value
            mlflow.log_metric(f"{param_name}_{value}_RMSE", rmse)
            sensitivity_results.append((value, rmse))
            print(f"RMSE for {param_name}={value}: {rmse}")

        # Optionally, plot results to visualize the sensitivity
        self.plot_sensitivity(param_name, sensitivity_results)
    
    def plot_sensitivity(self, param_name, results):
        storage_client = storage.Client()
        bucket_name = "airquality-mlops-rg"
        bucket = storage_client.bucket(bucket_name)
        """Plots hyperparameter sensitivity results."""
        values, rmses = zip(*results)
        plt.figure(figsize=(10, 6))
        plt.plot(values, rmses, marker='o')
        plt.xlabel(param_name)
        plt.ylabel("RMSE")
        plt.title(f"Hyperparameter Sensitivity: {param_name}")


        artifact_dir = os.path.join(os.getcwd(), 'artifacts')
        os.makedirs(artifact_dir, exist_ok=True)
        plot_path = os.path.join(artifact_dir, f'{param_name}_sensitivity_randomforest.png')
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        
        # Save plot and log it as an artifact
        destination_blob_name = f'artifacts/{param_name}_sensitivity_randomforest.png'
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(plot_path)
    
    def shap_analysis(self):
        storage_client = storage.Client()
        bucket_name = "airquality-mlops-rg"
        bucket = storage_client.bucket(bucket_name)
        # Initialize SHAP explainer for XGBoost
        explainer = shap.Explainer(self.model, self.X_train)
        
        # Calculate SHAP values for the test set
        shap_values = explainer(self.X_test)

        artifact_dir = os.path.join(os.getcwd(), 'artifacts')
        os.makedirs(artifact_dir, exist_ok=True)
        
        # Plot SHAP summary plot and save it as an artifact
        shap.summary_plot(shap_values, self.X_test, show=False)
        shap_plot_path = os.path.join(os.getcwd(), 'artifacts/shap_summary_plot_randomforest.png')
        plt.savefig(shap_plot_path)
        mlflow.log_artifact(shap_plot_path)
        print(f"SHAP summary plot saved at {shap_plot_path}")

        destination_blob_name = f'artifacts/shap_summary_plot_randomforest.png'
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(shap_plot_path)

    def evaluate(self):
        # Make predictions on the test data
        y_pred_boxcox = self.model.predict(self.X_test)

        # Evaluate the model on the transformed target (Box-Cox transformed PM2.5)
        rmse_boxcox = mean_squared_error(self.y_test, y_pred_boxcox, squared=False)
        print(f"RMSE (Box-Cox transformed target): {rmse_boxcox}")

        # Inverse Box-Cox transformation to get predictions back to the original PM2.5 scale
        y_pred_original = inv_boxcox(y_pred_boxcox, self.lambda_value)

        # Evaluate the model on the original PM2.5 scale (using inverse-transformed predictions)
        rmse_original = mean_squared_error(self.y_test_original, y_pred_original, squared=False)
        mlflow.log_metric("RMSE",rmse_original)
        mlflow.log_metric("RMSE_BoxCOX",rmse_boxcox)
        print(f"RMSE (Original PM2.5 target): {rmse_original}")

        return y_pred_original

    def load_weights(self):

        bucket_name = "airquality-mlops-rg"
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(self.model_save_path)

        temp_file_name = "/tmp/temp_model.pth"
        # Download the file from the bucket to the temporary path
        blob.download_to_filename(temp_file_name)

        try:
            with open(temp_file_name, 'rb') as f:
                self.model = pd.read_pickle(f)
            print("Model loaded successfully using pandas.read_pickle.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def plot_results(self, y_pred_original):
        storage_client = storage.Client()
        bucket_name = "airquality-mlops-rg"
        bucket = storage_client.bucket(bucket_name)
        # Plot actual vs predicted values
        plt.figure(figsize=(10,6))
        plt.plot(self.y_test_original.values, label='Actual PM2.5', color='blue')
        plt.plot(y_pred_original, label='Predicted PM2.5', color='red')
        plt.title('Actual vs Predicted PM2.5 Values')
        plt.xlabel('Time')
        plt.ylabel('PM2.5 Value')
        plt.legend()

        artifact_dir = os.path.join(os.getcwd(), 'artifacts')
        os.makedirs(artifact_dir, exist_ok=True)

        # Save the plot as a PNG file
        plot_path = os.path.join(os.getcwd(), 'artifacts/pm25_actual_vs_predicted_RandomForest.png')
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        print(f"Plot saved at {plot_path}")

        destination_blob_name = f'pm25_actual_vs_predicted_RandomForest.png'
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(plot_path)

def main():
    setup_mlflow_tracking()
    mlflow.set_experiment("PM2.5 Random Forest")
    # curr_dir = os.getcwd()

    bucket_name = "airquality-mlops-rg"
    train_file_gcs = f'processed/train/feature_eng_data.pkl'
    test_file_gcs = f'processed/test/feature_eng_data.pkl'
    model_save_path_gcs = f'weights/rf_model.pth'
    
    file_path = f'processed/test/anamoly_data.pkl'
    engineer = DataFeatureEngineer(file_path)
    engineer.load_data()
    chosen_column = engineer.handle_skewness(column_name='pm25')
    engineer.feature_engineering(chosen_column)
    fitting_lambda = engineer.get_lambda()
    mlflow.log_param("lambda_value", fitting_lambda)

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():
        rf_model = RandomForestPM25Model(train_file_gcs, test_file_gcs, fitting_lambda, model_save_path_gcs)
        rf_model.load_data()
        y_pred_original = rf_model.evaluate()
        rf_model.shap_analysis()
        rf_model.hyperparameter_sensitivity("n_estimators", [100, 200])
        rf_model.load_weights()
        rf_model.plot_results(y_pred_original)
    mlflow.end_run()

if __name__ == "__main__":
    main()
