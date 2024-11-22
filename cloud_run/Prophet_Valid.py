import sys
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.special import inv_boxcox
from prophet import Prophet
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.prophet
import mlflow.pyfunc
import time
import shap


class ProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, prophet_model):
        self.prophet_model = prophet_model
    
    def predict(self, context, model_input):
        # Assuming model_input is a DataFrame with 'ds' column (date)
        return self.prophet_model.predict(model_input)['yhat']

class ProphetPM25Model:
    def __init__(self, train_file, test_file, lambda_value, model_save_path):
        self.train_file = train_file
        self.test_file = test_file
        self.lambda_value = lambda_value
        self.model_save_path = model_save_path
        self.model = Prophet()
        with open(self.model_save_path, 'rb') as f:
            self.model = pickle.load(f)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_train_original = None
        self.y_test_original = None
    
    def load_data(self):
        # Load training and test data
        train_data = pd.read_pickle(self.train_file)
        test_data = pd.read_pickle(self.test_file)

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

    def shap_analysis(self):
        # Custom predict function for SHAP that works with Prophet
        def prophet_predict(df):
            future = pd.DataFrame({'ds': df['ds']})
            forecast = self.model.predict(future)
            return forecast['yhat'].values

        # Prepare the dataset for SHAP
        # Convert 'ds' to a numeric format (e.g., Unix timestamp)
        shap_data = pd.DataFrame({'ds': self.X_test.index.astype(np.int64) // 10**9})  # Convert to seconds

        # Initialize SHAP Explainer with the custom predict function
        explainer = shap.Explainer(prophet_predict, shap_data)

        # Calculate SHAP values for the test set
        shap_values = explainer(shap_data)

        # Plot SHAP summary and save it as an artifact
        shap.summary_plot(shap_values, shap_data, show=False)
        shap_plot_path = os.path.join(os.getcwd(), 'artifacts/shap_summary_plot_prophet.png')
        plt.savefig(shap_plot_path)
        mlflow.log_artifact(shap_plot_path)
        print(f"SHAP summary plot saved at {shap_plot_path}")

    def preprocess_data(self):
        # Prepare training data in Prophet format
        self.df_train = pd.DataFrame({
            'ds': self.X_train.index.tz_localize(None),  # Remove timezone
            'y': self.y_train  # Use Box-Cox transformed target
        })

    def evaluate(self):
        # Make future dataframe
        future = self.model.make_future_dataframe(periods=len(self.X_test))
        future['ds'] = future['ds'].dt.tz_localize(None)  # Remove timezone

        # Predict on the test data
        forecast = self.model.predict(future)
        y_pred_boxcox = forecast['yhat'][-len(self.X_test):].values

        # Inverse Box-Cox transformation to get predictions back to original PM2.5 scale
        y_pred_original = inv_boxcox(y_pred_boxcox, self.lambda_value)

        # Handle NaN values in predictions
        valid_idx = ~np.isnan(y_pred_original) & ~np.isnan(self.y_test_original)
        y_pred_original_valid = y_pred_original[valid_idx]
        y_test_original_valid = self.y_test_original[valid_idx]

        # Evaluate RMSE on the original PM2.5 scale
        rmse_original = mean_squared_error(y_test_original_valid, y_pred_original_valid, squared=False)
        mlflow.log_metric("RMSE",rmse_original)
        print(f"RMSE (Original PM2.5 target): {rmse_original}")

        return y_pred_original_valid

    def load_weights(self):
        # Load the Prophet model from the specified path
        with open(self.model_save_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {self.model_save_path}")

    def plot_results(self, y_pred_original_valid):
        # Plot actual vs predicted values
        plt.figure(figsize=(10,6))
        plt.plot(self.y_test_original.values, label='Actual PM2.5', color='blue')
        plt.plot(y_pred_original_valid, label='Predicted PM2.5', color='red')
        plt.title('Actual vs Predicted PM2.5 Values with Prophet')
        plt.xlabel('Time')
        plt.ylabel('PM2.5 Value')
        plt.legend()

        # Save the plot as a PNG file
        plot_path = os.path.join(os.getcwd(), 'artifacts/pm25_actual_vs_predicted_Prophet.png')
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        print(f"Plot saved at {plot_path}")
# Main function to orchestrate the workflow
def main():
    mlflow.set_experiment("PM2.5 Prophet")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns"))
    curr_dir = os.getcwd()
    print(curr_dir)
    data_prepocessing_path_pkl = os.path.join(curr_dir,'DataPreprocessing/src/data_store_pkl_files')
    # Step 1: Load Data using DataFeatureEngineer
    file_path = os.path.join(data_prepocessing_path_pkl, 'test_data/no_anamoly_test_data.pkl')
    sys.path.append(data_prepocessing_path_pkl)
    from DataPreprocessing.src.test.data_preprocessing.feature_eng import DataFeatureEngineer
    engineer = DataFeatureEngineer(file_path)
    engineer.load_data()
    chosen_column = engineer.handle_skewness(column_name='pm25')
    engineer.feature_engineering(chosen_column)
    fitting_lambda = engineer.get_lambda()
    mlflow.log_param("lambda_value", fitting_lambda)

    # Step 2: Define file paths
    train_file = os.path.join(data_prepocessing_path_pkl, 'train_data/feature_eng_train_data.pkl')
    test_file = os.path.join(data_prepocessing_path_pkl, 'test_data/feature_eng_test_data.pkl')
    model_save_path = os.path.join(curr_dir, 'weights/prophet_pm25_model.pth')
    
    if mlflow.active_run():
        mlflow.end_run()
    
    with mlflow.start_run() as run:
        prophet_model = ProphetPM25Model(train_file, test_file, fitting_lambda, model_save_path)
        prophet_model.load_data()
        prophet_model.preprocess_data()
        y_pred_original = prophet_model.evaluate()
        prophet_model.shap_analysis()
        prophet_model.load_weights()
        prophet_model.plot_results(y_pred_original)
    mlflow.end_run()
if __name__ == "__main__":
    main()
