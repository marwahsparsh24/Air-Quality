import xgboost as xgb
import shap
import sys
import os
from sklearn.metrics import mean_squared_error
from scipy.special import inv_boxcox
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import time
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV

class XGBoostPM25Model:
    def __init__(self, train_file, test_file, lambda_value, model_save_path):
        self.train_file = train_file
        self.test_file = test_file
        self.lambda_value = lambda_value
        self.model_save_path = model_save_path
        self.param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        self.model_save_path = model_save_path
        self.model = xgb.XGBRegressor(random_state=42)
        self.model.load_model(self.model_save_path)
        #self.model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=5, random_state=42)
        # mlflow.log_param("n_estimators",100)
        # mlflow.log_param("learning_rate",0.01)
        # mlflow.log_param("random_state",42)
        # mlflow.log_param("max_depth",5)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_train_original = None
        self.y_test_original = None
    
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
        """Plots hyperparameter sensitivity results."""
        values, rmses = zip(*results)
        plt.figure(figsize=(10, 6))
        plt.plot(values, rmses, marker='o')
        plt.xlabel(param_name)
        plt.ylabel("RMSE")
        plt.title(f"Hyperparameter Sensitivity: {param_name}")
        
        # Save plot and log it as an artifact
        plot_path = os.path.join(os.getcwd(), f'artifacts/{param_name}_sensitivity_xgboost.png')
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        print(f"Sensitivity plot saved at {plot_path}")
    

    def shap_analysis(self):
        # Initialize SHAP explainer for XGBoost
        explainer = shap.Explainer(self.model, self.X_train)
        
        # Calculate SHAP values for the test set
        shap_values = explainer(self.X_test)
        
        # Plot SHAP summary plot and save it as an artifact
        shap.summary_plot(shap_values, self.X_test, show=False)
        shap_plot_path = os.path.join(os.getcwd(), 'artifacts/shap_summary_plot_xgboost.png')
        plt.savefig(shap_plot_path)
        mlflow.log_artifact(shap_plot_path)
        print(f"SHAP summary plot saved at {shap_plot_path}")
    
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

    def evaluate(self):
        # Make predictions on the test data
        y_pred_boxcox = self.model.predict(self.X_test)

        # Evaluate the model on the transformed target (Box-Cox transformed PM2.5)
        rmse_boxcox = mean_squared_error(self.y_test, y_pred_boxcox, squared=False)
        print(f"RMSE (Box-Cox transformed target): {rmse_boxcox}")

        # Inverse Box-Cox transformation to get predictions back to the original PM2.5 scale
        y_pred_original = inv_boxcox(y_pred_boxcox, self.lambda_value)  # Use the lambda from the Box-Cox transformation

        # Evaluate the model on the original PM2.5 scale (using inverse-transformed predictions)
        rmse_original = mean_squared_error(self.y_test_original, y_pred_original, squared=False)
        mlflow.log_metric("RMSE",rmse_original)
        mlflow.log_metric("RMSE_BoxCOX",rmse_boxcox)
        print(f"RMSE (Original PM2.5 target): {rmse_original}")

        return y_pred_original

    def load_weights(self):
        # Load the model weights from the specified path
        self.model.load_model(self.model_save_path)
        print(f"Model loaded from {self.model_save_path}")

    def plot_results(self, y_pred_original):
        # Plot actual vs predicted values
        plt.figure(figsize=(10,6))
        plt.plot(self.y_test_original.values, label='Actual PM2.5', color='blue')
        plt.plot(y_pred_original, label='Predicted PM2.5', color='red')
        plt.title('Actual vs Predicted PM2.5 Values')
        plt.xlabel('Time')
        plt.ylabel('PM2.5 Value')
        plt.legend()

        # Save the plot as a PNG file
        plot_path = os.path.join(os.getcwd(), 'artifacts/pm25_actual_vs_predicted_XGBoost.png')
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        print(f"Plot saved at {plot_path}")

def main():
    mlflow.set_experiment("PM2.5 XGBoost Prediction")
    curr_dir = os.getcwd()
    data_prepocessing_path_pkl = os.path.join(curr_dir,'DataPreprocessing/src/data_store_pkl_files')
    sys.path.append(data_prepocessing_path_pkl)
    from dags.DataPreprocessing.src.test.data_preprocessing.feature_eng import DataFeatureEngineer
    # Step 1: Load Data using DataFeatureEngineer
    file_path = os.path.join(data_prepocessing_path_pkl, 'test_data/no_anamoly_test_data.pkl')
    # Initialize DataFeatureEngineer to preprocess the data and fetch the lambda value
    engineer = DataFeatureEngineer(file_path)
    engineer.load_data()
    chosen_column = engineer.handle_skewness(column_name='pm25')
    engineer.feature_engineering(chosen_column)
    fitting_lambda = engineer.get_lambda()
    mlflow.log_param("lambda_value", fitting_lambda)

    # Step 2: Define file paths
    train_file = os.path.join(data_prepocessing_path_pkl, 'train_data/feature_eng_train_data.pkl')
    test_file = os.path.join(data_prepocessing_path_pkl, 'test_data/feature_eng_test_data.pkl')
    model_save_path = os.path.join(curr_dir, 'dags/weights/xgboost_pm25_model.pth')

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():    
        xgb_model = XGBoostPM25Model(train_file, test_file, fitting_lambda, model_save_path)
        xgb_model.load_data()
        #xgb_model.train_model()
        y_pred_original = xgb_model.evaluate()
        xgb_model.shap_analysis()
        xgb_model.hyperparameter_sensitivity("n_estimators", [50, 100, 200, 300])
        xgb_model.hyperparameter_sensitivity("learning_rate", [0.01, 0.05, 0.1, 0.2])
        xgb_model.hyperparameter_sensitivity("max_depth", [3, 5, 7, 10])
        xgb_model.load_weights()
        xgb_model.plot_results(y_pred_original)
    mlflow.end_run()
if __name__ == "__main__":
    # mlflow.set_tracking_uri("file:///opt/airflow/dags/mlruns")
    main()
