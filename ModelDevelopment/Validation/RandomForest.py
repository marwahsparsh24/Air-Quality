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
        with open(model_save_path, 'rb') as f:
            self.model = pd.read_pickle(f)
        # mlflow.log_param("n_estimators",100)
        # mlflow.log_param("random_state",42)
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

    def hyperparameter_sensitivity(self, param_name, param_values):
        """Analyzes sensitivity of model performance to a specified hyperparameter.
        
        Parameters:
            param_name (str): The hyperparameter to vary.
            param_values (list): List of values to test for the specified hyperparameter.
        """
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
        plot_path = os.path.join(os.getcwd(), f'artifacts/{param_name}_sensitivity_randomforest.png')
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
        shap_plot_path = os.path.join(os.getcwd(), 'dags/artifacts/shap_summary_plot_randomforest.png')
        plt.savefig(shap_plot_path)
        mlflow.log_artifact(shap_plot_path)
        print(f"SHAP summary plot saved at {shap_plot_path}")

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
        # Load the model weights from the specified path
        model_save_path = self.model_save_path
        with open(model_save_path, 'rb') as f:
            self.model = pd.read_pickle(f)
        print(f"Model loaded from {model_save_path}")

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
        plot_path = os.path.join(os.getcwd(), 'artifacts/pm25_actual_vs_predicted_RandomForest.png')
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        print(f"Plot saved at {plot_path}")

def main():
    mlflow.set_experiment("PM2.5 Random Forest")
    curr_dir = os.getcwd()
    # main_path = os.path.abspath(os.path.join(curr_dir, '.'))
    # data_prepocessing_path = os.path.abspath(os.path.join(main_path, 'DataPreprocessing'))
    # data_prepocessing_path_pkl = os.path.abspath(os.path.join(main_path, 'DataPreprocessing/src/data_store_pkl_files'))
    data_prepocessing_path_pkl = os.path.join(curr_dir,'DataPreprocessing/src/data_store_pkl_files')
    file_path = os.path.join(data_prepocessing_path_pkl, 'test_data/no_anamoly_test_data.pkl')
    # sys.path.append(main_path)
    # sys.path.append(data_prepocessing_path)
    sys.path.append(data_prepocessing_path_pkl)
    from DataPreprocessing.src.test.data_preprocessing.feature_eng import DataFeatureEngineer
    engineer = DataFeatureEngineer(file_path)
    engineer.load_data()
    chosen_column = engineer.handle_skewness(column_name='pm25')
    engineer.feature_engineering(chosen_column)
    fitting_lambda = engineer.get_lambda()
    mlflow.log_param("lambda_value", fitting_lambda)
    train_file = os.path.join(data_prepocessing_path_pkl, 'train_data/feature_eng_train_data.pkl')
    test_file = os.path.join(data_prepocessing_path_pkl, 'test_data/feature_eng_test_data.pkl')
    model_save_path = os.path.join(curr_dir, 'weights/randomforest_pm25_model.pth')  # Save in .pth format

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():
        rf_model = RandomForestPM25Model(train_file, test_file, fitting_lambda, model_save_path)
        rf_model.load_data()
        y_pred_original = rf_model.evaluate()
        rf_model.shap_analysis()
        rf_model.hyperparameter_sensitivity("n_estimators", [100, 200])
        rf_model.load_weights()
        rf_model.plot_results(y_pred_original)
    mlflow.end_run()

if __name__ == "__main__":
    # mlflow.set_tracking_uri("file:///opt/airflow/dags/mlruns")
    main()
