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
    
    def train_model(self):
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        mlflow.sklearn.log_model(self.model,"RandomForest",input_example=self.X_train[:5])

        return y_pred_original

    def save_weights(self):
        # Save the model weights to the specified path
        model_save_path = self.model_save_path
        with open(model_save_path, 'wb') as f:
            pd.to_pickle(self.model, f)
        mlflow.log_artifact(self.model_save_path)
        print(f"Model saved at {model_save_path}")

def main():
    mlflow.set_experiment("PM2.5 Random Forest")
    curr_dir = os.getcwd()
    data_prepocessing_path_pkl = os.path.join(curr_dir,'dags/DataPreprocessing/src/data_store_pkl_files')
    sys.path.append(data_prepocessing_path_pkl)
    train_file = os.path.join(data_prepocessing_path_pkl, 'train_data/feature_eng_train_data.pkl')
    model_save_path = os.path.join(curr_dir, 'dags/weights/randomforest_pm25_model.pth')  # Save in .pth format

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():
        start_time = time.time()
        rf_model = RandomForestPM25Model(train_file, None, None, model_save_path)
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
