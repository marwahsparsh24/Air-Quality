import mlflow
import os
import pickle
from dags.ModelDevelopment.XGBoost import XGBoostPM25Model
from dags.ModelDevelopment.Prophet import ProphetPM25Model
from dags.ModelDevelopment.RandomForest import RandomForestPM25Model


def load_randomforest_model(filepath):
    randomforest = RandomForestPM25Model(train_file=None,test_file=None,lambda_value=None,model_save_path=filepath)
    return randomforest.load_weights()

def load_xgboost_model(filepath):
    xgbmodel = XGBoostPM25Model(train_file=None,test_file=None,lambda_value=None,model_save_path=filepath)
    return xgbmodel.load_weights()

def load_prophet_model(filepath):
    prophetmodel = ProphetPM25Model(train_file=None,test_file=None,lambda_value=None,model_save_path=filepath)
    return prophetmodel.load_weights()

def get_best_model_and_load_weights(experiment_names):

    # Set the tracking URI to log locally
    #mlflow.set_tracking_uri("file:///opt/airflow/dags/mlruns")

    best_rmse = float('inf')
    best_model_uri = None
    best_experiment_name = None
    best_run_id = None

    # Loop through each experiment to find the best model
    for experiment_name in experiment_names:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found.")
            continue
        experiment_id = experiment.experiment_id
        print(experiment_id)

        # Get all runs for the current experiment
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        if runs.empty:
            print(f"No runs found for experiment '{experiment_name}'.")
            continue

        # Identify the run with the lowest RMSE in this experiment
        best_run_in_experiment = runs.loc[runs['metrics.RMSE'].idxmin()]
        current_rmse = best_run_in_experiment['metrics.RMSE']

        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_experiment_name = experiment_name
            best_run_id = best_run_in_experiment['run_id']
            if best_experiment_name == "PM2.5 Prophet":
                best_model_uri = f"runs:/{best_run_in_experiment['run_id']}/prophet_pm25_model.pth"
                model_path = mlflow.artifacts.download_artifacts(best_model_uri)
                model = load_prophet_model(model_path)
                # model = mlflow.prophet.load_model(model_path)
                
            elif best_experiment_name == "PM2.5 Random Forest":
                best_model_uri = f"runs:/{best_run_in_experiment['run_id']}/randomforest_pm25_model.pth"
                model_path = mlflow.artifacts.download_artifacts(best_model_uri)
                # model = mlflow.sklearn.load_model(best_model_uri)
                model = load_randomforest_model(model_path)

            elif best_experiment_name == "PM2.5 XGBoost Prediction":
                best_model_uri = f"runs:/{best_run_in_experiment['run_id']}/xgboost_pm25_model.pth"
                model_path = mlflow.artifacts.download_artifacts(best_model_uri)
                # model = mlflow.xgboost.load_model(model_path)
                model = load_xgboost_model(model_path)

    if best_model_uri:
        # Load the best model as a .pth file
        print(f"Best model found in experiment '{best_experiment_name}' with run ID '{best_run_id}'")
        print(f"Validation RMSE: {best_rmse}")
        
        # if best_experiment_name == "PM2.5 XGBoost Prediction":
        #     self.model.load_model(self.model_save_path)
        # # Attempt to load the .pth file using pickle
        # with open(model_path, 'rb') as f:
        #     model_data = pickle.load(f)  # This assumes the .pth file can be read by pickle
        
        return model, best_rmse, best_experiment_name, best_run_id
    else:
        print("No valid models found across experiments.")
        return None, None, None, None

def main():
    # Usage
    experiment_names = ["PM2.5 Random Forest", "PM2.5 XGBoost Prediction", "PM2.5 Prophet"]
    model, best_rmse, best_experiment_name, best_run_id = get_best_model_and_load_weights(experiment_names)
    print(best_experiment_name)
    print(model)
    print(best_run_id)
    print(best_rmse)

if __name__ == "__main__":
    main()
