import mlflow
import os
import pickle
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

    if best_model_uri:
        # Load the best model as a .pth file
        print(f"Best model found in experiment '{best_experiment_name}' with run ID '{best_run_id}'")
        print(f"Validation RMSE: {best_rmse}")
        
        # Download model artifact as a .pth file from MLflow
        model_path = mlflow.artifacts.download_artifacts(best_model_uri)
        
        # Attempt to load the .pth file using pickle
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)  # This assumes the .pth file can be read by pickle
        
        return model_data, best_rmse, best_experiment_name, best_run_id
    else:
        print("No valid models found across experiments.")
        return None, None, None, None

def main():
    # Usage
    experiment_names = ["PM2.5 Random Forest", "PM2.5 XGBoost Prediction", "PM2.5 Prophet"]
    best_model_data, best_rmse, best_experiment_name, best_run_id = get_best_model_and_load_weights(experiment_names)
    print(best_experiment_name)
    print(best_run_id)
    print(best_rmse)
    print(best_model_data)

    if best_model_data:
        print(f"Loaded best model data from experiment '{best_experiment_name}' with RMSE: {best_rmse}")

if __name__ == "__main__":
    main()
