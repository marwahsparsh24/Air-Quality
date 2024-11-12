from airflow import DAG
import os
import sys
from airflow import configuration as conf
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.dummy import DummyOperator
from datetime import timedelta,datetime
from airflow.utils.dates import days_ago
# from dags.ModelDevelopment.LSTM import main as LSTM

from dags.ModelDevelopment.Training.RandomForest import main as Training_randomforest
from dags.ModelDevelopment.Validation.RandomForest import main as Validation_randomforest
from dags.ModelDevelopment.Training.Prophet import main as Training_prophet
from dags.ModelDevelopment.Validation.Prophet import main as Validation_prophet
from dags.ModelDevelopment.Training.XGBoost import main as Training_xgboost
from dags.ModelDevelopment.Validation.XGBoost import main as Validation_xgboost
from dags.ModelDevelopment.ModelBias.Model_bias import main_pipeline as Bias_check
from dags.bestmodel import main as bestmodel

conf.set('core', 'enable_xcom_pickling', 'True')
conf.set('core', 'enable_parquet_xcom', 'True')

default_args = {
    'owner': 'MLOPS',
    'start_date': datetime(2024, 10, 21),
    'retries': 1, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=1), # Delay before retries
}

dag = DAG(
    'modeling_pipeline',
    default_args=default_args,
    description='Modeling pipeline using Airflow DAG',
    schedule_interval=None,
    catchup=False
)

run_Prophet_train = PythonOperator(
    task_id = 'Prophet_model_training',
    python_callable = Training_prophet,
    dag=dag
)
run_Prophet_validation = PythonOperator(
    task_id = 'Prophet_model_validation',
    python_callable = Validation_prophet,
    dag=dag
)

run_xgboost_Training = PythonOperator(
    task_id = 'XGBoost_model_Train',
    python_callable = Training_xgboost,
    dag=dag
)

run_xgboost_Validation = PythonOperator(
    task_id = 'XGBoost_model_Validation',
    python_callable = Validation_xgboost,
    dag=dag
)

run_random_forest_Training = PythonOperator(
    task_id = 'Random_Forest_model_Train',
    python_callable = Training_randomforest,
    dag=dag
)

run_random_forest_validation = PythonOperator(
    task_id = 'Random_Forest_model_Validation',
    python_callable = Validation_randomforest,
    dag=dag
)

run_bestmodel = PythonOperator(
    task_id = 'Best_Model',
    python_callable = bestmodel,
    dag=dag
)

run_bias_check = PythonOperator(
    task_id = 'Bias_Model',
    python_callable = Bias_check,
    dag=dag
)

run_Prophet_train >> run_Prophet_validation
run_xgboost_Training >> run_xgboost_Validation
run_random_forest_Training >> run_random_forest_validation
[run_Prophet_validation, run_xgboost_Validation, run_random_forest_validation] >> run_bias_check >> run_bestmodel

if __name__ == "__main__":
    dag.cli()

