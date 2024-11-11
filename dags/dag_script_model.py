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
from dags.ModelDevelopment.Prophet import main as prophet
from dags.ModelDevelopment.RandomForest import main as randomforest
from dags.ModelDevelopment.XGBoost import main as xgboost
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

# run_LSTM = PythonOperator(
#     task_id = 'LSTM model',
#     python_callable = LSTM,
#     dag=dag
# )

run_Prophet = PythonOperator(
    task_id = 'Prophet_model',
    python_callable = prophet,
    dag=dag
)

run_xgboost = PythonOperator(
    task_id = 'XGBoost_model',
    python_callable = xgboost,
    dag=dag
)

run_random_forest = PythonOperator(
     task_id = 'Random_Forest_model',
    python_callable = randomforest,
    dag=dag
)

run_bestmodel = PythonOperator(
     task_id = 'Best_Model',
    python_callable = bestmodel,
    dag=dag
)


# order in which tasks are run
run_Prophet >> run_xgboost >> run_random_forest >> run_bestmodel

if __name__ == "__main__":
    dag.cli()

