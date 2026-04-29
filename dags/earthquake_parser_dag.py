from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def fetch_earthquakes():
    # сюда импортируй свой парсер
    print("Fetching earthquakes...")

def save_to_supabase():
    print("Saving to Supabase...")

with DAG(
    dag_id="earthquake_parser",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@hourly",  # каждый час
    catchup=False,
    tags=["earthquakes"],
) as dag:

    fetch = PythonOperator(
        task_id="fetch_earthquakes",
        python_callable=fetch_earthquakes,
    )

    save = PythonOperator(
        task_id="save_to_supabase",
        python_callable=save_to_supabase,
    )

    fetch >> save  # fetch запускается первым, затем save
