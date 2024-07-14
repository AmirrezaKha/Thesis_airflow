import os
import pandas as pd
from sqlalchemy import create_engine
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

def load_data_to_postgres():
    try:
        # Database credentials from environment variables
        db_user = os.getenv('POSTGRES_USER')
        db_password = os.getenv('POSTGRES_PASSWORD')
        db_host = 'postgres'
        db_port = '5432'
        db_name = os.getenv('POSTGRES_DB')

        print(f"User: {db_user}, Password: {db_password}, DB: {db_name}")

        if not db_user or not db_password or not db_name:
            raise ValueError("Database credentials are not set")

        # Create a connection to the database
        engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

        # Build the path to the CSV file
        base_dir = os.path.dirname(os.path.dirname(__file__))  # Go to the parent directory
        csv_path = os.path.join(base_dir, 'data', 'a500.csv')

        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Write the data to the database
        df.to_sql('a500', engine, if_exists='replace', index=False)

        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error: {e}")
        raise

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'retries': 1,
}

dag = DAG(
    'load_data_dag',
    default_args=default_args,
    description='Simple DAG to load data from CSV to PostgreSQL',
    schedule_interval=None,
)

load_data_task = PythonOperator(
    task_id='load_data_to_postgres',
    python_callable=load_data_to_postgres,
    dag=dag,
)
