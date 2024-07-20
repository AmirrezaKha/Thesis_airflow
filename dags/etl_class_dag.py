import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

# Add parent directory to system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

# Import the dataProcessor class from the parent directory
from libraries.data_process import dataProcessor

def load_data_to_postgres():
    """
    Loads data from a CSV file into a PostgreSQL database.

    The function performs the following steps:
    1. Retrieves database credentials from environment variables.
    2. Establishes a connection to the PostgreSQL database.
    3. Reads data from a CSV file located in the 'data' directory.
    4. Loads the data into a PostgreSQL table named 'a500'.
    5. Handles errors that may occur during the process.
    """
    try:
        # Retrieve database credentials from environment variables
        db_user = os.getenv('POSTGRES_USER')
        db_password = os.getenv('POSTGRES_PASSWORD')
        db_host = 'postgres'  # Default host for the database
        db_port = '5432'      # Default port for PostgreSQL
        db_name = os.getenv('POSTGRES_DB')

        if not db_user or not db_password or not db_name:
            raise ValueError("Database credentials are not set")

        # Create a connection to the PostgreSQL database
        engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

        # Build the path to the CSV file located in the 'data' directory
        base_dir = os.path.dirname(os.path.dirname(__file__))  # Move to the parent directory
        csv_path = os.path.join(base_dir, 'data', 'a500.csv')

        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_path)

        # Write the DataFrame data to the PostgreSQL table 'a500'
        df.to_sql('a500', engine, if_exists='replace', index=False)

        print("Data loaded successfully!")

    except Exception as e:
        print(f"Error: {e}")
        raise

def data_transformation_and_save(postgres_params, source_table):
    """
    Executes the data transformation and saves the processed DataFrame to PostgreSQL.

    Args:
        postgres_params (dict): Dictionary containing PostgreSQL connection parameters.
        source_table (str): Name of the source table in PostgreSQL.

    """
    # Create an instance of the dataProcessor with the provided parameters
    etl_processor = dataProcessor(
        num_lags=5,
        source_table=source_table,
        dest_table=source_table,  # Save with the same name as the source table
        db_params_df=pd.DataFrame([postgres_params])
    )

    # Run the preprocess_data method to get the processed DataFrame
    mining_df = etl_processor.preprocess_data()

    # Save the processed DataFrame to the PostgreSQL table
    db_user = postgres_params['POSTGRES_USER']
    db_password = postgres_params['POSTGRES_PASSWORD']
    db_host = postgres_params['POSTGRES_HOST']
    db_port = postgres_params['POSTGRES_PORT']
    db_name = postgres_params['POSTGRES_DB']

    engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

    # Write the DataFrame data to the PostgreSQL table with the same name as the source table
    mining_df.to_sql(source_table, engine, if_exists='replace', index=False)

    print(f"Data transformation and saving to PostgreSQL table '{source_table}' completed successfully!")

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'data_pipeline_dag',
    default_args=default_args,
    description='DAG to load data from CSV to PostgreSQL and transform it',
    schedule_interval=None,
)

# Define the task for loading data into PostgreSQL
load_data_task = PythonOperator(
    task_id='load_data_to_postgres',
    python_callable=load_data_to_postgres,
    dag=dag,
)

# Define the task for transforming data and saving it to PostgreSQL
data_transformation_task = PythonOperator(
    task_id='data_transformation_and_save',
    python_callable=data_transformation_and_save,
    op_kwargs={
        'postgres_params': {
            'POSTGRES_USER': os.getenv('POSTGRES_USER'),
            'POSTGRES_PASSWORD': os.getenv('POSTGRES_PASSWORD'),
            'POSTGRES_HOST': 'postgres',
            'POSTGRES_PORT': '5432',
            'POSTGRES_DB': os.getenv('POSTGRES_DB'),
        },
        'source_table': 'mining_table',  
    },
    dag=dag,
)

# Set task dependencies
load_data_task >> data_transformation_task
