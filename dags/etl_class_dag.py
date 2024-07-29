import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

# Add the 'libraries' directory to the Python path
current_directory = os.path.dirname(os.path.abspath(__file__))
libraries_directory = os.path.join(current_directory, 'libraries')
sys.path.append(libraries_directory)

# Now you can import the required classes
from data_process import dataProcessor
from parent_models import *
from RF_models import *
from fnn_models import *
from lstm_models import *

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

def data_transformation_and_save(postgres_params, source_table, destination_table):
    """
    Executes the data transformation and saves the processed DataFrame to PostgreSQL.

    Args:
        postgres_params (dict): Dictionary containing PostgreSQL connection parameters.
        source_table (str): Name of the source table in PostgreSQL.
        destination_table (str): Name of the destination table in PostgreSQL.
    """
    try:
        # Create an instance of the dataProcessor with the provided parameters
        etl_processor = dataProcessor(
            num_lags=5,
            source_table=source_table,
            dest_table=destination_table,  # Use the correct destination table name
            db_params_df=pd.DataFrame([postgres_params])
        )

        # Run the preprocess_data method to get the processed DataFrame
        mining_df = etl_processor.preprocess_data()

        # Verify the DataFrame content
        print(mining_df.head())

        # Extract database connection parameters
        db_user = postgres_params['POSTGRES_USER']
        db_password = postgres_params['POSTGRES_PASSWORD']
        db_host = postgres_params['POSTGRES_HOST']
        db_port = postgres_params['POSTGRES_PORT']
        db_name = postgres_params['POSTGRES_DB']

        # Create the SQLAlchemy engine
        engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

        # Write the DataFrame data to the PostgreSQL table with the specified name
        mining_df.to_sql(destination_table, engine, if_exists='replace', index=False)

        print(f"Data transformation and saving to PostgreSQL table '{destination_table}' completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

def classification_models(postgres_params, source_table, destination_table):
    """
    This function performs the following tasks:

    1. **Database Connection**:
        - Establishes a connection to a PostgreSQL database using credentials provided in `postgres_params`.
        - Retrieves data from the specified `source_table` into a Pandas DataFrame.

    2. **Model Training**:
        - Initializes and trains two classification models (`rf_model` and `fnn_model`) on the data fetched.
        - Extracts various performance metrics from each model:
            - `indices`: Important indices or features used by the models.
            - `best_model`: The best-performing model (could be a trained model or its parameters).
            - `train_loss`: Loss values recorded during training.
            - `test_loss`: Loss values recorded during testing.
            - `confusion_matrix`: A matrix showing the classification performance.
            - `best_train_error`: Best training error observed.
            - `best_test_error`: Best testing error observed.
            - `best_iteration`: The iteration at which the best performance was achieved.

    3. **Results Compilation**:
        - Compiles the results from both models into a Pandas DataFrame.
        - Converts all elements into strings to ensure uniformity and prevent data type issues.

    4. **Save Results**:
        - Saves the compiled results into a new table named `classification_results` in the PostgreSQL database.
        - Replaces any existing table with the same name.

    5. **Error Handling**:
        - Implements error handling to catch and print exceptions that occur during database connection, data fetching, model training, or results processing.

    Args:
        postgres_params (dict): Dictionary containing PostgreSQL connection parameters.
        source_table (str): Name of the table from which to fetch the data.

    Returns:
        None
    """
    try:
        # Create a connection to the PostgreSQL database
        db_user = postgres_params['POSTGRES_USER']
        db_password = postgres_params['POSTGRES_PASSWORD']
        db_host = postgres_params['POSTGRES_HOST']
        db_port = postgres_params['POSTGRES_PORT']
        db_name = postgres_params['POSTGRES_DB']
        
        engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

        # Query to fetch data from the source_table
        query = f"SELECT * FROM {source_table}"
        mining_df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error connecting to the database or fetching data: {e}")
        return

    target_column = 'Daily_Actual_Demand'
    random_state = 42

    try:
        rf_model_instance = rf_model(mining_df, target_column, random_state)
        fnn_model_instance = fnn_model(mining_df, target_column, random_state)

        indices_rf, best_model_rf, train_loss_rf, test_loss_rf, confusion_mat_rf, best_train_error_rf, best_test_error_rf, best_iteration_rf = rf_model_instance.class_main()
        indices_fnn, best_model_fnn, train_loss_fnn, test_loss_fnn, confusion_mat_fnn, best_train_error_fnn, best_test_error_fnn, best_iteration_fnn = fnn_model_instance.class_main()
    except Exception as e:
        print(f"Error training models: {e}")
        return

    try:
        # Ensure all elements are converted to strings
        results = pd.DataFrame({
            'model': ["RF", "FNN"],
            'indices': [str(indices_rf), str(indices_fnn)],
            'best_model': [str(best_model_rf), str(best_model_fnn)],
            'train_loss': [str(train_loss_rf), str(train_loss_fnn)],
            'test_loss': [str(test_loss_rf), str(test_loss_fnn)],
            'confusion_matrix': [str(confusion_mat_rf), str(confusion_mat_fnn)],
            'best_train_error': [str(best_train_error_rf), str(best_train_error_fnn)],
            'best_test_error': [str(best_test_error_rf), str(best_test_error_fnn)],
            'best_iteration': [str(best_iteration_rf), str(best_iteration_fnn)]
        })

        # Save the results to a new table in PostgreSQL
        results.to_sql(destination_table, engine, if_exists='replace', index=False)
        print("Classification results saved successfully!")
    except Exception as e:
        print(f"Error processing results or saving to database: {e}")
        return

def regression_models(postgres_params, source_table, destination_table):
    """
    This function performs the following tasks:

    1. **Database Connection**:
        - Establishes a connection to a PostgreSQL database using credentials provided in `postgres_params`.
        - Retrieves data from the specified `source_table` into a Pandas DataFrame.
        - Fetches indices from the `classification_results` table to be used in regression models.

    2. **Model Training**:
        - Initializes and trains two regression models (`rf_model` and `fnn_model`) on the data fetched, using the extracted indices from the classification results.
        - Extracts various performance metrics from each model:
            - `indices`: Important indices or features used by the models.
            - `best_model`: The best-performing model (could be a trained model or its parameters).
            - `result_df`: DataFrame containing the model results.
            - `train_mae`: Mean Absolute Error recorded during training.
            - `test_mae`: Mean Absolute Error recorded during testing.
            - `best_train_error`: Best training error observed.
            - `best_test_error`: Best testing error observed.
            - `best_iteration`: The iteration at which the best performance was achieved.

    3. **Results Compilation**:
        - Compiles the results from both models into a Pandas DataFrame.
        - Converts all elements into strings to ensure uniformity and prevent data type issues.

    4. **Save Results**:
        - Saves the compiled results into a new table named `destination_table` in the PostgreSQL database.
        - Replaces any existing table with the same name.

    5. **Error Handling**:
        - Implements error handling to catch and print exceptions that occur during database connection, data fetching, indices retrieval, model training, or results processing.

    Args:
        postgres_params (dict): Dictionary containing PostgreSQL connection parameters.
            - POSTGRES_USER (str): The username for the PostgreSQL database.
            - POSTGRES_PASSWORD (str): The password for the PostgreSQL database.
            - POSTGRES_HOST (str): The host address for the PostgreSQL database.
            - POSTGRES_PORT (str): The port number for the PostgreSQL database.
            - POSTGRES_DB (str): The name of the PostgreSQL database.
        source_table (str): Name of the table from which to fetch the data.
        destination_table (str): Name of the table to which the results will be saved.

    Returns:
        None
    """
    try:
        # Create a connection to the PostgreSQL database
        db_user = postgres_params['POSTGRES_USER']
        db_password = postgres_params['POSTGRES_PASSWORD']
        db_host = postgres_params['POSTGRES_HOST']
        db_port = postgres_params['POSTGRES_PORT']
        db_name = postgres_params['POSTGRES_DB']
        
        engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

        # Query to fetch data from the source_table
        query = f"SELECT * FROM {source_table}"
        mining_df = pd.read_sql(query, engine)
        
        # Query to fetch indices from classification_results table
        indices_query = "SELECT model, indices FROM classification_results"
        indices_df = pd.read_sql(indices_query, engine)

        # Extract indices from the classification_results table
        indices_rf = eval(indices_df[indices_df['model'] == 'RF']['indices'].values[0])
        indices_fnn = eval(indices_df[indices_df['model'] == 'FNN']['indices'].values[0])
    except Exception as e:
        print(f"Error connecting to the database or fetching data: {e}")
        return

    target_column = 'Daily_Actual_Demand'
    random_state = 42

    try:
        # Train regression models using the extracted indices
        rf_model_instance = rf_model(mining_df, target_column, random_state, indices=indices_rf)
        fnn_model_instance = fnn_model(mining_df, target_column, random_state, indices=indices_fnn)

        indices_rf, best_model_rf, result_df_rf, train_mae_rf, test_mae_rf, best_train_error_rf, best_test_error_rf, best_iteration_rf = rf_model_instance.reg_main()
        indices_fnn, best_model_fnn, result_df_fnn, train_mae_fnn, test_mae_fnn, best_train_error_fnn, best_test_error_fnn, best_iteration_fnn = fnn_model_instance.reg_main()
    except Exception as e:
        print(f"Error training models: {e}")
        return

    try:
        # Ensure all elements are converted to strings
        results = pd.DataFrame({
            'model': ["RF", "FNN"],
            'indices': [str(indices_rf), str(indices_fnn)],
            'best_model': [str(best_model_rf), str(best_model_fnn)],
            'train_error': [str(train_mae_rf), str(train_mae_fnn)],
            'test_error': [str(test_mae_rf), str(test_mae_fnn)],
            'result_df': [str(result_df_rf), str(result_df_fnn)],
            'best_train_error': [str(best_train_error_rf), str(best_train_error_fnn)],
            'best_test_error': [str(best_test_error_rf), str(best_test_error_fnn)],
            'best_iteration': [str(best_iteration_rf), str(best_iteration_fnn)]
        })

        # Save the results to a new table in PostgreSQL
        results.to_sql(destination_table, engine, if_exists='replace', index=False)
        print("Regression results saved successfully!")
    except Exception as e:
        print(f"Error processing results or saving to database: {e}")
        return


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
        'source_table': 'a500',
        'destination_table': 'mining_table', 
    },
    dag=dag,
)

# Define the task for Classification data and saving it to PostgreSQL
data_classification_task = PythonOperator(
    task_id='classification_and_saving_data',
    python_callable=classification_models,
    op_kwargs={
        'postgres_params': {
            'POSTGRES_USER': os.getenv('POSTGRES_USER'),
            'POSTGRES_PASSWORD': os.getenv('POSTGRES_PASSWORD'),
            'POSTGRES_HOST': 'postgres',
            'POSTGRES_PORT': '5432',
            'POSTGRES_DB': os.getenv('POSTGRES_DB'),
        },
        'source_table': 'mining_table', 
        'destination_table': 'classification_table',
    },
    dag=dag,
)

# Define the task for Regression data and saving it to PostgreSQL
data_regression_task = PythonOperator(
    task_id='regression_and_saving_data',
    python_callable=regression_models,
    op_kwargs={
        'postgres_params': {
            'POSTGRES_USER': os.getenv('POSTGRES_USER'),
            'POSTGRES_PASSWORD': os.getenv('POSTGRES_PASSWORD'),
            'POSTGRES_HOST': 'postgres',
            'POSTGRES_PORT': '5432',
            'POSTGRES_DB': os.getenv('POSTGRES_DB'),
        },
        'source_table': 'classification_table', 
        'destination_table': 'regression_table',
    },
    dag=dag,
)

# Set task dependencies
load_data_task >> data_transformation_task >> data_classification_task >> data_regression_task
