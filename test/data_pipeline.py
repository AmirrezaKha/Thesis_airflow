import os
import sys
import pandas as pd
from sqlalchemy import create_engine
import json

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

class DataPipeline:
    def __init__(self, postgres_params):
        self.postgres_params = postgres_params
        self.engine = create_engine(f"postgresql://{postgres_params['POSTGRES_USER']}:{postgres_params['POSTGRES_PASSWORD']}@{postgres_params['POSTGRES_HOST']}:{postgres_params['POSTGRES_PORT']}/{postgres_params['POSTGRES_DB']}")

    def load_data_to_postgres(self):
        """
        Loads data from a CSV file into a PostgreSQL database.
        """
        try:
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'a500.csv')
            df = pd.read_csv(csv_path)
            df.to_sql('a500', self.engine, if_exists='replace', index=False)
            print("Data loaded successfully!")
        except Exception as e:
            print(f"Error: {e}")
            raise

    def data_transformation_and_save(self, source_table, intermediate_table, destination_table):
        """
        Executes the data transformation and saves the processed DataFrame to PostgreSQL.
        """
        try:
            etl_processor = dataProcessor(
                num_lags=5,
                source_table=source_table,
                dest_table=destination_table,
                db_params_df=pd.DataFrame([self.postgres_params])
            )
            mining_df = etl_processor.preprocess_data()
            a500 = etl_processor.get_a500()
            
            a500.to_sql(intermediate_table, self.engine, if_exists='replace', index=False)
            mining_df.to_sql(destination_table, self.engine, if_exists='replace', index=False)
            print(f"Data transformation and saving to PostgreSQL table '{destination_table}' completed successfully!")
        except Exception as e:
            print(f"An error occurred: {e}")

    def classification_models(self, source_table, destination_table):
        """
        Performs classification model training and saves the results to PostgreSQL.
        """
        try:
            query = f"SELECT * FROM {source_table}"
            mining_df = pd.read_sql(query, self.engine)

            target_column = 'Daily_Actual_Demand'
            random_state = 42
            
            rf_model_instance = rf_model(mining_df, target_column, random_state)
            fnn_model_instance = fnn_model(mining_df, target_column, random_state)

            results_rf = rf_model_instance.class_main()
            results_fnn = fnn_model_instance.class_main()
            
            results = pd.DataFrame({
                'model': ["RF", "FNN"],
                'indices': [str(results_rf[0]), str(results_fnn[0])],
                'best_model': [str(results_rf[1]), str(results_fnn[1])],
                'train_loss': [str(results_rf[2]), str(results_fnn[2])],
                'test_loss': [str(results_rf[3]), str(results_fnn[3])],
                'confusion_matrix': [str(results_rf[4]), str(results_fnn[4])],
                'best_train_error': [str(results_rf[5]), str(results_fnn[5])],
                'best_test_error': [str(results_rf[6]), str(results_fnn[6])],
                'best_iteration': [str(results_rf[7]), str(results_fnn[7])]
            })

            results.to_sql(destination_table, self.engine, if_exists='replace', index=False)
            print("Classification results saved successfully!")
        except Exception as e:
            print(f"Error processing classification models: {e}")

    def regression_models(self, source_table, indices_table, destination_table):
        """
        Performs regression model training and saves the results to PostgreSQL.
        """
        try:
            mining_df = pd.read_sql(f"SELECT * FROM {source_table}", self.engine)
            indices_df = pd.read_sql(f"SELECT model, indices FROM {indices_table}", self.engine)

            indices_rf_str = indices_df[indices_df['model'] == 'RF']['indices'].values[0]
            indices_fnn_str = indices_df[indices_df['model'] == 'FNN']['indices'].values[0]

            indices_rf = json.loads(indices_rf_str)
            indices_fnn = json.loads(indices_fnn_str)

            target_column = 'Daily_Actual_Demand'
            random_state = 42

            rf_model_instance = rf_model(mining_df, target_column, random_state)
            fnn_model_instance = fnn_model(mining_df, target_column, random_state)

            results_rf = rf_model_instance.demand_function(indices=indices_rf)
            results_fnn = fnn_model_instance.demand_function(indices=indices_fnn)

            results = pd.DataFrame({
                'model': ["RF", "FNN"],
                'indices': [str(results_rf[0]), str(results_fnn[0])],
                'best_model': [str(results_rf[1]), str(results_fnn[1])],
                'result_df': [str(results_rf[2]), str(results_fnn[2])],
                'train_error': [str(results_rf[3]), str(results_fnn[3])],
                'test_error': [str(results_rf[4]), str(results_fnn[4])],
                'best_train_error': [str(results_rf[5]), str(results_fnn[5])],
                'best_test_error': [str(results_rf[6]), str(results_fnn[6])],
                'best_iteration': [str(results_rf[7]), str(results_fnn[7])]
            })

            results.to_sql(destination_table, self.engine, if_exists='replace', index=False)
            print("Regression results saved successfully!")
        except Exception as e:
            print(f"Error processing regression models: {e}")
