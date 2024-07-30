import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from sqlalchemy import create_engine
from your_module import DataPipeline  # Adjust import as necessary

class TestDataPipeline(unittest.TestCase):

    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_sql')
    @patch('sqlalchemy.create_engine')
    def test_load_data_to_postgres(self, mock_create_engine, mock_to_sql, mock_read_csv):
        # Setup mocks
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_read_csv.return_value = pd.DataFrame({'col1': [1], 'col2': [2]})
        
        pipeline = DataPipeline(postgres_params={})
        pipeline.load_data_to_postgres()
        
        # Check that the CSV file was read
        mock_read_csv.assert_called_once()
        
        # Check that the DataFrame was written to SQL
        mock_to_sql.assert_called_once_with('a500', mock_engine, if_exists='replace', index=False)

    @patch('your_module.dataProcessor')
    @patch('pandas.DataFrame.to_sql')
    @patch('sqlalchemy.create_engine')
    def test_data_transformation_and_save(self, mock_create_engine, mock_to_sql, mock_dataProcessor):
        # Setup mocks
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        mock_processor = MagicMock()
        mock_dataProcessor.return_value = mock_processor
        mock_processor.preprocess_data.return_value = pd.DataFrame({'transformed_col': [1]})
        mock_processor.get_a500.return_value = pd.DataFrame({'a500_col': [2]})
        
        pipeline = DataPipeline(postgres_params={})
        pipeline.data_transformation_and_save(
            source_table='a500',
            intermediate_table='a500',
            destination_table='mining_table'
        )
        
        # Check that the DataFrame was written to SQL
        mock_to_sql.assert_any_call('a500', mock_engine, if_exists='replace', index=False)
        mock_to_sql.assert_any_call('mining_table', mock_engine, if_exists='replace', index=False)

    @patch('your_module.rf_model')
    @patch('your_module.fnn_model')
    @patch('pandas.DataFrame.to_sql')
    @patch('sqlalchemy.create_engine')
    def test_classification_models(self, mock_create_engine, mock_to_sql, mock_fnn_model, mock_rf_model):
        # Setup mocks
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        mock_rf_instance = MagicMock()
        mock_rf_model.return_value = mock_rf_instance
        mock_rf_instance.class_main.return_value = (
            'indices_rf', 'best_model_rf', 'train_loss_rf', 'test_loss_rf', 'confusion_mat_rf',
            'best_train_error_rf', 'best_test_error_rf', 'best_iteration_rf'
        )
        
        mock_fnn_instance = MagicMock()
        mock_fnn_model.return_value = mock_fnn_instance
        mock_fnn_instance.class_main.return_value = (
            'indices_fnn', 'best_model_fnn', 'train_loss_fnn', 'test_loss_fnn', 'confusion_mat_fnn',
            'best_train_error_fnn', 'best_test_error_fnn', 'best_iteration_fnn'
        )
        
        pipeline = DataPipeline(postgres_params={})
        pipeline.classification_models(
            source_table='mining_table',
            destination_table='classification_table'
        )
        
        # Check that the DataFrame was written to SQL
        mock_to_sql.assert_called_once_with('classification_table', mock_engine, if_exists='replace', index=False)

    @patch('your_module.rf_model')
    @patch('your_module.fnn_model')
    @patch('pandas.DataFrame.to_sql')
    @patch('sqlalchemy.create_engine')
    def test_regression_models(self, mock_create_engine, mock_to_sql, mock_fnn_model, mock_rf_model):
        # Setup mocks
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        mock_rf_instance = MagicMock()
        mock_rf_model.return_value = mock_rf_instance
        mock_rf_instance.demand_function.return_value = (
            'indices_rf', 'best_model_rf', pd.DataFrame({'result_col': [1]}), 'train_mae_rf', 'test_mae_rf',
            'best_train_error_rf', 'best_test_error_rf', 'best_iteration_rf'
        )
        
        mock_fnn_instance = MagicMock()
        mock_fnn_model.return_value = mock_fnn_instance
        mock_fnn_instance.demand_function.return_value = (
            'indices_fnn', 'best_model_fnn', pd.DataFrame({'result_col': [2]}), 'train_mae_fnn', 'test_mae_fnn',
            'best_train_error_fnn', 'best_test_error_fnn', 'best_iteration_fnn'
        )
        
        pipeline = DataPipeline(postgres_params={})
        pipeline.regression_models(
            source_table='mining_table',
            indices_table='classification_table',
            destination_table='regression_table'
        )
        
        # Check that the DataFrame was written to SQL
        mock_to_sql.assert_called_once_with('regression_table', mock_engine, if_exists='replace', index=False)

if __name__ == '__main__':
    unittest.main()
