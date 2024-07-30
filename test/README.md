# Data Pipeline Project

## Overview

This project involves a data pipeline that performs the following tasks:
1. **Loading Data**: Load data from a CSV file into a PostgreSQL database.
2. **Data Transformation**: Transform data using a custom ETL processor and save the intermediate and transformed data into PostgreSQL tables.
3. **Classification Models**: Train classification models and save the results into a PostgreSQL table.
4. **Regression Models**: Train regression models using the best indices from classification results and save the results into a PostgreSQL table.

## Project Structure

- `data_process.py`: Contains the `dataProcessor` class used for ETL processes.
- `parent_models.py`, `RF_models.py`, `fnn_models.py`, `lstm_models.py`: Contains classes for different models used in the pipeline.
- `data_pipeline.py`: Contains the `DataPipeline` class that integrates all steps of the pipeline.
- `test_data_pipeline.py`: Contains unit tests for the `DataPipeline` class.

## Setup

1. **Dependencies**: Ensure you have the required Python packages installed. You can install them using pip:

    ```bash
    pip install pandas sqlalchemy
    ```

2. **PostgreSQL Database**: Ensure you have access to a PostgreSQL database and update the `postgres_params` in your code with the appropriate database credentials.

## Running the Pipeline

To run the data pipeline, follow these steps:

1. **Load Data to PostgreSQL**:

    ```python
    from data_pipeline import DataPipeline

    # Replace with your actual PostgreSQL parameters
    postgres_params = {
        'POSTGRES_USER': 'user',
        'POSTGRES_PASSWORD': 'password',
        'POSTGRES_HOST': 'localhost',
        'POSTGRES_PORT': '5432',
        'POSTGRES_DB': 'your_database'
    }

    pipeline = DataPipeline(postgres_params)
    pipeline.load_data_to_postgres()
    ```

2. **Transform and Save Data**:

    ```python
    pipeline.data_transformation_and_save(
        source_table='a500',
        intermediate_table='a500',
        destination_table='mining_table'
    )
    ```

3. **Run Classification Models**:

    ```python
    pipeline.classification_models(
        source_table='mining_table',
        destination_table='classification_table'
    )
    ```

4. **Run Regression Models**:

    ```python
    pipeline.regression_models(
        source_table='mining_table',
        indices_table='classification_table',
        destination_table='regression_table'
    )
    ```

## Running Tests

To ensure your code is working correctly, you should run the unit tests. This can be done using Python's built-in unittest framework or pytest. 

1. **Using unittest**:

    ```bash
    python -m unittest discover
    ```

2. **Using pytest**:

    ```bash
    pytest
    ```

## Notes

- Ensure all necessary files and directories (`data`, `libraries`, etc.) are correctly set up and accessible.
- Update any import paths or module names based on your project structure.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

