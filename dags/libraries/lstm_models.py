from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor 
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, log_loss, roc_curve,  roc_curve
from sklearn.model_selection import  TimeSeriesSplit

import pandas as pd
from parent_models import *
###############################################################################################################################################
# def create_lstm_model(input_shape, hidden_layer_size=50, activation='relu', alpha=0.001, learning_rate_init=0.01):
#         model = Sequential()
#         model.add(LSTM(hidden_layer_size, activation=activation, input_shape=input_shape))
#         model.add(Dense(1))
#         model.compile(optimizer='adam', loss='mean_squared_error')
#         return model
class lstm_model(parent_model):
    def __init__(self, mining_df, target_column, random_state):
        super().__init__(mining_df, target_column, random_state)

#######################################################################LSTM_model_classification########################################################################

    def temporal_prob_function(self, X_train, y_train, X_test, y_test, random_state=42):
        # Define the parameter grid for the LSTM model
        param_grid = {
            'n_lstm_units': [64, 128],
            'dropout_rate': [0.2, 0.5],
            'optimizer': ['adam', 'rmsprop'],
            'learning_rate': [0.001, 0.01]
        }

        # Function to create LSTM model
        def create_lstm_model(n_lstm_units=64, dropout_rate=0.2, optimizer='adam', learning_rate=0.001):
            model = Sequential([
                LSTM(units=n_lstm_units, input_shape=(X_train.shape[1], 1)),
                Dropout(dropout_rate),
                Dense(1, activation='sigmoid')
            ])
            if optimizer == 'adam':
                opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            else:
                opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            return model

        # Create KerasClassifier
        lstm_model = KerasClassifier(build_fn=create_lstm_model, verbose=1)

        # Initialize GridSearchCV with the specified parameter grid
        grid_search = GridSearchCV(estimator=lstm_model, param_grid=param_grid, cv=TimeSeriesSplit(n_splits=10), scoring='neg_log_loss', verbose=1, n_jobs=1)

        # Reshape X_train and X_test for LSTM
        X_train_lstm = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
        X_test_lstm = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))

        # Fit the model to the data
        grid_result = grid_search.fit(X_train_lstm, y_train)

        # Get the best hyperparameters from the grid search
        best_params = grid_result.best_params_
        print("Best Hyperparameters:", best_params)

        # Get the best model
        best_model = grid_result.best_estimator_

        # Evaluation results
        # Predictions on training set
        train_predictions = best_model.predict(X_train_lstm)
        # Predictions on test set
        test_predictions = best_model.predict(X_test_lstm)

        # Calculate MAE for training set
        train_loss = log_loss(y_train, train_predictions)
        # Calculate MAE for test set
        test_loss = log_loss(y_test, test_predictions)

        print("Log Loss - Training Set:", train_loss)
        print("Log Loss - Test Set:", test_loss)

        # Get feature importances
        self.plot_feature_weights_lstm(best_model, X_test_lstm, y_test, X_train.columns, "LSTM")

        # Use predictions and probabilities from the best iteration
        best_model.fit(X_train_lstm, y_train)  # Complete fitting on the best iteration
        best_y_test_proba = best_model.predict_proba(X_test_lstm)
        best_y_test_pred = best_model.predict(X_test_lstm)

        # Calculate metrics for best iteration
        auc_score, accuracy, f1, precision, recall, confusion_mat = self.calculate_metrics(y_test, best_y_test_proba, best_y_test_pred, "LSTM")

        self.plot_confusion_matrix(confusion_mat, "LSTM")

        # Get predicted probabilities for train and test sets
        y_train_pred_proba = best_model.predict_proba(X_train_lstm)[:, 1]
        y_test_pred_proba = best_model.predict_proba(X_test_lstm)[:, 1]

        # Concatenate predicted probabilities for train and test sets
        all_y_pred_proba = np.concatenate((y_train_pred_proba, y_test_pred_proba))

        # Get indices where predicted probabilities are 0
        indices = np.where(all_y_pred_proba == 0)[0]
        return indices, best_model, train_loss, test_loss, confusion_mat

    def iteration_prob_function(self, best_model, X_train_lstm, y_train, X_test, y_test):
        # Define lists to store losses
        train_error = []
        test_error = []
        best_train_error = float('inf')
        best_test_error = float('inf')
        best_iteration = 0
        best_total_error = float('inf')
        
        # Determine the total number of samples
        n_train_samples = len(X_train_lstm)
        
        # Define the bounds for random selection of increment fraction
        min_increment = 0.01
        max_increment = 0.05
        
        # Train the model for 50 iterations
        for i in range(50):
            # Generate a random increment fraction
            increment_fraction = np.random.uniform(min_increment, max_increment)
            
            # Calculate the current subset indices for training
            train_end_index = min(n_train_samples, int((i + 1) * n_train_samples * increment_fraction))
            
            X_train_subset = X_train_lstm[:train_end_index]
            y_train_subset = y_train[:train_end_index]

            # Fit the model to the training data subset
            history = best_model.fit(X_train_subset, y_train_subset, epochs=1, batch_size=64, validation_split=0.2, verbose=1)
            
            # Evaluate the model on the test data
            y_test_pred = best_model.predict(X_test)

            # Ensure y_test and y_test_pred are numerical
            y_test_numeric = y_test.astype(float)
            y_test_pred_numeric = y_test_pred.astype(float)
            
            test_loss = mean_absolute_error(y_test_numeric, y_test_pred_numeric)
            
            # Calculate total error
            total_error = history.history['val_loss'][0] + history.history['loss'][0] + test_loss
            
            # Append losses to lists
            train_error.append(history.history['loss'][0])
            test_error.append(test_loss)

            # Check for the best total error and its corresponding iteration
            if total_error < best_total_error:
                best_total_error = total_error
                best_train_error = history.history['val_loss'][0]
                best_test_error = test_loss
                best_iteration = i + 1

        best_train_error, best_test_error, best_iteration = self.plot_errors_class(test_error, train_error, 
                                                                                   best_iteration, best_test_error, best_train_error, "LSTM")
        return best_train_error, best_test_error, best_iteration



#######################################################################LSTM_model_regression########################################################################

    def temporal_reg_function(self, X_train, y_train, X_test, y_test, dates, pn, indices=None, random_state=42):

            # Define the parameter grid for the LSTM model
            param_grid = {
                'n_lstm_units': [64, 128],
                'dropout_rate': [0.2, 0.5],
                'optimizer': ['adam', 'rmsprop'],
                'learning_rate': [0.001, 0.01]
            }

            # Function to create LSTM model
            def create_lstm_model(n_lstm_units=64, dropout_rate=0.2, optimizer='adam', learning_rate=0.001):
                model = Sequential([
                    LSTM(units=n_lstm_units, input_shape=(X_train.shape[1], 1)),
                    Dropout(dropout_rate),
                    Dense(1, activation='linear')  # Changed activation to 'linear' for regression
                ])
                if optimizer == 'adam':
                    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                else:
                    opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
                model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_error'])  # Changed loss to MAE for regression
                return model


            # Create KerasRegressor 
            lstm_model = KerasRegressor(build_fn=create_lstm_model, verbose=1)

            # Initialize GridSearchCV with the specified parameter grid
            grid_search = GridSearchCV(estimator=lstm_model, param_grid=param_grid, cv=TimeSeriesSplit(n_splits=10), scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)

            # Reshape X_train and X_test for LSTM
            X_train_lstm = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
            X_test_lstm = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))

            # Fit the model to the data
            grid_result = grid_search.fit(X_train_lstm, y_train)

            # Get the best hyperparameters from the grid search
            best_params = grid_result.best_params_
            print("Best Hyperparameters:", best_params)

            # Get the best model
            best_model = grid_result.best_estimator_

            # Evaluation results
            # Predictions on training set
            train_predictions = best_model.predict(X_train)
            # Predictions on test set
            test_predictions = best_model.predict(X_test)

            if indices is not None:
                # Ensure indices are within the valid range of y_test_pred
                for index in indices:
                    if 0 <= index < len(test_predictions):
                        test_predictions[index] = 0

            # Calculate MAE for training set
            train_mae = mean_absolute_error(y_train, train_predictions)
            # Calculate MAE for test set
            test_mae = mean_absolute_error(y_test, test_predictions)

            print("Mean Absolute Error (MAE) - Training Set:", train_mae)
            print("Mean Absolute Error (MAE) - Test Set:", test_mae)

            # Get feature importances
            self.plot_feature_weights_lstm(best_model, X_test_lstm, y_test, X_train.columns, "LSTM")

            # Plot residual plot for the best iteration

            # Plot residual plot for the LSTM model
            self.plot_residuals(y_test, test_predictions, "LSTM")

            # Create the DataFrame with Date, PN, y_test, and test_predictions
            result_df = pd.DataFrame({
                'Date': dates,
                'PN': pn,
                'y_test': y_test,
                'test_predictions': test_predictions
            })

            return best_model, result_df, train_mae, test_mae
    
    def iteration_reg_function(self, best_model, X_train, y_train, X_test, y_test, indices=None):
        # Define lists to store losses
        train_error = []
        test_error = []
        best_train_error = float('inf')
        best_test_error = float('inf')
        best_iteration = 0
        best_total_error = float('inf')

        # Determine the total number of samples
        n_train_samples = len(X_train)
        
        # Define the bounds for random selection of increment fraction
        min_increment = 0.01
        max_increment = 0.05

        # Train the model for 50 iterations
        for i in range(50):
            # Generate a random increment fraction
            increment_fraction = np.random.uniform(min_increment, max_increment)
            
            # Calculate the current subset indices for training
            train_end_index = min(n_train_samples, int((i + 1) * n_train_samples * increment_fraction))
            
            X_train_subset = X_train[:train_end_index]
            y_train_subset = y_train[:train_end_index]

            # Fit the model to the training data subset
            history = best_model.fit(X_train_subset, y_train_subset, epochs=1, batch_size=64, validation_split=0.2, verbose=1)

            # Evaluate the model on the test data
            y_test_pred = best_model.predict(X_test)
            test_loss = mean_absolute_error(y_test, y_test_pred)

            # Apply condition if indices is not None
            if indices is not None:
                # Ensure indices are within the valid range of y_test_pred
                for index in indices:
                    if 0 <= index < len(y_test_pred):
                        y_test_pred[index] = 0

            # Calculate total error
            total_error = history.history['val_loss'][0] + history.history['loss'][0] + test_loss
            
            # Append losses to lists
            train_error.append(history.history['loss'][0])
            test_error.append(test_loss)

            # Check for the best total error and its corresponding iteration
            if total_error < best_total_error:
                best_total_error = total_error
                best_train_error = history.history['val_loss'][0]
                best_test_error = test_loss
                best_iteration = i + 1

        if indices is not None:
            best_train_error, best_test_error, best_iteration = self.plot_errors_actual(test_error, train_error, 
                                                                                        best_iteration, best_test_error, best_train_error, "LSTM")
        else:
            best_train_error, best_test_error, best_iteration = self.plot_errors_actual(train_error, test_error, 
                                                                                        best_iteration, best_train_error, best_test_error, "LSTM")
            return best_train_error, best_test_error, best_iteration


        

