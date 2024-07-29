from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split,  RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, log_loss, roc_auc_score, roc_curve, f1_score, confusion_matrix, precision_score, log_loss, roc_auc_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, learning_curve, GridSearchCV, cross_val_predict

from parent_models import *
#######################################################################class RF_model########################################################################

class rf_model(parent_model):
    def __init__(self, mining_df, target_column, random_state):
        super().__init__(mining_df, target_column, random_state)


#######################################################################RF_model_probabilistic########################################################################
    
    def temporal_prob_function(self, X_train, y_train, X_test, y_test, random_state=42):

            # Define the parameter distributions for Random Forest
            param_dist = {
                'n_estimators': np.random.randint(100, 500, 10),
                'max_depth': [None] + list(np.random.randint(10, 100, 10)),
                'min_samples_split': np.random.randint(2, 20, 10),
                'min_samples_leaf': np.random.randint(1, 20, 10),
                'max_features': ['auto', 'sqrt']
            }

            # Create a Random Forest classifier
            rf_classifier = RandomForestClassifier(random_state=random_state)

            # Initialize RandomizedSearchCV with the specified parameter distributions
            random_search = RandomizedSearchCV(rf_classifier, param_distributions=param_dist, n_iter=10, cv=TimeSeriesSplit(n_splits=10), scoring='neg_log_loss', verbose=1, n_jobs=-1)

            # Fit the model to the data
            random_search.fit(X_train, y_train)

            # Get the best hyperparameters from the random search
            best_params = random_search.best_params_
            print("Best Hyperparameters:", best_params)

            # Get the best model
            best_model = random_search.best_estimator_

            # Evaluation results
            # Predictions on training set
            train_predictions = best_model.predict(X_train)
            # Predictions on test set
            test_predictions = best_model.predict(X_test)

            # Calculate MAE for training set
            train_loss = log_loss(y_train, train_predictions)
            # Calculate MAE for test set
            test_loss = log_loss(y_test, test_predictions)

            print("Log Loss - Training Set:", train_loss)
            print("Log Loss - Test Set:", test_loss)

            # Get feature importances
            self.plot_feature_importance(best_model.feature_importances_, X_train.columns, "RF")

            # Use predictions and probabilities from the best iteration
            best_model.fit(X_train, y_train)  # Complete fitting on the best iteration
            best_y_test_proba = best_model.predict_proba(X_test)
            best_y_test_pred = best_model.predict(X_test)

            # Calculate metrics for best iteration
            auc_score, accuracy, f1, precision, recall, confusion_mat = self.calculate_metrics(y_test, best_y_test_proba, best_y_test_pred, "RF")

            # self.plot_auc(*roc_curve(y_test, best_y_test_proba[:, 1]), auc_score, "RF")

            self.plot_confusion_matrix(confusion_mat, "RF")

            # Get predicted probabilities for train and test sets
            y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
            y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]

            # Concatenate predicted probabilities for train and test sets
            all_y_pred_proba = np.concatenate((y_train_pred_proba, y_test_pred_proba))

            # Get indices where predicted probabilities are 0
            indices = np.where(all_y_pred_proba == 0)[0]
            return indices, best_model, train_loss, test_loss, confusion_mat
    
    def iteration_prob_function(self, best_model, X_train, y_train, X_test, y_test):
        # Initialize lists to store training and test errors
        training_errors = []
        test_errors = []
        best_train_error = float('inf')
        best_test_error = float('inf')
        best_iteration = 0
        best_total_error = float('inf')
        
        # Determine the total number of samples
        n_train_samples = len(X_train)
        n_test_samples = len(X_test)
        
        # Define the bounds for increment fraction
        min_fraction = 0.01
        max_fraction = 0.05

        # Initialize the current end index
        train_end_index = 0
        test_end_index = 0
        
        # Perform training for 50 iterations
        for i in range(50):
            # Generate a random increment fraction
            increment_fraction = np.random.uniform(min_fraction, max_fraction)
            increment_train = int(n_train_samples * increment_fraction)
            increment_test = int(n_test_samples * increment_fraction)
            
            # Calculate the current subset indices for training and testing
            train_end_index = min(n_train_samples, train_end_index + increment_train)
            test_end_index = min(n_test_samples, test_end_index + increment_test)
            
            X_train_subset = X_train.iloc[:train_end_index]
            y_train_subset = y_train.iloc[:train_end_index]
            
            X_test_subset = X_test.iloc[:test_end_index]
            y_test_subset = y_test.iloc[:test_end_index]

            # Fit the model to the training data
            best_model.fit(X_train_subset, y_train_subset)
            
            # Calculate training and test errors
            y_train_pred_proba = best_model.predict_proba(X_train_subset)
            y_test_pred_proba = best_model.predict_proba(X_test_subset)
            train_error = log_loss(y_train_subset, y_train_pred_proba)
            test_error = log_loss(y_test_subset, y_test_pred_proba)
            total_error = train_error + test_error

            # Append errors to the lists
            training_errors.append(train_error)
            test_errors.append(test_error)

            # Check for the best total error and its corresponding iteration
            if total_error < best_total_error:
                best_total_error = total_error
                best_train_error = train_error
                best_test_error = test_error
                best_iteration = i + 1

        best_train_error, best_test_error, best_iteration = self.plot_errors_class(training_errors, test_errors, best_iteration, best_train_error, best_test_error, "RF")
        return best_train_error, best_test_error, best_iteration

#######################################################################RF_model_regression########################################################################
   
    def temporal_reg_function(self, X_train, y_train, X_test, y_test, dates, pn, indices=None, random_state=42):
            # Define the parameter distributions for Random Forest
            param_dist = {
                'n_estimators': np.random.randint(100, 500, 10),
                'max_depth': [None] + list(np.random.randint(10, 100, 10)),
                'min_samples_split': np.random.randint(2, 20, 10),
                'min_samples_leaf': np.random.randint(1, 10, 10),
                'max_features': ['auto', 'sqrt']
            }

            # Create a Random Forest regressor
            rf_regressor = RandomForestRegressor(random_state=random_state)

            # Initialize RandomizedSearchCV with the specified parameter distributions
            random_search = RandomizedSearchCV(rf_regressor, param_distributions=param_dist, n_iter=10, cv=TimeSeriesSplit(n_splits=10), scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)
            
            # Fit the model to the data
            random_search.fit(X_train, y_train)

            # Get the best hyperparameters from the random search
            best_params = random_search.best_params_
            print("Best Hyperparameters:", best_params)

            # Get the best model
            best_model = random_search.best_estimator_
            
            best_model.fit(X_train, y_train)  # Complete fitting on the best iteration
            # Evaluation results
            # Predictions on training set
            train_predictions = best_model.predict(X_train)
            # Predictions on test set
            test_predictions = best_model.predict(X_test)

            # Calculate MAE for training set
            train_mae = mean_absolute_error(y_train, train_predictions)
            # Calculate MAE for test set
            test_mae = mean_absolute_error(y_test, test_predictions)

            print("Mean Absolute Error (MAE) - Training Set:", train_mae)
            print("Mean Absolute Error (MAE) - Test Set:", test_mae)

            # Get feature importances
            self.plot_feature_importance(best_model.feature_importances_, X_train.columns, "RF")

            # Plot residual plot for the best iteration

            best_y_test_pred = best_model.predict(X_test)

            if indices is not None:
                # Ensure indices are within the valid range of y_test_pred
                for index in indices:
                    if 0 <= index < len(best_y_test_pred):
                        best_y_test_pred[index] = 0

            self.plot_residuals(y_test, best_y_test_pred, "RF")


            # Create the DataFrame with Date, PN, y_test, and test_predictions
            result_df = pd.DataFrame({
                'Date': dates,
                'PN': pn,
                'y_test': y_test,
                'test_predictions': test_predictions
            })

            return best_model, result_df, train_mae, test_mae

    def iteration_reg_function(self, best_model, X_train, y_train, X_test, y_test, indices=None):
        # Initialize lists to store training and test errors
        training_errors = []
        test_errors = []
        best_train_error = float('inf')
        best_test_error = float('inf')
        best_iteration = 0
        best_total_error = float('inf')
        
        # Determine the total number of samples
        n_train_samples = len(X_train)
        n_test_samples = len(X_test)
        
        # Define the bounds for random selection of increment fraction
        min_increment = 0.01
        max_increment = 0.05
        
        # Perform training for 50 iterations
        for i in range(50):
            # Generate a random increment fraction
            increment_fraction = np.random.uniform(min_increment, max_increment)
            
            # Calculate the current subset indices for training and testing
            train_end_index = min(n_train_samples, int((i + 1) * n_train_samples * increment_fraction))
            test_end_index = min(n_test_samples, int((i + 1) * n_test_samples * increment_fraction))
            
            X_train_subset = X_train.iloc[:train_end_index]
            y_train_subset = y_train.iloc[:train_end_index]
            
            X_test_subset = X_test.iloc[:test_end_index]
            y_test_subset = y_test.iloc[:test_end_index]

            # Fit the model to the training data
            best_model.fit(X_train_subset, y_train_subset)
            
            # Calculate training and test errors
            y_train_pred = best_model.predict(X_train_subset)
            y_test_pred = best_model.predict(X_test_subset)

            if indices is not None:
                # Ensure indices are within the valid range of y_test_pred
                for index in indices:
                    if 0 <= index < len(y_test_pred):
                        y_test_pred[index] = 0

            train_error = mean_absolute_error(y_train_subset, y_train_pred)
            test_error = mean_absolute_error(y_test_subset, y_test_pred)
            total_error = train_error + test_error

            # Append errors to the lists
            training_errors.append(train_error)
            test_errors.append(test_error)

            # Check for the best total error and its corresponding iteration
            if total_error < best_total_error:
                best_total_error = total_error
                best_train_error = train_error
                best_test_error = test_error
                best_iteration = i + 1

        best_train_error, best_test_error, best_iteration = self.plot_errors_actual(test_errors, training_errors, best_iteration, best_test_error, best_train_error, "RF")
        return best_train_error, best_test_error, best_iteration