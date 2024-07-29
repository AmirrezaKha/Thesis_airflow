
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.model_selection import train_test_split,  RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, log_loss, roc_auc_score, roc_curve, f1_score, confusion_matrix, precision_score, roc_auc_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from sklearn.inspection import permutation_importance
from statsmodels.stats.contingency_tables import mcnemar

import seaborn as sns
# from sklearn.model_selection import TimeSeriesSplit, learning_curve, GridSearchCV, cross_val_predict
import os
#######################################################################class parent_model########################################################################

class parent_model:

    def __init__(self, df, target_column, random_state):
        self.df = df
        self.random_state = random_state
        self.target_column = target_column
        self.results_derivate = []
        self.results_demand = []
        self.results_probabilistic = []
    # Save Images function
    def save_images(self, folder, filename):
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_folder = os.path.join(parent_dir, 'images', folder)
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f'{filename}.png')
        
        plt.savefig(output_file, dpi=300)
        print(f"Image saved at {output_file}")

#######################################################################calculate_date_deviation########################################################################
    # calculate date_deviation column
    def calculate_date_deviation(self):
        
        self.df['Date_Deviation'] = 0  # Initialize with 0

        # Iterate over unique PN values
        for pn_value in self.df['PN'].unique():
            df_pn = self.df[self.df['PN'] == pn_value]
            non_zero_indices = df_pn[df_pn['Daily_Actual_Demand'] != 0].index

            # Iterate over non-zero indices within each PN group
            for i in range(1, len(non_zero_indices)):
                prev_index = non_zero_indices[i - 1]
                curr_index = non_zero_indices[i]
                
                # Calculate date deviation within the same 'PN' group
                deviation_days = (df_pn.loc[curr_index, 'Date'] - df_pn.loc[prev_index, 'Date']).days
                # Update 'Date_Deviation' column for the previous index
                self.df.loc[prev_index, 'Date_Deviation'] = deviation_days

#######################################################################Classification Visualization########################################################################
    # plot_errors of classification
    def plot_errors_class(self, training_errors, test_errors, best_iteration, best_train_error, best_test_error, model):
        
        sns.set()
        sns.set_style("whitegrid")
        sns.set_context("poster", font_scale=1.5)

        # Define the range for y-ticks with a step of 0.02
        y_min = min(min(training_errors), min(test_errors))
        y_max = max(max(training_errors), max(test_errors))
        y_ticks = np.arange(y_min, y_max, 0.02)
        
        # Slice y_ticks to show only every 10th tick
        y_ticks_sparse = y_ticks[::10]

        plt.figure(figsize=(10, 8))
        plt.plot(training_errors, label='Training Error (Log Loss)')
        plt.plot(test_errors, label='Test Error (Log Loss)')
        plt.title('Training and Test Errors Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Error (Log Loss)')
        plt.yticks(y_ticks_sparse) 
        plt.legend()
        self.save_images(model, 'Cl_Iterations')

        print("Lowest Training Error (Log Loss):", best_train_error)
        print("Lowest Test Error (Log Loss):", best_test_error)
        print("Best Iteration:", best_iteration)
        
        return best_train_error, best_test_error, best_iteration

    # plot_auc of classification
    def plot_auc(self, fpr, tpr, auc_score, model):
        
        sns.set()
        sns.set_style("whitegrid")
        sns.set_context("poster", font_scale=1.5) 
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        self.save_images(model, 'ROC')

    # plot_confusion_matrix of classification
    def plot_confusion_matrix(self, confusion_mat, model):
       
        sns.set()
        sns.set_style("whitegrid")
        sns.set_context("poster", font_scale=1.5) 
        plt.figure(figsize=(15, 10))
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Final Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        self.save_images(model, 'Confusion')

    # calculate_metrics of classification
    def calculate_metrics(self, y_test, best_y_test_proba, best_y_test_pred, model):
       
        auc_score = roc_auc_score(y_test, best_y_test_proba[:, 1])
        accuracy = accuracy_score(y_test, best_y_test_pred)
        f1 = f1_score(y_test, best_y_test_pred)
        precision = precision_score(y_test, best_y_test_pred)
        recall = recall_score(y_test, best_y_test_pred)
        confusion_mat = confusion_matrix(y_test, best_y_test_pred)
        
        fpr, tpr, _ = roc_curve(y_test, best_y_test_proba[:, 1])
        
        self.plot_auc(fpr, tpr, auc_score, model)
        
        print("AUC Score:", auc_score)
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        print("Precision:", precision)
        print("Recall:", recall)
        print("Confusion Matrix:")
        print(confusion_mat)
        
        return auc_score, accuracy, f1, precision, recall, confusion_mat

#######################################################################Regression Visualization########################################################################
    # plot_feature_importance of regression for RF
    def plot_feature_weights(self, best_model, feature_names, model):
        # Get the coefficients (weights) of the input layer
        weights = best_model.coefs_[0]

        # Calculate absolute weights for each feature
        absolute_weights = np.abs(weights)

        # Compute mean importance across all neurons in the input layer
        mean_importance = np.mean(absolute_weights, axis=1)

        # Create a DataFrame to store feature importances along with their corresponding column names
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': mean_importance})

        # Sort the DataFrame by feature importance in descending order
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Set seaborn style and context
        sns.set(style="whitegrid", context="poster", font_scale=0.7)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title('Feature Importance (Absolute Weights)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        self.save_images(model, 'Feature Importance')

    # plot_feature_importance of regression for FNN
    def plot_feature_importance(self, feature_importance, feature_names, model):
        # Create a DataFrame to store feature importances along with their corresponding column names
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

        # Sort the DataFrame by feature importance in descending order
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Set seaborn style and context
        sns.set(style="whitegrid", context="poster", font_scale=0.7)

        # Plot feature importances
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        self.save_images(model, 'Feature Importance')

   # plot_feature_importance of regression for LSTM
    def plot_feature_weights_lstm(self, best_model, X_test, y_test, feature_names, model):

        # Reshape X_test_lstm to (samples, features)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

        # Compute permutation importances
        result = permutation_importance(best_model, X_test_reshaped, y_test, n_repeats=10, random_state=42, n_jobs=-1)

        # Sort the features by their importance
        sorted_idx = result.importances_mean.argsort()[::-1]

        plt.figure(figsize=(10, 8))
        sns.set(style="whitegrid", context="poster", font_scale=0.7)  # Set seaborn style and context
        sns.barplot(x=result.importances_mean[sorted_idx], y=np.array(feature_names)[sorted_idx])
        plt.title('Feature Importance (Permutation)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        self.save_images(model, 'Feature Importance')

    # plot_errors_actual demand of regression iteration approach
    def plot_errors_actual(self, test_errors, training_errors, best_iteration, best_test_error, best_train_error, model):
        sns.set()

        sns.set_style("whitegrid")
        sns.set_context("poster", font_scale=1.5)

        y_min = min(min(training_errors), min(test_errors))
        y_max = max(max(training_errors), max(test_errors))

        if y_min == y_max:
            y_max += 1e-5  

        # Generate up to 10 y-ticks
        num_ticks = 10
        y_ticks = np.linspace(y_min, y_max, num_ticks)
        
        # Format y-ticks to two decimal places
        y_tick_labels = [f"{tick:.2f}" for tick in y_ticks]

        plt.figure(figsize=(10, 8))
        plt.plot(training_errors, label='Training Error (MAE)')
        plt.plot(test_errors, label='Test Error (MAE)')
        plt.title('Training and Test Errors Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Error (MAE)')
        plt.yticks(y_ticks, y_tick_labels)  
        plt.legend()
        self.save_images(model, 'Iterations')  

        print("Lowest Training Error (MAE):", best_train_error)
        print("Lowest Test Error (MAE):", best_test_error)
        print("Best Iteration:", best_iteration)

        return best_train_error, best_test_error, best_iteration

    # plot_residuals of regression
    def plot_residuals(self, y_true, y_pred, model):
        sns.set()

        sns.set_style("whitegrid")
        sns.set_context("poster", font_scale=1.5) 
        plt.figure(figsize=(10, 8))
        residuals = y_true - y_pred
        plt.scatter(y_true, residuals)
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        self.save_images(model, 'Residuals')

#######################################################################Classification_Main########################################################################

    # temporal_prob_function for classification as parent function with no definiton
    def temporal_prob_function(self, X_train, y_train, X_test, y_test):
        raise NotImplementedError("Temporal_prob_function method must be implemented in the child class.")
    
    # iteration_prob_function for classification as parent function with no definiton
    def iteration_prob_function(self, best_model, X_train, y_train, X_test, y_test):
        raise NotImplementedError("Iteration_prob_function method must be implemented in the child class.")
    
    # Main function for classification as parent function with feeding values and get results from functions above
    def class_main(self):
        if 'PN' in self.df.columns:
            self.df = self.df.sort_values(by='Date')

            df_encoded = self.df.fillna(0)

            # Step 1: Identify the split date
            unique_pn_values_count = df_encoded["PN"].nunique()
            unique_date_values_count = df_encoded["Date"].nunique()
            split_index = int(unique_pn_values_count * unique_date_values_count * 0.8)
            split_date = df_encoded.iloc[split_index]['Date']  

            # Step 2: Perform temporal splitting
            train_data = df_encoded[df_encoded['Date'] < split_date]
            test_data = df_encoded[df_encoded['Date'] >= split_date]

            # Extract features and target variable
            if 'Date_Deviation' in train_data.columns:
                X_train = train_data.drop(['Date_Deviation', 'Daily_Actual_Demand', 'PN', 'Date', 'Avg_Daily_Actual_Demand', 'Avg_Daily_Forecast_Demand'], axis=1)
            else:
                X_train = train_data.drop(['Daily_Actual_Demand', 'PN', 'Date', 'Avg_Daily_Actual_Demand', 'Avg_Daily_Forecast_Demand'], axis=1)
            y_train = train_data[self.target_column] != 0

            if 'Date_Deviation' in test_data.columns:
                X_test = test_data.drop(['Date_Deviation', 'Daily_Actual_Demand', 'PN', 'Date', 'Avg_Daily_Actual_Demand', 'Avg_Daily_Forecast_Demand'], axis=1)
            else:
                X_test = test_data.drop(['Daily_Actual_Demand', 'PN', 'Date', 'Avg_Daily_Actual_Demand', 'Avg_Daily_Forecast_Demand'], axis=1)
            y_test = test_data[self.target_column] != 0

            print("Train Start Date:", train_data['Date'].min())
            print("Train End Date:", train_data['Date'].max())
            print("Test Start Date:", test_data['Date'].min())
            print("Test End Date:", test_data['Date'].max())
            print("X_train shape:", X_train.shape)
            print("X_test shape:", X_test.shape)
            print("X Columns: ", X_train.columns)
            print("Target Column: ", self.target_column)

            indices, best_model, train_loss, test_loss, confusion_mat = self.temporal_prob_function(X_train, y_train, X_test, y_test)
            best_train_error, best_test_error, best_iteration = self.iteration_prob_function(best_model, X_train, y_train, X_test, y_test)
            return indices, best_model,  train_loss, test_loss, confusion_mat, best_train_error, best_test_error, best_iteration

        else:
            print("Column 'PN' not found in the DataFrame.")
            return None

#######################################################################Regression_Main########################################################################

    # temporal_reg_function for regression as parent function with no definiton
    def temporal_reg_function(self, X_train, y_train, X_test, y_test, dates, pn, indices=None, random_state=42): 
        raise NotImplementedError("Temporal_reg_function method must be implemented in the child class.")
    
    # iteration_reg_function for regression as parent function with no definiton
    def iteration_reg_function(self, best_model, X_train, y_train, X_test, y_test, indices):
        raise NotImplementedError("iteration_reg_function method must be implemented in the child class.")
    
    # Main function for regression as parent function with feeding values and get results from functions above
    def reg_main(self, target_column, indices=None):
        if 'PN' in self.df.columns:
            self.df = self.df.sort_values(by='Date')

            df_encoded = self.df.fillna(0)

            # Step 1: Identify the split date
            unique_pn_values_count = df_encoded["PN"].nunique()
            unique_date_values_count = df_encoded["Date"].nunique()
            split_index = int(unique_pn_values_count * unique_date_values_count * 0.8)
            split_date = df_encoded.iloc[split_index]['Date']  # Assuming 'Date' is the column indicating time

            # Step 2: Perform temporal splitting
            train_data = df_encoded[df_encoded['Date'] < split_date]
            test_data = df_encoded[df_encoded['Date'] >= split_date]

            # Extract features and target variable
            if 'Date_Deviation' in train_data.columns:
                X_train = train_data.drop(['Date_Deviation', 'Daily_Actual_Demand', 'PN', 'Date', 'Avg_Daily_Actual_Demand', 'Avg_Daily_Forecast_Demand'], axis=1)
            else:
                X_train = train_data.drop(['Daily_Actual_Demand', 'PN', 'Date', 'Avg_Daily_Actual_Demand', 'Avg_Daily_Forecast_Demand'], axis=1)
            y_train = train_data[target_column]

            if 'Date_Deviation' in test_data.columns:
                # Save PN and Date columns of X_test in separate vectors
                pn_test = test_data['PN'].values
                date_test = test_data['Date'].values
                X_test = test_data.drop(['Date_Deviation', 'Daily_Actual_Demand', 'PN', 'Date', 'Avg_Daily_Actual_Demand', 'Avg_Daily_Forecast_Demand'], axis=1)
            else:
                # Save PN and Date columns of X_test in separate vectors
                pn_test = test_data['PN'].values
                date_test = test_data['Date'].values
                X_test = test_data.drop(['Daily_Actual_Demand', 'PN', 'Date', 'Avg_Daily_Actual_Demand', 'Avg_Daily_Forecast_Demand'], axis=1)
            y_test = test_data[target_column]

            print("Train Start Date:", train_data['Date'].min())
            print("Train End Date:", train_data['Date'].max())
            print("Test Start Date:", test_data['Date'].min())
            print("Test End Date:", test_data['Date'].max())
            print("X_train shape:", X_train.shape)
            print("X_test shape:", X_test.shape)
            print("X Columns: ", X_train.columns)
            print("Target Column: ", target_column)

            best_model, result_df, train_mae, test_mae = self.temporal_reg_function(X_train = X_train, y_train= y_train,
                                                                               X_test = X_test, y_test = y_test, dates = date_test, pn = pn_test, indices = indices)
            best_train_error, best_test_error, best_iteration = self.iteration_reg_function(best_model = best_model, 
                                                                                            X_train = X_train, y_train = y_train, X_test = X_test,
                                         y_test = y_test, indices= indices)
            # print("result_df:", result_df.head())
            return indices, best_model, result_df, train_mae, test_mae, best_train_error, best_test_error, best_iteration
            
        else:
            print("Column 'PN' not found in the DataFrame.")

#######################################################################demand_function########################################################################
    
    # Main is feeded here and return reults for actual demand values
    def demand_function(self, indices=None):
        # indices = self.temporal_prob_function()
       indices, best_model, result_df = self.reg_main(target_column = self.target_column, indices = indices)
       return indices, best_model, result_df

#######################################################################derivative_function########################################################################
    
    # Main is feeded here and return reults for number of next days actual demand derivative_function values
    def derivative_function(self):

        # Calculate date deviation
        self.calculate_date_deviation()
        indices, best_model, result_df = self.reg_main(target_column = 'Date_Deviation')
        return indices, best_model, result_df

#######################################################################Confusion_test########################################################################
    
    #Confusion matrices test
    def compare_models_with_mcnemar(self, conf_matrix_A, conf_matrix_B, conf_matrix_C):
        def compute_mcnemar(matrix1, matrix2):
            # Calculate disagreements
            n_01 = matrix1[0, 1] + matrix2[0, 1]
            n_10 = matrix1[1, 0] + matrix2[1, 0]

            # Create 2x2 table for McNemar's test
            table = np.array([[0, n_01], [n_10, 0]])

            # Perform McNemar's test
            result = mcnemar(table, exact=True)
            return result.statistic, result.pvalue

        # Compare Model A vs Model B
        statistic_AB, pvalue_AB = compute_mcnemar(conf_matrix_A, conf_matrix_B)
        print(f"Model RF vs Model FNN: McNemar's Test Statistic = {statistic_AB:.6f}, p-value = {pvalue_AB:.6f}")

        # Compare Model A vs Model C
        statistic_AC, pvalue_AC = compute_mcnemar(conf_matrix_A, conf_matrix_C)
        print(f"Model RF vs Model LSTM: McNemar's Test Statistic = {statistic_AC:.6f}, p-value = {pvalue_AC:.6f}")

        # Compare Model B vs Model C
        statistic_BC, pvalue_BC = compute_mcnemar(conf_matrix_B, conf_matrix_C)
        print(f"Model FNN vs Model LSTM: McNemar's Test Statistic = {statistic_BC:.6f}, p-value = {pvalue_BC:.6f}")

        # Return the results in a structured format
        results = {
            "RF vs FNN": {"statistic": statistic_AB, "p-value": pvalue_AB},
            "RF vs LSTM": {"statistic": statistic_AC, "p-value": pvalue_AC},
            "FNN vs LSTM": {"statistic": statistic_BC, "p-value": pvalue_BC}
        }
        return results
    
#######################################################################Business_Problem########################################################################

    def optimize_order_quantity(df, T, k, C_inventory=1.0, C_order=2.0, C_penalty=3.0):
        # Calculate cost
        def calculate_cost(OQ, AD_t, IL_t):
            penalty_cost = max(0, (AD_t - IL_t)) * C_penalty
            return IL_t * C_inventory + OQ * C_order + penalty_cost

        # Order function
        def order_function(df, t, k):
            min_cost = float('inf')
            best_OQ = 0
            for OQ in range(1, 400):  
                total_cost = 0
                for t_ in range(t, t + k + 1):
                    rows = df[df['Date_ordinal'] == t_]
                    if not rows.empty:
                        for i in rows['PN'].unique():
                            AD_t = rows.loc[rows['PN'] == i, 'test_predictions'].values[0]
                            if t > df['Date_ordinal'].min():
                                IL_t_values = df.loc[(df['PN'] == i) & (df['Date_ordinal'] == t-1), 'IL'].values
                                IL_t = IL_t_values[0] if len(IL_t_values) > 0 else 0
                            else:
                                IL_t = 0
                            total_cost += calculate_cost(OQ, AD_t, IL_t)
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_OQ = OQ
            return best_OQ

        # Initialize IL and OQ columns
        df['IL'] = 0
        df['OQ'] = 0
        
        # Convert Date to ordinal
        df['Date_ordinal'] = df['Date'].apply(lambda x: x.toordinal())
        
        for t in range(df['Date_ordinal'].min(), df['Date_ordinal'].max() + 1):
            for i in df['PN'].unique():
                if t > df['Date_ordinal'].min():
                    IL_prev_values = df.loc[(df['PN'] == i) & (df['Date_ordinal'] == t-1), 'IL'].values
                    OQ_prev_values = df.loc[(df['PN'] == i) & (df['Date_ordinal'] == t-1), 'OQ'].values
                    if len(IL_prev_values) > 0 and len(OQ_prev_values) > 0:
                        df.loc[(df['PN'] == i) & (df['Date_ordinal'] == t), 'IL'] = IL_prev_values[0] + OQ_prev_values[0]
            best_OQ = order_function(df, t, k)
            df.loc[df['Date_ordinal'] == t, 'OQ'] = best_OQ

        # Drop the auxiliary column
        df.drop(columns=['Date_ordinal'], inplace=True)

        return df
