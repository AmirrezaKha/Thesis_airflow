
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

class dataProcessor:
    def __init__(self, num_lags, source_table, dest_table, db_params_df):
        self.num_lags = num_lags
        self.source_table = source_table
        self.dest_table = dest_table
        self.db_params_df = db_params_df
        self.mining_df = None
        self.a500_df = None
        self.engine = self.create_engine_from_params()

    def create_engine_from_params(self):
        db_params = self.db_params_df.iloc[0]
        db_user = db_params['POSTGRES_USER']
        db_password = db_params['POSTGRES_PASSWORD']
        db_host = db_params['POSTGRES_HOST']
        db_port = db_params['POSTGRES_PORT']
        db_name = db_params['POSTGRES_DB']

        if not db_user or not db_password or not db_host or not db_port or not db_name:
            raise ValueError("Database credentials are not set properly in the DataFrame")

        return create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

    def get_postgres_data(self):
        query = f"SELECT * FROM {self.source_table}"
        self.a500_df = pd.read_sql(query, self.engine)

    def pre(self):
        self.a500_df['Sent.Date'] = pd.to_datetime(self.a500_df['Sent.Date'])
        threshold_date = pd.to_datetime('2020-01-01')
        self.a500_df['Production.Date'] = pd.to_datetime(self.a500_df['Production.Date'])
        self.a500_df = self.a500_df[(self.a500_df['Sent.Date'] >= threshold_date) & (self.a500_df['Production.Date'] >= threshold_date)]
        self.a500_df.rename(columns={'Modelo': 'Model'}, inplace=True)

        unique_values = self.a500_df['Model'].unique()
        label_mapping = {value: f'm{i+1}' for i, value in enumerate(unique_values)}
        self.a500_df['Model'] = self.a500_df['Model'].map(label_mapping)

        unique_values = self.a500_df['PN'].unique()
        label_mapping = {value: f'PN{i+1}' for i, value in enumerate(unique_values)}
        self.a500_df['PN'] = self.a500_df['PN'].map(label_mapping)

        unique_values = self.a500_df['Item.Category.Code'].unique()
        label_mapping = {value: f'ICC{i+1}' for i, value in enumerate(unique_values)}
        self.a500_df['Item.Category.Code'] = self.a500_df['Item.Category.Code'].map(label_mapping)

    def create_mining(self):
        self.a500_df['Production.ID'] = range(1, len(self.a500_df) + 1)
        self.a500_df_SentDate = self.a500_df.groupby(['PN', 'Sent.Date']).agg(Daily_Actual_Demand=('Production.ID', 'count')).reset_index()
        self.a500_df_ProductionDate = self.a500_df.groupby(['PN', 'Production.Date']).agg(Daily_Forecast_Demand=('Production.ID', 'count')).reset_index()

        from itertools import product

        min_date = min(self.a500_df['Production.Date'].min(), self.a500_df['Sent.Date'].min())
        max_date = max(self.a500_df['Production.Date'].max(), self.a500_df['Sent.Date'].max())
        date_range = pd.date_range(min_date, max_date, freq='B')
        calendar_df = pd.DataFrame(list(product(date_range, self.a500_df['PN'].unique())), columns=['Date', 'PN'])
        calendar_df['Date'] = pd.to_datetime(calendar_df['Date'])
        self.a500_df_SentDate['Sent.Date'] = pd.to_datetime(self.a500_df_SentDate['Sent.Date'])
        self.a500_df_ProductionDate['Production.Date'] = pd.to_datetime(self.a500_df_ProductionDate['Production.Date'])

        self.mining_df = pd.merge(calendar_df, self.a500_df_SentDate, left_on=['Date', 'PN'], right_on=['Sent.Date', 'PN'], how='left')
        self.mining_df = pd.merge(self.mining_df, self.a500_df_ProductionDate, left_on=['Date', 'PN'], right_on=['Production.Date', 'PN'], how='left')

        self.mining_df['Daily_Actual_Demand'].fillna(0, inplace=True)
        self.mining_df['Daily_Forecast_Demand'].fillna(0, inplace=True)
        self.mining_df = self.mining_df[["Date", "PN", "Daily_Actual_Demand", "Daily_Forecast_Demand"]]

        date_range_days = self.mining_df.groupby('PN').cumcount()
        sum_by_pn = self.mining_df.groupby('PN')['Daily_Actual_Demand'].cumsum() - self.mining_df['Daily_Actual_Demand']
        self.mining_df['Avg_Daily_Actual_Demand'] = sum_by_pn / date_range_days
        self.mining_df['Diff_Avg_Daily_Actual_Demand'] = (self.mining_df.groupby('PN')['Daily_Actual_Demand'].shift(1) - self.mining_df['Avg_Daily_Actual_Demand']) / self.mining_df['Avg_Daily_Actual_Demand']
        self.mining_df['Diff_Avg_Daily_Actual_Demand'].replace(np.inf, np.nan, inplace=True)

        date_range_days = self.mining_df.groupby('PN')['Daily_Actual_Demand'].transform('count') - 1
        sum_by_pn = self.mining_df.groupby('PN')['Daily_Forecast_Demand'].transform('sum') - self.mining_df['Daily_Forecast_Demand']
        self.mining_df['Avg_Daily_Forecast_Demand'] = sum_by_pn / date_range_days
        self.mining_df['Diff_Avg_Daily_Forecast_Demand'] = (self.mining_df.groupby('PN')['Daily_Forecast_Demand'].shift(1) - self.mining_df['Avg_Daily_Forecast_Demand']) / self.mining_df['Avg_Daily_Forecast_Demand']
        self.mining_df['Diff_Avg_Daily_Forecast_Demand'].replace(np.inf, np.nan, inplace=True)

    def add_lag(self):
        for lag in range(1, self.num_lags + 1):
            lag_column_name = f'Diff_Avg_Daily_Actual_Demand{lag}'
            self.mining_df[lag_column_name] = self.mining_df.groupby('PN')['Diff_Avg_Daily_Actual_Demand'].shift(lag)

        for lag in range(1, self.num_lags + 1):
            lag_column_name = f'Diff_Avg_Daily_Forecast_Demand{lag}'
            self.mining_df[lag_column_name] = self.mining_df.groupby('PN')['Diff_Avg_Daily_Forecast_Demand'].shift(lag)

        self.mining_df['Date'] = pd.to_datetime(self.mining_df['Date'])

        grouped_df = self.mining_df.groupby(['PN', self.mining_df['Date'].dt.day_name()])['Daily_Actual_Demand'].agg(['sum', 'count'])
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        for weekday in weekdays:
            col_name_sum = f'Avg_{weekday}_sum'
            col_name_count = f'Avg_{weekday}_count'
            col_name_avg = f'Avg_Actual_{weekday}'
            self.mining_df[col_name_sum] = self.mining_df.apply(lambda row: grouped_df.loc[(row['PN'], weekday), 'sum'], axis=1) - self.mining_df['Daily_Actual_Demand']
            self.mining_df[col_name_count] = self.mining_df.apply(lambda row: grouped_df.loc[(row['PN'], weekday), 'count'], axis=1)
            self.mining_df[col_name_avg] = self.mining_df[col_name_sum] / self.mining_df[col_name_count]
            self.mining_df.drop(columns=[col_name_sum, col_name_count], inplace=True)

        grouped_df = self.mining_df.groupby(['PN', self.mining_df['Date'].dt.day_name()])['Daily_Forecast_Demand'].agg(['sum', 'count']) - 1
        for weekday in weekdays:
            col_name_sum = f'Avg_{weekday}_sum'
            col_name_count = f'Avg_{weekday}_count'
            col_name_avg = f'Avg_Forecast_{weekday}'
            self.mining_df[col_name_sum] = self.mining_df.apply(lambda row: grouped_df.loc[(row['PN'], weekday), 'sum'], axis=1) - self.mining_df['Daily_Forecast_Demand']
            self.mining_df[col_name_count] = self.mining_df.apply(lambda row: grouped_df.loc[(row['PN'], weekday), 'count'], axis=1) - 1
            self.mining_df[col_name_avg] = self.mining_df[col_name_sum] / self.mining_df[col_name_count]
            self.mining_df.drop(columns=[col_name_sum, col_name_count], inplace=True)

    def calculate_days_since_last_non_zero_demand(group):
            date_diffs = []
            prev_non_zero_demand_date = None
            for index, row in group.iterrows():
                if row['Daily_Actual_Demand'] != 0:
                    if prev_non_zero_demand_date is not None:
                        days_diff = (row['Date'] - prev_non_zero_demand_date).days
                    else:
                        days_diff = 0
                    prev_non_zero_demand_date = row['Date']
                else:
                    if prev_non_zero_demand_date is not None:
                        days_diff = (row['Date'] - prev_non_zero_demand_date).days
                    else:
                        days_diff = None
                date_diffs.append(days_diff)
            group['Days_since_last_non_zero_demand'] = date_diffs
            return group

    def preprocess_data(self):
        self.get_postgres_data()
        self.pre()
        self.create_mining()
        self.add_lag()

        self.mining_df['Year'] = self.mining_df['Date'].dt.year
        self.mining_df['Month'] = self.mining_df['Date'].dt.month
        self.mining_df['Day'] = self.mining_df['Date'].dt.day
        self.mining_df['DayOfWeek'] = self.mining_df['Date'].dt.dayofweek
        self.mining_df['Date'] = pd.to_datetime(self.mining_df['Date'])



        self.mining_df = self.mining_df.groupby('PN').apply(self.calculate_days_since_last_non_zero_demand).reset_index(drop=True)
        return self.mining_df
