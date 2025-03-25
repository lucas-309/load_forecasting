import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler


DATA_DIR = 'data/'
LOAD_FILE = 'load/nyiso_load.csv'
WEATHER_FILE = 'weather/nyiso_weather.csv'

# Get all of the data for the forecasts
def get_data():
    # Get the data files
    df_weather = pd.read_csv(os.path.join(DATA_DIR, WEATHER_FILE), parse_dates=['DateTime'])
    df_load = pd.read_csv(os.path.join(DATA_DIR, LOAD_FILE), parse_dates=['DateTime'])
    # Merge Weather and Load on DateTime and TZ (Timezone = EST, EDT)
    df = pd.merge(df_weather, df_load, left_on=['DateTime', 'TZ'], right_on=['DateTime', 'TZ'], how='left')
    return df

# Prepare the data for forecasts
def prepare_data(df):
    # Convert the DateTime field to a datetime
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    # Set the index
    df.set_index(['DateTime', 'TZ'], inplace=True)
    # Reduce to two weather stations, say JFK and SYR ...
    df = df[['KJFK', 'KSYR', 'Load']]
    # Rename Stations
    df.rename(columns={'KJFK': 'Temp1', 'KSYR': 'Temp2'}, inplace=True)
    return df

def add_features(df):
    # Add Temp1 squared and Temp2 squared
    df['Temp1_squared'] = df['Temp1'] ** 2
    df['Temp2_squared'] = df['Temp2'] ** 2
    # Extract hour of the day
    df['hour'] = df.index.get_level_values('DateTime').hour
    # Extract day of the week and create dummies
    df['day_of_week'] = df.index.get_level_values('DateTime').dayofweek
    day_dummies = pd.get_dummies(df['day_of_week'], prefix='day', drop_first=True).astype(int)  # Drop the first dummy and convert to int
    df = pd.concat([df, day_dummies], axis=1)
    # Create weekend dummy
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0).astype(int)
    # Create holiday dummy
    cal = calendar()
    holidays = cal.holidays(start=df.index.get_level_values('DateTime').min(), end=df.index.get_level_values('DateTime').max())
    df['is_holiday'] = df.index.get_level_values('DateTime').normalize().isin(holidays).astype(int)
    # Create weekend/holiday indicator
    df['weekend_holiday'] = df[['is_weekend', 'is_holiday']].max(axis=1).astype(int)
    # Add season dummy (e.g., Winter, Spring, Summer, Fall)
    df['month'] = df.index.get_level_values('DateTime').month
    df['season'] = df['month'] % 12 // 3 + 1
    season_dummies = pd.get_dummies(df['season'], prefix='season', drop_first=True).astype(int)  # Drop the first dummy and convert to int
    df = pd.concat([df, season_dummies], axis=1)
    # Drop temporary columns
    df.drop(columns=['day_of_week', 'is_weekend', 'is_holiday', 'month', 'season'], inplace=True)
    # Reorder columns to move Load to the end
    load_column = df.pop('Load')
    df['Load'] = load_column
    df['Load_smooth'] = (
        df['Load']
        .rolling(window=3, center=True)
        .mean()
        .fillna(method='bfill')
        .fillna(method='ffill')
    )
    return df

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def create_model(df, train_start_date, train_end_date):
    # Filter the data for the training period
    df_train = df[(df.index.get_level_values('DateTime') >= train_start_date) &
                  (df.index.get_level_values('DateTime') <= train_end_date)]
    models = {}
    scalers = {}
    results = {'hour': [], 'train_rmse': [], 'test_rmse': [], 'train_mape': [], 'test_mape': [], 'cv_rmse': [], 'cv_rmse_std': []}
    tscv = TimeSeriesSplit(n_splits=5)
    for hour in range(24):
        print(f"Processing hour: {hour}")
        # Filter data for the current hour
        df_hour = df_train[df_train['hour'] == hour]
        X_hour = df_hour.drop(columns=['Load'])
        y_hour = df_hour['Load_smooth']
        # Decompose into train and test set
        X_train, X_test, y_train, y_test = train_test_split(X_hour, y_hour, test_size=0.2, random_state=42)
        # Scale the feature data
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
        # Initialize the model
        model = RandomForestRegressor(n_estimators=300, max_depth = 20, random_state=42, min_samples_split=3)
        # Time series cross-validation to evaluate the model
        cv_rmse_scores = []
        for train_index, test_index in tscv.split(X_train_scaled):
            X_cv_train, X_cv_test = X_train_scaled[train_index], X_train_scaled[test_index]
            y_cv_train, y_cv_test = y_train.iloc[train_index], y_train.iloc[test_index]
            model.fit(X_cv_train, y_cv_train)
            y_cv_pred = model.predict(X_cv_test)
            cv_rmse_scores.append(np.sqrt(mean_squared_error(y_cv_test, y_cv_pred)))
        cv_rmse_scores = np.array(cv_rmse_scores)
        # Fit the model on the full training data
        model.fit(X_train_scaled, y_train)
        # Predict on the training and test sets
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        # Calculate RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        # Calculate MAPE
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
        # Save the trained model and scaler
        models[hour] = model
        scalers[hour] = scaler
        # Save results
        results['hour'].append(hour)
        results['train_rmse'].append(train_rmse)
        results['test_rmse'].append(test_rmse)
        results['train_mape'].append(train_mape)
        results['test_mape'].append(test_mape)
        results['cv_rmse'].append(cv_rmse_scores.mean())
        results['cv_rmse_std'].append(cv_rmse_scores.std())
        print(f"Hour {hour}: In-Sample RMSE: {train_rmse}, Out-of-Sample RMSE: {test_rmse}, Train MAPE: {train_mape}, Test MAPE: {test_mape}, CV RMSE: {cv_rmse_scores.mean()} (std: {cv_rmse_scores.std()})")
    # Convert results to DataFrame for better readability
    results_df = pd.DataFrame(results)
    return models, scalers, results_df


def plot_predictions(df, models, scalers, train_end_date, num_periods=10):
    # Filter the data for dates beyond train_end_date
    df_test = df[df.index.get_level_values('DateTime') > train_end_date]
    test_dates = df_test.index.get_level_values('DateTime').normalize().unique()
    # random_dates = np.random.choice(test_dates, num_periods, replace=False)
    for date in ['2024-02-15', '2024-04-04', '2024-06-26', '2024-09-13', '2024-11-10']:
        df_period = df_test[df_test.index.get_level_values('DateTime').normalize() == date]
        actual_loads = df_period['Load'].values
        predicted_loads = []
        for hour in range(24):
            df_hour = df_period[df_period['hour'] == hour]
            X_hour = df_hour.drop(columns=['Load'])
            model = models[hour]
            scaler = scalers[hour]
            X_scaled = scaler.transform(X_hour)  # Use the saved scaler for transformation
            predicted_load = model.predict(X_scaled)
            predicted_loads.append(predicted_load[0])

        plt.figure(figsize=(12, 6))
        plt.plot(range(24), actual_loads, label='Actual Load')
        plt.plot(range(24), predicted_loads, label='Predicted Load', linestyle='--')
        plt.xlabel('Hour')
        plt.ylabel('Load')
        plt.title(f'Load Prediction for Date: {pd.to_datetime(date).date()}')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    df = get_data()
    df = prepare_data(df)
    df = add_features(df)
    # Define the training period
    train_start_date = dt.datetime(2020, 1, 1)
    train_end_date = dt.datetime(2023, 12, 31)
    models, scalers, results_df = create_model(df, train_start_date, train_end_date)
    print(results_df)
    print("Average Test RMSE: ", sum(results_df['test_rmse']) / 24)
    # Plot predictions for 10 random 24-hour periods after the training end date
    plot_predictions(df, models, scalers, train_end_date=train_end_date, num_periods=10)