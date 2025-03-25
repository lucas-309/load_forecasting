import os
import pandas as pd
import datetime as dt
from model import get_data, prepare_data, add_features, create_model


# Constants
FORECAST_DATES = ['2024-02-15', '2024-04-04', '2024-06-26', '2024-09-13', '2024-11-10', '2025-03-25']
DATA_DIR = 'data/'
def forecast_load(models, scalers, forecast_date, df):
    forecast_date = pd.to_datetime(forecast_date)
    df_forecast = df[df.index.get_level_values('DateTime').normalize() == forecast_date]
    forecasted_loads = []
    for hour in range(24):
        df_hour = df_forecast[df_forecast['hour'] == hour]
        X_hour = df_hour.drop(columns=['Load'])
        model = models[hour]
        scaler = scalers[hour]
        X_scaled = scaler.transform(X_hour)
        forecasted_load = model.predict(X_scaled)
        forecasted_loads.append(forecasted_load[0])
    return forecasted_loads


def save_forecast_to_csv(forecast_date, forecasted_loads):
    forecast_date = pd.to_datetime(forecast_date)
    times = [forecast_date + dt.timedelta(hours=i) for i in range(24)]
    df_output = pd.DataFrame({'DateTime': times, 'Load': forecasted_loads})
    output_file = os.path.join(DATA_DIR, forecast_date.strftime('%Y%m%d') + '.csv')
    df_output.to_csv(output_file, index=False)


def main():
    df = get_data()
    df = prepare_data(df)
    df = add_features(df)
    # Define the training period
    train_start_date = dt.datetime(2020, 1, 1)
    train_end_date = dt.datetime(2023, 12, 31)
    models, scalers, _ = create_model(df, train_start_date, train_end_date)
    # Forecast for each date in FORECAST_DATES
    for forecast_date in FORECAST_DATES:
        forecasted_loads = forecast_load(models, scalers, forecast_date, df)
        save_forecast_to_csv(forecast_date, forecasted_loads)
        print(f"Forecast for {forecast_date} saved to CSV.")


if __name__ == '__main__':
    main()