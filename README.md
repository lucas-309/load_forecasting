# Millennium Load Forecasting Challenge 2025

This project provides a comprehensive framework for forecasting electricity load, specifically tailored for the Millennium Load Forecasting Challenge 2025. It includes scripts for data acquisition, feature engineering, model training, and forecasting.

## Team Members

- Lucas He
- Ratchaphon Lertdamrongwong

## Project Overview

The core of this project is a machine learning model that predicts hourly electricity load. The model is trained on historical load data from the New York Independent System Operator (NYISO) and weather data from various weather stations in New York. The final output is a CSV file with 24-hour load forecasts for specified dates.

## Features

- **Automated Data Retrieval:** Scripts to download historical load data from NYISO and weather data from Meteostat.
- **Feature Engineering:** Creates a rich set of features including:
    - Temperature and its square
    - Time-based features (hour, day of the week, month, season)
    - Holiday and weekend indicators
    - Lagged load values
    - Rolling mean of the load
- **Hourly Models:** Trains a separate Random Forest Regressor for each hour of the day to capture distinct daily patterns.
- **Time Series Cross-Validation:** Uses time series cross-validation to evaluate model performance robustly.
- **Scalability:** The framework can be easily adapted for different date ranges and forecasting horizons.

## Getting Started

### Prerequisites

- Python 3.x
- The required Python libraries can be installed via pip:
  ```bash
  pip install pandas requests meteostat scikit-learn matplotlib
  ```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd load_forecasting
    ```

2.  **Create the data directories:**
    The scripts will automatically create the necessary `data/load` and `data/weather` directories if they don't exist.

### Running the Project

The main driver script for the project is `forecast_challenge_driver.py`.

1.  **Data Acquisition:**
    - Run `get_load_data.py` to fetch the latest load data from NYISO.
    - Run `get_weather_data.py` to fetch the latest weather data.

    ```bash
    python get_load_data.py
    python get_weather_data.py
    ```
    *Note: These scripts can take a while to run, especially on the first execution as they download several years of historical data.*

2.  **Generate Forecasts:**
    - Modify the `FORECAST_DATES` list in `forecast_challenge_driver.py` to include the dates you want to forecast.
    - Run the script:
      ```bash
      python forecast_challenge_driver.py
      ```
    - The forecasts will be saved as CSV files in the `data/` directory, with filenames like `YYYYMMDD.csv`.

## File Descriptions

- **`forecast_challenge_driver.py`**: The main script to run the forecasting model. It loads data, trains the models, and generates forecasts for the specified dates.
- **`get_load_data.py`**: Fetches historical and current electricity load data from the NYISO website.
- **`get_weather_data.py`**: Fetches historical and current weather data for various NY stations using the `meteostat` library.
- **`model.py`**: Contains the core logic for data preparation, feature engineering, model training (`RandomForestRegressor`), and evaluation.
- **`data/`**: This directory stores all the data.
    - `load/`: Contains the raw and processed load data.
    - `weather/`: Contains the raw and processed weather data.
- **`README.md`**: This file.

## Model Details

The forecasting approach consists of 24 individual `RandomForestRegressor` models, one for each hour of the day. This allows the model to learn the specific load patterns associated with each hour.

The features used for training include:
- **Weather:** Temperature from two key weather stations (JFK and Syracuse) and their squared values.
- **Time-based:** Hour of the day, day of the week, month, and season dummies.
- **Events:** Binary flags for weekends and US federal holidays.
- **Historical Load:** Lagged load from the previous hour and the same hour on the previous day, plus a 7-day rolling average of the load.

The data is scaled using `StandardScaler` before being fed into the model.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

## License

## License

This project is licensed under the MIT License.