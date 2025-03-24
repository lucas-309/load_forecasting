from meteostat import Stations, Hourly # (pip install meteostat if necessary)

import datetime as dt

import pandas as pd

import os

import calendar

import time


# List of weather station identifiers

WEATHER_STATIONS = [

    'KALB','KART','KBGM','KELM','KELZ',

    'KFOK','KFRG','KGFL','KHPN','KIAG',

    'KISP','KITH','KJFK','KLGA','KMSS',

    'KMSV','KNYC','KPLB','KPOU','KROC',

    'KSLK','KSWF','KSYR','KUCA','KBGM'

]

 

# Create the directory if it doesn't exist

DATA_DIR = r'C:\temp\nyiso_load'

os.makedirs(DATA_DIR, exist_ok=True)

 

LOAD_FILE = 'nyiso_weather.csv'

 

LOAD_FROM_FILE = True

 

# Function to fetch weather stations in a specific region

def get_stations(country='US', state='NY'):

    country = 'US'

    state = 'NY'

    stations = Stations()

    stations = stations.region(country, state)

    df_stations = stations.fetch()

   

    return df_stations

 

# Function to get station IDs for NYISO weather stations

def get_nyiso_station_ids():

    df_stations = get_stations(country='US', state='NY')

    df_stations = df_stations[df_stations['icao'].isin(WEATHER_STATIONS)]

    df_stations = df_stations[['icao']]

    df_stations.columns = ['station']

   

    return df_stations

 

# Function to determine the timezone name based on daylight saving time

def get_tz_name(dt):

    if dt.dst() != pd.Timedelta(0):

        return 'EDT'

    else:

        return 'EST'

 

# Function to fetch hourly weather data for a specific station and time range

def get_hourly_data(station, dt_from, dt_to):

    data = Hourly(station, dt_from, dt_to, timezone='America/New_York')

    df = data.fetch()

   

    return df

 

def clean_data(df):

    # Remove all nan columns

    df = df.dropna(axis=1, how='all')

   

    # Drop columns with more than 50% NaNs

    threshold = len(df) * 0.5

    df = df.dropna(axis=1, thresh=threshold)

   

    # Get the max datetime in the index

    max_datetime = df.index.max()

   

    # Define the time window for the last 48 hours

    time_window_start = max_datetime - pd.Timedelta(hours=48)

   

    # Drop columns with NaNs in the last 48 hours

    cols_to_drop = df.loc[time_window_start:max_datetime].isna().any()

    df = df.drop(columns=cols_to_drop[cols_to_drop].index)

   

    # Drop KSWF (missing historical data)

    del df['KSWF']

   

    # Forward fill missing data

    # Note: There are better ways to fill missing data! E.g. fill gaps with yesterday if possible

    df = df.ffill()

   

    return df

 

# Function to get hourly weather data for all NYISO stations

def get_nyiso_hourly_weather_data():

   

    if not LOAD_FROM_FILE:

        # Get NYISO weather station list

        df_stations = get_nyiso_station_ids()

       

        # Set date range

        dt_today = dt.datetime(dt.datetime.today().year, dt.datetime.today().month, dt.datetime.today().day)

        dt_from = dt.datetime(2020, 1, 1)

        dt_to = dt.datetime(dt_today.year, dt_today.month, dt_today.day, 23, 59) + dt.timedelta(days=1)

       

        # Loop through weather stations by id and get hourly data

        df = pd.DataFrame()

        station_id_list = df_stations.index.values.tolist()

        for station_id in station_id_list:

            df_temp = get_hourly_data(station_id, dt_from, dt_to)

            df_temp = df_temp[['temp']]

            station = df_stations.loc[station_id].iat[0]

            df_temp.columns = [station]

           

            if len(df) == 0:

                df = df_temp

            else:

                df = df.join(df_temp, how='left')

               

            time.sleep(2)

           

        # Add timezone indicator

        df['TZ'] = df.index.map(get_tz_name)

        df.index.name = 'DateTime'

        df.index = df.index.tz_localize(None)

       

        # Clean data

        df = clean_data(df)

   

        # Save file

        file_dir = os.path.dirname(__file__)

        df.to_csv(os.path.join(DATA_DIR,LOAD_FILE))

   

    else:

        df = df.read_csv(os.path.join(DATA_DIR,LOAD_FILE))

   

    return df

   

if __name__ == '__main__':

    df = get_nyiso_hourly_weather_data()

