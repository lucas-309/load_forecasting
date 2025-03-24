import os

import requests

import pandas as pd

from io import BytesIO

from zipfile import ZipFile

import datetime as dt

import time

import os

 

# Create the directory if it doesn't exist

DATA_DIR = r'C:\temp\nyiso_load'

os.makedirs(DATA_DIR, exist_ok=True)

 

HIST_LOAD_FILE = 'nyiso_hist_load.csv'

CURR_LOAD_FILE = 'nyiso_curr_load.csv'

LOAD_FILE = 'nyiso_load.csv'

# Get load data from NYISO

def get_load_data_from_nyiso(start_date, end_date):

    # Initialize an empty dataframe for the master data

    df_load_raw = pd.DataFrame()

   

    # Loop through each month in the date range

    current_date = start_date

    while current_date <= end_date:

        print(f'loading file: {current_date}')

       

        # Format the date for the URL

        date_str = current_date.strftime('%Y%m%d')

        url = f'https://mis.nyiso.com/public/csv/palIntegrated/{date_str}palIntegrated_csv.zip'

   

        # Download the zip file

        try:

            response = requests.get(url)

        except:

            print('error loading file -- retry after 30 sec')

            time.sleep(30)

            continue

       

        if response.status_code == 200:

            with ZipFile(BytesIO(response.content)) as z:

                # Extract all files in the zip

                for file_name in z.namelist():

                    if file_name.endswith('.csv'):

                        with z.open(file_name) as f:

                            # Read the CSV file into a dataframe

                            df_daily = pd.read_csv(f)

                            # Append to the master dataframe

                            df_load_raw = pd.concat([df_load_raw, df_daily], ignore_index=True)

        else:

            print(f"Failed to download {url}")

   

        # Move to the next month

        next_month = current_date.month + 1

        next_year = current_date.year + (next_month // 13)

        next_month = next_month % 12 or 12

        current_date = dt.datetime(next_year, next_month, 1)

        time.sleep(5)

       

    # Transform the df_load_raw dataframe into df_load

    df_load_raw['DateTime'] = pd.to_datetime(df_load_raw['Time Stamp'])

    df_load_raw['Hour'] = df_load_raw['DateTime'].dt.floor('H')

    df_load = df_load_raw.groupby(['Hour', 'Time Zone'])['Integrated Load'].sum().reset_index()

    df_load.rename(columns={'Hour': 'DateTime', 'Time Zone': 'TZ', 'Integrated Load': 'Load'}, inplace=True)

   

    return df_load

# Read historical load from local file or NYISO

def get_hist_load_data():

   

    hist_load_file_path = os.path.join(DATA_DIR, HIST_LOAD_FILE)

    if os.path.exists(hist_load_file_path):

        df_hist_load = pd.read_csv(hist_load_file_path)

    else:

        start_date = dt.datetime(2020, 1, 1)

        end_date = dt.datetime(2025, 1, 1)

        df_hist_load = get_load_data_from_nyiso(start_date, end_date)

       

        # Save the historical load file

        df_hist_load.to_csv(hist_load_file_path, index=False)

       

    return df_hist_load

# Read current load from NYISO

def get_curr_load_data():

   

    curr_load_file_path = os.path.join(DATA_DIR, CURR_LOAD_FILE)

    start_date = dt.datetime(2025, 2, 1)

    end_date = dt.datetime(dt.datetime.today().year, dt.datetime.today().month, 1)

    df_curr_load = get_load_data_from_nyiso(start_date, end_date)

   

    # Save the historical load file

    df_curr_load.to_csv(curr_load_file_path, index=False)

       

    return df_curr_load


# Get the NYISO load data    

def get_load_data():

    df_hist_load = get_hist_load_data()

    df_curr_load = get_curr_load_data()

   

    df_load = pd.concat([df_hist_load,df_curr_load])

   

    # Save the load file

    load_file_path = os.path.join(DATA_DIR, LOAD_FILE)

    df_load.to_csv(load_file_path, index=False)

   

    return df_load

   

 

if __name__ == '__main__':

    df_load = get_load_data()