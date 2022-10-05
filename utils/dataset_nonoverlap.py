""" Dataset loader for the nonoverlapping dataset and mobility data in training Spatial CPC """

import torch.utils.data as data
import pandas as pd
import numpy as np
import logging
import pickle
from glob import glob

## Get the same logger from main"
logger = logging.getLogger("scpc")

class COVID1C(data.Dataset):
    def __init__(self, config):

        """
        Args:
            config (box): hyperparameters file
        """
        # Hyperparameters
        self.data_path = config.dataset.data_path

        # load data in subfolder <data>
        logger.info("Loading in dataset from %s", self.data_path)
        self.windows = np.load(self.data_path)
        self.windows = self.windows.astype('f')
        # self.windows = np.lib.stride_tricks.sliding_window_view(self.windows, (8, 7))

    def __getitem__(self, i):
        return self.windows[i]

    def __len__(self):
        return len(self.windows)

class COVIDCounties():
    def __init__(self, config):

        """
        Args:
            config (box): hyperparameters file
        """
        # Hyperparameters
        self.data_path = config.dataset.data_folder
        self.counties = config.dataset.counties

        # load data in subfolder <data>
        logger.info("Loading in dataset from %s", self.data_path)
        
        datatypes = {'fips': 'string'}

        # Import data from all 3 years
        us2020 = pd.read_csv(f'{self.data_path}/us-counties-2020.csv', index_col=0, parse_dates=True, dtype=datatypes)
        us2021 = pd.read_csv(f'{self.data_path}/us-counties-2021.csv', index_col=0, parse_dates=True, dtype=datatypes)
        us2022 = pd.read_csv(f'{self.data_path}/us-counties-2022.csv', index_col=0, parse_dates=True, dtype=datatypes)

        # Concatenate all 3 dataframes together
        data = pd.concat([us2020, us2021, us2022])

        # Drop NA values.
        data = data.dropna()

        # Use correct datatypes for fips, cases, and deaths
        data = data.astype({'cases': 'int', 'deaths': 'int'})

        # Reduce dataframe to only data from the selected counties
        data = data[data['fips'].isin(self.counties)]

        # For each county, select the rows from the dataframe for that county, and then do the following:
        # 1) Convert the cases column into a 7-day rolling average of cases.
        # 2) Drop the NA values from the column (due to taking the rolling average)
        # 3) Convert the cases column into a numpy array
        # 4) Use the np.lib.slide_tricks.sliding_window_view function to generate 7-day sliding windows
        # 5) Use the function again to take a 8-long sliding window of the 7-day sliding windows.
        # 6) Add that to a larger array.

        self.data = {}

        for county in self.counties:
            # Select county
            mask = data['fips'] == county
            cases = data.loc[mask, 'cases'].copy(deep=True)

            # Compute new cases, drop NA
            cases = cases.diff().dropna()

            # Convert to rolling average, drop NA
            cases = cases.rolling(7).mean().dropna()

            # Add percent change
            #cases = cases.pct_change(periods=1)

            # Fill nan values with 0
            cases = cases.fillna(0)

            # Replace inf with 1, -inf with 0
            cases = cases.replace(np.inf, 1)
            cases = cases.replace(-np.inf, 0)

            # Convert to numpy array & to floats
            pct_cases_county = cases.to_numpy()
            pct_cases_county = pct_cases_county.astype('f')

            # Use sliding_window_view to group into weeks
            weekly_data = np.lib.stride_tricks.sliding_window_view(pct_cases_county, 7)

            # Use sliding_window_view again to group into six weeks of data
            grouped_weekly_data = np.lib.stride_tricks.sliding_window_view(weekly_data, (28, 7))

            # Reshape the window so that it's in the correct format, and only select the
            # distinct weeks with no overlap
            grouped_weekly_data = grouped_weekly_data.reshape((grouped_weekly_data.shape[0], 28, 7))[:, ::7]

            self.data[county] = CountyData(county, grouped_weekly_data)

    def __len__(self):
        return len(self.data)

class CountyData(data.Dataset):
    def __init__(self, county, data):

        """
        Args:
            config (box): hyperparameters file
        """
        
        self.county = county
        self.data = data

    def get_county(self):
        return self.county

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)
    
class MobilityData():
    def __init__(self, config):
        list_fips = config.dataset.counties
        path = config.dataset.data_path
        with open(path, 'rb') as f:
            list_mob = pickle.load(f)
        trunc_mob = dict()
        for val in list_fips:
            total_inner = 0
            for v in list_mob[val]:
                total_inner += list_mob[val][v]
            inner_dict = dict()
            for v in list_mob[val]:
                inner_dict[v] = list_mob[val][v] / total_inner
            trunc_mob[val] = inner_dict
        self.data = trunc_mob