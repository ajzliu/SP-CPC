import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

from box import box_from_file
from pathlib import Path
import pandas as pd

config = box_from_file(Path('config.yaml'), file_type='yaml')

# Hyperparameters
data_path = config.dataset.data_folder
counties = config.dataset.counties

datatypes = {'fips': 'string'}

# Import data from all 3 years
us2020 = pd.read_csv(f'{data_path}/us-counties-2020.csv', index_col=0, parse_dates=True, dtype=datatypes)
us2021 = pd.read_csv(f'{data_path}/us-counties-2021.csv', index_col=0, parse_dates=True, dtype=datatypes)
us2022 = pd.read_csv(f'{data_path}/us-counties-2022.csv', index_col=0, parse_dates=True, dtype=datatypes)

# Concatenate all 3 dataframes together
data = pd.concat([us2020, us2021, us2022])

# Drop NA values.
data = data.dropna()

# Use correct datatypes for fips, cases, and deaths
data = data.astype({'cases': 'int', 'deaths': 'int'})

# Reduce dataframe to only data from the selected counties
data = data[data['fips'].isin(counties)]

all_X, all_y = [], []

for county in counties:
    # Select county
    mask = data['fips'] == county
    cases = data.loc[mask, 'cases'].copy(deep=True)

    # Compute new cases, drop NA
    cases = cases.diff().dropna()

    # Convert to rolling average, drop NA
    cases = cases.rolling(7).mean().dropna()

    # Fill nan values with 0
    cases = cases.fillna(0)

    # Replace inf with 1, -inf with 0
    cases = cases.replace(np.inf, 1)
    cases = cases.replace(-np.inf, 0)

    # Convert to numpy array & to floats
    pct_cases_county = cases.to_numpy()
    pct_cases_county = pct_cases_county.astype('f')
    
    # Group data into weeks
    grouped_weeks = np.lib.stride_tricks.sliding_window_view(pct_cases_county, 13)

    # Pair together data and targets
    combined = [(grouped_weeks[i], grouped_weeks[i + 13][:7]) for i in range(len(grouped_weeks) - 13)]

    # Unzip data into X and y.
    X, y = zip(*combined)

    all_X += X
    all_y += y

# Remove all targets and corresponding X values with values less than 100
assert len(all_X) == len(all_y)

removed_X, removed_y = [], []

for i in range(len(all_y)):
    if np.any(all_y[i] < 100):
        continue

    removed_X.append(all_X[i])
    removed_y.append(all_y[i])

X_train, X_test, y_train, y_test = train_test_split(removed_X, removed_y, test_size=0.2, random_state=42)

reg = LinearRegression().fit(X_train, y_train)

y_pred = reg.predict(X_test)
y_pred_test = reg.predict(X_train)

print("Training set: " + str(mean_absolute_percentage_error(y_train, y_pred_test)))
print("Testing set: " + str(mean_absolute_percentage_error(y_test, y_pred)))