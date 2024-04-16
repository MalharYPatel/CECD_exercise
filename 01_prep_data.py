#0. Imports and config
#update system path
import os
import sys
wd = os.path.dirname(__file__) 
os.chdir(wd)
if wd in sys.path:
    sys.path.insert(0, wd)

#imports
#from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import configparser
import ast
from functions.data_functions import *
#config
config = configparser.ConfigParser()
config.read('config.ini')


# 1. Load Data
#1. Load each sheet as its own table
gdp_df = pd.read_excel("data/Forecasting_dataset.xlsx", 'GDP')
monthly_df = pd.read_excel("data/Forecasting_dataset.xlsx", 'Monthly surveys')
quarterly_df = pd.read_excel("data/Forecasting_dataset.xlsx", 'Quarterly surveys')


# 2. Clean Data
# Inspection in the variable explorer shows us that the column names are messy, the dat column has no name,
# and the monthly dataframe has a trailing tail of missing information. We need to clean this up.
#Also of note, is that the monthly series doesn't extend as far back. If this is used, it will limit our training data
gdp_df = clean_data(gdp_df, ast.literal_eval(config['data']['gdp_columns']))
monthly_df = clean_data(monthly_df, ast.literal_eval(config['data']['monthly_columns']))
quarterly_df = clean_data(quarterly_df, ast.literal_eval(config['data']['quarterly_columns']))



#3. Feature engineer
#Only some basic engineering has been done, due to time restrictions of excercise
#Stationarity tests (eg using adf test) have also not been done due to time limitations

# Example code for future implementation
# #Check stationarity by Testing null hypothesis of order 1 integration
# cols = df.iloc[:,1:-1].columns # list of columns to test
# adf_result = []
# for col in cols:
#     adf_result.append(adfuller(df[col], maxlag = 1)[1])

# stationarity_df = pd.DataFrame({'variable':cols, 'p_value':adf_result})


gdp_cols = gdp_df.columns.copy()
gdp_df = add_lags(gdp_df,  gdp_cols, 4)
gdp_df = add_diffs(gdp_df, gdp_df, 4)

monthly_cols = monthly_df.columns.copy()
monthly_df = add_lags(monthly_df, monthly_cols, 4)
monthly_df = add_diffs(monthly_df, monthly_cols, 4)

quarterly_cols = quarterly_df.columns.copy()
quarterly_df = add_lags(quarterly_df, quarterly_cols, 12)
quarterly_df = add_diffs(quarterly_df, quarterly_cols, 12)

# 4. Merge data
#we make 2 data sets
#A) data which is just GDP and quarterly survey data
full_df = gdp_df.merge(quarterly_df, how = 'left', left_on='date', right_on='date').dropna()

#B) data which is gdp growth, quarterly and monthly survey data. There are less data points, but more features
partial_df = full_df.merge(monthly_df, how = 'left', left_on='date', right_on='date').dropna()

# 5. Save result for convenience
partial_df.to_csv(f"{wd}{config['data']['partial_data_output_path']}", index = False)
full_df.to_csv(f"{wd}{config['data']['full_data_output_path']}",index = False)
