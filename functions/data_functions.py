import pandas as pd
import numpy as np

def clean_data(df, cols):
    '''
    Simple cleaning. Removes NAs, and selects only relevant columns
    PARAMS
    df - the input df
    cols - a list of columns to select
    '''
    df = df.dropna()
    df.columns = cols
    return df


def add_lags(df, cols, n_lags):
    '''
    Adds past values (lags) from select columns as new column values.
    PARAMS
    df - the input df
    cols - a list of columns to make lags from
    n_lags - the functions makes this many number of lags. eg if set to 3, it will make a column, for lag1, lag2 and lag3 for each variable
    '''
    for col in cols:
        if col == 'date':
            continue
        for lag in range(1, n_lags + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
            
    df = df.dropna()
    return(df)

def add_diffs(df, cols, n):
    '''
    Adds fractional differences releative to past values, for select columns, and puts them in a new column.
    PARAMS
    df - the input df
    cols - a list of columns to make differences from
    n - the functions makes this many number of lags to make fractional differences from
    eg if set to 3, it will make a column, for current value/lag1, current value/lag2 and current value/lag3, for each variable
    '''
    for col in cols:
        if col == 'date':
            continue
        for lag in range(1, n + 1):
            df[f"{col}_diff{lag}"] = df[col]/df[col].shift(lag)
    df = df.dropna()
    return(df)

# Potential to do further work with stationarity using this:
# from statsmodels.tsa.stattools import adfuller
# or with technical indicators including Movign averages, using the ta library