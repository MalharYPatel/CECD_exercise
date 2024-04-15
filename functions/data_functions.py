from statsmodels.tsa.stattools import adfuller
import configparser
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import pandas_ta as ta

#Get config variables
config_path = __file__
config_path = config_path.split('functions\data_functions.py')[0]
config_path = config_path + 'config.ini'
config = configparser.ConfigParser()
config.read(config_path)

def clean_gdp(df, cols):
    df = df.dropna()
    df.columns = cols
    return df


def clean_monthly(df, cols):
    df = df.dropna()
    df.columns = cols
    return df

def clean_quarterly(df, cols):
    df = df.dropna()
    df.columns = cols    
    return df

def add_lags(df, cols, n_lags):
    for col in df.columns:
        if col == 'date':
            continue
        for lag in range(1, n_lags + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
            
    df = df.dropna()
    return(df)

def add_diffs(df, cols, n):
    for col in df.columns:
        if col == 'date':
            continue
        for lag in range(1, n + 1):
            df[f"{col}_diff{lag}"] = df[col]/df[col].shift(lag)
    df = df.dropna()
    return(df)
