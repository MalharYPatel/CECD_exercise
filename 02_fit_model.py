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
from sklearn.preprocessing import StandardScaler #for scaling
from sklearn.model_selection import train_test_split #for seperating train and test samples
from functions.model_functions import *
from model_config import *
from sklearn.ensemble import RandomForestRegressor #Random Forest Model
from xgboost import XGBRegressor#XGBoost model
from sklearn.linear_model import ARDRegression #ARD
from sklearn.model_selection import GridSearchCV # gridsearch wrapper for hyperparameters
from sklearn.metrics import mean_squared_error as MSE #evaluation metric used to construct RMSE
from joblib import dump
import time
#config
config = configparser.ConfigParser()
config.read('config.ini')


# 1. Load Data
t0 = time.time()
partial_df = pd.read_csv(f"{wd}{config['data']['partial_data_output_path']}")
full_df = pd.read_csv(f"{wd}{config['data']['full_data_output_path']}")
t1 = time.time()
print("Data loading took", (t1 - t0), "seconds")

#2. Scale_and_split both sets of data
t0 = time.time()
partial_df_data = scale_and_split(df = partial_df, horizon = ast.literal_eval(config['model_prep']['horizon']), test_frac = 0.25,
                                                   scaler_save_path = f"{wd}/{config['model_prep']['scaler_path']}",
                                                   shuffle_split = False, seed = 7)

full_df_data = scale_and_split(df = full_df, horizon = 1, test_frac = 0.25,
                                                   scaler_save_path = f"{wd}/{config['model_prep']['scaler_path']}",
                                                   shuffle_split = False, seed = 7)

data = [['full', full_df_data], ['partial', partial_df_data]]
t1 = time.time()
print("Data scaling and splitting took", (t1 - t0), "seconds")

#Fit models
t0 = time.time()
optimised_models_list= make_optimised_model_list(data_list = data, model_list = model_list, cv = 5)
t1 = time.time()
print("Optimising fit for all models took", (t1 - t0)/60, "minutes")

#Save list of optimised models
t0 = time.time()
dump(optimised_models_list, f"{wd}/{config['model_prep']['optimised_models_list_path']}")
t1 = time.time()
print("Saving all models took", (t1 - t0), "seconds")

#Save processed data
t0 = time.time()
dump(data, f"{wd}/{config['model_prep']['scaled_data_path']}")
t1 = time.time()
print("Saving all data", (t1 - t0), "seconds")