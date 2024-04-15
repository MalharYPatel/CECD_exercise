from statsmodels.tsa.stattools import adfuller
import configparser
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler #for scaling
from sklearn.model_selection import train_test_split #for seperating train and test samples
from joblib import dump
from sklearn.ensemble import RandomForestRegressor #Random Forest Model
from xgboost import XGBRegressor#XGBoost model
from sklearn.linear_model import ARDRegression #ARD
from sklearn.model_selection import GridSearchCV # gridsearch wrapper for hyperparameters

#Get config variables
config_path = __file__
upper_path = config_path.split('functions\model_functions.py')[0]
config_path = upper_path + 'config.ini'
config = configparser.ConfigParser()
config.read(config_path)


def scale_and_split(df, horizon, test_frac, scaler_save_path, shuffle_split, seed):
    #create appropriate taret variable, based o user specified horizon
    df['y'] = df['gdp_growth'].shift(-horizon)
    df = df.replace([np.inf, -np.inf], 0)
    df = df.dropna()
    #split into feature and target arrays
    X = df.drop(['date', 'y'], axis=1)
    y = df['y']
    
    X_cols = X.columns # save column names for later
    
    #Split into training and validation samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=seed, shuffle = shuffle_split)


    # initiate scaler and fit to training data
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)

    #save scaler model
    dump(scaler, f'{upper_path}{scaler_save_path}')

    #apply transforms
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def fit_model(model, params, X, y, cv):
    gs_model = GridSearchCV(model,
                          param_grid=params,
                          cv=cv)
    gs_model.fit(X, y)
    #gs_xg.best_params_
    #gs_xg.score(X_train, y_train)
    return gs_model