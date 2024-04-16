import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler #for scaling
from sklearn.model_selection import train_test_split #for seperating train and test samples
from joblib import dump
from sklearn.ensemble import RandomForestRegressor #Random Forest Model
from xgboost import XGBRegressor#XGBoost model
from sklearn.linear_model import ARDRegression #ARD
from sklearn.model_selection import GridSearchCV # gridsearch wrapper for hyperparameters


def scale_and_split(df, horizon, test_frac, scaler_save_path, shuffle_split, seed):
    '''
    Function which scales the input data, splits it into training and testing samples, creates a y/target variable which is the future estimation horizon,
    and splits the data into this target volumn, and a dataframe of variables for estimating the model. It returns these 4 split parts of the data
    PARAMS
    df - the input data is the final merged and feature engineered table
    horizon - for making our target variable, this is how many quarters ahead we want to estimate
    test_frac - the data is split into trainign and testing samples. This si the fraction held out for testing. A decimal value between 0 and 1
    scaler_save_path - once the data is scaled, this si the path the scaler model is saved to for future use.
    shuffle_split - if set to True, it will shuffle the data before splitting it. As this is a time series, it is best left as False
    seed - the random seed for the scaler - keeping the same value ensures reproduceability
    '''
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
    dump(scaler, scaler_save_path)

    #apply transforms
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def fit_model(model, params, X, y, cv):
    '''
    A sub-function which uses a grid search to return an optimised, fit model
    PARAMS
    model - the model to be fit
    params - a dictionary of hyperparameters to optimise the model with
    X - The data being used to train the model
    y - data which makes up the target variable the model is estimating
    cv - the optimiser uses cross validation for robustness, The number of folds can be specified here. 5 or 10 are common values
    '''
    gs_model = GridSearchCV(model,
                          param_grid=params,
                          cv=cv)
    gs_model.fit(X, y)
    #gs_xg.best_params_
    #gs_xg.score(X_train, y_train)
    return gs_model


def make_optimised_model_list(data_list, model_list, cv):
    '''
    A Function which loops through model options and uses the 'fit model' subfunction to fit each model. 
    It returns a list of tuples, where the first value is a name for the model, and the second is the model itself
    PARAMS
    data_list - the input data made previously. This is a list of tuples, with each tuple containing a description of the data, and the data itself
    model_list - a list, found in the model_config, which contains tuples of the model itself, and a dictionary of hyperparameters to optimise that model with
    cv - cv value to pass to the 'fit_model' subfunction
    '''
    optimised_models_list = []
    
    for data_set in data_list:
        X_train, X_test, y_train, y_test = data_set[1]
        for model in model_list:
            optimised_model =  fit_model(model = model[0], 
                       params = model[1],
                       X = X_train, y = y_train, cv = cv)
            
            model_name = f"{model[2]}_{data_set[0]}"
            result = (model_name, optimised_model)
            optimised_models_list.append(result)
    return optimised_models_list
    
