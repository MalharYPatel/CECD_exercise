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
#config
config = configparser.ConfigParser()
config.read('config.ini')

#%%
# 1. Load Data
partial_df = pd.read_csv(f"{wd}{config['data']['partial_data_output_path']}")
full_df = pd.read_csv(f"{wd}{config['data']['full_data_output_path']}")

#2. Scale_and_split both sets of data

partial_df_data = scale_and_split(df = partial_df, horizon = 1, test_frac = 0.25,
                                                   scaler_save_path = config['model_prep']['scaler_path'],
                                                   shuffle_split = False, seed = 7)

full_df_data = scale_and_split(df = full_df, horizon = 1, test_frac = 0.25,
                                                   scaler_save_path = config['model_prep']['scaler_path'],
                                                   shuffle_split = False, seed = 7)

data = [(full_df_data, 'full'), (partial_df_data, 'partial')]

'''
pca = PCA()
pca.fit(X_train)
evr = pd.Series(pca.explained_variance_ratio_)
evrsum = evr.cumsum()
evrsum.iloc[25:35]

#based on the above, we will use 33 components, which captures 95% variance
pca = PCA(n_components = 33)
pca.fit(X_train) #fit model

#dump(pca, 'pca.joblib') #save model

#apply transforms
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
'''

#Fit and evaluate models
model_name = []
data_used = []
RMSE_list = []
fitted_models = []

for data_set in data:
    X_train, X_test, y_train, y_test = data_set[0]
    for model in model_list:
        optimised_model =  fit_model(model = model[0], 
                   params = model[1],
                   X = X_train, y = y_train, cv = 5)
        y_pred = optimised_model.predict(X_test)
        RMSE = MSE(y_test, y_pred)**0.5
        
        model_name.append(model[2])
        data_used.append(data_set[1])
        RMSE_list.append(RMSE)
        fitted_models.append(optimised_model)
   
summary_df = pd.DataFrame({'Model':model_name,
                           'Data': data_used,
                           'RMSE': RMSE_list})        



#%% View models
#we can see from len(y_test) that the test was only on 21 data points. We can use these for our evaluation


eval_df = partial_df.copy()
eval_df['Next_Q_gdp_growth'] = eval_df['gdp_growth'].shift(-1)
eval_df = eval_df.replace([np.inf, -np.inf], 0)
eval_df = eval_df.dropna()
eval_df = eval_df.tail(21)
X = eval_df.drop(['date', 'Next_Q_gdp_growth', 'y'], axis=1)


for n in range(0, 3):
    x = X.iloc[:, :(full_df.shape[1] - 2)]
    name = summary_df.loc[n, 'Model']
    sample = summary_df.loc[n, 'Data']
    eval_df[f"prediction_{name}_{sample}"] = fitted_models[n].predict(x)
    
for n in range(3, 6):
    name = summary_df.loc[n, 'Model']
    sample = summary_df.loc[n, 'Data']
    eval_df[f"prediction_{name}_{sample}"] = fitted_models[n].predict(X)

chart_df = eval_df.iloc[:, :(summary_df.shape[0]+1)]


def chart_results():
    return

#gs_rf.best_params_
#gs_rf.score(X_train, y_train)




