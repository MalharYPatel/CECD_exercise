from sklearn.ensemble import RandomForestRegressor #Random Forest Model
from xgboost import XGBRegressor#XGBoost model
from sklearn.linear_model import ARDRegression #ARD


'''
Due to the way the python config parser is set up, we need to add our model list into a seperate file, so that
it can be read as a python object - a list of tuples
The first item in each tuple is the model
The second is the hyperparameters dictionary
The third is the model name

'''
    
model_list = [
    (RandomForestRegressor(n_jobs = -1),
    [{  'n_estimators': [100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth' : [4,6,8]}],
    'Random Forest'),

    (ARDRegression(),
    [{  'max_iter' : [200,300]}],
    'ARDR'),

    (XGBRegressor(n_jobs = -1),
    [{'max_depth': [5, 10, 15],
             'n_estimators': [50,100,200], 
             'subsample': [0.5,0.7,0.9], 
             'eta': [0.2,0.3]}],
    'XGBoost')]