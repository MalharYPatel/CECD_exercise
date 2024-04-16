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
from functions.evaluation_functions import *
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error #evaluation metric used to construct RMSE
from joblib import dump, load
import time
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

#config
config = configparser.ConfigParser()
config.read('config.ini')


# 1. Load Data
t0 = time.time()
partial_df = pd.read_csv(f"{wd}{config['data']['partial_data_output_path']}")
full_df = pd.read_csv(f"{wd}{config['data']['full_data_output_path']}")
optimised_models_list = load(f"{wd}/{config['model_prep']['optimised_models_list_path']}")
data = load(f"{wd}/{config['model_prep']['scaled_data_path']}")
t1 = time.time()
print("Data loading took", (t1 - t0), "seconds")

#2. Produce summary stats     
t0 = time.time() 
summary_df = evaluate_models(data, optimised_models_list)
summary_df.sort_values(by = 'RMSE')
t1 = time.time()
print("Summary evaluation measures took", (t1 - t0), "seconds")

#3. View model predictions

t0 = time.time() 
eval_df = make_predictions_df(partial_df, full_df, optimised_models_list, horizon = ast.literal_eval(config['model_prep']['horizon']))
eval_df['Naive AR model'] = eval_df['gdp_growth'].shift(ast.literal_eval(config['model_prep']['horizon']) -1) # A Naive lag model
predictions_df = eval_df.iloc[:, -(len(optimised_models_list) +2):]
predictions_df['date'] = eval_df['date']
t1 = time.time()
print("Making Estimations took", (t1 - t0), "seconds")

#4. Plot Time Chart

start_date = config['evaluation']['start_date']
end_date = config['evaluation']['end_date']
palette = ast.literal_eval(config['evaluation']['boe_palette'])

plot_interactive_chart(predictions_df, start_date, end_date, palette)

plot_interactive_chart(predictions_df, "2000", "2024", palette)
