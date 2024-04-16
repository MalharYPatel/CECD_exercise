##############################################################################
'''

'''
#%%import modules
from configparser import ConfigParser
from datetime import datetime
import os
import sys


#%%add file path
file_path = __file__
file_path = file_path.split('config\make_config_file.py')[0]
import configparser


#%%Get the configparser object
config_object = ConfigParser()

#%% Data
config_object['data'] = {
    "data_path" : "data/Forecasting_dataset.xlsx", 
    "gdp_columns" : ['date', 'gdp_growth'],
    "monthly_columns" : ['date', 'pmi_current', 'pmi_expectations',
           'lloyds_business_activity', 'lloyds_optimism',
           'gfK_cc_past','gfK_cc_future'],
    "quarterly_columns" : ['date', 'bcc_past_3m', 'bcc_next_3m'],
    "partial_data_output_path" :"\data\clean_partial_data.csv",
    "full_data_output_path" :"\data\clean_full_data.csv"
    }

#%% Model
config_object['model_prep'] = {
    "scaler_path" : "models/scaler.joblib",
    "optimised_models_list_path": "models/optimised_models_list.joblib",
    "scaled_data_path": "data/scaled_data.joblib",
    "horizon": 1
    
    }


#%% Evaluation
config_object['evaluation'] = {
    "start_date" : "2002-09-30",
    "end_date": "2019-12-30",
    "boe_palette": ["#AC98DB", "#A31A7E","#EEAF30","#7AB800","#63B1E5","#d53647","#6773B6","#005E6E","#69923A","#A79E70","#CAC0B6","#57068C","#752864","#165788","#A51140","#E05206","#DF7A00"]
    
    }
#%%write config file
with open(os.path.join(file_path, 'config.ini'), 'w') as conf:
    config_object.write(conf)
