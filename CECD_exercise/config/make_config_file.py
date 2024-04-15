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

#%%Block2


#%%write config file
with open(os.path.join(file_path, 'config.ini'), 'w') as conf:
    config_object.write(conf)
