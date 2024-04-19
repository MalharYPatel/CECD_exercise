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

# Potential to do further work :
#a) with stationarity using this:
# from statsmodels.tsa.stattools import adfuller

# b) Using automated fetaure creation an selecion with tsfresh

    # from tsfresh import extract_features
    # extracted_features = extract_features(df, column_id="date", column_sort="date"
                                          
    # from tsfresh import select_features
    # from tsfresh.utilities.dataframe_functions import impute
    # impute(extracted_features)
    # features_filtered = select_features(extracted_features, y) # here y would have to be out shifted() gp growth variable
    
# c) or with technical indicators including Moving averages, using the ta library
# In theory, the use of MAs and lagged variables, created a fully specfied VARIMA(X) model

# d) Feature reduction AFTER feaure egineering. THis could be interpretable eg RFE, 
# stepwise feature selection with AIC ec, or fature rojecton with eg PCA. Here is some example code
# pca = PCA()
# pca.fit(X_train)
# evr = pd.Series(pca.explained_variance_ratio_)
# evrsum = evr.cumsum()
# evrsum.iloc[25:35]

# #based on the above, we can chose the number of components, which captures 95% variance
# pca = PCA(n_components = 33) #g 33 components in this example
# pca.fit(X_train) #fit model

# #dump(pca, 'pca.joblib') #save model

# #apply transforms
# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)





