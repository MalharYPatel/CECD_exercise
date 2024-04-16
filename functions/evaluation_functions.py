import pandas as pd
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestRegressor #Random Forest Model
from xgboost import XGBRegressor#XGBoost model
from sklearn.linear_model import ARDRegression #ARD
from sklearn.model_selection import GridSearchCV # gridsearch wrapper for hyperparameters
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error #evaluation metric used to construct RMSE
import plotly.express as px


def evaluate_models(data, optimised_models_list):
    '''
    Loops through models in a list and returns evaluation metrics for each
    PARAMS
    data - the input data made previously. This is a list of tuples, with each tuple containing a description of the data, and the data itself
    optimised_models_list - the model list made previously. This is a list of tuples, with each tuple containing a name/description of the model, and the model itself
    '''
    model_name = []
    RMSE_list = []
    MAE_list = []
    R2_list =[]

    for data_list in data:
        data_tuple = data_list[1]
        X_train, X_test, y_train, y_test = data_tuple
        for model_list in optimised_models_list:
            model = model_list[1]
            name = model_list[0]
            #half the models are fit to one dataset, and half to the other
            #to avoid problems, we use exception handling to pass the incompatible cases
            try:
                y_pred = model.predict(X_test)
                RMSE = mean_squared_error(y_test, y_pred)**0.5
                MAE = mean_absolute_error(y_test, y_pred)
                R2 = r2_score(y_test, y_pred)
                
                model_name.append(name)
                RMSE_list.append(RMSE)
                R2_list.append(R2)
                MAE_list.append(MAE)
            except:
                continue
            
    summary_df = pd.DataFrame({'Model_details':model_name,
                                'RMSE': RMSE_list,
                                'MAE': MAE_list,
                                'R2': R2_list}) 
    return summary_df

def make_predictions_df(partial_df, full_df, optimised_models_list, horizon):
    '''
    Loops through models in a list and makes estimations from each. The function returns the input dataframe, but with the model predictions added
    PARAMS
    partial_df - the input data made previously. Partial df is needed as, although this only containsa partial subset of rows, it contains all the columns needed for the models 
    full_df - the models fit to the full_df use less columns in estimation. We pass this data so we know how to reduce the X/input data size for fitting these models
    optimised_models_list - the model list made previously. This is a list of tuples, with each tuple containing a name/description of the model, and the model itself
   horizon - The horizon over which the models were estimated. eg 1 quarter would be 1
    '''
    eval_df = partial_df.copy()
    eval_df['Next_Q_gdp_growth'] = eval_df['gdp_growth'].shift(-horizon)
    eval_df = eval_df.replace([np.inf, -np.inf], 0)

    X_vals = eval_df.drop(['date', 'Next_Q_gdp_growth'], axis=1)
    for model_list in optimised_models_list:
        model = model_list[1]
        try:
            eval_df[f"{model_list[0]}"] = model.predict(X_vals)
        except:
            try:
                x_narrow = X_vals.iloc[:, :(full_df.shape[1] - 1)]
                eval_df[f"{model_list[0]}"] = model.predict(x_narrow)
            except:
                print(f'{model_list[0]} failed')
    return eval_df



def plot_interactive_chart(predictions_df, start_date, end_date, palette):
    '''
    Loops through models in a list and makes estimations from each. The function returns the input dataframe, but with the model predictions added
    PARAMS
    predictions_df - data produced from the 'make_predictions_df' function, which has been reduced to only a date column, a column for the real target variable, and 
    columns for model estimated values
    start_date - the data we want our chart to begin
    end_date - the data we want our chart to end
    palette - for better data viz, we specify a color palette in the config eg a boe charts colour palette
    '''
    predictions_df = predictions_df[predictions_df['date'] >= start_date]
    predictions_df = predictions_df[predictions_df['date'] <= end_date]
    chart_df = predictions_df.melt(id_vars=['date'], var_name='Key', value_name='GDP Growth')
    fig = px.line(chart_df, x='date', y='GDP Growth', color='Key',
                  markers=True, color_discrete_sequence = palette,
                  labels={
                         "date": "Date",
                         "GDP Growth": "GDP Growth"
                     },
                    title="Real GDP Growth Rate vs Model Predictions")
    fig.update_layout(plot_bgcolor="#12273F", paper_bgcolor = "#12273F", font_color= 'white', font_family= 'arial')
    fig.update_xaxes(showgrid=False)
    fig.show()

