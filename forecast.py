from numpy.core.numeric import cross
import pandas as pd
import numpy as np
import itertools
from pandas.core.reshape.tile import cut
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot, plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics
import math
import datetime
import time
import urllib.request
from io import StringIO

# constants
bootstrap_samples = 100
max_trend_length = 6 # months
significance_level = 0.95

# functions
def bootstrap_data(time_series_df, col_nm_str, time_col_str, bootstrap_samples, output_df, max_trend_length, depth):

    mean = time_series_df[col_nm_str].mean()
    # compute cusum
    time_series_df[time_col_str] = pd.to_datetime(time_series_df[time_col_str])
    time_series_df['orig-avg'] = time_series_df[col_nm_str] - mean
    time_series_df['CUSUM'] = time_series_df['orig-avg'].cumsum()
    s_diff_orig = time_series_df['CUSUM'].max() - time_series_df['CUSUM'].min()
    # bootstrap
    bootstrap_results = [] # True if bootstrapped s_diff is less than original, else False
    bootstrapped = time_series_df[col_nm_str].dropna().to_numpy() # use this array to bootstrap in place
    for _ in range(bootstrap_samples):
        np.random.shuffle(bootstrapped)
        bootstrapped_sales_avg = bootstrapped - mean
        bootstrapped_cusum = bootstrapped_sales_avg.cumsum()
        s_diff_boot = bootstrapped_cusum.max() - bootstrapped_cusum.min()
        if s_diff_boot < s_diff_orig:
            bootstrap_results.append(True)
        else:
            bootstrap_results.append(False)
    # compute confidence level
    confidence = sum(bootstrap_results) / bootstrap_samples
    if confidence > significance_level: # there is a change in trend
        # estimate where the change occurred
        day_change = time_series_df[abs(time_series_df['CUSUM']) == abs(time_series_df['CUSUM']).max()][time_col_str]
        if len(day_change) == 1:
            day_change = day_change.iloc[0]
        else:
            day_change.reset_index(drop = True, inplace = True)
            ind = math.floor(len(day_change) / 2)
            day_change = day_change[ind].iloc[0]
        time_series_df.drop(columns = ['orig-avg', 'CUSUM'], inplace = True)
        time_series_df_left = time_series_df[time_series_df[time_col_str] < day_change]
        time_series_df_right = time_series_df[time_series_df[time_col_str] > day_change]
        # add this trend to output_df
        if (len(time_series_df_left) > max_trend_length) and (len(time_series_df_right) > max_trend_length):
            output_df.loc[len(output_df)] = [day_change, confidence]
        print(depth)
        # call bootstrap function again on two segments split by day_change
        if len(time_series_df_left) > max_trend_length:
            output_df = bootstrap_data(time_series_df_left, col_nm_str, time_col_str, bootstrap_samples, output_df, max_trend_length, depth + 1)
        if len(time_series_df_right) > max_trend_length:
            output_df = bootstrap_data(time_series_df_right, col_nm_str, time_col_str, bootstrap_samples, output_df, max_trend_length, depth + 1)
        
    return output_df

# https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html

# read zillow housing data
epoch_time = str(time.time())
url = "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv?t=" + epoch_time
with urllib.request.urlopen(url) as f:
    html = f.read().decode('utf-8')

housing_data = pd.read_csv(StringIO(html), sep=",")

# filter for Austin, TX data
austin_data = housing_data[housing_data['RegionName'] == 'Austin, TX']
austin_data = austin_data.iloc[: , 5:].reset_index(drop = True) # remove unneeded descriptive columns
austin_data = austin_data.T # transpose df 
austin_data = austin_data.reset_index().rename(columns = {'index': 'ds', 0: 'y'})
austin_data['ds'] = pd.to_datetime(austin_data['ds'])
print(austin_data.head())  

# override trend changepoint identification with cusum-based method
output_df = pd.DataFrame(columns = ['Day', 'Confidence Level'])
output_df = bootstrap_data(austin_data, 'y', 'ds', bootstrap_samples, output_df, max_trend_length, 0)
changepoints_list = output_df['Day'].sort_values().tolist()
print(changepoints_list)

# hyperparameter tuning
# changepoint_prior_scale: flexibility of trend and how much the trend changes at the changepoints - larger = potential to overfit
# seasonality_prior_scale: flexibility of seasonality - smaller means there is a very small seasonal effect
# seasonality_mode: whether seasonality is additive or multiplicative
param_grid = {
    'changepoints': [changepoints_list],
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'seasonality_mode': ['additive', 'multiplicative'],
}
# generate combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = [] # store RMSEs for each param combo here
cutoffs = pd.date_range(start='2002-01-01', end='2019-01-01', freq='12MS') # cutoffs for cv
# use cross validation to evaluate each combination of parameters
for params in all_params:
    m = Prophet(**params)
    m.add_seasonality(name = 'yearly', period = 365, fourier_order = 12,)
    m.fit(austin_data)
    df_cv = cross_validation(m, horizon = '730 days', cutoffs = cutoffs)
    df_p = performance_metrics(df_cv)
    rmses.append(df_p['rmse'].values[0])
# find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
tuning_results.drop(columns=['changepoints'], inplace = True)
tuning_results.to_csv('output/tuning_results.csv', index = False)
best_params = all_params[np.argmin(rmses)]
print('BEST PARAMETERS:', best_params)

# Generate Prophet model
# m = Prophet(changepoints = changepoints_list, # override changepoint locations
#             changepoint_prior_scale = 0.03,
#             seasonality_mode = 'multiplicative')
m = Prophet(**best_params)
m.add_seasonality(name = 'yearly', period = 365, fourier_order = 12,)
m.fit(austin_data)

# Forecast
future = m.make_future_dataframe(periods = 24, freq = 'MS')
forecast = m.predict(future)

# Cross Validation
# this cuts the data off every year (1/2 the horizon) starting 6 years in (3 times the horizon) 
# and trains the model each time with the cutoff data
cutoffs = pd.date_range(start='2002-01-01', end='2019-01-01', freq='12MS')
df_cv = cross_validation(m, horizon = '730 days', cutoffs = cutoffs)
# get useful stats about the prediction performance -> MSE, RMSE, MAE, MAPE, etc.
df_p = performance_metrics(df_cv)
df_cv.to_csv('output/df_cv.csv', index = False)
df_p.to_csv('output/df_p.csv', index = False)
# visualize CV performance metrics - dots show the abs % error for each prediction in df_cv and the blue line shows the MAPE
# where the mean is taken over a rolling window of the dots
fig_cv = plot_cross_validation_metric(df_cv, metric = 'mape')
fig_cv.savefig('figures/forecast_cv_mape.png', index = False)

# Plots
pd.plotting.register_matplotlib_converters()
fig1 = m.plot(forecast)
# a = add_changepoints_to_plot(fig1.gca(), m, forecast)
fig1.savefig('figures/forecast.png', index = False)
fig2 = m.plot_components(forecast)
fig2.savefig('figures/forecast_components.png', index = False)
