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
from optparse import OptionParser
import logging

begin_time = datetime.datetime.now()

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
        bootstrapped_avg = bootstrapped - mean
        bootstrapped_cusum = bootstrapped_avg.cumsum()
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
        # logging.info(depth)
        # call bootstrap function again on two segments split by day_change
        if len(time_series_df_left) > max_trend_length:
            output_df = bootstrap_data(time_series_df_left, col_nm_str, time_col_str, bootstrap_samples, output_df, max_trend_length, depth + 1)
        if len(time_series_df_right) > max_trend_length:
            output_df = bootstrap_data(time_series_df_right, col_nm_str, time_col_str, bootstrap_samples, output_df, max_trend_length, depth + 1)
        
    return output_df

def tune_parameters(param_grid, cutoffs, str_label):

    # generate combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = [] # store RMSEs for each param combo here
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
    tuning_results.to_csv('output/tuning_results_housing_'+str_label+'.csv', index = False)
    best_params = all_params[np.argmin(rmses)]
    min_rmse = tuning_results['rmse'].min()
    logging.info('BEST PARAMETERS:')
    logging.info(str(best_params))
    logging.info('RMSE:', str(min_rmse))
    return best_params, min_rmse

# https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html

if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option('', '--hyperparam', dest = 'hyperparam', help = 'whether to run hyperparameter tuning', default = True)
    parser.add_option('', '--changepoint_prior_scale', dest = 'changepoint_prior_scale', help = 'value for changepoint_prior_scale parameter', default = None)
    parser.add_option('', '--seasonality_prior_scale', dest = 'seasonality_prior_scale', help = 'value for seasonality_prior_scale parameter', default = None)
    parser.add_option('', '--seasonality_mode', dest = 'seasonality_mode', help = 'additive or multiplicative', default = None)

    (options, args) = parser.parse_args()

    logging.basicConfig(filename="logging_forecast.log", level=logging.INFO)

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

    # override trend changepoint identification with cusum-based method
    output_df = pd.DataFrame(columns = ['Day', 'Confidence Level'])
    output_df = bootstrap_data(austin_data, 'y', 'ds', bootstrap_samples, output_df, max_trend_length, 0)
    changepoints_list = output_df['Day'].sort_values().tolist()

    # hyperparameter tuning
    # changepoint_prior_scale: flexibility of trend and how much the trend changes at the changepoints - larger = potential to overfit
    # seasonality_prior_scale: flexibility of seasonality - smaller means there is a very small seasonal effect
    # seasonality_mode: whether seasonality is additive or multiplicative

    if options.hyperparam == True: # hyperparameter tuning

        changepoint_prior_scale_full_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
        seasonality_prior_scale_full_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

        changepoint_prior_scale_initial_list = [changepoint_prior_scale_full_list[i] for i in [0, 3, 6, 9]]
        seasonality_prior_scale_initial_list = [seasonality_prior_scale_full_list[i] for i in [0, 3, 6, 9]]

        # test range of parameters
        param_grid_initial = {
            'changepoints': [changepoints_list],
            'changepoint_prior_scale': changepoint_prior_scale_initial_list, # [0.001, 0.01, 0.1, 0.5]
            'seasonality_prior_scale': seasonality_prior_scale_initial_list, # [0.01, 0.1, 1.0, 10.0]
            'seasonality_mode': ['additive', 'multiplicative'],
        }
        cutoffs = pd.date_range(start='2002-01-01', end='2019-01-01', freq='12MS') # cutoffs for cv
        first_round_params, first_round_rmse = tune_parameters(param_grid_initial, cutoffs, 'first_round')

        i_c = changepoint_prior_scale_full_list.index(first_round_params['changepoint_prior_scale'])
        i_s = seasonality_prior_scale_full_list.index(first_round_params['seasonality_prior_scale'])

        indices_changepoint_final = list(set([i_c-2, i_c-1, i_c+1, i_c+2]) & set(range(0,10))) # find intersection
        indices_seasonality_final = list(set([i_s-2, i_s-1, i_s+1, i_s+2]) & set(range(0,10))) # find intersection

        changepoint_prior_scale_final_list = [changepoint_prior_scale_full_list[i] for i in indices_changepoint_final]
        seasonality_prior_scale_final_list = [seasonality_prior_scale_full_list[i] for i in indices_seasonality_final]
        
        # test narrower range of parameters
        param_grid_final = {
            'changepoints': [changepoints_list],
            'changepoint_prior_scale': changepoint_prior_scale_final_list,
            'seasonality_prior_scale': seasonality_prior_scale_final_list,
            'seasonality_mode': [first_round_params['seasonality_mode']]
        }
        second_round_params, second_round_rmse = tune_parameters(param_grid_final, cutoffs, 'final')

        if first_round_rmse < second_round_rmse:
            logging.info('First round params had lowest RMSE')
            m = Prophet(**first_round_params)
        else:
            logging.info('Second round params had lowest RMSE')
            m = Prophet(**second_round_params)

    else: # run with parameter values given in command line input

        m = Prophet(changepoints = changepoints_list, # override changepoint locations
                    changepoint_prior_scale = options.changepoint_prior_scale,
                    seasonality_prior_scale = options.seasonality_prior_scale,
                    seasonality_mode = options.seasonality_mode)
    
    m.add_seasonality(name = 'yearly', period = 365, fourier_order = 12,)
    m.fit(austin_data)

    # Forecast
    future = m.make_future_dataframe(periods = 24, freq = 'MS')
    forecast = m.predict(future)
    forecast.to_csv('output/forecast_housing.csv', index = False)

    # Cross Validation
    # this cuts the data off every year (1/2 the horizon) starting 6 years in (3 times the horizon) 
    # and trains the model each time with the cutoff data
    cutoffs = pd.date_range(start='2002-01-01', end='2019-01-01', freq='12MS')
    df_cv = cross_validation(m, horizon = '730 days', cutoffs = cutoffs)
    # get useful stats about the prediction performance -> MSE, RMSE, MAE, MAPE, etc.
    df_p = performance_metrics(df_cv)
    df_cv.to_csv('output/df_cv_housing.csv', index = False)
    df_p.to_csv('output/df_p_housing.csv', index = False)
    # visualize CV performance metrics - dots show the abs % error for each prediction in df_cv and the blue line shows the MAPE
    # where the mean is taken over a rolling window of the dots
    fig_cv = plot_cross_validation_metric(df_cv, metric = 'mape')
    fig_cv.savefig('figures/forecast_cv_mape_housing.png', index = False)

    # Plots
    pd.plotting.register_matplotlib_converters()
    fig1 = m.plot(forecast)
    fig1.savefig('figures/forecast_housing.png', index = False)
    a = add_changepoints_to_plot(fig1.gca(), m, forecast)
    fig1.savefig('figures/forecast_housing_w_changepoints.png', index = False)
    fig2 = m.plot_components(forecast)
    fig2.savefig('figures/forecast_components_housing.png', index = False)

    logging.info('Time to run program:')
    logging.info(str(datetime.datetime.now() - begin_time))