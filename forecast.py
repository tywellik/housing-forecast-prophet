import pandas as pd
import numpy as np
from fbprophet import Prophet
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import math
import datetime

# read zillow housing data
housing_data = pd.read_csv('input_data/zillow_monthly_metro_housing_prices.csv')

# filter for Austin, TX data
austin_data = housing_data[housing_data['RegionName'] == 'Austin, TX']
austin_data = austin_data.iloc[: , 5:].reset_index(drop = True) # remove unneeded descriptive columns
austin_data = austin_data.T # transpose df 
austin_data = austin_data.reset_index().rename(columns = {'index': 'ds', 0: 'y'})
austin_data['ds'] = pd.to_datetime(austin_data['ds'])

# TODO: determine holidays? 2008? Covid?
# need df with columns holiday, Month, lower_window, upper_window (not sure what these last two mean)

# Generate Prophet model
# m = Prophet(holidays = holiday_df, seasonality_mode = 'multiplicative')
m = Prophet(changepoint_range = 1.0,  # use all of the data 
            changepoint_prior_scale = 0.05, # decrease flexibility of the model to reduce overfitting (default is 0.05)
            seasonality_mode = 'multiplicative')
m.add_seasonality(name = 'yearly', period = 365, fourier_order = 12,)
m.fit(austin_data)

# Forecast
future = m.make_future_dataframe(periods = 24, freq = 'M')
forecast = m.predict(future)

# Plots
pd.plotting.register_matplotlib_converters()
fig1 = m.plot(forecast)
a = add_changepoints_to_plot(fig1.gca(), m, forecast)
fig1.savefig('figures/forecast.png', index = False)
fig2 = m.plot_components(forecast)
fig2.savefig('figures/forecast_components.png', index = False)
