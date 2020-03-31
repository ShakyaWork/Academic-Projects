# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 19:41:24 2019

@author: Shakya Work
"""
#Importing Initial libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime # Date Manipulation
import seaborn # Vizualization

#Turning of the warnings
import warnings
warnings.filterwarnings("ignore")

#Time Series
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot

# Importing files
dateparse = lambda dates: pd.datetime.strptime(dates, '%d.%m.%Y')

sales_train = pd.read_csv('sales_train.csv', parse_dates=['date'], 
                          index_col='date',date_parser=dateparse)
sales_test = pd.read_csv('test.csv')
items = pd.read_csv('items.csv')
items_categories = pd.read_csv('item_categories.csv')
shops = pd.read_csv('shops.csv')


# Formatting Dates
#print(sales_train.info()) #Notice date datatype is object. Convert into datetime
#sales_train.date = sales_train.date.apply(lambda a:datetime.datetime.strptime(a, '%d.%m.%Y'))

print(sales_train.info()) #Notice, date is converted to datetime


#We need forecast at month level, thus, aggregate sales at month level
#monthly_train = sales_train.groupby(['date_block_num','shop_id','item_id'])['date','item_price','item_cnt_day'].agg({'date':['min','max'],'item_price':'mean','item_cnt_day':'sum'})
#monthly_train.head(10)

#Time Series Analysis
#monthly_item_cnt = sales_train.groupby(["date_block_num"])["item_cnt_day"].sum()
monthly_item_cnt = sales_train.resample(rule='M')['item_cnt_day'].sum()
monthly_item_cnt.astype('float')
plt.figure(figsize=(12,8))
plt.title("Total Item Sales of the Company")
plt.ylabel("Count of Items")
plt.xlabel("Month")
plt.plot(monthly_item_cnt)

# Decomposition
import statsmodels.api as sm

#Additive Model
decom_add = sm.tsa.seasonal_decompose(monthly_item_cnt, model = "additive", freq=12)
fig = decom_add.plot()

#Multiplicative Model
decom = sm.tsa.seasonal_decompose(monthly_item_cnt, model = "multiplicative", freq=12)
fig = decom.plot()


#Checking Stationarity using Dickey-Fuller test
def dickey_fuller(ts):
    
    #Determing rolling statistics
    rolmean = ts.rolling(12).mean()
    rolstd = ts.rolling(12).std()

    #Plot rolling statistics:
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    df_test = adfuller(ts, autolag = "AIC")
    df_result = pd.Series(df_test[0:4], index=['Test Statistic','p-value','Lags',
                       'Number of Observations Used'])
    for key, value in df_test[4].items():
        df_result['Critical Value (%s)'%key] = value
    print(df_result)

dickey_fuller(monthly_item_cnt)

#Estimating and Eliminating Trend
ts_log_item = np.log(monthly_item_cnt)
plt.plot(ts_log_item)

#Moving Average
moving_avg = ts_log_item.rolling(12).mean()
plt.plot(ts_log_item)
plt.plot(moving_avg, color='red')

ts_log_moving_avg_diff = ts_log_item - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)
dickey_fuller(ts_log_moving_avg_diff)

#Differencing
ts_log_diff = ts_log_item - ts_log_item.shift()
plt.plot(ts_log_diff)

ts_log_diff.dropna(inplace=True)
dickey_fuller(ts_log_diff)

#Non Log Variable De-Trending
ts_diff_nl = monthly_item_cnt - monthly_item_cnt.shift()

#Decomposing or Removing Seasonality

from pandas import Series
def differ(data, interval=1):
    diff = list()
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return Series(diff)

ts_log_decom = differ(ts_log_diff,12) 

#Non Log Variable De-seasonalizing
ts_decom_nl = differ(ts_diff_nl, 12)
#####################################################################################################
#Differencing
#Elimination Trend
"""from pandas import Series
def differ(data, interval=1):
    diff = list()
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced forecast
def inverse_differ(last_obs, value):
    return value + last_obs

plt.subplot(311)
plt.title('After Removing Trend')
plt.xlabel('Months')
plt.ylabel('Total Item Sales of the Company')
new_monthly_item_cnt=differ(monthly_item_cnt)
plt.plot(new_monthly_item_cnt)
plt.plot()

plt.subplot(313)
plt.title('After Removing Seasonality')
plt.xlabel('Months')
plt.ylabel('Total Item Sales of the Company')
new_monthly_item_cnt=differ(monthly_item_cnt,12)       # assuming the seasonality is 12 months long
plt.plot(new_monthly_item_cnt)
plt.plot()

#Testing Stationarity
dickey_fuller(new_monthly_item_cnt)"""

#Checking ACF and PACF
lag_acf = acf(ts_log_item, nlags=20)
lag_pacf = pacf(ts_log_item, nlags=20, method='ols')
upper_CI = 1.96/np.sqrt(len(ts_log_item))
lower_CI = -1.96/np.sqrt(len(ts_log_item))

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=lower_CI,linestyle='--',color='gray')
plt.axhline(y=upper_CI,linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.xlabel('Lags')
plt.ylabel('ACF')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=lower_CI,linestyle='--',color='gray')
plt.axhline(y=upper_CI,linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.xlabel('Lags')
plt.ylabel('PACF')

#ACF and PACF on Detrended and Deseasonal data
#Checking ACF and PACF
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
upper_CI = 1.96/np.sqrt(len(ts_log_diff))
lower_CI = -1.96/np.sqrt(len(ts_log_diff))

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=lower_CI,linestyle='--',color='gray')
plt.axhline(y=upper_CI,linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.xlabel('Lags')
plt.ylabel('ACF')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=lower_CI,linestyle='--',color='gray')
plt.axhline(y=upper_CI,linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.xlabel('Lags')
plt.ylabel('PACF')


#Implementing ARIMA Model
#AR Model
model = ARIMA(ts_log_item, order=(1,1,0))
results_AR = model.fit(disp=-1)
plt.plot(ts_log_item)
plt.plot(results_AR.fittedvalues, color = 'red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))


#MA Model
model = ARIMA(ts_log_item, order=(0,1,1))
results_MA = model.fit(disp=-1)
plt.plot(ts_log_item)
plt.plot(results_MA.fittedvalues, color = 'red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))

#ARIMA Model
model = ARIMA(ts_log_item, order=(1,1,1))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_item)
plt.plot(results_ARIMA.fittedvalues, color = 'red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))

#Predicted values
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

#Cumulative Sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_log = pd.Series(ts_log_item.ix[0], index = ts_log_item.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)


predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(monthly_item_cnt)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-monthly_item_cnt)**2)/len(monthly_item_cnt)))

#############################################################################################
