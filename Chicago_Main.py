"""
Chicago Crime Analysis and Prediction

Copyright (c) 2019
Licensed
Written by Shakya Munghate
Date: 10/26/2019
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split #train_test_split
from sklearn import metrics #for accuracy calculation

data_url = 'https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD'
def read_file(csv):
    file = pd.read_csv(csv)
    return file

def type_conv(data):
    date_col = ['Date', 'Updated On']
    int_col = ['District', 'Ward', 'Community Area', 'X Coordinate', 'Y Coordinate', 
               'Historical Wards 2003-2015', 'Zip Codes', 'Community Areas', 'Census Tracts', 'Wards', 
               'Boundaries - ZIP Codes', 'Police Districts', 'Police Beats']
    #string to date
    data.loc[:, date_col] = data.loc[:, date_col].apply(pd.to_datetime) 
    #float to int
    data.loc[:, int_col] = data.loc[:, int_col].astype('int64')
    
    return print("data types of the columns have been converted")

def data_preprocess(data):
    #Dropping unnecessary columns
    data = data.drop(columns = ['X Coordinate', 'Y Coordinate',  
                                  'Police Beats', 'Case Number', 'Census Tracts','Case Number', 
                                  'Census Tracts', 'Historical Wards 2003-2015'])
    
    null_numeric = ['District', 'Ward', 'Community Area', 'Zip Codes', 'Community Areas', 
                'Wards', 'Boundaries - ZIP Codes', 'Police Districts', 'Latitude', 'Longitude']
    null_string = ['Location Description', 'Location']
    
    data.loc[:, null_numeric] = data.loc[:, null_numeric].fillna(value = 0)
    data.loc[:, null_string] = data.loc[:, null_string].fillna(value = 'unspecified')
    
    #Getting hourly, monthly and yearly dimensions
    data['hour'] = data['Date'].apply(lambda a: a.hour).astype('int64')
    data['month'] = data['Date'].apply(lambda a: a.month).astype('int64')
    data['year'] = data['Date'].apply(lambda a: a.year).astype('int64')
    data['day of the week'] = data['Date'].apply(lambda a: a.weekday())
    
    return data

#Get Season
def get_season(tran_month):
    if tran_month==12:
        season = 'Winter'
    elif tran_month==1:
        season = 'Winter'
    elif tran_month==2:
        season = 'Winter'
    elif tran_month==3:
        season = 'Spring'
    elif tran_month==4:
        season = 'Spring'
    elif tran_month==5:
        season = 'Spring'
    elif tran_month==6:
        season = 'Summer'
    elif tran_month==7:
        season = 'Summer'
    elif tran_month==8:
        season = 'Summer'
    else:
        season = 'Fall'
    return season

def get_weekend(day):
    if day == 'Saturday':
        weekend=1
    elif day == 'Sunday':
        weekend=1
    else:
        weekend=0
    return weekend

#Year over year crimes
def yoy(data):
    plt.figure(figsize = (12, 6))
    data.resample('Y').size().plot(legend=False)
    plt.title('Number of crimes per year 2001-2019')
    plt.xlabel('Year')
    plt.ylabel('Number of crimes')
    plt.show()

#Month over month crimes
def mom(data):
    plt.figure(figsize = (12, 6))
    data.resample('M').size().plot(legend=False)
    plt.title('Number of crimes per month 2001-2019')
    plt.xlabel('Month')
    plt.ylabel('Number of crimes')
    plt.show()

def top3(data): 
   #Top Crimes every year
    crime_cnt = data.groupby(['year','Primary Type'])['Primary Type'].count().to_frame('count')
    cnt = crime_cnt.reset_index()
    g = crime_cnt['count'].groupby(level=0, group_keys=False)
    top_crimes = g.nlargest(3).reset_index()
    top5_crimes = g.nlargest(5).reset_index()
    
    names17 = list(top_crimes[top_crimes['year'] == 2017]['Primary Type'])
    values17 = list(top_crimes[top_crimes['year'] == 2017]['count']*100/sum(cnt[cnt['year'] == 2017]['count']))
    
    names18 = list(top_crimes[top_crimes['year'] == 2018]['Primary Type'])
    values18 = list(top_crimes[top_crimes['year'] == 2018]['count']*100/sum(cnt[cnt['year'] == 2018]['count']))
    
    names19 = list(top_crimes[top_crimes['year'] == 2019]['Primary Type'])
    values19 = list(top_crimes[top_crimes['year'] == 2019]['count']*100/sum(cnt[cnt['year'] == 2019]['count']))
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
    axs[0].bar(names17, values17)
    axs[1].bar(names18, values18)
    axs[2].bar(names19, values19)
    axs[0].title.set_text('2017')
    axs[0].set_ylabel('Percentage (%)')
    axs[1].title.set_text('2018')
    axs[2].title.set_text('2019')
    fig.suptitle('Top 3 types of crimes in Chicago 2017-2019')
    
def MoM_Curr(data):
    plt.figure(figsize = (12, 6))
    data[data['year'].isin([2017,2018,2019])].resample('M').size().plot(legend=False)
    plt.title('Number of crimes per month 2017-2019')
    plt.xlabel('Month')
    plt.ylabel('Number of crimes')
    plt.show()
    
def mom_prim(data):
    crime_cnt = data.groupby(['year','Primary Type'])['Primary Type'].count().to_frame('count')
    cnt = crime_cnt.reset_index()
    g = crime_cnt['count'].groupby(level=0, group_keys=False)
    top5_crimes = g.nlargest(5).reset_index()
    top5 = data[data['Primary Type'].isin(top5_crimes['Primary Type'].unique())]
    fig, ax = plt.subplots(figsize=(15,7))
    top5.groupby(['year','Primary Type']).count()['ID'].unstack().plot(ax=ax)
    
def roll_yoy(data):
    #Rolling sum of crimes by primary type year over year
    cnt_dt = data.pivot_table('ID', aggfunc=np.size, columns='Primary Type', index = data.index.date, fill_value = 0)
    cnt_dt.index = pd.DatetimeIndex(cnt_dt.index)
    cnt_dt.rolling(365).sum().plot(figsize=(15, 30), subplots=True, layout=(-1, 3), sharex=False, sharey=False)
    return ("Rolling Sum YoY")

def roll_prim(data):
    #Rolling sum of crimes by primary type year over year
    cnt_dt = data.pivot_table('ID', aggfunc=np.size, columns='Primary Type', index = data.index.date, fill_value = 0)
    cnt_dt.index = pd.DatetimeIndex(cnt_dt.index)
    cnt_dt.rolling(365).sum().plot(figsize=(15, 30), subplots=True, layout=(-1, 3), sharex=False, sharey=False)
    return ("Rolling Sum YoY by Type")
 
def arrest(data):
    #Subsetting arrest data 
    Arrest_data = data[data['Arrest'] == True]

    #Rolling sum of arrests
    plt.figure(figsize=(11,4))
    Arrest_data.resample('D').size().rolling(365).sum().plot()
    plt.title('Rolling sum of all arrests from 2001 - 2019')
    plt.ylabel('Number of arrests')
    plt.xlabel('Days')
    plt.show()
    return ("Rolling sum of Arrest")

def top_comm(data):
    #Crimes by community area
    cmt_cnt = data_current.groupby(['year','Community Areas'])['ID'].count().to_frame('count')
    d = cmt_cnt['count'].groupby(level=0, group_keys=False)
    top_unsafe_comm = d.nlargest(10).reset_index()
    comm_cnt = top_unsafe_comm.pivot_table('count', columns='year', index='Community Areas', aggfunc=np.sum).sort_values(by = 2019, ascending=False).reset_index()
    return comm_cnt

def comm(data):
    #Investigating COmmunity 26
    df26 = data[data['Community Area'] == 26].copy()
    # plot data
    cntdf26 = df26.pivot_table('ID', aggfunc=np.size, columns='Primary Type', index=pd.DatetimeIndex(df26.index.date), fill_value=0)
    cntdf26.rolling(365).sum().plot(figsize=(15, 30), subplots=True, layout=(-1, 3), sharex=False, sharey=False)
    
def beat(data):
    #Crimes by beat
    recent_yrs = [2016,2017,2018,2019]
    bt_cnt = data[data['year']==2019].groupby(['year','Beat'])['ID'].count().to_frame('count')
    bt = bt_cnt['count'].groupby(level=0, group_keys=False)
    top_unsafe_bt = bt.nlargest(10).reset_index()
    beat_cnt = top_unsafe_bt.pivot_table('count', columns='year', index='Beat', aggfunc=np.sum).sort_values(by = 2019, ascending=False).reset_index()

    bt_19 = top_unsafe_bt['Beat']
    bt_cnt_all = data[(data['year'].isin(recent_yrs)) & (data['Beat'].isin(bt_19))].groupby(['year','Beat'])['ID'].count().to_frame('count').reset_index()

    xpos=np.arange(len(bt_19))
    plt.figure(figsize=(12,5))
    plt.xticks(xpos, bt_19)
    plt.bar(x=xpos-0.4,width=0.2, height=bt_cnt_all[bt_cnt_all['year']==2016]['count'], label='2016')
    plt.bar(x=xpos-0.2,width=0.2, height=bt_cnt_all[bt_cnt_all['year']==2017]['count'], label='2017')
    plt.bar(x=xpos,width=0.2, height=bt_cnt_all[bt_cnt_all['year']==2018]['count'], label='2018')
    plt.bar(x=xpos+0.2,width=0.2, height=bt_cnt_all[bt_cnt_all['year']==2019]['count'], label='2019')
    plt.legend()
    plt.title("Number of Crime by Beats 2016-2019")
    plt.xlabel("Beat")
    plt.ylabel("No. of crimes")
    
recent_yrs = [2017, 2018, 2019]
def rec_crimes(data):
    #No. of crimes by time and year
    cnt_hr = data[data['year'].isin(recent_yrs)].pivot_table('ID', aggfunc=np.size, columns='year', index = data[data['year'].isin(recent_yrs)].index.hour, fill_value = 0)
    return cnt_hr.plot(figsize=(16, 5), subplots=True, layout=(-1, 4), sharex=False, sharey=True)


def hypothesis(data):
    #Hypothesis Testing
    #1. Least number of crimes occur during winters
    burgThft = data[data['year']==2019]
    rand_winter = burgThft[burgThft['season']=='Winter']
    rand_summer = burgThft[burgThft['season']=='Summer']
    rand_fall = burgThft[burgThft['season']=='Fall']
    rand_spring = burgThft[burgThft['season']=='Spring']
    winter_cnt = rand_winter.groupby(['hour'])['ID'].count().to_frame('count').reset_index()
    summer_cnt = rand_summer.groupby(['hour'])['ID'].count().to_frame('count').reset_index()
    fall_cnt = rand_fall.groupby(['hour'])['ID'].count().to_frame('count').reset_index()
    spring_cnt = rand_spring.groupby(['hour'])['ID'].count().to_frame('count').reset_index()

    plt.figure(figsize=(12,5))
    plt.plot(winter_cnt['count'], label='Winter')
    plt.plot(summer_cnt['count'], label='Summer')
    plt.plot(fall_cnt['count'], label='Fall')
    plt.plot(spring_cnt['count'], label='Spring')
    plt.legend()
    plt.title('Hourly trend of crimes - Winter vs Rest 2019')
    plt.xlabel('hours')
    plt.ylabel('No. of crimes')
    
def modeling(data):
    df_ml = data.copy()
    #Factorizing the string data
    df_ml['Primary Type'] = pd.factorize(df_ml["Primary Type"])[0]
    df_ml['Block'] = pd.factorize(df_ml["Block"])[0]
    df_ml['IUCR'] = pd.factorize(df_ml["IUCR"])[0]
    df_ml['Location Description'] = pd.factorize(df_ml["Location Description"])[0]
    df_ml['FBI Code'] = pd.factorize(df_ml["FBI Code"])[0]
    df_ml['Location'] = pd.factorize(df_ml["Location"])[0]
    return df_ml

def corr(data):
    #Using Pearson Correlation
    plt.figure(figsize=(20,10))
    cor = data.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()
    #Identifying important variables
    cor_y = abs(cor['Primary Type'])
    imp_var = cor_y[(cor_y>0.2) & (cor_y<=0.7)]
    print("Only following are the variables (with correlation coefficient) important for modeling\n{}".format(imp_var))


if __name__ == "__main__" :
    #Decision Tree Regression
    X = df_ml[['IUCR','FBI Code']]
    y = df_ml[['Primary Type']]


    #Test train Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

    # Create Decision Tree classifer object
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)

    #Predict
    y_pred = pd.DataFrame(regressor.predict(X_test)).astype('int64')


    # Model Evaluation

    accuracy = metrics.accuracy_score(y_test , y_pred)

    confusion_m = metrics.confusion_matrix(y_test , y_pred)









