# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:13:41 2020

@author: anama
"""

from preprocessing.series import select_data, downsample
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from preprocessing.alignment import missing_values
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def get_dataset(min_date=None, max_date=None):
    df = pd.read_csv("F:\\manual\\Tese\\exploratory\\wisdom\\dataset\\infraquinta\\temperaturas\\temperatures_tg.csv", header=0)
    df['datahora'] = pd.to_datetime(df['datahora'])
    df = df.rename(columns={"datahora": "date", "valor": "value"})
    df = df.drop(['nome', 'unidade'], axis=1)
    df.set_index(['date'], drop=True, inplace=True)
    df = select_data(df, 'date', min_date, max_date)
    minutes = '15min'
    df = downsample(df, minutes)
   
    df['value'] = df['value'].astype(np.float)
    print(df['value'].isnull().values.any())
    df = missing_values(df)
    print(df['value'].isnull().values.any())
    print(np.isfinite(df['value']).all())
    return df

def means_per_day(df):
    df_means = df.resample('d').mean()
    return df_means
    
def change_temperature(temp):
    diff = list()
    for i in range(1, len(temp)):
        value = temp[i] - temp[i - 1]
        diff.append(value)
    return diff

def temperature_correction_factor(error_vector, dates, time):
    df_error = pd.DataFrame(error_vector, index=dates, columns=['error'])
    print("shape", df_error.shape)
    df_temp = get_dataset(min_date=min(dates), max_date=max(dates))
    values = change_temperature(df_temp.values)
    new_df_temp = pd.DataFrame(values, index=df_temp.index[1:], columns=df_temp.columns)
    df_join = new_df_temp.join(df_error)
    print("df_join columns", df_join.columns)
    print("shape df join", df_join.shape)
    fig1, ax1 = plt.subplots()
    ax1.scatter(df_join['value'], df_join['error'], marker='o')
    plt.show()
    
    df_means = means_per_day(df_temp)
    fig2, ax2 = plt.subplots()
    ax2.plot(df_means, marker='o')
    plt.show()
    
    print(df_join['error'].isnull().values.any())
    print(np.isfinite(df_join['value']).all())
    
    df_join = missing_values(df_join)
    split_index = round(df_join.shape[0] * 0.8)
    X_train = df_join['value'].iloc[:split_index].values.reshape(-1, 1)
    y_train = df_join['error'].iloc[:split_index].values.reshape(-1, 1)
    X_test = df_join['value'].iloc[split_index:].values.reshape(-1, 1)
    y_test = df_join['error'].iloc[split_index:].values.reshape(-1, 1)
    
    print(X_train[:5])
    reg = LinearRegression().fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    
    fig3, ax3 = plt.subplots()
    # Plot outputs
    ax3.scatter(X_test, y_test,  color='black')
    ax3.plot(X_test, y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())
    
    plt.show()
    
    print("Slope:%.3f" % reg.coef_[0])
    print("Intercept:%.3f" % reg.intercept_)

    # The coefficients
    print('Coefficients: \n', reg.coef_)
    # The mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
    
    #the derivative of the fitted line is the slope
    return reg.coef_[0]
