#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:08:31 2018

@author: dimm

identifies stocks that should outperform S&P 500
"""


import numpy as np
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#Import random forest model
from sklearn.ensemble import RandomForestRegressor
#Import cross-validation pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
#from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import os
import time
from datetime import datetime
import quandl
import re
import urllib.request



API_KEY = open("api_key.txt", "r").read()

intraQuarterPath = "intraQuarter"


FEATURES = ['Total Debt/Equity',
             'Trailing P/E',
             'Price/Sales',
             'Price/Book',
             'Profit Margin',
             'Operating Margin',
             'Return on Assets',
             'Return on Equity',
             'Revenue Per Share',
             'Market Cap',
             'Enterprise Value',
             'Forward P/E',
             'PEG Ratio',
             'Enterprise Value/Revenue',
             'Enterprise Value/EBITDA',
             'Revenue',
             'Gross Profit',
             'EBITDA',
             'Net Income Avl to Common ',
             'Diluted EPS',
             'Earnings Growth',
             'Revenue Growth',
             'Total Cash',
             'Total Cash Per Share',
             'Total Debt',
             'Current Ratio',
             'Book Value Per Share',
             'Cash Flow',
             'Beta',
             'Held by Insiders',
             'Held by Institutions',
             'Shares Short (as of',
             'Short Ratio',
             'Short % of Float',
             'Shares Short (prior ']


DFColumns = ['Status',
             'Date',
             'Unix',
             'Ticker',
             'Price',
             'stock_p_change',
             'SP500',
             'sp500_p_change',
             'Difference'] + FEATURES


FEATURES_CURRENT=["Total Debt/Equity",
                  'Trailing P/E',
                  'Price/Sales',
                  'Price/Book',
                  'Profit Margin',
                  'Operating Margin',
                  'Return on Assets',
                  'Return on Equity',
                  'Revenue Per Share',
                  'Market Cap',
                  'Enterprise Value',
                  'Forward P/E',
                  'PEG Ratio',
                  'Enterprise Value/Revenue',
                  'Enterprise Value/EBITDA',
                  'Revenue',
                  'Gross Profit',
                  'EBITDA',
                  'Net Income Avi to Common',
                  'Diluted EPS',
                  'Earnings Growth',
                  'Revenue Growth',
                  'Total Cash',
                  'Total Cash Per Share',
                  'Total Debt',
                  'Current Ratio',
                  'Book Value Per Share',
                  'Cash Flow',
                  'Beta',
                  'Held by Insiders',
                  'Held by Institutions',
                  'Shares Short',
                  'Short Ratio',
                  'Short % of Float',
                  'Shares Short (prior ']

def Query_Quandle(name, start_date, end_date):    
    return quandl.get(name, 
                      start_date = start_date, 
                      end_date = end_date, 
                      api_key=API_KEY)
    
   
def CalculateSP500Value(unix_time, df, col_name):
    sp500_date = datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d')
    val = df[(df.index == sp500_date)][col_name]
    
    return float(val)


def CalculateStockPerformance(unix_time, df, column, time_period):
    try:
        val = CalculateSP500Value(unix_time, df, column)#                       
    except:
        val = CalculateSP500Value(unix_time-259200, df, column)#-3 day
        
    later = int(unix_time + time_period)
    
    try:
        val_later = CalculateSP500Value(later, df, column)
    except Exception as e:
        try:
            val_later = CalculateSP500Value(later - 259200, df, column)   #-3 day                         
        except Exception as e:
            print("CalculateStockValues exception: ", str(e))
            
    return val, round(((val_later - val) / val) * 100, 2)


def Pull_Stock_Prices(start_date, end_date, force_query = False):
    
    if not force_query and os.path.exists('stock_prices.csv'):
        df = pd.read_csv('stock_prices.csv', index_col=0)
        return df
    
    print(start_date, end_date)
    df = pd.DataFrame()
    statspath = intraQuarterPath + "/_KeyStats"
    
    stock_list = [x[0] for x in os.walk(statspath)]
    
    for each_dir in stock_list[1:]:
        try:            
            ticker = each_dir.split('/')[2]
            print(ticker)
            
            name = "WIKI/" + ticker.upper()
            data = Query_Quandle(name, start_date, end_date)
        except Exception as e:
            print(str(e))
            time.sleep(10)
            try:
                data = Query_Quandle(name, start_date, end_date)
            except Exception as e:
                print(str(e))
            
        data[ticker.upper()] = data["Adj. Close"]
        df = pd.concat([df, data[ticker.upper()]], axis = 1)
        
    df.to_csv('stock_prices.csv')
    return df



def Parse_Key_Stats(outperform_threshold, NA_threshold, sp500_df, stock_df, time_period, force_query = False):
        
    if not force_query and os.path.exists('key_stats.csv'):
        df = pd.read_csv('key_stats.csv', index_col=0)
        return df
                
    statspath = intraQuarterPath + "/_KeyStats"
    stock_list = [x[0] for x in os.walk(statspath)]
    
    df = pd.DataFrame(columns = DFColumns)
            
    for each_dir in stock_list[1:]:        
        each_file = os.listdir(each_dir)
        
        if len(each_file) > 0:
            
            ticker = each_dir.split("/")[2].upper()
            
            for file in each_file:
                date_stamp = datetime.strptime(file, '%Y%m%d%H%M%S.html')                
                unix_time = time.mktime(date_stamp.timetuple())                
                full_file_path = each_dir + '/' + file                
                source = open(full_file_path, 'r').read()

                try:
                    
                    value_list = []
                    for each_data in FEATURES:                        
                        try:
                            
                            regex = re.escape(each_data) + r'.*?(\d{1,8}\.\d{1,8}M?B?|N/A)%?</td>'
                            
                            value = re.search(regex, source)
                            value = (value.group(1))
                            
                            if "B" in value:
                                value = float(value.replace("B", ''))*1000000000
                            elif "M" in value:
                                value = float(value.replace("M", ''))*1000000
                                
                            value_list.append(value)
                            
                        except:
                            #print(str(e), ticker, file)
                            value = np.nan
                            value_list.append(value)
                    
                    
                    sp_500_value, sp500_p_change = CalculateStockPerformance(unix_time, sp500_df, 'Adj Close', time_period)
                    
                    stock_price, stock_p_change = CalculateStockPerformance(unix_time, stock_df, ticker, time_period)
                                                            
                    difference = stock_p_change-sp500_p_change
                    
                    if difference >= outperform_threshold: #if outperfors S&P by outperform_threshold%
                        status = 1#"outperform"
                    else:
                        status = 0#"underperform"
                    
                    
                    if value_list.count(np.nan) > NA_threshold: #value_list.count("N/A") > 0:
                        pass
                    else:                        
                    
                        df = df.append({'Date':date_stamp,
                                        'Unix':unix_time,
                                        'Ticker':ticker,                                        
                                        'Price':stock_price,
                                        'stock_p_change':stock_p_change,
                                        'SP500':sp_500_value,
                                        'sp500_p_change':sp500_p_change,
                                        'Difference':difference,
                                        'Total Debt/Equity':value_list[0],
                                        'Trailing P/E':value_list[1],
                                        'Price/Sales':value_list[2],
                                        'Price/Book':value_list[3],
                                        'Profit Margin':value_list[4],
                                        'Operating Margin':value_list[5],
                                        'Return on Assets':value_list[6],
                                        'Return on Equity':value_list[7],
                                        'Revenue Per Share':value_list[8],
                                        'Market Cap':value_list[9],
                                        'Enterprise Value':value_list[10],
                                        'Forward P/E':value_list[11],
                                        'PEG Ratio':value_list[12],
                                        'Enterprise Value/Revenue':value_list[13],
                                        'Enterprise Value/EBITDA':value_list[14],
                                        'Revenue':value_list[15],
                                        'Gross Profit':value_list[16],
                                        'EBITDA':value_list[17],
                                        'Net Income Avl to Common ':value_list[18],
                                        'Diluted EPS':value_list[19],
                                        'Earnings Growth':value_list[20],
                                        'Revenue Growth':value_list[21],
                                        'Total Cash':value_list[22],
                                        'Total Cash Per Share':value_list[23],
                                        'Total Debt':value_list[24],
                                        'Current Ratio':value_list[25],
                                        'Book Value Per Share':value_list[26],
                                        'Cash Flow':value_list[27],
                                        'Beta':value_list[28],
                                        'Held by Insiders':value_list[29],
                                        'Held by Institutions':value_list[30],
                                        'Shares Short (as of':value_list[31],
                                        'Short Ratio':value_list[32],
                                        'Short % of Float':value_list[33],
                                        'Shares Short (prior ':value_list[34],
                                        'Status':status}, ignore_index = True)       
                except:
                    pass
            
    
    df = df.replace(np.nan, 0).replace("N/A", 0)
    df.to_csv("key_stats.csv")
    return df
 
def Query_Yahoo(ticker):
    link = "https://finance.yahoo.com/quote/{0}/key-statistics/".format(ticker)
    try:
        print(ticker, "...")
        resp = urllib.request.urlopen(link).read()       
        
    except Exception as e:    
        print(str(e))
        time.sleep(2)
        try:
            resp = urllib.request.urlopen(link).read()
        except:
            resp = ''            
        
    return str(resp)
        
    
def Parse_Today_Key_Stats(NA_threshold, force_query = False):
    if not force_query and os.path.exists('forward_sample.csv'):
        df = pd.read_csv('forward_sample.csv', index_col=0)
        return df
        
    df = pd.DataFrame(columns = DFColumns)
        
    statspath = intraQuarterPath+"/_KeyStats"    
    stock_list = [x[0] for x in os.walk(statspath)]
    print(5*"_", "Pulling today's key statistics for {0} stocks".format(len(stock_list) - 1), 5*"_")
    
    for e in stock_list[1:]:
        ticker = e.split("/")[2].upper()        
        
        source = Query_Yahoo(ticker)
                        
        if (len(source) == 0):
            continue
                
        try:            
            value_list = []
            for each_data in FEATURES_CURRENT:                        
                try:                    
                    regex = re.escape(each_data) + r'.*?(\d{1,8}\.\d{1,8}M?B?|N/A)%?</td>'
                    
                    value = re.search(regex, source)
                    value = (value.group(1))
                    
                    if "B" in value:
                        value = float(value.replace("B", ''))*1000000000
                    elif "M" in value:
                        value = float(value.replace("M", ''))*1000000
                        
                    value_list.append(value)
                    #print('Parsed', each_data, 'from', ticker)
        
                except:
                    #print(str(e), ticker, each_data)
                    value = np.nan
                    value_list.append(value)
            
     
            if value_list.count(np.nan) > NA_threshold:
                pass
            else:            
                df = df.append({'Date':np.nan,
                                'Unix':np.nan,
                                'Ticker':ticker,                                        
                                'Price':np.nan,
                                'stock_p_change':np.nan,
                                'SP500':np.nan,
                                'sp500_p_change':np.nan,
                                'Difference':np.nan,
                                'Total Debt/Equity':value_list[0],
                                'Trailing P/E':value_list[1],
                                'Price/Sales':value_list[2],
                                'Price/Book':value_list[3],
                                'Profit Margin':value_list[4],
                                'Operating Margin':value_list[5],
                                'Return on Assets':value_list[6],
                                'Return on Equity':value_list[7],
                                'Revenue Per Share':value_list[8],
                                'Market Cap':value_list[9],
                                 'Enterprise Value':value_list[10],
                                 'Forward P/E':value_list[11],
                                 'PEG Ratio':value_list[12],
                                 'Enterprise Value/Revenue':value_list[13],
                                 'Enterprise Value/EBITDA':value_list[14],
                                 'Revenue':value_list[15],
                                 'Gross Profit':value_list[16],
                                 'EBITDA':value_list[17],
                                 'Net Income Avl to Common ':value_list[18],
                                 'Diluted EPS':value_list[19],
                                 'Earnings Growth':value_list[20],
                                 'Revenue Growth':value_list[21],
                                 'Total Cash':value_list[22],
                                 'Total Cash Per Share':value_list[23],
                                 'Total Debt':value_list[24],
                                 'Current Ratio':value_list[25],
                                 'Book Value Per Share':value_list[26],
                                 'Cash Flow':value_list[27],
                                 'Beta':value_list[28],
                                 'Held by Insiders':value_list[29],
                                 'Held by Institutions':value_list[30],
                                 'Shares Short (as of':value_list[31],
                                 'Short Ratio':value_list[32],
                                 'Short % of Float':value_list[33],
                                 'Shares Short (prior ':value_list[34],
                                 'Status':np.nan}, ignore_index = True)       
        except:
            pass
            

    df = df.replace(np.nan, 0).replace("N/A", 0)
    df.to_csv("forward_sample.csv")        
    return df
  
def Performance_Calc(stock, sp500, outperform):
    if stock - sp500 >= outperform:
        return 1
    else:
        return 0

def Build_X_Feature(data_df):    
            
    X = np.array(data_df[FEATURES].values)        
    
    if len(X) != 0:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
    
    return X

def Build_y_Feature(data_df, outperform):   
    data_df["Status"] = list(map(lambda s, sp: Performance_Calc(s, sp, outperform), data_df['stock_p_change'], data_df['sp500_p_change']))                   
        
    return data_df["Status"].values.tolist()

def TrainModel_Linear__(data_df, outperform, test_size):
    
    data_df = data_df.reindex(np.random.permutation(data_df.index))#random shuffling
    
    print(5*"_", 'Building features for specified data set', 5*"_")
    X = Build_X_Feature(data_df)
    y = Build_y_Feature(data_df, outperform)
                
    print(5*"_", 'Creating and training model, test size={0}'.format(test_size), 5*"_")
    clf = svm.SVC(kernel="linear", C = 1.0)
    clf.fit(X[:-test_size], y[:-test_size])
    
    return clf

def TrainModel_Linear(data_df, outperform, test_size):
    
    #data_df = data_df.reindex(np.random.permutation(data_df.index))#random shuffling
    
    print(5*"_", 'Building features for specified data set', 5*"_")
    X = Build_X_Feature(data_df)
    y = Build_y_Feature(data_df, outperform)
                
    #Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)
    
    print(5*"_", 'Creating and training model', 5*"_")
    clf = svm.SVC(kernel="linear", C = 1.0)
    clf.fit(X_train, y_train)
    
    
    #Predict a new set of data
    y_pred = clf.predict(X_test)

    #evaluate model performance. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse)
    print('')
    print('R^2 (coefficient of determination) regression score: ' + str(r2_score(y_test, y_pred)))
    print('Mean squared error regression loss: ' + str(mean_squared_error(y_test, y_pred)))
    print('')
    
    return clf

def TrainModel_KNeighbor(data_df, outperform, test_size, K):
    
    print(5*"_", 'Building features for specified data set', 5*"_")
    X = Build_X_Feature(data_df)
    y = Build_y_Feature(data_df, outperform)
    
    #Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/10, random_state=55)
    
    print(5*"_", 'Creating and training model', 5*"_")
    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(X_train, y_train) 
    
    #evaluate model performance. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse)
    print('')    
    print('Mean accuracy on the given test data and labels:' + str(neigh.score(X_test, y_test)))
    print('')
    
    return neigh



def TrainModel_Rand_Forest(data_df, outperform, test_size):
    
    #data_df = data_df.reindex(np.random.permutation(data_df.index))#random shuffling
        
    print(5*"_", 'Building features for specified data set', 5*"_")
    X = np.array(data_df[FEATURES].values)
    X = preprocessing.scale(X)
    
    y = Build_y_Feature(data_df, outperform)
    
    #Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
    
    
    #Pipeline with preprocessing and model
    #Declare data preprocessing steps
    pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))
    
    #Declare hyperparameters to tune
    hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 
                       'randomforestregressor__max_depth': [None, 5, 3, 1]}
    
    print(5*"_", 'Creating and training model', 5*"_")
    #Sklearn cross-validation with pipeline. 
    #GridSearchCV essentially performs cross-validation across the entire "grid" (all possible permutations) of hyperparameters.
    clf = GridSearchCV(pipeline, hyperparameters, cv=10)
     
    # Fit and tune model
    clf.fit(X_train, y_train)
    
    #Predict a new set of data
    y_pred = clf.predict(X_test)  
    
    #evaluate model performance. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse)
    print('')
    print('R^2 (coefficient of determination) regression score: ' + str(r2_score(y_test, y_pred)), 'Best possible score is 1.0')
    print('Mean squared error regression loss: ' + str(mean_squared_error(y_test, y_pred)))
    print('')
    
    return clf

def Analysis(model, tickers, X_list):
    invest_list = []
    
    for i in range(len(X_list)):
        p = model.predict([X_list[i]])[0]        
        if p == 1:
            invest_list.append(tickers[i])
            
    
    return invest_list
    

NA_threshold = 2
outperform_threshold = 18
test_size = 3

#time_period_perf_calc =  31536000 #year
time_period_perf_calc = 7884000 #decide abour performance each quarter
#time_period_perf_calc = 15768000 #half year

print(5*"_", 'Pulling stock prices', 5*"_")
stock_df = Pull_Stock_Prices("2000-01-01", datetime.now().strftime('%Y-%m-%d'))#up to today

print(5*"_", 'Reading S&P 500 prices (GSPC)', 5*"_")
sp500_df = pd.read_csv("GSPC.csv", index_col=0)

print(5*"_", 'Parsing key stats and comparing stocks prices to S&P 500 prices', 5*"_")
key_stats_df = Parse_Key_Stats(outperform_threshold, NA_threshold, sp500_df, stock_df, time_period_perf_calc)

print(5*"_", "Pulling and Parsing today's key statistics", 5*"_")
forward_df = Parse_Today_Key_Stats(NA_threshold)

print(5*"_", 'Performing analysis of key statistics data set', 5*"_")
tickers = forward_df['Ticker'].values.tolist()

#clf = TrainModel_Linear(key_stats_df, outperform_threshold, test_size)
#clf = TrainModel_Rand_Forest(key_stats_df, outperform_threshold, test_size)
clf = TrainModel_KNeighbor(key_stats_df, outperform_threshold, test_size, 5)

print(5*"_", 'Building features for forward data set', 5*"_")
X = Build_X_Feature(forward_df)

stock_list = Analysis(clf, tickers, X)

print(15*"_", 'Finished', 15*"_")
print('Following stocks will outperform S&P 500 by at least', outperform_threshold, '%')
for s in stock_list:
    print(s)
        
   
#tickers = ['VRNS', 'MSFT', 'DAL', 'WFC', 'GBTC']
#
#print(5*"_", 'Checking custom list', 5*"_")
#print(tickers)
#
#X_list = []
#for t in tickers:
#    X = np.array(forward_df[forward_df['Ticker'] == t][FEATURES].values)
#        
#    if len(X) != 0: 
#        X = preprocessing.scale(X[0])
#        X_list.append(X)
#
#stock_list = Analysis(clf, tickers, X_list)
#for s in stock_list:
#    print(s)
#    
#    
    
    
    
    