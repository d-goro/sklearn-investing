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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score

import pandas as pd
import os
import time
from datetime import datetime
import quandl
import re
import json
import urllib.request

from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()

intraQuarterPath = "intraQuarter"

QUANDL_API_KEY = open("api_key.txt", "r").read()
BARCHART_API_KEY = open("barchart_api_key.txt", "r").read()


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


DFColumns = ['Date',
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
                  'Quarterly Earnings Growth',
                  'Quarterly Revenue Growth',
                  'Total Cash',
                  'Total Cash Per Share',
                  'Total Debt',
                  'Current Ratio',
                  'Book Value Per Share',
                  'Cash Flow',
                  'Beta',
                  '% Held by Insiders',
                  '% Held by Institutions',
                  'Shares Short',
                  'Short Ratio',
                  'Short % of Float',
                  'Shares Short (prior ']

def Request_Link(link):
    headers={'User-agent' : 'Mozilla/5.0'}
    req = urllib.request.Request(link, None, headers)
    resp = urllib.request.urlopen(req).read()
    return resp

def Query_BarChart(name, start_date, end_date):
    s = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
    e = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
    link = "https://marketdata.websol.barchart.com/getHistory.json?apikey={0}&symbol={1}&type=daily&startDate={2}&endDate={3}".format(BARCHART_API_KEY, name, s, e)
    
    try:
        print(name)
        resp = Request_Link(link)
        
    except Exception as e:    
        print(str(e))
        time.sleep(2)
        try:
            resp = Request_Link(link)
        except:
            resp = ''
    
    df = pd.DataFrame(columns=['Date', 'Symbol', 'Adj. Close'])
    data = json.loads(resp)

    if len(data['results']) < 200:
        return df    

    for d in data['results']:        
        df = df.append({'Date':pd.Timestamp(d['tradingDay']),
                        'Symbol':d['symbol'],
                        'Adj. Close':d['close']}, ignore_index=True)
    
    df = df.set_index('Date')
    return df
    

def Query_Quandle(name, start_date, end_date):    
    return quandl.get(name, 
                      start_date = start_date, 
                      end_date = end_date, 
                      api_key=QUANDL_API_KEY)
    
def Query_Yahoo(name, start_date, end_date):
    stock_ohlc = pdr.get_data_yahoo(name, start=start_date, end=end_date)
    return stock_ohlc.rename(mapper = {'Adj Close':'Adj. Close'}, axis=1)
    
   
def GetStockValue(unix_time, df, col_name):
    date = datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d')
    val = df[(df.index == date)][col_name]
    
    return float(val)


def CalculateStockPerformance(unix_time, df, column, time_period):
    try:
        val = GetStockValue(unix_time, df, column)#                       
    except:
        try:
            unix_time = unix_time-259200
            val = GetStockValue(unix_time, df, column)#-3 day
        except:
            try:
                unix_time = unix_time-259200
                val = GetStockValue(unix_time, df, column)#-3 day
            except Exception as e:
                print("CalculateStockValues for earlier value exception: ", str(e))
                raise e 
        
    later = int(unix_time + time_period)
    
    try:
        val_later = GetStockValue(later, df, column)
    except:
        try:
            later = later - 259200
            val_later = GetStockValue(later, df, column)   #-3 day
        except:
            try:
                later = later - 259200
                val_later = GetStockValue(later, df, column)   #-3 day
            except Exception as e:
                val_later = val
                print("CalculateStockValues for later value exception: ", str(e))
                raise e 
            
    return val, round(((val_later - val) / val) * 100, 2)


def Pull_Stock_Prices(stock_list, start_date, end_date, force_query = False):
    
    stocks_to_remove = []
    
    if not force_query and os.path.exists('stock_prices.csv'):
        df = pd.read_csv('stock_prices.csv', index_col=0)
        return df, stocks_to_remove
    
    print(start_date, end_date)
    df = pd.DataFrame()
    
    for ticker in stock_list:
        try:
            ticker = ticker.upper()
            print(ticker)
            time.sleep(1)
            
            data = Query_Quandle("WIKI/" + ticker, start_date, end_date)
            
            if len(data) < 100:
                print('Not enough data for', ticker, 'on Quandl, querying Yahoo...')
                data = Query_Yahoo(ticker, start_date, end_date)
                        
            if len(data) < 100:
                print('Not enough data for', ticker, 'on Yahoo, querying Barchart...')      
                data = Query_BarChart(ticker, start_date, end_date)
                
            if len(data) < 100:
                print('Still not enough data for', ticker, 'removing it from list')                                    
                stocks_to_remove.append(ticker)
                continue
        except Exception as e:
            print(str(e))   
            
            try:
                print('Pulling from Yahoo...')
                data = Query_Yahoo(ticker, start_date, end_date)
                if len(data) < 100:
                    raise Exception('Not enough data for {0} on Yahoo'.format(ticker))
            except Exception as e:
                print(str(e))     
                print('Pulling from Barchart...')
                try:
                    data = Query_BarChart(ticker, start_date, end_date)
                    if len(data) < 100:
                        print('Not enough data for', ticker)
                        stocks_to_remove.append(ticker)
                        continue                
                except Exception as e:
                    print(str(e))
                    continue
            
        data[ticker] = data["Adj. Close"]
        df = pd.concat([df, data[ticker]], axis = 1)
        
    df.to_csv('stock_prices.csv')
    
    return df, stocks_to_remove


def Load_Key_Stats():
    df = pd.read_csv('key_stats.csv', index_col=0)
    return df
 
def Query_Yahoo_Stats(ticker):
    link = "https://finance.yahoo.com/quote/{0}/key-statistics/".format(ticker)
    try:
        print(ticker, "...")
        resp = Request_Link(link)
        
    except Exception as e:    
        print(ticker, "threw exception", str(e), "retrying...")
        time.sleep(2)
        try:
            resp = Request_Link(link)
        except:
            print(ticker, "unavailable on yahoo")
            resp = ''            
        
    return str(resp)

        
def Update_Key_Stats_Outperform_Status(key_stats_df, outperform_threshold):
    key_stats_df['Status'] = list(map(lambda x: 1 if x >= outperform_threshold else 0, key_stats_df.Difference))
    

def Parse_Key_Stats(NA_threshold, sp500_df, stock_df, time_period, force_query = False):
        
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
                            #regex = r'>' + re.escape(each_data) + r'.*?(\-?\d+\.*\d*K?M?B?|N/A[\\n|\s]*|>0|NaN)%?' \
                            #r'(</td>|</span>)'
                            #regex = re.escape(each_data) + r'.*?(\d{1,8}\.\d{1,8}M?B?|N/A)%?</td>'
                            regex = re.escape(each_data) + r'.*?([+-]?\d{1,8}\.?\d{1,8}?M?B?K?|N/A[\\n|\s]*|>0|NaN)%?(</td>|</span>)'
                            value = re.search(regex, source) 
                            
                            if value == None:
                                #print('No feature', each_data, 'for ticker', ticker, 'going to next one...')
                                value = np.nan                                
                            else:                                                                              
                                if value.span()[1] - value.span()[0] > 500:
                                    value = np.nan
                                else:
                                    value = (value.group(1))
                                    if value[0] == '>':
                                        value = value[1:]
                                    
                                    if "B" in value:
                                        value = float(value.replace("B", ''))*1000000000
                                    elif "M" in value:
                                        value = float(value.replace("M", ''))*1000000
                                    elif "K" in value:
                                        value = float(value.replace("K", ''))*1000
                                
                            value_list.append(value)
                            
                        except:
                            #print(str(e), ticker, file)
                            value = np.nan
                            value_list.append(value)
                    
                    
                    #print(ticker, ': N/A count:', value_list.count('N/A'))
                    if value_list.count('N/A') + value_list.count(np.nan) > NA_threshold:
                        print(ticker, ': N/A count:', value_list.count('N/A') + value_list.count(np.nan), 'passes')
                        pass
                    else:                    
                        sp_500_value, sp500_p_change = CalculateStockPerformance(unix_time, sp500_df, 'Adj Close', time_period)
                    
                        stock_price, stock_p_change = CalculateStockPerformance(unix_time, stock_df, ticker, time_period)
                                                                
                        difference = stock_p_change-sp500_p_change
                                                
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
                                        'Shares Short (prior ':value_list[34]}, ignore_index = True)       
                except:
                    pass
            
    
    df = df.replace(np.nan, 0).replace("N/A", 0)
    df.to_csv("key_stats.csv")
    return df


def Parse_Today_Key_Stats(stock_list, NA_threshold, force_query = False):
    if not force_query and os.path.exists('forward_sample.csv'):
        df = pd.read_csv('forward_sample.csv', index_col=0)
        return df
        
    df = pd.DataFrame(columns = DFColumns)
            
    print(5*"_", "Pulling today's key statistics for {0} stocks".format(len(stock_list)), 5*"_")
    
    for ticker in stock_list:
        source = Query_Yahoo_Stats(ticker)
                        
        if (len(source) == 0):
            continue
                
        try:            
            value_list = []
            for each_data in FEATURES_CURRENT:                        
                try:                    
                    regex = re.escape(">{0}".format(each_data)) + r'.*?>([+-]?\d{1,8}\.?\d{1,8}?M?B?K?|N/A)%?<'
                    value = re.search(regex, source) 
                    
                    if value == None:
                        print('No feature', each_data, 'for ticker', ticker, 'going to next one...')
                        value = np.nan                        
                    else:                                 
                        if value.span()[1] - value.span()[0] > 500:
                            value = np.nan
                        else:
                            value = (value.group(1))
                            if "B" in value:
                                value = float(value.replace("B", ''))*1000000000
                            elif "M" in value:
                                value = float(value.replace("M", ''))*1000000
                            elif "K" in value:
                                value = float(value.replace("K", ''))*1000
                    
                    
                    value_list.append(value)
                    #print('Parsed', each_data, 'from', ticker, value_list)
        
                except Exception as e:
                    print(str(e), ticker, each_data)
                    value = np.nan
                    value_list.append(value)
            
     
            #print('N/A count:', value_list.count('N/A'))
            if value_list.count('N/A') + value_list.count(np.nan) > NA_threshold:
                print(ticker, ': N/A count:', value_list.count('N/A') + value_list.count(np.nan), '- not added to the table')
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
                                 'Shares Short (prior ':value_list[34]}, ignore_index = True)       
        except Exception as e:
            print(str(e), ticker)
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
    
    print(5*"_", 'Creating and training KNeighbors model', 5*"_")
    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(X_train, y_train) 
    
    #Predict a new set of data    
    y_pred = neigh.predict(X_test)
    
    #evaluate model performance. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse)
    print('')    
    print("KNeighbors classifier performance\n", "=" * 20)
    print(f"Accuracy score on the given test data and labels: {neigh.score(X_test, y_test): .2f}")
    print(f"Precision score: {precision_score(y_test, y_pred): .2f}")
    print('')
    
    return neigh



def TrainModel_Rand_Forest(data_df, outperform, test_size):
    
    #data_df = data_df.reindex(np.random.permutation(data_df.index))#random shuffling
        
    print(5*"_", 'Building features for specified data set', 5*"_")
    X = np.array(data_df[FEATURES].values)
    X = preprocessing.scale(X)
    
    y = Build_y_Feature(data_df, outperform)
    
    #Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/10, random_state=99)
    
    
    print(5*"_", 'Creating and training Random Forest model', 5*"_")
    clf = RandomForestClassifier(n_estimators=100, random_state=99)
    clf.fit(X_train, y_train)

    #Predict a new set of data    
    y_pred = clf.predict(X_test)
    print('')
    print("Random Forest Classifier performance\n", "=" * 20)
    print(f"Accuracy score on the given test data and labels: {clf.score(X_test, y_test): .2f}")
    print(f"Precision score: {precision_score(y_test, y_pred): .2f}")
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
outperform_threshold = 15
test_size = 2

#time_period_perf_calc =  31536000 #year
time_period_perf_calc = 7884000 #decide abour performance each quarter
#time_period_perf_calc = 15768000 #half year

print(5*"_", 'Pulling stocks', 5*"_")
stocks = pd.read_csv("Tickers.csv")['Symbol'].values.tolist()
print('There is {0} stocks'.format(len(stocks)))

print(5*"_", 'Pulling stock prices', 5*"_")
stock_df, stocks_to_remove = Pull_Stock_Prices(stocks, "2000-01-01", "2018-03-27")

print(len(stocks_to_remove), 'stocks will be removed due to lack of data')
stocks= list(set(stocks).difference(set(stocks_to_remove)))


print(5*"_", 'Reading S&P 500 prices (GSPC)', 5*"_")
sp500_df = pd.read_csv("GSPC.csv", index_col=0)

print(5*"_", 'Parsing key stats and comparing stocks prices to S&P 500 prices', 5*"_")
#key_stats_df = Load_Key_Stats()
key_stats_df = Parse_Key_Stats(NA_threshold, sp500_df, stock_df, time_period_perf_calc, True)

Update_Key_Stats_Outperform_Status(key_stats_df, outperform_threshold)

print(5*"_", "Pulling and Parsing today's key statistics", 5*"_")
forward_df = Parse_Today_Key_Stats(stocks, NA_threshold, True)
print('We have', len(forward_df), 'stocks with', NA_threshold, 'N/A threshold to analyze')

print(5*"_", 'Performing analysis of key statistics data set', 5*"_")
#clf = TrainModel_Linear(key_stats_df, outperform_threshold, test_size)

clfRF = TrainModel_Rand_Forest(key_stats_df, outperform_threshold, test_size)
clfKN = TrainModel_KNeighbor(key_stats_df, outperform_threshold, test_size, 5)

print(5*"_", 'Building features for forward data set', 5*"_")
X = Build_X_Feature(forward_df)
stocks_to_check = forward_df.Ticker.values.tolist()

print('Analyzing stocks with Random Forest model')
stock_list = Analysis(clfRF, stocks_to_check, X)

print(15*"_", 'Finished', 15*"_")
print('Regarding Random Forest model, following stocks will outperform S&P 500 by at least', outperform_threshold, '%')
for s in stock_list:
    print(s)
    
print('')
print('Analyzing stocks with KNeighbor model')    
stock_list = Analysis(clfKN, stocks_to_check, X)

print(15*"_", 'Finished', 15*"_")
print('Regarding KNeighbor model, following stocks will outperform S&P 500 by at least', outperform_threshold, '%')
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
    
    
    
    