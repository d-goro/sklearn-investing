#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 21:18:40 2018

Extarcts list of companies from Nasdaq and NYSE, filters out companies without data
and those companie that have MarketCap less than specified

@author: dimm
"""
import pandas as pd
import os


intraQuarterPath = "intraQuarter"

def Extract_Initial_Stock_List():
    statspath = intraQuarterPath + "/_KeyStats"
    stock_list = [x[0] for x in os.walk(statspath)]
    
    
    
    tickers = []
    for each_dir in stock_list[1:]:        
            each_file = os.listdir(each_dir)
            
            if len(each_file) > 0:            
                ticker = each_dir.split("/")[2].upper()            
                tickers.append(ticker)
    
    df = pd.DataFrame(columns = ['Symbol'], data = tickers)
    #df.to_csv("Tickers.csv")
    return df
    

def f (value):
    value = value.replace('$', '')
    if "B" in value:
        return float(value.replace("B", ''))*1000000000
    elif "M" in value:
        return float(value.replace("M", ''))*1000000





df1 = pd.read_csv('companylist_nyse.csv')[['Symbol', 'MarketCap']]
df1 = df1[df1.MarketCap.notnull()]
df1['MarketCap'] = df1['MarketCap'].apply(lambda x: f(x))


df2 = pd.read_csv('companylist_nasdaq.csv')[['Symbol', 'MarketCap']]
df2 = df2[df2.MarketCap.notnull()]
df2['MarketCap'] = df2['MarketCap'].apply(lambda x: f(x))

#df_c = df1.merge(df2, on='Symbol', how='outer')
df_c = pd.concat([df1, df2], ignore_index=True, verify_integrity=True)

df_c = df_c[df_c.MarketCap >= 900000000]


if os.path.exists('Tickers.csv'):
    tickers = pd.read_csv("Tickers.csv")[['Symbol']]
else:
    tickers = Extract_Initial_Stock_List()[['Symbol']]
    
    
new_df = tickers.join(df_c.set_index('Symbol'), how='outer', on='Symbol')[['Symbol']]

new_df.to_csv("Tickers.csv")
