#!/usr/bin/env python

import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression

SHORT_TERM = 3
LONG_TERM = 30
PRICE_RANGE = 4

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
	
    day = prcSoFar.shape[1]
    train_data = data_process(prcSoFar)

    global currentPos

    currentPrices = prcSoFar[:,day-1] # price of last day

    amp = range_so_far(train_data, day)
	
    # Get long term and short term average prices
    for stock in range(50):
        single_stock_data = train_data[train_data['stock'] == stock]
        single_stock_data.index = range(len(single_stock_data))

        #
        long_mean = single_stock_data.loc[day - (LONG_TERM-1): (day-1), 'closePrice'].mean()
        short_mean = single_stock_data.loc[day - (SHORT_TERM-1): (day-1), 'closePrice'].mean()
        today_sign = np.sign(short_mean - long_mean)

        # Determine crossover
        # if today_sign != yesterday_sign: crossover = True
        # else: crossover = False

        n_day_range = max(single_stock_data.loc[day-PRICE_RANGE:day-1, 'closePrice']) - min(single_stock_data.loc[day-PRICE_RANGE:day-1, 'closePrice'])
        n_day_diff = single_stock_data.loc[day-PRICE_RANGE, 'closePrice'] - single_stock_data.loc[day-1, 'closePrice']
        
        # if crossover: 
        #     value = 10000 * today_sign
        #     currentPos[stock] = value//currentPrices[stock]
        
        # elif n_day_range > amp[stock] / 4:
        #     value = 0


        # if n_day_range >= amp[stock]/4:
        #     value = 0
            
        # else:
        #     value = 2019.5003*np.arctanh((today_sign * n_day_range)/(amp[stock]/4))
            
        if n_day_diff <= amp[stock]/8:
            # value = 0
            # currentPos[stock] = value//currentPrices[stock]
            pass
        elif np.abs(n_day_diff) >= amp[stock]/1.5:
            value = 0
            currentPos[stock] = value//currentPrices[stock]
            
        else:
            # value = 1638.5303*np.arctanh((today_sign * n_day_diff)/(amp[stock]/2))
            # value = 20000/np.pi * np.arctan(5*(n_day_diff-amp[stock]/6)/(amp[stock]/2-amp[stock]/6))
            value = today_sign * 10000
            currentPos[stock] = value//currentPrices[stock]
	
            
        # Update to curr pos
        
	# Use Cstar to calculate the principal denoted to a given pair, using a selected kernal function. If Cstar is postive, j has a higher price 
	# and i has a lower price -> return positive principal, short i and long j. If Cstar is negative, then return negative principal, long i and short j.
    
    return currentPos





def data_process(prcAll):
    """
    Convert raw price data into closeprice columns 
    """
    closePrice = []
    for stock in range(50):
        closePrice.extend(prcAll[stock])

    dates = []
    for stock in range(50):
        for day in range(prcAll.shape[1]):
            dates.append(day)

    stocks = []

    for stock in range(50):
        stocks.extend([stock]*prcAll.shape[1])
    full_data = pd.DataFrame({'date': dates, 'stock': stocks, 'closePrice': closePrice})
    full_data['log_closePrice'] = np.log(full_data['closePrice'])
    return full_data


def range_so_far(data, day):
    amp = []
    for j in range(50):
        single_stock_data = data[data['stock'] == j]
        single_stock_data.index = range(len(single_stock_data))
        single_stock_data['diff_closePrice'] = single_stock_data['closePrice'].diff()

        base_range = max(single_stock_data.loc[0:day-1, 'closePrice']) -  min(single_stock_data.loc[0:100, 'closePrice'])
        amp.append(base_range)
    return amp