#!/usr/bin/env python

import numpy as np
import pandas as pd

SHORT_TERM = 3              # Short term average
LONG_TERM = 30              # Long term average
PRICE_RANGE = 5             # The period to calculate price difference
AMP_WINDOW = 100            # The period to get stock amplitude

INCREASE_HOLDING = 500      # Price to increase holding
DECREASE_HOLDING = 500      # Price to decrease holding
AMP_LO_THRESHOLD = 10       # The threshold when price movement is considered low
AMP_HI_THRESHOLD = 1.5      # The threshold when price movement is considered high

nInst = 50
currentPos = np.zeros(nInst)

SMOOTH_INCREASE_STOCKS = []
SMOOTH_DECREASE_STOCKS = []
NUTURAL_STOCKS = []

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

        # Use short term and long term average to determine sign
        long_mean = single_stock_data.loc[day - LONG_TERM: (day-1), 'closePrice'].mean()
        short_mean = single_stock_data.loc[day - SHORT_TERM: (day-1), 'closePrice'].mean()
        today_sign = np.sign(short_mean - long_mean)

        # Use a price window to make decision
        n_day_diff = single_stock_data.loc[day-PRICE_RANGE, 'closePrice'] - single_stock_data.loc[day-1, 'closePrice']
        n_day_range = max(single_stock_data.loc[day-PRICE_RANGE:day-1, 'closePrice']) - min(single_stock_data.loc[day-PRICE_RANGE:day-1, 'closePrice'])
        

        if np.abs(n_day_diff) <= amp[stock]/AMP_LO_THRESHOLD:
            pass
            
        elif np.abs(n_day_diff) >= amp[stock]/AMP_HI_THRESHOLD:
            value = today_sign * DECREASE_HOLDING
            currentPos[stock] -= value//currentPrices[stock]
    
        else:
            value = today_sign * INCREASE_HOLDING
            currentPos[stock] += value//currentPrices[stock]

	
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

        base_range = max(single_stock_data.loc[day-AMP_WINDOW:day-1, 'closePrice']) -  min(single_stock_data.loc[day-AMP_WINDOW:day-1, 'closePrice'])
        amp.append(base_range)
    return amp