#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


SHORT_TERM = 4              # Short term average
LONG_TERM = 15              # Long term average
PRICE_RANGE = 7             # The period to calculate price difference
AMP_WINDOW = 75            # The period to get stock amplitude

CHANGE_HOLDING = 500            # Price to change holding
AMP_LO_THRESHOLD = 7.5       # The threshold when price movement is considered low
AMP_HI_THRESHOLD = 1
PRICE_CHANGE_THRESHOLD = 0.01
MSE_THRESHOLD_2 = 0.04
SLOPE_THRESHOLD_2 = 2   # The threshold when price movement is considered high
MSE_THRESHOLD_1 = 0.05       # The threshold to control price volatility
SLOPE_THRESHOLD_1 = 0.05       # The threshold to control LR slope

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
	
    global currentPos

    currentPrices = prcSoFar[:,-1] # price of last day

    amp = range_so_far(prcSoFar)
	
    # Get long term and short term average prices
    for stock in range(50):
        single_stock_data = prcSoFar[stock]

        # Use short term and long term average to determine sign
        long_mean = single_stock_data[-LONG_TERM:].mean()
        short_mean = single_stock_data[-SHORT_TERM:].mean()
        today_sign = np.sign(short_mean - long_mean)

        # Use a price window to make decision
        n_day_diff = single_stock_data[-PRICE_RANGE] - single_stock_data[-1]
        n_day_range = np.max(single_stock_data[-PRICE_RANGE:]) - np.min(single_stock_data[-PRICE_RANGE:])
        # two_day_diff = single_stock_data.loc[day-3, 'closePrice'] - single_stock_data.loc[day-1, 'closePrice']

        # Calculate the MSE of price movement during the range
        n_day_gap = np.diff(single_stock_data[-PRICE_RANGE:])
        LR = LinearRegression(n_jobs=-1).fit(np.array(range(PRICE_RANGE-1)).reshape(-1, 1), n_day_gap.reshape(-1,1))
        n_day_mse = mean_squared_error(n_day_gap, LR.predict(np.array(range(PRICE_RANGE-1)).reshape(-1, 1)))
        
        # Position decition making

        # Stop loss
        if currentPos[stock] * n_day_range * np.sign(n_day_diff) > np.abs(currentPos[stock] * currentPrices[stock]) * PRICE_CHANGE_THRESHOLD and n_day_mse > np.abs(n_day_diff*MSE_THRESHOLD_2) or ((LR.coef_ * currentPos[stock] < 0)[0][0] and (np.abs(LR.coef_) > SLOPE_THRESHOLD_2)[0][0]):
            currentPos[stock] = 0

        # Keep current position unchanged
        elif np.abs(n_day_diff) <= amp[stock]/AMP_LO_THRESHOLD or (n_day_mse > np.abs(n_day_diff*MSE_THRESHOLD_1) and (np.abs(LR.coef_) < SLOPE_THRESHOLD_1)[0][0]):
            pass
            
        # Decrease holding
        elif np.abs(n_day_diff) >= amp[stock]/AMP_HI_THRESHOLD:
            value = today_sign * CHANGE_HOLDING
            currentPos[stock] -= value//currentPrices[stock]
    
        # Increase holding 
        else:
            value = today_sign * CHANGE_HOLDING
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


def range_so_far(data):
    amp = []
    for j in range(50):
        single_stock_data = data[j]

        base_range = np.max(single_stock_data[-AMP_WINDOW:]) -  np.min(single_stock_data[-AMP_WINDOW:])
        amp.append(base_range)
    return amp