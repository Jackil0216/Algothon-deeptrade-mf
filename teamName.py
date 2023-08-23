#!/usr/bin/env python

import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression

nInst=50
currentPos = np.zeros(nInst)
selected_pairs = 0
call = 0

def getMyPosition(prcSoFar):
	global call
	day = prcSoFar.shape[1]
	train_data = data_process(prcSoFar)
	# print(train_data)
        
	if (day-1)%5 == 0 or call == 0:
		colint_stock_list_ct = find_coint(train_data)
		moving_avg = trend_smoothing(train_data, day)
		trend_moving_avg = find_trend(moving_avg, colint_stock_list_ct)
		global selected_pairs 
		selected_pairs = find_detrended_pairs(trend_moving_avg, train_data, colint_stock_list_ct)

	global currentPos
	nInst, nt = prcSoFar.shape
	currentPrices = prcSoFar[:,nt-1] # price of last day
	# For selected pairs, calculate the price difference on the last day
	currentDiff = []
	for index, row in selected_pairs.iterrows():
		pair, slope, intercept, sd = row
		stock_i, stock_j = pair
		currdiff = currentPrices[stock_i] - currentPrices[stock_j] # close price difference
		currtrend = slope*nt + intercept # linear trend predicted using historical data
		currentDiff.append(currdiff - currtrend) # spread = price difference - predicted trend
	C = currentDiff/selected_pairs["StandardDeviation"] # divide the spread by the historical standard deviation
	
	# Set an upper bound of 1.5 and lower bound of -1.5 standard deviations. Similar to a threshold.
	Cstar = [] 
	for c in C:
		if c >= 1.5:
			Cstar.append(1.5)
		elif c <= -1.5:
			Cstar.append(-1.5)
		else:
			Cstar.append(c)
	Cstar = np.array(Cstar)
	
	# Use Cstar to calculate the principal denoted to a given pair, using a selected kernal function. If Cstar is postive, j has a higher price 
	# and i has a lower price -> return positive principal, short i and long j. If Cstar is negative, then return negative principal, long i and short j.
	principal = -(1500)*np.cos((Cstar+1.5)/3*np.pi)
	# principal = 1500*np.tanh(0.8*Cstar)
	# principal = -1500*(32*(Cstar/3)**5)
	
	# make trading decisions.
	for i in range(len(selected_pairs["Pair"])):
		stock_i, stock_j = selected_pairs["Pair"][i]
		principal_ij = principal[i]
		
		currentPos[stock_i] -= np.floor(principal_ij/currentPrices[stock_i])
		currentPos[stock_j] += np.floor(principal_ij/currentPrices[stock_j])
	
	call += 1
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


def find_coint(train_data):
    """
    Find stock pairs with cointegration
    """
    colint_stock_list_ct = []
    for stock_i in range(50):
        stock_i_df = train_data[train_data['stock'] == stock_i]

        for stock_j in range(stock_i+1, 50):
            stock_j_df = train_data[train_data['stock'] == stock_j]
            outcome = coint(stock_i_df['log_closePrice'], stock_j_df['log_closePrice'], trend = 'ct')
            if outcome[1] < 0.05:
                colint_stock_list_ct.append(([stock_i, stock_j], outcome))
    return colint_stock_list_ct


def trend_smoothing(train_data, num_days):
    """
    Smooth the trend using window
    """
    WINDOW = 5
    moving_avg = pd.DataFrame(columns=['stock', 'mid_date', 'avg_price'])

    for stock in range(50):
        single_stock = train_data[train_data['stock'] == stock]

        for date in range(num_days-WINDOW):
            window = single_stock[(single_stock['date'] >= date) & (single_stock['date'] < date+WINDOW)]
            avg_price = window['closePrice'].mean()

            new_row = {'stock': stock, 'mid_date': date+(WINDOW-1)/2, 'avg_price': avg_price}
            moving_avg = pd.concat([moving_avg, pd.DataFrame([new_row])], ignore_index=True)

    return moving_avg


def find_trend(moving_avg, colint_stock_list_ct):
    """
    Fit the data using Linear Regression
    """
    trend_moving_avg = pd.DataFrame(columns=['Pair', 'Slope', 'intercept'])
    for ([stock_i, stock_j], outcome) in colint_stock_list_ct:
        stock_i_df = moving_avg[moving_avg['stock'] == stock_i]
        stock_j_df = moving_avg[moving_avg['stock'] == stock_j]

        # Price difference df
        diff = pd.DataFrame()
        diff['difference'] = (stock_i_df['avg_price'].reset_index() - stock_j_df['avg_price'].reset_index())['avg_price']
        diff = np.array(diff['difference']).reshape(-1, 1)
        date = np.array(stock_i_df['mid_date']).reshape(-1, 1)
        
        # Fit linear regression
        LR = LinearRegression(n_jobs=-1).fit(date, diff)

        # Append new data
        new_data = {
            'Pair': [(stock_i, stock_j)],
            'Slope': [LR.coef_],
            'intercept': [LR.intercept_]
        }
        new_df = pd.DataFrame(new_data)
        trend_moving_avg = pd.concat([trend_moving_avg, new_df], ignore_index=True)

    return trend_moving_avg


def find_detrended_pairs(trend_moving_avg, train_data, colint_stock_list_ct):
    """
    Detrend the stock pairs, filter again
    """
    selected_pairs = pd.DataFrame(columns=["Pair", "LinearSlope", "LinearIntercept", "StandardDeviation"])

    for ([stock_i, stock_j], outcome) in colint_stock_list_ct:
        stock_i_df = train_data[train_data['stock'] == stock_i]
        stock_j_df = train_data[train_data['stock'] == stock_j]
        trend_slope = float(trend_moving_avg[trend_moving_avg["Pair"] == (stock_i,stock_j)]["Slope"])
        trend_intercept = float(trend_moving_avg[trend_moving_avg["Pair"] == (stock_i,stock_j)]["intercept"])
        
        # detrend the difference in prices using the linear regression results
        diff = (stock_i_df['closePrice'].reset_index() - stock_j_df['closePrice'].reset_index())["closePrice"]
        dates = np.linspace(0,len(stock_i_df)-1, len(stock_i_df))
        trend = trend_slope*dates + trend_intercept
        diff = diff-trend
        
        # Use the augemented dickey-fuller test to filter the pairs with stationary detrended price differences for trading. append to selected pairs.
        adf = adfuller(diff, regression="ct")
        if adf[1] <= 0.05:
            new_data = {
                'Pair' : [(stock_i, stock_j)],
                'LinearSlope' : [trend_slope],
                'LinearIntercept' : [trend_intercept],
                'StandardDeviation' : [np.std(diff)]
            }
            # print(stock_i, stock_j)      
            # plt.plot(dates,diff)
            # plt.show()
            new_df = pd.DataFrame(new_data)
            selected_pairs = pd.concat([selected_pairs, new_df], ignore_index=True)
        
    return selected_pairs