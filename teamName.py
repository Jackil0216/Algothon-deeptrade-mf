#!/usr/bin/env python

import numpy as np

nInst=50
currentPos = np.zeros(nInst)
# def getMyPosition (prcSoFar):
#     global currentPos
#     (nins,nt) = prcSoFar.shape
#     if (nt < 2):
#         return np.zeros(nins)
#     lastRet = np.log(prcSoFar[:,-1] / prcSoFar[:,-2])
#     rpos = np.array([int(x) for x in 2000000 * lastRet / prcSoFar[:,-1]])
#     currentPos = np.array([int(x) for x in currentPos+rpos])
#     return currentPos

    
def getMyPosition(pred_df, date):
        
    daily = [0 for i in range(50)]

    for stock in pred_df['stock'].unique():
        daily[stock-1] = 10000 if pred_df[(pred_df['stock'] == stock) & (pred_df['date'] == date)]['pred'].values[0] == 1 else -10000

    return daily
