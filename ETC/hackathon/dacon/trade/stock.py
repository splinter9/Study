# -*- coding: utf-8 -*-
"""
DACON AI Bit Trader Competition Season 2

Main script
"""

'''
Need to save script below as a .py file in the same directory for using custom 
functions while running multiprocessing.pool with window system (Not necessary for linux).

#%% Import modules
import itertools as it
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller

#%% Custom functions
def multi_adfuller_pval(seq):
    return adfuller(seq)[1]


def train_invest(y_train_open, y_train_mdl, invest_strat, use_weight=False):
    if use_weight:
        buy_q, sell_t, p_thld, weight = invest_strat
    else:
        buy_q, sell_t, p_thld = invest_strat
        weight=None
    
    if len(y_train_mdl) > 1 and weight != None:
        if weight < 0 or weight > 1: 
            print('Weight must from 0 to 1')
            raise ValueError
        elif len(y_train_mdl) > 2:
            print('More than two prediction results are exist')
            raise ValueError
        
        y_train_mdl = weight*y_train_mdl[0] + (1-weight)*y_train_mdl[-1]
        # Change y_train_mdl as a weighted sum of y_train_mdl elements
    
        rslt_invest = calc_train_investment(
            y_train_open, make_submission(y_train_mdl,
                                          buy_q,
                                          sell_t,
                                          profit_thld=p_thld))
        return [buy_q, sell_t, p_thld, weight, rslt_invest[0], rslt_invest[1]]

    rslt_invest = calc_train_investment(
        y_train_open, make_submission(y_train_mdl,
                                      buy_q,
                                      sell_t,
                                      profit_thld=p_thld))
    
    return [buy_q, sell_t, p_thld, rslt_invest[0], rslt_invest[1]]


def calc_train_investment(df_y_open_true, subm):
    money = 10000
    for _idx, _val in subm.iterrows():
        money = money*_val.buy_quantity*\
            df_y_open_true.loc[_val.sell_time, _val.sample_id]*(0.9995)**2 + \
            money*(1-_val.buy_quantity)
    num_invest = len(subm[subm.buy_quantity!=0])
    
    return money, num_invest


def arima_prediction(df_open):
    import warnings
    
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    
    warnings.simplefilter('ignore', ConvergenceWarning) 
    # Ignore convergence warning
    
    p = [1, 2, 4, 6, 8]
    d = q = range(0, 2)
    params_arima = list(it.product(p,d,q))
    
    combs = {}
    aics = []
    
    for i, param in enumerate(params_arima):
        try:
            m = ARIMA(df_open, 
                      order=param,
                      enforce_invertibility=False,
                      enforce_stationarity=False)
            m_fit = m.fit()
            combs.update({m_fit.aic : param})
            aics.append(m_fit.aic)
            
        except: continue
        
    m_arima_best_aic_idx = min(aics)        
    m_arima = ARIMA(df_open,
                    order=combs[m_arima_best_aic_idx],
                    enforce_invertibility=False,
                    enforce_stationarity=False)
    m_arima_fit = m_arima.fit()            
    
    return m_arima_fit.forecast(120)
'''
