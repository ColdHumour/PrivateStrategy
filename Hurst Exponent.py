import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
from heapq import nsmallest, nlargest
from copy import deepcopy

import seaborn
from matplotlib import pylab

import DataAPI
import quartz
from quartz.api import *


def rs(nlist):
    d = len(nlist)
    m = 1. * sum(nlist) / d
    nlist = [n-m for n in nlist]
    z = [sum(nlist[:i+1]) for i in range(d)]
    r = max(z) - min(z)
    nslist = [n*n for n in nlist]
    s = (sum(nslist) / d) ** 0.5
    return r/s

def LinearRegression(y, x):
    y = np.array(y)
    x = np.column_stack((x,))
    x = sm.add_constant(x, prepend=True)
    res = sm.OLS(y,x).fit(method = 'qr')
    return res.params[1]

def hurst(nlist):
    ms = [1, 2, 4, 8, 16, 32, 64]
    ns = [len(nlist) / m for m in ms]
    x = map(math.log, ns)
    y = []
    for m,n in zip(ms, ns):
        temp = [nlist[i*n:(i+1)*n] for i in range(m)]
        out = map(rs, temp)
        out = [s for s in out if not np.isnan(s)]
        out = (math.log(sum(out) / len(out))) if out else np.nan
        y.append(out)
    return LinearRegression(y, x)

def hseries(nlist, window=512, gap=20):
    if window > len(nlist):
        return None
    
    output = []
    for i in range(window, len(nlist), gap):
        output.append(hurst(nlist[i-512:i]))
    return output


start = '2009-01-01'
end   = '2015-05-01'
benchmark = 'HS300'
universe = set_universe('HS300')
capital_base = 200000.

sim_params = quartz.sim_condition.env.SimulationParameters(start, end, benchmark, universe, capital_base)
idxmap_all, data_all = quartz.sim_condition.data_generator.get_daily_data(sim_params)



refresh_rate = 40

def initialize(account):
    account.size = 10
    
def handle_data(account):
    print account.current_date
    
    cls = account.get_attribute_history('closePrice', 632)
    pcl = account.get_attribute_history('preClosePrice', 632)
    
    buylist, sellist = [], []
    for stock in account.universe:
        ret = cls[stock] / pcl[stock] - 1.
        x = hseries(ret)
        d = len(x)
    
        if sum(x[:3])/3 > 0.57 and x[-1] < 0.50:
            m = sum(cls[stock][-120:]) / 120
            if cls[stock][-1] < m:
                buylist.append(stock)
            elif stock in account.valid_secpos:
                sellist.append(stock)
    
    p = len(account.valid_secpos)
    for stock in sellist:
        order_to(stock, 0)
        p -= 1
    
    p += len(buylist)
    if not p: return
    v = account.referencePortfolioValue / p
    for stock in account.valid_secpos:
        if stock not in sellist:
            order_to(stock, v / cls[stock][-1])
    
    for stock in buylist:
        order(stock, v / cls[stock][-1])

    
strategy = quartz.sim_condition.strategy.TradingStrategy(initialize, handle_data)        
bt, acct = quartz.quick_backtest(sim_params, strategy, idxmap_all, data_all, refresh_rate = refresh_rate)
perf = quartz.perf_parse(bt, acct)

out_keys = ['annualized_return', 'volatility', 'information_ratio', 
            'sharpe', 'max_drawdown', 'alpha', 'beta']
print '\nHybrid Regime Switch Performance:'
for k in out_keys:
    print '    %s%.2f' % (k + ' '*(20-len(k)), perf[k])
print '\n'

fig = pylab.figure(figsize=(12, 6))
perf['cumulative_returns'].plot()
perf['benchmark_cumulative_returns'].plot()
pylab.legend(['Hybrid Regime Switch', 'HS300'], loc='upper left')