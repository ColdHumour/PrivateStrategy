import pandas as pd
import numpy as np
from heapq import nsmallest
from copy import deepcopy

import seaborn
from matplotlib import pylab

import quartz
from quartz.api import *

start = '2010-01-01'
end   = '2015-03-01'
benchmark = 'HS300'
universe = set_universe('HS300')
capital_base = 20000.

sim_params = quartz.sim_condition.env.SimulationParameters(start, end, benchmark, universe, capital_base)
idxmap_all, data_all = quartz.sim_condition.data_generator.get_daily_data(sim_params)

longest_history = 60

max_t = 10     # 持仓时间
max_n = 10     # 持仓数量

def initialize(account):
    account.hold_days = {}
    account.free_cash = 0.
    account.to_sell = set([])
    
def handle_data(account):
    prxmap = account.get_attribute_history('closePrice', longest_history)
    retmap = {}
    for stock, p in prxmap.items():
        if stock in account.universe and np.isnan(p).sum() <= longest_history * 0.33 and len(p) >= longest_history * 0.67:
            retmap[stock] = p[-1] / p[0]
        prxmap[stock] = p[-1]
        
    buylist = nsmallest(max_n, retmap.keys(), key=retmap.get)        
    rebalance(account, buylist, prxmap)

def rebalance(account, buylist, prxmap):
    account.free_cash = account.cash
    
    # 卖出前一日无法卖出的
    for stock in deepcopy(account.to_sell):
        if stock in account.valid_secpos:
            order_to(stock, 0)
            account.free_cash += prxmap.get(stock, 0) * account.valid_secpos[stock]
            if stock in buylist:
                buylist.remove(stock)
        else:
            account.to_sell.remove(stock)
    
    # 更新前一日买到的
    for stock in account.valid_secpos:
        if stock not in account.to_sell.union(account.hold_days):
            account.hold_days[stock] = 1
    
    # 卖出当日应卖出的
    for stock in account.hold_days.keys():
        if account.hold_days[stock] == max_t:
            if stock in buylist:
                account.hold_days[stock] = 1
                buylist.remove(stock)
            else:
                account.free_cash += prxmap.get(stock, 0) * account.valid_secpos[stock]
                order_to(stock, 0)
                account.to_sell.add(stock)
                del account.hold_days[stock]
        else:
            if stock in buylist:
                buylist.remove(stock)
            account.hold_days[stock] += 1
    
    # 买入当日应买入的
    n = max_n - len(account.hold_days)
    if n <= 0: return
    
    buylist = buylist[:n]
    account.free_cash *= 0.9
    if not buylist or account.free_cash < min(map(prxmap.get, buylist)) * 100: return
    
    exp_amount, c = {}, account.free_cash
    for stock in buylist:     
        a = int(c / len(buylist) / prxmap[stock]) / 100 * 100
        exp_amount[stock] = a
        account.free_cash -= a * prxmap[stock]

    while account.free_cash > min(map(prxmap.get, buylist)) * 100:
        for stock in sorted(buylist, key=exp_amount.get):
            if account.free_cash > 100 * prxmap[stock]:
                exp_amount[stock] += 100
                account.free_cash -= 100 * prxmap[stock]
    
    for stock,a in exp_amount.items():
        order(stock, a)
        
strategy = quartz.sim_condition.strategy.TradingStrategy(initialize, handle_data)        
bt, idxmap, data = quartz.quick_backtest(
    sim_params, strategy, idxmap_all, data_all,
    longest_history = longest_history)
perf = quartz.perf_parse(bt, idxmap, data)

out_keys = ['annualized_return', 'volatility', 'information_ratio', 
            'sharpe', 'max_drawdown', 'alpha', 'beta']
print '\nSimple Mean Reversion Performance:'
for k in out_keys:
    print '    %s%.2f' % (k + ' '*(20-len(k)), perf[k])
print '\n'

fig = pylab.figure(figsize=(12, 6))
perf['cumulative_returns'].plot()
perf['benchmark_cumulative_returns'].plot()
pylab.legend(['Mean Reversion', 'HS300'], loc=1)