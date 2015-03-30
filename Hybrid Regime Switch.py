import pandas as pd
import numpy as np
from heapq import nsmallest, nlargest
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

# Simple Version

longest_history = 60
refresh_rate = 5

max_n = 10     # 持仓数量

def initialize(account):
    account.to_sell = set([])
    
def handle_data(account):
    prx = account.get_attribute_history('closePrice', 60)
    uret, dret = {}, {}
    for stock, p in prx.items():
        if stock in account.universe and np.isnan(p).sum() <= longest_history * 0.33 and len(p) >= longest_history * 0.67:
            uret[stock] = p[-1] / p[-20]
            dret[stock] = p[-1] / p[0]
    
    buylist = []
    
    down = [v for v in dret.values() if v < 1]
    dpct = 1.*len(down)/len(dret)
    if dpct > 0.75:
        buylist += nsmallest(max_n, dret, key=dret.get)            
    else:
        up = [v for v in uret.values() if v > 1]
        upct = 1.*len(up)/len(uret)
        rbar = sum(up)/len(up)
        if 0.5 < upct < 0.75 and rbar < 1.1:
            buylist += nlargest(max_n, uret, key=uret.get)

    rebalance(account, buylist)

def rebalance(account, buylist):
    for stock in account.valid_secpos:
        order_to(stock, 0)
    
    v = c = account.referencePortfolioValue
    p = account.referencePrice
    
    if not buylist: return
    
    exp_amount = {}
    for stock in buylist:
        a = int(v/len(buylist)/p[stock]/100)*100
        exp_amount[stock] = a
        c -= p[stock] * a
        
    while c > min(map(p.get, buylist)) * 100:
        for stock in sorted(buylist, key=exp_amount.get):
            if c > 100 * p[stock]:
                exp_amount[stock] += 100
                c -= p[stock] * 100
    
    for stock,a in exp_amount.items():
        order(stock, a)
    
strategy = quartz.sim_condition.strategy.TradingStrategy(initialize, handle_data)        
bt, acct = quartz.quick_backtest(
    sim_params, strategy, idxmap_all, data_all,
    refresh_rate = refresh_rate,
    longest_history = longest_history)
perf = quartz.perf_parse(bt, acct)



# More Detailed Version

longest_history = 60

max_t = 5      # 持仓时间
max_n = 10     # 持仓数量

def initialize(account):
    account.hold_days = {}
    account.free_cash = 0.
    account.to_sell = set([])
    
def handle_data(account):
    prx = account.get_attribute_history('closePrice', 60)
    uret, dret = {}, {}
    for stock, p in prx.items():
        if stock in account.universe and np.isnan(p).sum() <= longest_history * 0.33 and len(p) >= longest_history * 0.67:
            uret[stock] = p[-1] / p[-20]
            dret[stock] = p[-1] / p[0]
    
    buylist = []
    
    down = [v for v in dret.values() if v < 1]
    dpct = 1.*len(down)/len(dret)
    if dpct > 0.75:
        buylist += nsmallest(max_n, dret.keys(), key=dret.get)            
    else:
        up = [v for v in uret.values() if v > 1]
        upct = 1.*len(up)/len(uret)
        rbar = sum(up)/len(up)
        if 0.5 < upct < 0.75 and rbar < 1.1:
            buylist += nlargest(max_n, uret.keys(), key=uret.get)

    rebalance(account, buylist)

def rebalance(account, buylist):
    p = account.referencePrice
    account.free_cash = account.cash
    
    # 卖出前一日无法卖出的
    for stock in deepcopy(account.to_sell):
        if stock in account.valid_secpos:
            order_to(stock, 0)
            account.free_cash += p.get(stock, 0) * account.valid_secpos[stock]
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
        if account.hold_days[stock] >= max_t:
            if stock in buylist:
                account.hold_days[stock] = 1
                buylist.remove(stock)
            else:
                account.free_cash += p.get(stock, 0) * account.valid_secpos[stock]
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
    if not buylist or account.free_cash < min(map(p.get, buylist)) * 100: return
    
    exp_amount, c = {}, account.free_cash
    for stock in buylist:     
        a = int(c / len(buylist) / p[stock]) / 100 * 100
        exp_amount[stock] = a
        account.free_cash -= a * p[stock]

    while account.free_cash > min(map(p.get, buylist)) * 100:
        for stock in sorted(buylist, key=exp_amount.get):
            if account.free_cash > 100 * p[stock]:
                exp_amount[stock] += 100
                account.free_cash -= 100 * p[stock]
    
    for stock,a in exp_amount.items():
        order(stock, a)
        
strategy = quartz.sim_condition.strategy.TradingStrategy(initialize, handle_data)        
bt, acct = quartz.quick_backtest(
    sim_params, strategy, idxmap_all, data_all,
    longest_history = longest_history)
perf = quartz.perf_parse(bt, acct)