import pandas as pd
import numpy as np
from heapq import nsmallest, nlargest
from copy import deepcopy

import seaborn
from matplotlib import pylab

import quartz
from quartz.api import *

start = '2010-01-01'
end   = '2015-04-01'
benchmark = 'HS300'
universe = set_universe('HS300')
capital_base = 20000.

sim_params = quartz.sim_condition.env.SimulationParameters(start, end, benchmark, universe, capital_base)
idxmap_all, data_all = quartz.sim_condition.data_generator.get_daily_data(sim_params)




# Backtest Version

longest_history = 80
refresh_rate = 1

max_t = {1: 10, 2: 5, 3: 5}   # 持仓时间
max_n = 10                    # 持仓数量
v_thres = 4                   # 交易量倍数
r_thres = 0.05                # 收益率上限

class MyPosition:
    def __init__(self):
        self.buydate = {}
        self.sigtype = {}
    
    def rebalance(self, account, buylist):
        tradeDates = account.get_symbol_history('tradeDate', longest_history)
        effectPrx  = account.get_attribute_history('openPrice', longest_history)
        refPrxMap  = account.referencePrice
        refSecPos  = account.valid_secpos
        dt2ix = dict(zip(tradeDates, range(len(tradeDates))))
        
        # 确认已买入
        for stock in self.buydate.keys():
            if stock not in refSecPos:
                del self.buydate[stock]
                del self.sigtype[stock]

        # 更新已持有
        for stock, signal in buylist[:]:
            if stock in refSecPos:
                self.buydate[stock] = account.current_date
                self.sigtype[stock] = signal
                buylist.remove((stock, signal))

        # 卖出
        free_cash = account.cash
        for stock, dt in self.buydate.items():
            if dt < tradeDates[0]:
                dt = tradeDates[0]
            elif dt > tradeDates[-1]:
                continue
                
            if len(filter(None, effectPrx[stock][dt2ix[dt]:])) >= max_t[self.sigtype[stock]]:
                free_cash += refPrxMap[stock] * refSecPos[stock] 
                order_to(stock, 0)

        # 是否需要买入
        if (not buylist) or (len(refSecPos) > max_n * 0.7) or \
           (free_cash < account.referencePortfolioValue * 0.3):
            return

        # 买入
        n = min(len(buylist), max_n - len(refSecPos))
        buylist = buylist[:n]

        symbols, amount, c = zip(*buylist)[0], {}, free_cash
        for stock in symbols:     
            a = int(c / len(symbols) / refPrxMap[stock]) / 100 * 100
            amount[stock] = a
            free_cash -= a * refPrxMap[stock]

        while free_cash > min(map(refPrxMap.get, symbols)) * 100:
            for stock in sorted(symbols, key=amount.get):
                if free_cash > 100 * refPrxMap[stock]:
                    amount[stock] += 100
                    free_cash -= 100 * refPrxMap[stock]

        for stock, signal in buylist:
            if amount[stock]:
                self.buydate[stock] = account.current_date
                self.sigtype[stock] = signal
                order(stock, amount[stock])
    
def initialize(account):
    account.myPos = MyPosition()
    
def handle_data(account):
    prx = account.get_attribute_history('closePrice', 60)
    uret, dret = {}, {}
    for stock, p in prx.items():
        if stock in account.universe and np.isnan(p).sum() <= longest_history * 0.33 and \
           not np.isnan(p[-1]) and not np.isnan(p[0]) and not np.isnan(p[-20]):
            uret[stock] = p[-1] / p[-20]
            dret[stock] = p[-1] / p[0]
    
    buylist = []
    
    down = filter(lambda x: x < 1, dret.values())
    dpct = 1.*len(down)/len(dret)
    if dpct > 0.75:
        buylist = nsmallest(max_n, dret, key=dret.get)
        signal  = 1
    
    up = filter(lambda x: x > 1, uret.values())
    upct = 1.*len(up)/len(uret)
    rbar = sum(up)/len(up)
    if 0.5 < upct < 0.75 and rbar < 1.1 and not buylist:
        buylist = nlargest(max_n, uret, key=uret.get)
        signal  = 2
    
    tv = account.get_attribute_history('turnoverVol', 80)
    volmap = {}
    for stock,ts in tv.items():
        if ts[-1]:
            ts = filter(None, ts)
            v = 1. * sum(ts[:-1]) / (len(ts) - 1)
            if len(ts) >= 60 and ts[-1] >= v_thres * v and 0 < account.referenceReturn[stock] < r_thres:
                volmap[stock] = 1.*ts[-1]/v
    if not buylist:
        buylist = nlargest(max_n, volmap, key=volmap.get)
        signal  = 3
    
    buylist = [(stock, signal) for stock in buylist]
    account.myPos.rebalance(account, buylist)
        
strategy = quartz.sim_condition.strategy.TradingStrategy(initialize, handle_data)        
bt, acct = quartz.quick_backtest(
    sim_params, strategy, idxmap_all, data_all,
    refresh_rate = refresh_rate, longest_history = longest_history)
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




# PMS Version

import pandas as pd
import numpy as np
from datetime import datetime
from heapq import nsmallest, nlargest

import quartz
from quartz.api import *
from CAL.PyCAL import *



today = Date.todaysDate()
cal = Calendar('China.SSE')
start = cal.advanceDate(today, '-80B', BizDayConvention.Following)
end   = cal.advanceDate(today, '-1B',  BizDayConvention.Following)

start = datetime(start.year(), start.month(), start.dayOfMonth())
end   = datetime(end.year(),   end.month(),   end.dayOfMonth())

print 'start:\t', start
print 'end:  \t', end

trading_days = quartz.utils.tradingcalendar.get_trading_days(start, end)

assert len(trading_days) == 80

universe = set_universe('HS300')
idxmap_univ, idxmap_cols, data_all = quartz.data.load_stocks_data(universe, trading_days)
opn_all = [np.array(d[idxmap_cols['openPrice']])   for d in data_all]
prx_all = [np.array(d[idxmap_cols['closePrice']])  for d in data_all]
vol_all = [np.array(d[idxmap_cols['turnoverVol']]) for d in data_all]

max_t = {1: 10, 2: 5, 3: 5}   # 持仓时间
max_n = 10                    # 持仓数量
v_thres = 4                   # 交易量倍数
r_thres = 0.05                # 收益率上限

dret, uret, volmap, prx = {}, {}, {}, {}
for stock,i in idxmap_univ.items():
    p = prx_all[i]
    if np.isnan(p).sum() <= 20 and not np.isnan(p[-1]) and \
       not np.isnan(p[0]) and not np.isnan(p[-20]):
        dret[stock] = p[-1] / p[0]
        uret[stock] = p[-1] / p[-20]
        prx[stock] = p[-1]
    
    ts = vol_all[i]
    if not ts[-1]:
        continue
    
    ts = filter(None, ts)
    v = 1. * sum(ts[:-1]) / (len(ts) - 1)
    if len(ts) >= 60 and ts[-1] >= v_thres * v and 1 < p[-1]/p[-2] < 1+r_thres:
        volmap[stock] = 1.*ts[-1]/v
        prx[stock] = p[-1]

buylist = []
down = filter(lambda x: x < 1, dret.values())
dpct = 1.*len(down)/len(dret)
if dpct > 0.75:
    buylist = nsmallest(max_n, dret, key=dret.get)            
    signal = 1
    
up = filter(lambda x: x > 1, uret.values())
upct = 1.*len(up)/len(uret)
rbar = sum(up)/len(up)
if 0.5 < upct < 0.75 and rbar < 1.1 and not buylist:
    buylist = nlargest(max_n, uret, key=uret.get)
    signal = 2
    
if not buylist:
    buylist = nlargest(max_n, volmap, key=volmap.get)
    signal = 3

buylist = [(stock, signal) for stock in buylist]    
    
print '\nbuylist:'
for stock, signal in buylist:
    print stock, signal



cash = 20000.

position = {

}

buydate = {

}

sigtype = {

}

tradeDates = trading_days
effectPrx  = opn_all
refPrxMap  = prx
refSecPos  = position
dt2ix = dict(zip(tradeDates, range(len(tradeDates))))

# 更新已持有
for stock, signal in buylist[:]:
    if stock in refSecPos:
        buydate[stock] = trading_days[-1]
        sigtype[stock] = signal
        buylist.remove((stock, signal))

# 卖出
free_cash = portfolio_value = cash
to_sell = []
for stock, dt in buydate.items():
    portfolio_value += refPrxMap[stock] * refSecPos[stock]
    
    if dt < tradeDates[0]:
        dt = tradeDates[0]
    elif dt > tradeDates[-1]:
        continue
    
    if len(filter(None, effectPrx[stock][dt2ix[dt]:])) >= max_t[sigtype[stock]]:
        free_cash += refPrxMap[stock] * refSecPos[stock]
        to_sell.append(stock)

# 买入
if (not buylist) or (len(refSecPos) > max_n * 0.7) or (free_cash < portfolio_value * 0.3):
    amount = {}
else:
    n = min(len(buylist), max_n - len(refSecPos))
    buylist = buylist[:n]

    symbols, amount, c = zip(*buylist)[0], {}, free_cash
    for stock in symbols:     
        a = int(c / len(symbols) / refPrxMap[stock]) / 100 * 100
        amount[stock] = a
        free_cash -= a * refPrxMap[stock]

    while free_cash > min(map(refPrxMap.get, symbols)) * 100:
        for stock in sorted(symbols, key=amount.get):
            if free_cash > 100 * refPrxMap[stock]:
                amount[stock] += 100
                free_cash -= 100 * refPrxMap[stock]

    for stock, signal in buylist:
        if amount[stock]:
            buydate[stock] = trading_days[-1]
            sigtype[stock] = signal
        else:
            amount.remove(stock)
    for stock in to_sell:
        amount[stock] = 0

print "OPERATIONS:\n"
for stock,a in amount.items():
    print '%s: %d' % (stock, amount[stock]-position.get(stock, 0))
    position[stock] = a
    
print "\n\nSTATUS:"
print "\nposition = {"
for stock, a in position.items():
    print '    \'%s\': %d,' % (stock, a)
print '}'

print "\nbuydate = {"
for stock, a in buydate.items():
    print '    \'%s\': %s,' % (stock, repr(a).split('.')[1])
print '}'
    
print "\nsigtype = {"
for stock, a in sigtype.items():
    print '    \'%s\': %d,' % (stock, a)
print '}'