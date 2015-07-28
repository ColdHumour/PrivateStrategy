import pandas as pd
import numpy as np
from heapq import nsmallest, nlargest
from copy import deepcopy

import seaborn
from matplotlib import pylab

import quartz
from quartz.api import *

def mixed_universe(index='HS300', start=2010, end=2015):
    universe = set([])
    for y in range(start, end+1):
        universe = universe.union(set_universe(index, '{0}-01-01'.format(y)))
    return sorted(list(universe))

start = '2011-01-01'
end   = '2015-07-20'
benchmark = 'HS300'
# universe = mixed_universe('HS300')
universe = set_universe('HS300')
capital_base = 20000.

sim_params = quartz.sim_condition.env.SimulationParameters(start, end, benchmark, universe, capital_base)
idxmap_all, data_all = quartz.sim_condition.data_generator.get_daily_data(sim_params)


# Position Specifying

class HoldingDaysPosition:
    from heapq import nsmallest, nlargest
    
    def __init__(self, maxSecNum=10):
        self.maxSecNum = maxSecNum  # 最大证券数量
        self.goodShelf = {}         # 现持有证券
        self.wareHouse = {}         # 待买入证券
    
    def refresh_position(self, account, buylist):
        """
        根据股票持仓判断前一交易日的指令成交情况，并更新goodShelf和wareHouse中的数据
        buylist的格式为: [(stock1, days1), (stocks2, days2), ...]
        """
        
        refPrxMap = account.referencePrice
        refSecPos = account.valid_secpos
        
        # 1. 更新倒计时
        for sec in self.goodShelf:
            if sec in account.universe:
                self.goodShelf[sec] -= 1
        for sec in self.wareHouse.keys():
            if sec in account.universe:
                self.wareHouse[sec] -= 1
            if self.wareHouse[sec] <= 0:
                del self.wareHouse[sec]
        
        # 2. 更新goodShelf
        for sec, n in self.goodShelf.items():
            if sec not in refSecPos:
                if n > 0:
                    self.wareHouse[sec] = n
                del self.goodShelf[sec]
            
        # 3. 根据buylist更新goodShelf和wareHouse
        for sec, n in buylist[:]:
            if sec in self.goodShelf:
                self.goodShelf[sec] = n
                buylist.remove((sec, n))
            elif sec in self.wareHouse:
                del self.wareHouse[sec]
                
        return buylist

    def rebalance_freecash(self, account, buylist):
        """
        根据剩余资金进行下单
        """
        
        refPrxMap = account.referencePrice
        refSecPos = account.valid_secpos
        
        # 1. 根据goodShelf卖出
        free_cash = account.cash
        for sec, n in self.goodShelf.items():
            if n <= 0:
                free_cash += refPrxMap[sec] * refSecPos[sec] 
                order_to(sec, 0)
        
        # 2. 确定buylist
        b = int(self.maxSecNum * free_cash / account.referencePortfolioValue)
        supplement = [sec for sec in self.wareHouse if sec in account.universe]
        # 1) 没有调仓空间，buylist全部添加进wareHouse
        if b == 0:
            for sec, n in buylist:
                self.wareHouse[sec] = n
            return
        # 2) 没有可买股票
        elif len(buylist) + len(supplement) == 0:
            return
        # 3) buylist足够，多的添加进wareHouse
        elif len(buylist) >= b:
            for sec, n in buylist[b:]:
                self.wareHouse[sec] = n
            buylist = buylist[:b]
        # 4) buylist不够，但是加上仓库的足够
        elif len(buylist) + len(supplement) >= b:
            supplement = nlargest(b-len(buylist), supplement, key=self.wareHouse.get)
            for sec in supplement:
                buylist.append((sec, self.wareHouse[sec]))
                del self.wareHouse[sec]
        # 5) 加上仓库的也不够
        else:
            for sec in supplement:
                if sec in account.universe:
                    buylist.append((sec, self.wareHouse[sec]))
                    del self.wareHouse[sec]
        
        # 3. 买入
        symbols, amount, c = zip(*buylist)[0], {}, free_cash
        # 1) 尝试平均买入
        for sec in symbols:
            a = int(c / len(symbols) / refPrxMap[sec]) / 100 * 100
            amount[sec] = a
            free_cash -= a * refPrxMap[sec]

        # 2) 最大限度利用现金
        while free_cash > min(map(refPrxMap.get, symbols)) * 100:
            for sec in sorted(symbols, key=amount.get):
                if free_cash > 100 * refPrxMap[sec]:
                    amount[sec] += 100
                    free_cash -= 100 * refPrxMap[sec]

        # 3) 下单，钱不够的添加进wareHouse
        for sec, n in buylist:
            if amount[sec]:
                self.goodShelf[sec] = n
                order(sec, amount[sec])
            else:
                self.wareHouse[sec] = n
    
    def rebalance_equalweight(self, account, buylist):
        """
        等权重买入剩余已有证券+买入列表
        """
        
        refPrxMap = account.referencePrice
        refSecPos = account.valid_secpos

        # 1. 根据goodShelf卖出
        b = len(self.goodShelf)
        for sec, n in self.goodShelf.items():
            if n <= 0:
                b -= 1
                order_to(sec, 0)

        # 2. 确定buylist
        b = max_n - b
        supplement = [sec for sec in self.wareHouse if sec in account.universe]
        # 1) 没有调仓空间，buylist全部添加进wareHouse
        if b == 0:
            for sec, n in buylist:
                self.wareHouse[sec] = n
            return
        # 2) 没有可买股票
        elif len(buylist) + len(supplement) == 0:
            return
        # 3) buylist足够，多的添加进wareHouse
        elif len(buylist) >= b:
            for sec, n in buylist[b:]:
                self.wareHouse[sec] = n
            buylist = buylist[:b]
        # 4) buylist不够，但是加上仓库的足够
        elif len(buylist) + len(supplement) >= b:
            supplement = nlargest(b-len(buylist), supplement, key=self.wareHouse.get)
            for sec in supplement:
                buylist.append((sec, self.wareHouse[sec]))
                del self.wareHouse[sec]
        # 4) buylist不够，加上仓库的也不够
        else:
            for sec in supplement:
                if sec in account.universe:
                    buylist.append((sec, self.wareHouse[sec]))
                    del self.wareHouse[sec]
        
        # 3. 买入
        symbols, amount, v = list(zip(*buylist)[0]), {}, account.referencePortfolioValue
        # 1) 补完调仓列表
        for sec, n in self.goodShelf.items():
            if n > 0:
                symbols.append(sec)
        
        # 2) 计算调仓数额
        for sec in symbols:
            amount[sec] = int(v / len(symbols) / refPrxMap[sec]) / 100 * 100

        # 3) 先卖出
        for sec, n in self.goodShelf.items():
            if n > 0:
                order_to(sec, amount[sec])
        
        # 4) 后买入
        for sec, n in buylist:
            if amount[sec]:
                self.goodShelf[sec] = n
                order(sec, amount[sec])
            else:
                self.wareHouse[sec] = n


# Backtest Version

# Plan A

refresh_rate = 1

max_t = {1: 10, 2: 5, 3: 5}   # 持仓时间
max_n = 10                    # 持仓数量
v_thres = 4                   # 交易量倍数
r_thres = 0.05                # 收益率上限

def initialize(account):
    account.myPos = HoldingDaysPosition(max_n)
    
def handle_data(account):
    # 1. 计算过去三个月和一个月的累计收益
    opn = account.get_attribute_history('openPrice', 60)
    prx = account.get_attribute_history('closePrice', 60)
    uret, dret = {}, {}
    for sec, p in prx.items():
        if sec in account.universe and len(filter(None, opn[sec])) >= 40 and\
           all(opn[sec][-5:]) and not np.isnan(p[-1]) and \
           not np.isnan(p[0]) and not np.isnan(p[-20]):
            uret[sec] = p[-1] / p[-20]
            dret[sec] = p[-1] / p[0]
    
    buylist = []
    
    # 2. 判断是否进入反转期，并获得反转买单
    down = filter(lambda x: x < 1, dret.values())
    dpct = 1.*len(down)/len(dret)
    if dpct > 0.75:
        buylist = nsmallest(max_n, dret, key=dret.get)
        signal  = 1
    
    # 3. 判断是否进入惯性期，并获得惯性买单
    up = filter(lambda x: x > 1, uret.values())
    upct, rbar = 1.*len(up)/len(uret), sum(up)/len(up)
    if 0.5 < upct < 0.75 and rbar < 1.1 and not buylist:
        buylist = nlargest(max_n, uret, key=uret.get)
        signal  = 2
    
    # 4. 判断是否存在交易量择时机会，并获得交易量择时买单
    tv = account.get_attribute_history('turnoverVol', 80)
    volmap = {}
    for sec,ts in tv.items():
        if ts[-1]:
            ts = filter(None, ts)
            v = 1. * sum(ts[:-1]) / (len(ts) - 1)
            r = account.referenceReturn[sec]
            if len(ts) >= 60 and ts[-1] >= v_thres * v and 0 < r < r_thres:
                volmap[sec] = 1.*ts[-1]/v
    if not buylist:
        buylist = nlargest(max_n, volmap, key=volmap.get)
        signal  = 3
    
    buylist = [(sec, max_t[signal]) for sec in buylist]
    account.myPos.refresh_position(account, buylist)
    account.myPos.rebalance_equalweight(account, buylist)
        
strategy = quartz.sim_condition.strategy.TradingStrategy(initialize, handle_data)        
bt, acct = quartz.quick_backtest(sim_params, strategy, idxmap_all, data_all, refresh_rate = refresh_rate)
perf = quartz.perf_parse(bt, acct)

out_keys = ['annualized_return', 'volatility', 'information_ratio', 
            'sharpe', 'max_drawdown', 'alpha', 'beta']
print '\nHybrid Regime Switch Performance:'
for k in out_keys:
    print '    %s%.2f' % (k + ' '*(20-len(k)), perf[k])
print '\n'

fig = pylab.figure(figsize=(10, 5))
perf['cumulative_returns'].plot()
perf['benchmark_cumulative_returns'].plot()
pylab.legend(['Hybrid Regime Switch', 'HS300'], loc='upper left')


"""
Hybrid Regime Switch Performance:
    annualized_return   0.69
    volatility          0.27
    information_ratio   1.84
    sharpe              2.51
    max_drawdown        0.30
    alpha               0.50
    beta                0.61
"""



# Plan B

refresh_rate = 1

max_t = {1: 10, 2: 5, 3: 5}   # 持仓时间
max_n = 10                    # 持仓数量
v_thres = 4                   # 交易量倍数
r_thres = 0.05                # 收益率上限

def initialize(account):
    account.myPos = HoldingDaysPosition(max_n)
    
def handle_data(account):
    # 1. 计算过去三个月和一个月的累计收益
    opn = account.get_attribute_history('openPrice', 60)
    prx = account.get_attribute_history('closePrice', 60)
    uret, dret = {}, {}
    for sec, p in prx.items():
        if sec in account.universe and len(filter(None, opn[sec])) >= 40 and\
           all(opn[sec][-5:]) and not np.isnan(p[-1]) and \
           not np.isnan(p[0]) and not np.isnan(p[-20]):
            uret[sec] = p[-1] / p[-20]
            dret[sec] = p[-1] / p[0]
    
    buylist = []
    
    # 2. 判断是否进入反转期，并获得反转买单
    down = filter(lambda x: x < 1, dret.values())
    dpct = 1.*len(down)/len(dret)
    if dpct > 0.75:
        buylist = nsmallest(max_n, dret, key=dret.get)
        signal  = 1
    
    # 3. 判断是否进入惯性期，并获得惯性买单
    up = filter(lambda x: x > 1, uret.values())
    upct, rbar = 1.*len(up)/len(uret), sum(up)/len(up)
    if 0.5 < upct < 0.75 and rbar < 1.1 and not buylist:
        buylist = nlargest(max_n, uret, key=uret.get)
        signal  = 2
    
    # 4. 判断是否存在交易量择时机会，并获得交易量择时买单
    tv = account.get_attribute_history('turnoverVol', 80)
    volmap = {}
    for sec,ts in tv.items():
        if ts[-1]:
            ts = filter(None, ts)
            v = 1. * sum(ts[:-1]) / (len(ts) - 1)
            r = account.referenceReturn[sec]
            if len(ts) >= 60 and ts[-1] >= v_thres * v and 0 < r < r_thres:
                volmap[sec] = 1.*ts[-1]/v
    if not buylist:
        buylist = nlargest(max_n, volmap, key=volmap.get)
        signal  = 3
    
    buylist = [(sec, max_t[signal]) for sec in buylist]
    account.myPos.refresh_position(account, buylist)
    account.myPos.rebalance_freecash(account, buylist)
        
strategy = quartz.sim_condition.strategy.TradingStrategy(initialize, handle_data)        
bt, acct = quartz.quick_backtest(sim_params, strategy, idxmap_all, data_all, refresh_rate = refresh_rate)
perf = quartz.perf_parse(bt, acct)

out_keys = ['annualized_return', 'volatility', 'information_ratio', 
            'sharpe', 'max_drawdown', 'alpha', 'beta']
print '\nHybrid Regime Switch Performance:'
for k in out_keys:
    print '    %s%.2f' % (k + ' '*(20-len(k)), perf[k])
print '\n'

fig = pylab.figure(figsize=(10, 5))
perf['cumulative_returns'].plot()
perf['benchmark_cumulative_returns'].plot()
pylab.legend(['Hybrid Regime Switch', 'HS300'], loc='upper left')


"""
Hybrid Regime Switch Performance:
    annualized_return   0.52
    volatility          0.35
    information_ratio   1.22
    sharpe              1.47
    max_drawdown        0.38
    alpha               0.40
    beta                0.74
"""





# Daily Version

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
    if len(filter(None, opn_all[i])) >= 54 and not np.isnan(p[-1]) and \
       all(opn_all[i][-5:]) and not np.isnan(p[0]) and not np.isnan(p[-20]):
        uret[stock] = p[-1] / p[-20]
        dret[stock] = p[-1] / p[0]
        prx[stock] = p[-1]
    
    ts = vol_all[i]
    if not ts[-1]:
        continue
    
    ts = filter(None, ts)
    if len(ts) < 60: 
        continue
    
    v = 1. * sum(ts[:-1]) / (len(ts) - 1)
    r = p[-1] / p[-2] - 1.
    if len(ts) >= 60 and ts[-1] >= v_thres * v and 0 < r < r_thres:
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

buylist = [(stock, max_t[signal]) for stock in buylist]    
    
print '\nbuylist:'
for stock, n in buylist:
    print stock, n



cash = 30000.

position = {
}

goodShelf = {
}

wareHouse = {  
}
    
# 1. 更新倒计时
for sec in goodShelf:
    if sec in prx:
        goodShelf[sec] -= 1
for sec in wareHouse.keys():
    if sec in prx:
        wareHouse[sec] -= 1
    if wareHouse[sec] <= 0:
        del wareHouse[sec]

# 2. 根据position更新goodShelf
for sec, n in goodShelf.items():
    if sec not in position:
        if n > 0:
            wareHouse[sec] = n
        del goodShelf[sec]

# 3. 根据buylist更新goodShelf和wareHouse
for sec, n in buylist[:]:
    if sec in goodShelf:
        goodShelf[sec] = n
        buylist.remove((sec, n))
    elif sec in wareHouse:
        del wareHouse[sec]

# 4. 根据goodShelf卖出
amount, v = {}, cash
b = len(goodShelf)
for sec, n in goodShelf.items():
    if n <= 0:
        b -= 1
        amount[sec] = 0
        v += position[sec] * prx[sec]

# 5. 确定buylist
b = max_n - b
flag = 1
supplement = [sec for sec in wareHouse if sec in prx]
# 1) 没有调仓空间，buylist全部添加进wareHouse
if b == 0:
    for sec, n in buylist:
        wareHouse[sec] = n
    flag = 0
# 2) 没有可买股票
elif len(buylist) + len(supplement) == 0:
    flag = 0
# 3) buylist足够，多的添加进wareHouse
elif len(buylist) >= b:
    for sec, n in buylist[b:]:
        wareHouse[sec] = n
    buylist = buylist[:b]
# 4) buylist不够，但是加上仓库的足够
elif len(buylist) + len(supplement) >= b:
    supplement = nlargest(b-len(buylist), supplement, key=wareHouse.get)
    for sec in supplement:
        buylist.append((sec, wareHouse[sec]))
        del wareHouse[sec]
# 4) buylist不够，加上仓库的也不够
else:
    for sec in supplement:
        if sec in account.universe:
            buylist.append((sec, wareHouse[sec]))
            del wareHouse[sec]

# 6. 买入
symbols = list(zip(*buylist)[0]) if buylist else []
for sec, n in position.items():
    v += n * prx[sec]

if flag or amount:
    # 1) 补完调仓列表
    for sec, n in goodShelf.items():
        if n > 0:
            symbols.append(sec)

    # 2) 计算调仓数额
    for sec in symbols:
        amount[sec] = int(v / len(symbols) / prx[sec]) / 100 * 100

    for sec, n in buylist:
        if amount[sec]:
            goodShelf[sec] = n
        else:
            wareHouse[sec] = n

print "OPERATIONS:\n"
for stock,a in amount.items():
    print '%s: %d' % (stock, amount[stock]-position.get(stock, 0))
    position[stock] = a
    
print "\n\nSTATUS:"
print "\nposition = {"
for stock, a in position.items():
    print '    \'%s\': %d,' % (stock, a)
print '}'

print "\ngoodShelf = {"
for stock, a in goodShelf.items():
    print '    \'%s\': %s,' % (stock, a)
print '}'
    
print "\nwareHouse = {"
for stock, a in wareHouse.items():
    print '    \'%s\': %d,' % (stock, a)
print '}'