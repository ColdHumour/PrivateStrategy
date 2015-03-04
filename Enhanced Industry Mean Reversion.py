import pandas as pd
import numpy as np
from heapq import nsmallest
from collections import defaultdict
from copy import deepcopy

import seaborn
from matplotlib import pylab

import quartz
from quartz.api import *

SEC_IND = {
    '000630.XSHE':1030305, '601857.XSHG':1030302, '002001.XSHE':1030317, '600100.XSHG':1030325, '600252.XSHG':1030317, '600718.XSHG':1030325, '600547.XSHG':1030305, 
    '601866.XSHG':1030319, '601088.XSHG':1030302, '002129.XSHE':1030312, '600900.XSHG':1030318, '002008.XSHE':1030312, '000937.XSHE':1030302, '000069.XSHE':1030320, 
    '600332.XSHG':1030317, '600597.XSHG':1030314, '601166.XSHG':1030321, '000156.XSHE':1030326, '601333.XSHG':1030319, '002375.XSHE':1030307, '600578.XSHG':1030318, 
    '002051.XSHE':1030307, '600741.XSHG':1030311, '002422.XSHE':1030317, '600518.XSHG':1030317, '601928.XSHG':1030326, '000960.XSHE':1030305, '601009.XSHG':1030321, 
    '600535.XSHG':1030317, '600809.XSHG':1030314, '600436.XSHG':1030317, '000413.XSHE':1030312, '000002.XSHE':1030320, '000869.XSHE':1030314, '601288.XSHG':1030321, 
    '600887.XSHG':1030314, '000402.XSHE':1030320, '000538.XSHE':1030317, '002252.XSHE':1030317, '002470.XSHE':1030303, '000800.XSHE':1030311, '603288.XSHG':1030314, 
    '002038.XSHE':1030317, '601258.XSHG':1030311, '600104.XSHG':1030311, '601098.XSHG':1030326, '000001.XSHE':1030321, '600352.XSHG':1030303, '002653.XSHE':1030317, 
    '300058.XSHE':1030326, '601600.XSHG':1030305, '002673.XSHE':1030322, '600011.XSHG':1030318, '600549.XSHG':1030305, '600588.XSHG':1030325, '600705.XSHG':1030322, 
    '600837.XSHG':1030322, '002294.XSHE':1030317, '002415.XSHE':1030325, '601633.XSHG':1030311, '601398.XSHG':1030321, '002570.XSHE':1030314, '600309.XSHG':1030303, 
    '002399.XSHE':1030317, '601607.XSHG':1030317, '000061.XSHE':1030323, '600583.XSHG':1030302, '000581.XSHE':1030311, '000559.XSHE':1030311, '600221.XSHG':1030319, 
    '601390.XSHG':1030307, '000983.XSHE':1030302, '601668.XSHG':1030307, '000776.XSHE':1030322, '000793.XSHE':1030326, '600188.XSHG':1030302, '300251.XSHE':1030326, 
    '000423.XSHE':1030317, '300146.XSHE':1030314, '000858.XSHE':1030314, '600642.XSHG':1030318, '601688.XSHG':1030322, '600867.XSHG':1030317, '600028.XSHG':1030303, 
    '600362.XSHG':1030305, '600633.XSHG':1030326, '600196.XSHG':1030317, '600208.XSHG':1030320, '000625.XSHE':1030311, '002310.XSHE':1030307, '601318.XSHG':1030322, 
    '600783.XSHG':1030322, '300133.XSHE':1030326, '601808.XSHG':1030302, '002142.XSHE':1030321, '603993.XSHG':1030305, '300015.XSHE':1030317, '600703.XSHG':1030312, 
    '600674.XSHG':1030318, '000060.XSHE':1030305, '600115.XSHG':1030319, '000686.XSHE':1030322, '600143.XSHG':1030303, '601800.XSHG':1030307, '601555.XSHG':1030322, 
    '600373.XSHG':1030326, '600649.XSHG':1030320, '002304.XSHE':1030314, '002024.XSHE':1030323, '603000.XSHG':1030326, '002236.XSHE':1030325, '600688.XSHG':1030303, 
    '601225.XSHG':1030302, '601933.XSHG':1030323, '601018.XSHG':1030319, '000166.XSHE':1030322, '601958.XSHG':1030305, '002241.XSHE':1030312, '600068.XSHG':1030307, 
    '002146.XSHE':1030320, '600832.XSHG':1030326, '300070.XSHE':1030318, '600369.XSHG':1030322, '002344.XSHE':1030323, '601216.XSHG':1030303, '600648.XSHG':1030320, 
    '002153.XSHE':1030325, '600111.XSHG':1030305, '000878.XSHE':1030305, '600340.XSHG':1030320, '000338.XSHE':1030311, '002007.XSHE':1030317, '000024.XSHE':1030320, 
    '002450.XSHE':1030303, '000783.XSHE':1030322, '600998.XSHG':1030317, '600066.XSHG':1030311, '000629.XSHE':1030302, '000826.XSHE':1030318, '600489.XSHG':1030305, 
    '601898.XSHG':1030302, '601377.XSHG':1030322, '601699.XSHG':1030302, '600048.XSHG':1030320, '600516.XSHG':1030305, '000027.XSHE':1030318, '600660.XSHG':1030311, 
    '600637.XSHG':1030326, '600395.XSHG':1030302, '601117.XSHG':1030307, '601158.XSHG':1030318, '600166.XSHG':1030311, '600157.XSHG':1030302, '002603.XSHE':1030317, 
    '002081.XSHE':1030307, '600079.XSHG':1030317, '000568.XSHE':1030314, '600267.XSHG':1030317, '002475.XSHE':1030312, '000883.XSHE':1030318, '600315.XSHG':1030303, 
    '600663.XSHG':1030320, '600519.XSHG':1030314, '600880.XSHG':1030326, '000963.XSHE':1030317, '601006.XSHG':1030319, '600018.XSHG':1030319, '000729.XSHE':1030314, 
    '002065.XSHE':1030325, '601929.XSHG':1030326, '002500.XSHE':1030322, '000917.XSHE':1030326, '600664.XSHG':1030317, '600030.XSHG':1030322, '600873.XSHG':1030314, 
    '601336.XSHG':1030322, '600999.XSHG':1030322, '600497.XSHG':1030305, '000598.XSHE':1030318, '601988.XSHG':1030321, '601231.XSHG':1030312, '601998.XSHG':1030321, 
    '600827.XSHG':1030323, '600036.XSHG':1030321, '601111.XSHG':1030319, '600256.XSHG':1030303, '600000.XSHG':1030321, '601899.XSHG':1030305, '002456.XSHE':1030312, 
    '000895.XSHE':1030314, '002230.XSHE':1030325, '600276.XSHG':1030317, '600863.XSHG':1030318, '600023.XSHG':1030318, '600600.XSHG':1030314, '601618.XSHG':1030307, 
    '601168.XSHG':1030305, '002292.XSHE':1030326, '000536.XSHE':1030312, '000623.XSHE':1030317, '600008.XSHG':1030318, '600170.XSHG':1030307, '601628.XSHG':1030322, 
    '000999.XSHE':1030317, '300027.XSHE':1030326, '600058.XSHG':1030323, '601169.XSHG':1030321, '600348.XSHG':1030302, '600804.XSHG':1030326, '600886.XSHG':1030318, 
    '600570.XSHG':1030325, '601901.XSHG':1030322, '601328.XSHG':1030321, '600029.XSHG':1030319, '601601.XSHG':1030322, '600027.XSHG':1030318, '000831.XSHE':1030305, 
    '600109.XSHG':1030322, '600655.XSHG':1030323, '600085.XSHG':1030317, '600009.XSHG':1030319, '000728.XSHE':1030322, '600016.XSHG':1030321, '600271.XSHG':1030325, 
    '000750.XSHE':1030322, '002400.XSHE':1030326, '601939.XSHG':1030321, '002594.XSHE':1030311, '002410.XSHE':1030325, '000725.XSHE':1030312, '600153.XSHG':1030319, 
    '600415.XSHG':1030323, '000503.XSHE':1030326, '002416.XSHE':1030323, '600015.XSHG':1030321, '600383.XSHG':1030320, '000970.XSHE':1030305, '601818.XSHG':1030321, 
    '600795.XSHG':1030318, '601669.XSHG':1030307, '601186.XSHG':1030307, '000792.XSHE':1030303, '600739.XSHG':1030323
}

IND_CON = {
    1030302: ['000937.XSHE', '000983.XSHE', '600188.XSHG', '600348.XSHG', '600583.XSHG', '601699.XSHG', '601088.XSHG', '601857.XSHG', '601808.XSHG', '601898.XSHG', '600395.XSHG', '000629.XSHE', '600157.XSHG', '601225.XSHG'] ,
    1030303: ['000792.XSHE', '600028.XSHG', '600256.XSHG', '600309.XSHG', '600143.XSHG', '600352.XSHG', '600315.XSHG', '002450.XSHE', '600688.XSHG', '002470.XSHE', '601216.XSHG'] ,
    1030305: ['000060.XSHE', '000630.XSHE', '000878.XSHE', '000960.XSHE', '600362.XSHG', '600549.XSHG', '600489.XSHG', '600497.XSHG', '600547.XSHG', '601600.XSHG', '600111.XSHG', '601168.XSHG', '601899.XSHG', '601958.XSHG', '600516.XSHG', '000970.XSHE', '603993.XSHG', '000831.XSHE'] ,
    1030307: ['600170.XSHG', '600068.XSHG', '601390.XSHG', '601186.XSHG', '601668.XSHG', '601618.XSHG', '601117.XSHG', '002310.XSHE', '002081.XSHE', '601669.XSHG', '002375.XSHE', '601800.XSHG', '002051.XSHE'] ,
    1030311: ['000625.XSHE', '000800.XSHE', '600104.XSHG', '600660.XSHG', '600741.XSHG', '600066.XSHG', '000338.XSHE', '600166.XSHG', '000581.XSHE', '601258.XSHG', '002594.XSHE', '601633.XSHG', '000559.XSHE'] ,
    1030312: ['600703.XSHG', '000725.XSHE', '002241.XSHE', '000536.XSHE', '002129.XSHE', '002456.XSHE', '601231.XSHG', '000413.XSHE', '002008.XSHE', '002475.XSHE'] ,
    1030314: ['000568.XSHE', '000729.XSHE', '000858.XSHE', '600519.XSHG', '600600.XSHG', '600809.XSHG', '000895.XSHE', '002304.XSHE', '600887.XSHG', '000869.XSHE', '600873.XSHG', '002570.XSHE', '600597.XSHG', '300146.XSHE', '603288.XSHG'] ,
    1030317: ['600085.XSHG', '600196.XSHG', '000538.XSHE', '000423.XSHE', '000623.XSHE', '002001.XSHE', '600664.XSHG', '000999.XSHE', '600518.XSHG', '002007.XSHE', '601607.XSHG', '600276.XSHG', '600535.XSHG', '002399.XSHE', '002422.XSHE', '600267.XSHG', '600252.XSHG', '002038.XSHE', '002603.XSHE', '000963.XSHE', '600332.XSHG', '600436.XSHG', '002294.XSHE', '600079.XSHG', '002653.XSHE', '600867.XSHG', '002252.XSHE', '600998.XSHG', '300015.XSHE'] ,
    1030318: ['600642.XSHG', '600795.XSHG', '600900.XSHG', '600674.XSHG', '600863.XSHG', '601158.XSHG', '600011.XSHG', '600886.XSHG', '600027.XSHG', '000598.XSHE', '000826.XSHE', '600008.XSHG', '000027.XSHE', '000883.XSHE', '600023.XSHG', '600578.XSHG', '300070.XSHE'] ,
    1030319: ['600009.XSHG', '600029.XSHG', '600153.XSHG', '600221.XSHG', '601006.XSHG', '601111.XSHG', '601333.XSHG', '601866.XSHG', '600115.XSHG', '601018.XSHG', '600018.XSHG'] ,
    1030320: ['000002.XSHE', '000024.XSHE', '000069.XSHE', '000402.XSHE', '600649.XSHG', '600383.XSHG', '600048.XSHG', '600208.XSHG', '002146.XSHE', '600340.XSHG', '600648.XSHG', '600663.XSHG'] ,
    1030321: ['000001.XSHE', '600000.XSHG', '600015.XSHG', '600016.XSHG', '600036.XSHG', '601988.XSHG', '601398.XSHG', '601166.XSHG', '601998.XSHG', '601328.XSHG', '002142.XSHE', '601009.XSHG', '601169.XSHG', '601939.XSHG', '601288.XSHG', '601818.XSHG'] ,
    1030322: ['600030.XSHG', '601628.XSHG', '601318.XSHG', '600837.XSHG', '600109.XSHG', '000686.XSHE', '000728.XSHE', '000783.XSHE', '601601.XSHG', '600369.XSHG', '600999.XSHG', '601688.XSHG', '000776.XSHE', '601377.XSHG', '002500.XSHE', '600783.XSHG', '601901.XSHG', '601555.XSHG', '601336.XSHG', '000750.XSHE', '002673.XSHE', '600705.XSHG', '000166.XSHE'] ,
    1030323: ['000061.XSHE', '600058.XSHG', '600739.XSHG', '002024.XSHE', '600415.XSHG', '600655.XSHG', '601933.XSHG', '002344.XSHE', '600827.XSHG', '002416.XSHE'] ,
    1030325: ['600100.XSHG', '600271.XSHG', '600588.XSHG', '600718.XSHG', '002415.XSHE', '002236.XSHE', '002065.XSHE', '002230.XSHE', '002410.XSHE', '600570.XSHG', '002153.XSHE'] ,
    1030326: ['600832.XSHG', '600804.XSHG', '601098.XSHG', '601928.XSHG', '600637.XSHG', '000156.XSHE', '000793.XSHE', '603000.XSHG', '000503.XSHE', '000917.XSHE', '600633.XSHG', '600880.XSHG', '002292.XSHE', '002400.XSHE', '601929.XSHG', '600373.XSHG', '300027.XSHE', '300058.XSHE', '300133.XSHE', '300251.XSHE']
}

start = '2010-01-01'
end   = '2015-03-01'
benchmark = 'HS300'
universe = SEC_IND.keys()
univ_sh50 = set(set_universe('SH50'))
capital_base = 20000.

sim_params = quartz.sim_condition.env.SimulationParameters(start, end, benchmark, universe, capital_base)
idxmap_all, data_all = quartz.sim_condition.data_generator.get_daily_data(sim_params)

longest_history = 80

mr_pct = 0.25     # 行业均值回归股票百分比
mr_lb = 0.05      # 行业均值回归收益率下界
mr_ub = 0.08      # 行业均值回归收益率上界

zs_window = 80    # 成交量择时历史窗口
zs_r_ub = 0.05    # 成交量单择时收益率上界
zs_v_lb = 4       # 成交量择时交易量倍数下界

max_t = 20        # 持仓时间
max_n = 30        # 持仓数量


def initialize(account):
    account.hold_period = {}

def handle_data(account):
    r = account.referenceReturn
    score = defaultdict(list)
    for stock in account.universe:
        score[SEC_IND[stock]].append(r[stock])
    for ind in score:
        score[ind] = sum(score[ind]) / len(score[ind])
    indlist = [ind for ind, v in score.items() if mr_lb <= v <= mr_ub]
    
    buylist = set([])
    for ind in indlist:
        n = int(mr_pct * len(IND_CON[ind]))
        slist = nsmallest(n, IND_CON[ind], key=r.get)
        buylist = buylist.union(slist)
        buylist = buylist.intersection(account.universe)
    
    if not buylist:
        tv = account.get_attribute_history('turnoverVol', zs_window)
        for stock in univ_sh50.intersection(account.universe):
            v = sum(tv[stock][:-1]) / (zs_window - 1)
            if tv[stock][-1] >= zs_v_lb * v and 0 < r[stock] <= zs_r_ub:
                buylist.add(stock)

    rebalance(account, buylist)
    
def rebalance(account, buylist):
    prxmap = account.referencePrice
    c = account.cash
    n = 0
    for stock in account.hold_period.keys():
        if stock in buylist:
            if stock in account.valid_secpos:  # 刷新状态
                account.hold_period[stock] = 0

        if account.hold_period[stock] == max_t:
            if stock in account.valid_secpos:  # 到期卖出
                order_to(stock, 0)
                c += prxmap[stock] * account.valid_secpos[stock]
            else:                              # 更新状态
                del account.hold_period[stock]
        else:                                  # 继续持有
            account.hold_period[stock] += 1
            n += 1
        
    buylist = list(buylist.difference(account.hold_period.keys()))
    if n == max_n or c < 1000 or not buylist:
        return
    
    b = min(max_n - n, len(buylist))
    for stock in buylist[:b]:
        amount = int(c / b / prxmap[stock]) / 100 * 100
        if amount:
            order_to(stock, amount)
            account.hold_period[stock] = 1

strategy = quartz.sim_condition.strategy.TradingStrategy(initialize, handle_data)        
bt, idxmap, data = quartz.quick_backtest(
    sim_params, strategy, idxmap_all, data_all,
    longest_history = longest_history)
perf = quartz.perf_parse(bt, idxmap, data)

out_keys = ['annualized_return', 'volatility', 'information_ratio', 
            'sharpe', 'max_drawdown', 'alpha', 'beta']
print '\nEnhanced Industry Mean Reversion Performance:'
for k in out_keys:
    print '    %s%.2f' % (k + ' '*(20-len(k)), perf[k])
print '\n'

fig = pylab.figure(figsize=(12, 6))
perf['cumulative_returns'].plot()
perf['benchmark_cumulative_returns'].plot()
pylab.legend(['Enhanced Industry Mean Reversion', 'HS300'], loc=1)



# 另一种rebalance的写法，收益会低一些

def rebalance(account, buylist):
    account.free_cash = account.cash
    prxmap = account.referencePrice
    
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
    
    observe('hold_days', account.hold_days)
    
    # 买入当日应买入的
    n = max_n - len(account.hold_days)
    if n <= 0: return
    
    buylist = list(buylist)[:n]
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