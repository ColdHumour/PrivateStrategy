from heapq import nsmallest

start = datetime(2010, 1, 1)
end   = datetime(2015, 2, 1)
benchmark = 'HS300'
universe = set_universe('HS300')
capital_base = 20000.

window = 60
max_t = 10
max_n = 10
ls_pct = 0.9

def initialize(account):
    add_history('hist', window)
    account.hold_days = {}
    account.hold_cost = {}
    account.free_cash = 0.
    account.n = 0
    account.max_v = 0.

def handle_data(account):
    flag = loss_stop(account)
    if flag: return
    
    prxmap, retmap = {}, {}
    test = {}
    for stock in account.universe:
        prx = account.hist[stock]['closePrice']
        if len(prx) >= window * 0.67:
            prxmap[stock] = prx.iloc[-1]
            retmap[stock] = prx.iloc[-1] / prx.iloc[0]
            test[stock] = prx.iloc[-1] / prx.iloc[-5]
    
    buylist = nsmallest(max_n, retmap.keys(), key=retmap.get)
    smart_sell(account, buylist, prxmap)
    smart_buy(account, buylist, prxmap)

def loss_stop(account):
    v = account.cash
    for stock,a in account.stkpos.items():
        if not a: continue
        v += account.hist[stock]['closePrice'].iloc[-1] * a
    account.max_v = max(account.max_v, v)
    if v / account.max_v < ls_pct:
        account.max_v *= ls_pct
        for stock,a in account.stkpos.items():
            if a: order_to(stock, 0)
        return True
    return False
    
def smart_sell(account, buylist, prxmap):
    account.free_cash = account.cash 
    account.n = 0
    for stock in account.hold_days.keys():
        if stock in buylist:
            if account.stkpos.get(stock, 0) > 0:  # 刷新状态
                account.hold_days[stock] = 1
                account.hold_cost[stock] = prxmap[stock]
                
        if account.hold_days[stock] == max_t:
            if account.stkpos.get(stock, 0) > 0:  # 到期卖出
                order_to(stock, 0)
                account.free_cash += prxmap.get(stock, 0) * account.stkpos.get(stock, 0)
            else:                                 # 更新状态
                del account.hold_days[stock]
                del account.hold_cost[stock]
        else:                                     # 继续持有
            account.hold_days[stock] += 1
            account.n += 1

def smart_buy(account, buylist, prxmap):
    n = max_n - account.n
    if n <= 0: return
    
    buylist = [x for x in buylist if account.hold_days.get(x, max_t) == max_t][:n]
    if account.free_cash <= min(map(prxmap.get, buylist)) * 100: return
    
    amount = dict.fromkeys(buylist, 0)
    
    # 首次尝试平均买入
    cash_now = account.free_cash
    for stock in buylist:     
        a = int(cash_now / len(buylist) / prxmap[stock]) / 100 * 100
        amount[stock] += a
        account.free_cash -= a * prxmap[stock]

    # 资金使用最大化
    while account.free_cash >= min(map(prxmap.get, buylist)) * 100:
        for stock in sorted(buylist, key=amount.get):
            if account.free_cash < 100 * prxmap[stock]: continue
            amount[stock] += 100
            account.free_cash -= 100 * prxmap[stock]
    
    # 下单
    for stock, i in amount.items():
        if not i: continue
        order_to(stock, i)
        account.hold_days[stock] = 1
        account.hold_cost[stock] = prxmap[stock]