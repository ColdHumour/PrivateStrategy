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