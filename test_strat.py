import pandas as pd
import yfinance as yf
import datetime
from math import sin, floor
import random
import matplotlib.pyplot as plt
# from get_data import get_stock_data
start = datetime.datetime(2015,1,1)
end = datetime.datetime(2024,10,10)
# data = get_stock_data('QQQ',start , end)
# data = yf.download('QQQ', start, end)
# pd.DataFrame(data).to_csv('QQQ.csv')
def gen_test_data(time_length):
    data_arr = []
    for i in range(time_length):
        new_data = 2+ sin(i/100)
        new_data = new_data*20 + random.randint(1,100)/50 + i/100
        data_arr.append(new_data)
    return data_arr

data = gen_test_data(1000)
t_data = range(0,len(data))
plt.plot(data)
# plt.show()

class Portfolio:
    liquid =0
    shares =0
    actions = []

    def __init__(self, liquid, shares):
        self.liquid = liquid
        self.shares=shares

    def buy(self, nr_shares, price, cycle=0):
        if nr_shares*price > self.liquid:
            nr_shares = floor(self.liquid/price)
            return self.buy(nr_shares, price,cycle)
        self.shares += nr_shares
        self.liquid -= nr_shares*price
        if nr_shares > 0:
            self.liquid -= 10
            print('bought ',nr_shares,'for',price)
            self.store_action('buy',cycle, price, nr_shares)

    def sell(self, nr_shares, price,cycle=0):
        if self.shares < nr_shares:
            nr_shares = self.shares
            return self.sell(nr_shares, price,cycle)
        self.shares -= nr_shares
        self.liquid += nr_shares*price
        if nr_shares > 0:
            self.liquid -= 10
            print('sold ',nr_shares,'for',price)
            self.store_action('sold',cycle, price, nr_shares)
    
    def total(self, curr_price):
        return self.liquid + self.shares*curr_price
    
    def store_action(self,type, cycle, price, quantity):
        action = {
            'type': type,
            'cycle': cycle,
            'price': price,
            'quantity': quantity
        }
        self.actions.append(action)

    def get_actions(self):
        return pd.DataFrame(self.actions)


def test_shares():
    p_test = Portfolio(liquid=1012, shares= 0)

    p_test.sell(10,10)
    assert p_test.liquid == 1006
    assert p_test.shares == 0
        
    p_test.buy(12,100)
    assert p_test.liquid == 0
    assert p_test.shares == 10

# test_shares()
def sig9_sim(data):
    nr_share = 100
    entry_price = data[0]
    lev_price = 3*entry_price
    liquid = 100*lev_price
    p_test = Portfolio(liquid,nr_share)
    p_hodl = Portfolio(liquid,nr_share)
    print('current total',p_test.total(lev_price))
    last_price = lev_price
    period_t = 50
    for i, entry in enumerate(data):
        lev_price = 3*entry
        if i%period_t == 0:
            p_change = (lev_price - last_price)/last_price
            print('p_change',p_change,'prev',last_price,'curr',lev_price)
            last_price = lev_price

            if p_change > 0.09:
                frac_sell = p_change-0.09
                frac_sell = min(frac_sell,0.2)
                nr_shares = int(p_test.shares * frac_sell)
                p_test.sell(nr_shares,lev_price)
            elif p_change < 0.09:
                frac_buy = 0.09 - p_change
                frac_buy = min(frac_buy,0.2)
                nr_shares = int(p_test.shares * frac_buy)
                p_test.buy(nr_shares,lev_price)
    
    print('owned',p_test.shares*lev_price)
    print('liquid',p_test.liquid)
    print('buy hold',liquid+100*lev_price)

    print('done')
# sig9_sim(data)
