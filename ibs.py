import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from test_strat import Portfolio
import pandas as pd
import numpy as np
from os.path import expanduser, isfile, join


# import matplotlib.pyplot as plt

# # Sample data
# # Array of 100 points for the continuous line
# y_continuous = np.sin(np.linspace(0, 2 * np.pi, 100))

# # Array of 7 x-coordinates for scatter points
# x_scatter = np.array([1, 5, 8, 32, 67, 78, 67])

# # Array of 7 y-values for scatter points
# y_scatter = np.array([0.5, 0.7, 0.3, 0.9, 0.6, 0.8, 0.4])

# # Plot the continuous line
# plt.figure(figsize=(12, 6))
# plt.plot(y_continuous, label='Continuous Line', color='blue')

# # Plot the scatter points
# plt.scatter(x_scatter, y_scatter, color='red', label='Scatter Points')

# # Add labels and title
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Value')
# plt.title('Continuous Line with Scatter Points')
# plt.legend()
# plt.show()
def get_store_data(ticker,start,end):
    file_name = ticker+'_data'
    file_name=file_name.replace('.','_')+'.csv'
    file_path = join(expanduser('~/Downloads/'),file_name)

    if isfile(file_path):
        df = pd.read_csv(file_path)
    else:
        df = yf.download(ticker, start=start, end=end, progress=False,ignore_tz=True)
        df.dropna(inplace=True)
        
        df.to_csv(file_path)
        df = pd.read_csv(file_path)

    df=df[2:]
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    # Ensure all other columns are of type float
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
    df.reset_index(inplace=True)
    return df

def run_ibs_strategy(ticker='TQQQ', start='2021-01-01', end='2025-06-20',
                     ibs_buy_threshold=0.15, ibs_sell_threshold=0.8):

    # Load data from Yahoo Finance
    # start=datetime.datetime.fromisoformat(start)
    
    df = get_store_data(ticker,start,end)
    df_qqq3=get_store_data('QQQ3.L',start,end)
    df_qqq3 = df_qqq3.rename(columns=lambda x: x + '_qqq3' if x != 'Price' else x)
# Ensure the first column is of type string
    merged_df = pd.merge(df, df_qqq3, on='Price', how='inner')
    start_with =10000

    p_test = Portfolio(liquid=start_with, shares= 0)
    p_hold = Portfolio(liquid=start_with, shares= 0)
    ibs_strat = []
    buy_hold =[]
    buy_actions = []
    sell_actions =[]

    action_at = 'Close'
    action_next = 'nothing'
    start_loc = 700
    last_price = df[action_at].iloc[start_loc]
    p_hold.buy(100000,last_price,0)

    for i, entry in merged_df[start_loc:].iterrows():
        division = (entry['High'] - entry['Low'])
        ibs = 0.5
        if division >0:
            ibs = (entry[action_at] - entry['Low']) / (entry['High'] - entry['Low'])
        # portf = Portfolio(10000,0)
        # curr_price = entry[action_at]
        print( p_test.total(entry['Close']),'liq',p_test.liquid)
        # if action_next == 'buy':
        #     p_test.buy(100000,entry['Open'],i)
        #     print( p_test.total(entry['Open']),'liq',p_test.liquid)
        #     action_next = 'nothing'
        # elif action_next == 'sell':
        #     p_test.sell(100000,entry['Open'],i)
        #     print( p_test.total(entry['Open']),'liq',p_test.liquid)
        #     action_next = 'nothing'
        if i+1 < merged_df.shape[0]:
            curr_price = merged_df['Open'].iloc[i+1]
        buy_hold.append((curr_price-last_price)/last_price)
        
        # Signal: Buy at close if IBS < threshold, Sell (exit) next day at close
        if ibs < ibs_buy_threshold:
            action_next = 'buy'
        if ibs > ibs_sell_threshold:
            action_next = 'sell'
        
        if action_next == 'buy':
            p_test.buy(100000,curr_price,i)
            # print( p_test.total(entry['Close']),'liq',p_test.liquid)
            action_next = 'nothing'
        elif action_next == 'sell':
            p_test.sell(100000,curr_price,i)
            # print( p_test.total(entry['Close']),'liq',p_test.liquid)
            action_next = 'nothing'
        
        final_nr = p_test.total(curr_price)
        final_hold = p_hold.total(entry[action_at])
        p_change =(final_nr-start_with)/start_with
        p_change_hodl =(final_hold-start_with)/start_with

        ibs_strat.append(p_change)
    
    actions = p_test.get_actions()
    buys = actions[actions['type'] == 'buy']
    sells =actions[actions['type'] == 'sold']
    x_scatter = buys['cycle']
    y_scatter = buys['price']*0

    x_scatter_sell = sells['cycle']
    y_scatter_sell = sells['price']*0
    print('end result',p_change,p_test.shares)
    print('end result hodl',p_change_hodl ,final_hold)
    print('buy and hold')

    # df.set_index('Date', inplace=True)
    plt.figure(figsize=(12, 6))
    plt.plot(buy_hold, label='Buy & Hold', color='blue')
    plt.plot(ibs_strat, label='IBS Strategy', color='red')
    # plt.scatter(list(x_scatter), list(y_scatter), color='green', label='buy')
    # plt.scatter(list(x_scatter_sell), list(y_scatter_sell), color='red', label='sell')

    plt.title(f'{ticker} - IBS Strategy vs. Buy & Hold')
    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Close Price and IBS Strategy')
    plt.legend()
    plt.show()
    # Show the plot
    plt.show()

    # return entry[[action_at, 'IBS', 'Position', 'IBS Strategy Return',]()]()

run_ibs_strategy()