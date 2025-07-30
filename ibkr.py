from ib_insync import *

# Connect to IBKR Gateway or TWS
ib = IB()
ib.connect('127.0.0.1', 666, clientId=666)

# Function to get the portfolio
def get_portfolio():
    portfolio = ib.portfolio()
    for position in portfolio:
        print(f"Contract: {position.contract.symbol}, Quantity: {position.position}, Market Value: {position.marketValue}")

# Function to buy a stock
def buy_stock(symbol, quantity):
    contract = Stock(symbol, 'SMART', 'USD')
    order = MarketOrder('BUY', quantity)
    trade = ib.placeOrder(contract, order)
    print(f"Buy Order placed for {quantity} shares of {symbol}")

# Function to sell a stock
def sell_stock(symbol, quantity):
    contract = Stock(symbol, 'SMART', 'USD')
    order = MarketOrder('SELL', quantity)
    trade = ib.placeOrder(contract, order)
    print(f"Sell Order placed for {quantity} shares of {symbol}")

# Main function to demonstrate the usage
def main():
    # Get and display the portfolio
    print("Current Portfolio:")
    get_portfolio()

    # Example: Buy 10 shares of AAPL
    # buy_stock('AAPL', 10)

    # Example: Sell 5 shares of AAPL
    # sell_stock('AAPL', 5)

    # Disconnect from IBKR
    ib.disconnect()

if __name__ == "__main__":
    main()