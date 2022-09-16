import pandas as pd
import numpy as np
import time
import multiprocessing as mp

# local imports
from backtester import engine, tester
from backtester import API_Interface as api

training_period = 20 # How far the rolling average takes into calculation
standard_deviations = 2 # Number of Standard Deviations from the mean the Bollinger Bands sit

interval = 14

'''
logic() function:
    Context: Called for every row in the input data.

    Input:  account - the account object
            lookback - the lookback dataframe, containing all data up until this point in time

    Output: none, but the account object will be modified on each call
'''




    # lookback is going to be a pandas dataframe e.g. 'Open':[100, 87, 69, 11]
                                                    # 'Volume':[23, 45, 76, 93]
    
def logic(account, lookback): # Logic function to be used for each time interval in backtest 

    last_index = len(lookback)-1

    if(last_index <= training_period): # If the lookback is long enough to calculate the Bollinger Bands
        return 

    close_price = lookback['close'][last_index]
    if(close_price < lookback['BOLD'][last_index]): # If current price is below lower Bollinger Band, enter a long position
        for position in account.positions: # Close all current positions
            account.close_position(position, 1, close_price)
        if(account.buying_power > 0):
            account.enter_position('long', account.buying_power, close_price) # Enter a long position

    if(close_price > lookback['BOLU'][last_index]): # If today's price is above the upper Bollinger Band, enter a short position
        for position in account.positions: # Close all current positions
            account.close_position(position, 1, close_price)
        if(account.buying_power > 0):
                account.enter_position('short', account.buying_power, close_price) # Enter a short position

'''
preprocess_data() function:
    Context: Called once at the beginning of the backtest. TOTALLY OPTIONAL. 
             Each of these can be calculated at each time interval, however this is likely slower.

    Input:  list_of_stocks - a list of stock data csvs to be processed

    Output: list_of_stocks_processed - a list of processed stock data csvs
'''
def preprocess_data(list_of_stocks):
    list_of_stocks_processed = []  # create empty list to append to 
    
    for stock in list_of_stocks:
        df = pd.read_csv("data/" + stock + ".csv", parse_dates=[0])
        df['TP'] = (df['close'] + df['low'] + df['high'])/3 # Calculate Typical Price
        df['std'] = df['TP'].rolling(training_period).std() # Calculate Standard Deviation
        df['MA-TP'] = df['TP'].rolling(training_period).mean() # Calculate Moving Average of Typical Price
        df['BOLU'] = df['MA-TP'] + standard_deviations*df['std'] # Calculate Upper Bollinger Band
        df['BOLD'] = df['MA-TP'] - standard_deviations*df['std'] # Calculate Lower Bollinger Band
        df['sum']=df['close'].cumsum()
        df['-DM'] = df['low'].shift(1) - df['low'] # Directional Movement : previous low minus current low
        df['+DM'] = df['high'] - df['high'].shift(1) # Directional Movement : current high minus previous high 
        df['+DM'] = np.where((df['+DM'] > df['-DM']) & (df['+DM']>0), df['+DM'], 0.0)
        df['-DM'] = np.where((df['-DM'] > df['+DM']) & (df['-DM']>0), df['-DM'], 0.0)

        df['TR_TMP1'] = df['high'] - df['low']
        df['TR_TMP2'] = np.abs(df['high'] - df['close'].shift(1))
        df['TR_TMP3'] = np.abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['TR_TMP1', 'TR_TMP2', 'TR_TMP3']].max(axis=1)
        df['TR'+str(interval)] = df['TR'].rolling(interval).sum()
        df['+DMI'+str(interval)] = df['+DM'].rolling(interval).sum()
        df['-DMI'+str(interval)] = df['-DM'].rolling(interval).sum()
        df['+DI'+str(interval)] = df['+DMI'+str(interval)] /   df['TR'+str(interval)]*100
        df['-DI'+str(interval)] = df['-DMI'+str(interval)] / df['TR'+str(interval)]*100
        df['DI'+str(interval)+'-'] = abs(df['+DI'+str(interval)] - df['-DI'+str(interval)])
        df['DI'+str(interval)] = df['+DI'+str(interval)] + df['-DI'+str(interval)]
        df['DX'] = (df['DI'+str(interval)+'-'] / df['DI'+str(interval)])*100
        df['ADX'] = df['DX'].rolling(interval).mean()
        df['ADX'+str(interval)] =   df['ADX'+str(interval)].fillna(df['ADX'+str(interval)].mean())
        del df['TR_TMP1'], df['TR_TMP2'], df['TR_TMP3'], df['TR'], df['TR'+str(interval)]
        del df['+DMI'+str(interval)], df['DI'+str(interval)+'-']
        del df['DI'+str(interval)], df['-DMI'+str(interval)]
        del df['+DI'+str(interval)], df['-DI'+str(interval)]
        del df['DX']

        df.to_csv("data/" + stock + "_Processed_test.csv", index=False) # Save to CSV
        list_of_stocks_processed.append(stock + "_Processed")
    return list_of_stocks_processed


if __name__ == "__main__":
    list_of_stocks = ["TSLA_2020-03-01_2022-01-20_1min"] 
    # list_of_stocks = ["TSLA_2020-03-09_2022-01-28_15min", "AAPL_2020-03-24_2022-02-12_15min"] # List of stock data csv's to be tested, located in "data/" folder 
    list_of_stocks_proccessed = preprocess_data(list_of_stocks) # Preprocess the data

    results = tester.test_array(list_of_stocks_proccessed, logic, chart=True) # Run backtest on list of stocks using the logic function
# passing logic function as parameter. 
    print("training period " + str(training_period))
    print("standard deviations " + str(standard_deviations))
    df = pd.DataFrame(list(results), columns=["Buy and Hold","Strategy","Longs","Sells","Shorts","Covers","Stdev_Strategy","Stdev_Hold","Stock"]) # Create dataframe of results
    df.to_csv("results/Test_Results.csv", index=False) # Save results to csv



 
 
  