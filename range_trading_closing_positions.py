from distutils.command.build_py import build_py
import pandas as pd
import numpy as np

# local imports
from backtester import  tester


TRAINING_PERIOD = 20 # How far the rolling average takes into calculation
STD_DV_CONST = 3.5 # Number of Standard Deviations from the mean the Bollinger Bands sit
STD_MULTIPLIER = 2  #Multiples of Standard Devs Const used for Boll Bands 

INTERVAL = 14 # Interval for bollinger bands and ADX calculations 

# bounds for RSI
RSI_BOUNDS = {
    "upper": 70,
    "lower": 30,
} 

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

    if(last_index <= TRAINING_PERIOD): # If the lookback is long enough to calculate the Bollinger Bands
        return   # A coding style that is the same as tabbed if statement.
    ADX = lookback['ADX'+str(INTERVAL)][last_index]

    RSI = lookback['RSI'][last_index]

    close_price = lookback['close'][last_index]


    if ADX >= 20 and ADX < 25:
        coeff = STD_MULTIPLIER*STD_DV_CONST
    elif ADX > 10 and ADX < 20:
        coeff = 1*STD_DV_CONST 
    else:
        coeff = 0.5*STD_DV_CONST

    
    BOLU = lookback['MA-TP'][last_index] + coeff*lookback['std'][last_index] # Calculate Upper Bollinger Band
    BOLD = lookback['MA-TP'][last_index] - coeff*lookback['std'][last_index] # Calculate Lower Bollinger Band

    # when ADX is >= 25, only close positions
    if ADX >= 25:
        if(close_price < BOLD):
            for position in account.positions: # Close all current positions
                if position.type_ == 'short':
                    account.close_position(position, 1, close_price)

        if(close_price > BOLU):
            for position in account.positions: # Close all current positions
                if position.type_ == 'long':
                    account.close_position(position, 1, close_price)
        return

    if(close_price < BOLD): # If current price is below lower Bollinger Band, enter a long position
        
        # if RSI > 70 - don't buy (Overvalued)
        # if RSI > RSI_BOUNDS["lower"]:
        #     return

        for position in account.positions: # Close all current positions
            account.close_position(position, 1, close_price)
        if(account.buying_power > 0):
            account.enter_position('long', account.buying_power, close_price) # Enter a long position (enter full position)

    if(close_price > BOLU): # If today's price is above the upper Bollinger Band, enter a short position
        
        # if RSI < 30 - don't sell (Undervalued)
        # if RSI < RSI_BOUNDS["upper"]:
        #     return
        
        for position in account.positions: # Close all current positions
            account.close_position(position, 1, close_price)
        if(account.buying_power > 0):
                account.enter_position('short', account.buying_power, close_price) # Enter a short position
    

def calc_rsi(data, periods=14):
    close_delta = data['close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True,
                       min_periods=periods).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

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
        df['std'] = df['TP'].rolling(TRAINING_PERIOD).std() # Calculate Standard Deviation
        df['MA-TP'] = df['TP'].rolling(TRAINING_PERIOD).mean() # Calculate Moving Average of Typical Price
        df['sum']=df['close'].cumsum()

        df['-DM'] = df['low'].shift(1) - df['low'] # Directional Movement : previous low minus current low
        df['+DM'] = df['high'] - df['high'].shift(1) # Directional Movement : current high minus previous high 
        df['+DM'] = np.where((df['+DM'] > df['-DM']) & (df['+DM']>0), df['+DM'], 0.0)
        df['-DM'] = np.where((df['-DM'] > df['+DM']) & (df['-DM']>0), df['-DM'], 0.0)
        df['TR_TMP1'] = df['high'] - df['low']
        df['TR_TMP2'] = np.abs(df['high'] - df['close'].shift(1))
        df['TR_TMP3'] = np.abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['TR_TMP1', 'TR_TMP2', 'TR_TMP3']].max(axis=1)
        df['TR'+str(INTERVAL)] = df['TR'].rolling(INTERVAL).sum()
        df['+DMI'+str(INTERVAL)] = df['+DM'].rolling(INTERVAL).sum()
        df['-DMI'+str(INTERVAL)] = df['-DM'].rolling(INTERVAL).sum()
        df['+DI'+str(INTERVAL)] = df['+DMI'+str(INTERVAL)] /   df['TR'+str(INTERVAL)]*100
        df['-DI'+str(INTERVAL)] = df['-DMI'+str(INTERVAL)] / df['TR'+str(INTERVAL)]*100
        df['DI'+str(INTERVAL)+'-'] = abs(df['+DI'+str(INTERVAL)] - df['-DI'+str(INTERVAL)])
        df['DI'+str(INTERVAL)] = df['+DI'+str(INTERVAL)] + df['-DI'+str(INTERVAL)]
        df['DX'] = (df['DI'+str(INTERVAL)+'-'] / df['DI'+str(INTERVAL)])*100
        df['ADX'+str(INTERVAL)] = df['DX'].rolling(INTERVAL).mean()

        
        df['ADX'+str(INTERVAL)] =  df['ADX'+str(INTERVAL)].fillna(df['ADX'+str(INTERVAL)].mean()) 
        del df['TR_TMP1'], df['TR_TMP2'], df['TR_TMP3'], df['TR'], df['TR'+str(INTERVAL)]
        del df['+DMI'+str(INTERVAL)], df['DI'+str(INTERVAL)+'-']
        del df['DI'+str(INTERVAL)], df['-DMI'+str(INTERVAL)]       # interval is 14 (ADX14)
        del df['+DI'+str(INTERVAL)], df['-DI'+str(INTERVAL)]
        del df['DX']

        # RSI
        df["RSI"] = calc_rsi(df, INTERVAL)
        df.to_csv("data/" + stock + "_Processed.csv", index=False) # Save to CSV


        list_of_stocks_processed.append(stock + "_Processed")
    return list_of_stocks_processed


if __name__ == "__main__":
    
    REMAKE_DATA = False
    # remake *_Processed.csv for all stocks in "stocks_to_process"
    if REMAKE_DATA:
        stocks_to_process = ["WMT_2020-10-05_2022-08-26_15min", "NDAQ_2020-10-05_2022-08-26_15min"]
        list_of_stocks = preprocess_data(
            stocks_to_process)  # Preprocess the data
    # resuse cvs from "list_of_stocks"
    else:
        # List of stock data csv's to be tested, located in "data/" folder
        #"TSLA_2020-03-09_2022-01-28_15min_Processed"
        list_of_stocks = ["WMT_2020-10-05_2022-08-26_15min_Processed", "NDAQ_2020-10-05_2022-08-26_15min_Processed"]
    results = tester.test_array(list_of_stocks, logic, chart=True) # Run backtest on list of stocks using the logic function
# passing logic function as parameter. 
    print("training period " + str(TRAINING_PERIOD))
    print("standard deviations " + str(STD_DV_CONST))
    df = pd.DataFrame(list(results), columns=["Buy and Hold","Strategy","Longs","Sells","Shorts","Covers","Stdev_Strategy","Stdev_Hold","Stock"]) # Create dataframe of results
    df.to_csv("results/Test_Results.csv", index=False) # Save results to csv

