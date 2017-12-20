import pandas as pd
import numpy as np

def symbol_to_path(symbol):
    return 'data/{}.csv'.format(symbol)

def get_data_frame(symbol, start_date, end_date):
    date_range = pd.date_range(start_date, end_date)
    data_frame = pd.DataFrame(index = date_range)
    
    symbol_data_frame = pd.read_csv(symbol_to_path(symbol), index_col = 'Date', parse_dates = True, na_values = ['NaN'])
    data_frame = data_frame.join(symbol_data_frame)
    
    # fill NaN values
    data_frame = data_frame.fillna(method='ffill')
    data_frame = data_frame.fillna(method='bfill')
    
    data_frame['value to predict'] = data_frame['Adj Close'].shift(periods = -5)
    data_frame = data_frame.dropna(axis=0, how='any')
    
    return data_frame

# data frame with manual indicators 
def get_manual_data_frame(symbol, start_date, end_date):
    date_range = pd.date_range(start_date, end_date)
    data_frame = pd.DataFrame(index = date_range)
    
    symbol_data_frame = pd.read_csv(symbol_to_path(symbol), index_col = 'Date', parse_dates = True, usecols = ['Date', 'Adj Close'], na_values = ['NaN'])
    symbol_data_frame = symbol_data_frame.rename(columns = {'Adj Close': 'stock value'})
    data_frame = data_frame.join(symbol_data_frame)
    
    # fill NaN values
    data_frame = data_frame.fillna(method='ffill')
    data_frame = data_frame.fillna(method='bfill')
    
    # build indicator database
    rolling_mean = data_frame['stock value'].rolling(window=5,center=False).mean()
    data_frame['bb_value'] = (data_frame['stock value'] - rolling_mean) / (data_frame['stock value'].rolling(window=5,center=False).std() * 2) 
    data_frame['momentum'] = (data_frame['stock value']/data_frame['stock value'].shift(periods = -5)) - 1
    data_frame['volatility'] = ((data_frame['stock value']/data_frame['stock value'].shift(periods = -1)) - 1).rolling(window=5,center=False).std()
    data_frame['value to predict'] = data_frame['stock value'].shift(periods = -5)
    
    # some values will become nan after calculation, and I drop those rows
    data_frame = data_frame.dropna(axis=0, how='any')
    
    #drop infinite numbers after calculation
    data_frame = data_frame.replace([np.inf, -np.inf], np.nan)
    data_frame = data_frame.fillna(method='ffill')
    data_frame = data_frame.fillna(method='bfill')
    
    return data_frame
    
def test_run():
    start_date = '2008-01-01'
    end_date = '2014-01-31'
    
    data_frame = get_data_frame('IBM', start_date, end_date)
    #data_frame = get_manual_data_frame('IBM', start_date, end_date)
    print(data_frame)
  
if __name__ == '__main__':
    test_run()
