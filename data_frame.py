import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def symbol_to_path(symbol):
    return 'data/{}.csv'.format(symbol)

def get_data_frame(symbol, start_date, end_date):
    date_range = pd.date_range(start_date, end_date)
    data_frame = pd.DataFrame(index = date_range)
    
    symbol_data_frame = pd.read_csv(symbol_to_path(symbol),
                                     index_col = 'Date',
                                     parse_dates = True,
                                     usecols = ['Date', 'Adj Close'],
                                     na_values = ['NaN'])
    symbol_data_frame = symbol_data_frame.rename(columns = {'Adj Close': 'Adj Close'})
    data_frame = data_frame.join(symbol_data_frame)
    
    # fill NaN values
    data_frame = data_frame.fillna(method='ffill')
    data_frame = data_frame.fillna(method='bfill')
    
    #data_frame['value to predict'] = data_frame['Adj Close'].shift(periods = -5)
    data_frame = data_frame.dropna(axis=0, how='any')
    
    return data_frame

# data frame with manual indicators 
def get_self_made_data_frame(symbol, start_date, end_date, y_windows=0):
    date_range = pd.date_range(start_date, end_date)
    data_frame = pd.DataFrame(index = date_range)
    
    symbol_data_frame = pd.read_csv(symbol_to_path(symbol), 
                                    index_col = 'Date',
                                    parse_dates = True,
                                    usecols = {'Date', 'Adj Close'},                           
                                    na_values = ['NaN'])
    data_frame = data_frame.join(symbol_data_frame)
    '''
    # add SPY indicator
    symbol_data_frame = pd.read_csv(symbol_to_path('SPY'), 
                                    index_col = 'Date', 
                                    parse_dates = True,                           
                                    na_values = ['NaN'])
    symbol_data_frame = symbol_data_frame.rename(columns = {'Adj Close':'SPY Adj Close',
                                                            'Open':'SPY Open',
                                                            'High':'SPY High',
                                                            'Low':'SPY Low',
                                                            'Close':'SPY Close',
                                                            'Volume':'SPY Volume'})
    data_frame = data_frame.join(symbol_data_frame)
    '''
    # fill NaN values
    data_frame = data_frame.fillna(method='ffill')
    data_frame = data_frame.fillna(method='bfill')
    
    #data_frame['Adj Close_20'] = data_frame['Adj Close'].shift(periods = 20)
    #data_frame['Adj Close_40'] = data_frame['Adj Close'].shift(periods = 40)
    
    # measures how intensely the stock price is fluctuating
    data_frame['volatility_20'] = ((data_frame['Adj Close']/data_frame['Adj Close'].shift(periods = 1)) - 1).rolling(window=20,center=False).std()
    data_frame['volatility_40'] = ((data_frame['Adj Close']/data_frame['Adj Close'].shift(periods = 1)) - 1).rolling(window=40,center=False).std()
    
    # momentum
    #data_frame['momentum_20'] = data_frame['Adj Close'] - data_frame['Adj Close'].shift(periods = 20)
    #data_frame['momentum_40'] = data_frame['Adj Close'] - data_frame['Adj Close'].shift(periods = 40)
    
    # build indicator database
    # %b value, which measures the effect of Bollinger Bands
    rolling_mean = data_frame['Adj Close'].rolling(window=20,center=False).mean()
    rolling_std = data_frame['Adj Close'].rolling(window=20,center=False).std()
    data_frame['b_value_20'] = (data_frame['Adj Close'] - (rolling_mean - rolling_std * 2)) / (rolling_std * 4)
    
    rolling_mean = data_frame['Adj Close'].rolling(window=40,center=False).mean()
    rolling_std = data_frame['Adj Close'].rolling(window=40,center=False).std() 
    data_frame['b_value_40'] = (data_frame['Adj Close'] - (rolling_mean - rolling_std * 2)) / (rolling_std * 4)
    
    
    # value to predict
    data_frame['classification'] = data_frame['Adj Close'].shift(periods = -1) - data_frame['Adj Close'].shift(periods = y_windows)
    for i in range(len(data_frame['classification'])):
        if data_frame['classification'][i] > 0: data_frame['classification'][i] = 1
        else: data_frame['classification'][i] = 0
    
    # some values will become nan after calculation, so drop those rows
    data_frame = data_frame.dropna(axis=0, how='any')
    
    # drop infinite numbers after calculation
    data_frame = data_frame.replace([np.inf, -np.inf], np.nan)
    data_frame = data_frame.fillna(method='ffill')
    data_frame = data_frame.fillna(method='bfill')
    
    # preprocess data for sklearn
    scaler = MinMaxScaler()
    data_frame.iloc[:,0:-1] = scaler.fit_transform(data_frame.iloc[:,0:-1])
    
    return data_frame
    
def test_run():
    start_date = '2016-01-01'
    end_date = '2018-01-01'
    
    #data_frame = get_data_frame('IBM', start_date, end_date)
    data_frame = get_self_made_data_frame('IBM', start_date, end_date, y_windows=20)
    
    print(data_frame)
    
   
if __name__ == '__main__':
    test_run()
