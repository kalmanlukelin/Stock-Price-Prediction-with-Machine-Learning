from data_frame import get_data_frame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
import pandas as pd
from sklearn import neighbors
from pandas.tests.io.msgpack.test_format import testArray

def test_run():
    
    #get stock value
    start_date = '2012-01-01' #'2012-01-01'
    end_date = '2014-01-01' #'2012-09-30'
    dates=pd.date_range(start_date, end_date)
    
    data_frame = get_data_frame('IBM', start_date, end_date, dropna=False)
    data_frame['IBM'] = data_frame['IBM'].fillna(method='ffill')
    data_frame['IBM'] = data_frame['IBM'].fillna(method='bfill')
    
    # build indicator database
    data_frame['actual value'] = data_frame['IBM']
    del data_frame['IBM']
    # data_frame['rolling_mean'] = data_frame['actual value'].rolling(window=5,center=False).mean()
    rolling_mean = data_frame['actual value'].rolling(window=5,center=False).mean()
    data_frame['bb_value'] = data_frame['actual value'] - rolling_mean
    data_frame['bb_value'] = data_frame['bb_value'] / (data_frame['actual value'].rolling(window=5,center=False).std() * 2) 
    data_frame['momentum'] = (data_frame['actual value']/data_frame['actual value'].shift(periods = -5)) - 1
    data_frame['volatility'] = ((data_frame['actual value']/data_frame['actual value'].shift(periods = -1)) - 1).rolling(window=5,center=False).std()
    data_frame['y_values'] = data_frame['actual value'].shift(periods = -5)
    data_frame = data_frame.dropna(axis=0, how='any')
    
    trainX = data_frame.iloc[:,0:-1].truncate(after='2012-12-31')
    trainY = data_frame.iloc[:,-1].truncate(after='2012-12-31')
    testX = data_frame.iloc[:,0:-1].truncate(before='2013-01-01')
    testY = data_frame.iloc[:,0:1].truncate(before='2013-01-01')
    
    #print(trainX)
    #print(trainY)
    #print(testX)
    #print(testY)
    
    
    # KNN training
    n_neighbors = 5
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
    predictedY = knn.fit(trainX, trainY).predict(testX)

    testY = np.array(testY).ravel()
    
    print("The accuracy:")
    print(sum(abs(predictedY-testY))/len(testY))
     
    dateRange=testX.index
    plt.plot(dateRange, testY, 'k', dateRange, predictedY, 'r')
    plt.show()
    
if __name__ == '__main__':
  test_run()