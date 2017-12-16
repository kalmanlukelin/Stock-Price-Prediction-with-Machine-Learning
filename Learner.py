from data_frame import get_data_frame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
import pandas as pd
from sklearn import neighbors
import datetime

def test_run():
    #get stock value
    start_date = '2012-01-01' #'2012-01-01'
    end_date = '2014-01-01' #'2012-09-30'
    data_frame = get_data_frame('IBM', start_date, end_date, dropna=False)
    data_frame['IBM'] = data_frame['IBM'].fillna(method='ffill')
    data_frame['IBM'] = data_frame['IBM'].fillna(method='bfill')
    
    # build index
    data_frame = data_frame.rename(index=str, columns={'IBM':'actual value'})
    dateIndex = [i.replace('00:00:00','') for i in data_frame.index]
    data_frame.index = dateIndex
    #data_frame['actual value'] = data_frame['IBM']
    #del(data_frame['IBM'])
    
    # build indicator database
    rolling_mean = data_frame['actual value'].rolling(window=5,center=False).mean()
    data_frame['bb_value'] = data_frame['actual value'] - rolling_mean
    data_frame['bb_value'] = data_frame['bb_value'] / (data_frame['actual value'].rolling(window=5,center=False).std() * 2) 
    data_frame['momentum'] = (data_frame['actual value']/data_frame['actual value'].shift(periods = -5)) - 1
    data_frame['volatility'] = ((data_frame['actual value']/data_frame['actual value'].shift(periods = -1)) - 1).rolling(window=5,center=False).std()
    data_frame['value to predict'] = data_frame['actual value'].shift(periods = -5)
    data_frame = data_frame.dropna(axis=0, how='any')
    
    # get training data
    trainX = data_frame.iloc[:,0:-1].truncate(after='2012-12-31')
    trainY = data_frame.iloc[:,-1].truncate(after='2012-12-31')
    
    # build test database
    test_data_frame = data_frame.iloc[:,0:-1].truncate(before='2013-01-01')
    test_data_frame['value to predict'] = data_frame.iloc[:,0:1].shift(periods = -5).truncate(before='2013-01-01')
    test_data_frame = test_data_frame.dropna(axis=0, how='any')
    
    #get test data
    testX = test_data_frame.iloc[:,0:-1]
    testY = test_data_frame.iloc[:,-1]
    
    # KNN training
    n_neighbors = 5
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
    predictedY = knn.fit(trainX, trainY).predict(testX)
    
    # print accuracy
    print("The accuracy:")
    testY = np.array(testY).ravel()
    print(sum(abs(predictedY-testY))/len(testY))
    
    # build data plot frame
    data_frame_plot = pd.DataFrame(index = testX.index)
    data_frame_plot['actual value'] = testY
    data_frame_plot['predicted value'] = predictedY
    
    # plot figure
    data_frame_plot.plot()
    plt.show()
    
if __name__ == '__main__':
  test_run()