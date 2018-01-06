from data_frame import get_self_made_data_frame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors, neural_network, svm, linear_model, tree, ensemble, discriminant_analysis
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd

def plot_figure(data, separate=False, sub_plot=False):
    if separate == True:
        for i in range(1, len(data.columns)):
            data.iloc[:,[0,i]].plot()
    else: data.plot(subplots=sub_plot)
    
def test_run():
     
    #get stock value
    start_date = '2007-01-01' #'2012-01-01'
    end_date = '2016-01-01' #'2014-01-01'
    stock_frame = get_self_made_data_frame('IBM', start_date, end_date, y_windows=13)
    
    # get training data
    train_start_date=start_date
    train_end_date='2014-12-31'
    trainX = stock_frame.iloc[:,0:-1].truncate(before=train_start_date, after=train_end_date)
    trainY = stock_frame.iloc[:,-1].truncate(before=train_start_date, after=train_end_date)
    
    # get test data
    test_start_date='2015-01-01'
    test_end_date=end_date
    testX = stock_frame.iloc[:,0:-1].truncate(before=test_start_date, after=test_end_date)
    testY = stock_frame.iloc[:,-1].truncate(before=test_start_date, after=test_end_date)
    
    selector = SelectKBest(chi2, k='all').fit(trainX,trainY)
    x_new = selector.transform(trainX) # not needed to get the score
    scores = selector.scores_
    
    for item1, item2 in zip(stock_frame.columns, scores):
        print(item1,":",item2)
    
if __name__ == '__main__':
  test_run()
