from data_frame import get_data_frame, get_manual_data_frame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors, neural_network, svm, linear_model, tree
import pandas as pd

def test_run():
    #get stock value
    start_date = '2012-01-01' #'2012-01-01'
    end_date = '2013-12-31' #'2014-01-01'
    stock_frame = get_manual_data_frame('IBM', start_date, end_date)
    
    # get training data
    train_end_date='2012-12-31'
    trainX = stock_frame.iloc[:,0:-1].truncate(after=train_end_date)
    trainY = stock_frame.iloc[:,-1].truncate(after=train_end_date)
    
    # get test data
    test_start_date='2013-01-01'
    testX = stock_frame.iloc[:,0:-1].truncate(before=test_start_date, after=end_date)
    testY = stock_frame.iloc[:,-1].truncate(before=test_start_date, after=end_date)
    testY = np.array(testY).ravel()
    
    # Miscellaneous machine learning methods. Test score means the correlation between training data and target data, so it's the smaller the better.
    # KNN training
    n_neighbors = 5
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
    predictY_knn = knn.fit(trainX, trainY).predict(testX)
    score_knn = knn.score(trainX, trainY) 
    
    # ANN training
    ann = neural_network.MLPRegressor(hidden_layer_sizes=(40,20), max_iter=2000) # hidden_layer_sizes=(first layer nodes, second layer nodes, ...) 10,5
    predictY_ann = ann.fit(trainX, trainY).predict(testX)
    score_ann = ann.score(trainX, trainY)
    
    # SVM training
    svm_method = svm.SVR()
    predictY_svm = svm_method.fit(trainX, trainY).predict(testX)
    score_svm = svm_method.score(trainX, trainY)
    
    
    # linear regression training
    linear_method = linear_model.LinearRegression()
    predictY_linear = linear_method.fit(trainX, trainY).predict(testX)
    score_linear = linear_method.score(trainX, trainY)
    
    # Decision tree training
    deci_tree_method = tree.DecisionTreeRegressor()
    predictY_deci = deci_tree_method.fit(trainX, trainY).predict(testX)
    score_deci = deci_tree_method.score(trainX, trainY)
    
    # print data frame
    #print(trainX)
    #print(trainY)
    #print(testX)
    #print(testY)
    
    # print accuracy
    print("KNN accuracy:"+str(score_knn))
    print("ANN accuracy:"+str(score_ann))
    print("SVM accuracy:"+str(score_svm))
    print("Linear Regression accuracy:"+str(score_linear))
    print("Decision Tree accuracy:"+str(score_deci))
    
    # build data plot frame
    data_frame_plot = pd.DataFrame(index = testX.index)
    data_frame_plot['actual value'] = testY
    data_frame_plot['KNN predicted value'] = predictY_knn
    data_frame_plot['ANN predicted value'] = predictY_ann
    data_frame_plot['SVM predicted value'] = predictY_svm
    data_frame_plot['Linear Regression predicted value'] = predictY_linear
    data_frame_plot['Decision Tree predicted value'] = predictY_deci
    
    # plot figure
    data_frame_plot.plot()
    plt.show()
    
if __name__ == '__main__':
  test_run()
