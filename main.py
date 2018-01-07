from data_frame import get_self_made_data_frame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors, neural_network, svm, ensemble
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def test_run():
    
    #get stock value
    start_date = '2015-01-01'
    end_date = '2018-01-01'
    stock_frame = get_self_made_data_frame('IBM', start_date, end_date, y_windows=15)
    
    # get training data
    train_start_date=start_date
    train_end_date='2016-12-31'
    trainX = stock_frame.iloc[:,0:-1].truncate(before=train_start_date, after=train_end_date)
    trainY = stock_frame.iloc[:,-1].truncate(before=train_start_date, after=train_end_date)
    
    # get test data
    test_start_date='2017-01-01'
    test_end_date=end_date
    testX = stock_frame.iloc[:,0:-1].truncate(before=test_start_date, after=test_end_date)
    testY = stock_frame.iloc[:,-1].truncate(before=test_start_date, after=test_end_date)
    
    # Miscellaneous machine learning methods. Test score means the correlation between training data and target data, so it's the smaller the better.
    # KNN training
    KNN = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
    KNN = KNN.fit(trainX, trainY)
    trainY_KNN = KNN.predict(trainX)
    testY_KNN = KNN.predict(testX)
    
    # MLP training
    MLP = neural_network.MLPClassifier(hidden_layer_sizes=(5, ), max_iter=3000) # hidden_layer_sizes=(first layer nodes, second layer nodes, ...)
    MLP = MLP.fit(trainX, trainY)
    trainY_MLP = MLP.predict(trainX)
    testY_MLP = MLP.predict(testX)
    
    # SVM training
    SVM_ = svm.SVC()
    SVM_ = SVM_.fit(trainX, trainY)
    trainY_SVM = SVM_.predict(trainX)
    testY_SVM = SVM_.predict(testX)
    
    # RanF training
    RanF = ensemble.RandomForestClassifier()
    RanF = RanF.fit(trainX, trainY)
    trainY_RanF = RanF.predict(trainX)
    testY_RanF = RanF.predict(testX)
    
    print("Training data accuracy")
    print("KNN:"+str(KNN.score(trainX, trainY)))
    print("MLP:"+str(MLP.score(trainX, trainY)))
    print("SVM:"+str(SVM_.score(trainX, trainY)))
    print("RanF:"+str(RanF.score(trainX, trainY)))
    print("")
    
    print("Test data accuracy")
    print("KNN:"+str(KNN.score(testX, testY)))
    print("MLP:"+str(MLP.score(testX, testY)))
    print("SVM:"+str(SVM_.score(testX, testY)))
    print("RanF:"+str(RanF.score(testX, testY)))
    print("")
    
    '''
    plt.matshow(confusion_matrix(trainY, trainY_KNN), cmap=plt.cm.gray)
    plt.matshow(confusion_matrix(testY, testY_KNN), cmap=plt.cm.gray)
    print(confusion_matrix(trainY, trainY_KNN))
    print(confusion_matrix(testY, testY_KNN))
    plt.show()
    '''
    
if __name__ == '__main__':
  test_run()
