from data_frame import get_self_made_data_frame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors, neural_network, svm, ensemble
import pandas as pd

def test_run():
     
    #get stock value
    start_date = '2007-01-01' #'2012-01-01'
    end_date = '2016-01-01' #'2014-01-01'
    stock_frame = get_self_made_data_frame('IBM', start_date, end_date, y_windows=13)
    
    num_feaures=len(stock_frame.columns)-1
    
    KNN_score, MLP_score, SVM_score, RanF_score = [], [], [], []
    
    for i in range(1, num_feaures):
        # get training data
        train_start_date=start_date
        train_end_date='2014-12-31'
        trainX = stock_frame.iloc[:,0:i].truncate(before=train_start_date, after=train_end_date)
        trainY = stock_frame.iloc[:,-1].truncate(before=train_start_date, after=train_end_date)
    
        # get test data
        test_start_date='2015-01-01'
        test_end_date=end_date
        testX = stock_frame.iloc[:,0:i].truncate(before=test_start_date, after=test_end_date)
        testY = stock_frame.iloc[:,-1].truncate(before=test_start_date, after=test_end_date)
    
        # Miscellaneous machine learning methods. Test score means the correlation between training data and target data, so it's the smaller the better.
        # KNN training
        KNN = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
        KNN = KNN.fit(trainX, trainY)
    
        # MLP training
        MLP = neural_network.MLPClassifier(hidden_layer_sizes=(20,10,5), max_iter=3000) # hidden_layer_sizes=(first layer nodes, second layer nodes, ...)
        MLP = MLP.fit(trainX, trainY)
        # SVM training
        SVM_ = svm.SVC()
        SVM_ = SVM_.fit(trainX, trainY)
        
        # RanF training
        RanF = ensemble.RandomForestClassifier()
        RanF = RanF.fit(trainX, trainY)
        
        #append scores
        KNN_score.append(KNN.score(trainX, trainY))
        MLP_score.append(MLP.score(trainX, trainY))
        SVM_score.append(SVM_.score(trainX, trainY))
        RanF_score.append(RanF.score(trainX, trainY))
    
    print("Scores of each methods")
    print("KNN:"+str(KNN_score))
    print("MLP:"+str(MLP_score))
    print("SVM:"+str(SVM_score))
    print("RanF:"+str(RanF_score))
    print("")
    
    plt.plot(KNN_score,label="KNN")
    plt.plot(MLP_score,label="MLP")
    plt.plot(SVM_score,label="SVM")
    plt.plot(RanF_score,label="RanF")
    plt.legend()
    plt.axis((0,num_feaures,0.2,1.1))
    plt.show()

if __name__ == '__main__':
  test_run()