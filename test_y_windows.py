from data_frame import get_self_made_data_frame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors, neural_network, svm, ensemble
import pandas as pd

def plot_figure(data, separate=False, sub_plot=False):
    if separate == True:
        for i in range(1, len(data.columns)):
            data.iloc[:,[0,i]].plot()
    else: data.plot(subplots=sub_plot)
    
def test_run():
    
    KNN_score, MLP_score, SVM_score, RanF_score = [], [], [], [] 
    y_windows=50
    
    for i in range(y_windows):
        #get stock value
        start_date = '2015-01-01' #'2012-01-01'
        end_date = '2018-01-01' #'2014-01-01'
        stock_frame = get_self_made_data_frame('IBM', start_date, end_date, y_windows=i)
    
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
    
        # MLP training
        MLP = neural_network.MLPClassifier(hidden_layer_sizes=(5,), max_iter=3000) # hidden_layer_sizes=(first layer nodes, second layer nodes, ...)
        MLP = MLP.fit(trainX, trainY)
    
        # SVM training
        SVM_ = svm.SVC()
        SVM_ = SVM_.fit(trainX, trainY)
    
        # RanF training
        RanF = ensemble.RandomForestClassifier()
        RanF = RanF.fit(trainX, trainY)
        
        #append scores
        KNN_score.append(KNN.score(testX, testY))
        MLP_score.append(MLP.score(testX, testY))
        SVM_score.append(SVM_.score(testX, testY))
        RanF_score.append(RanF.score(testX, testY))
    
    # print training data accuracy
    KNN_score=np.array(KNN_score)
    MLP_score=np.array(MLP_score)
    SVM_score=np.array(SVM_score)
    RanF_score=np.array(RanF_score)
    total=KNN_score+MLP_score+SVM_score+RanF_score
    
    print("Scores of each methods")
    print("KNN:"+str(KNN_score))
    print("MLP:"+str(MLP_score))
    print("SVM:"+str(SVM_score))
    print("RanF:"+str(RanF_score))
    print("Total:"+str(total))
    print("")
    
    print("KNN Max:"+str(max(KNN_score))+" "+"index:"+str(KNN_score.tolist().index(max(KNN_score))))
    print("MLP Max:"+str(max(MLP_score))+" "+"index:"+str(MLP_score.tolist().index(max(MLP_score))))
    print("SVM Max:"+str(max(SVM_score))+" "+"index:"+str(SVM_score.tolist().index(max(SVM_score))))
    print("RanF Max:"+str(max(RanF_score))+" "+"index:"+str(RanF_score.tolist().index(max(RanF_score))))
    print("total Max:"+str(max(total))+" "+"index:"+str(total.tolist().index(max(total))))
    
    plt.plot(KNN_score,label="KNN")
    plt.plot(MLP_score,label="MLP")
    plt.plot(SVM_score,label="SVM")
    plt.plot(RanF_score,label="RanF")
    plt.legend()
    plt.title('Long-Term Prediction Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Time Windows (days)')
    plt.axis((0,y_windows,0.6,1.1))
    plt.show()

if __name__ == '__main__':
  test_run()
