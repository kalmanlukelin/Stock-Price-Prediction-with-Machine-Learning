from data_frame import get_data_frame, get_manual_data_frame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors, neural_network, svm, linear_model, tree
import pandas as pd

def plot_figure(data, separate=False):
    if separate == True:
        for i in range(1, len(data.columns)):
            data.iloc[:,[0,i]].plot()
    else: data.plot()
    
def test_run():
    #get stock value
    start_date = '2009-01-01' #'2012-01-01'
    end_date = '2013-12-31' #'2014-01-01'
    stock_frame = get_manual_data_frame('IBM', start_date, end_date)
    #stock_frame = get_data_frame('IBM', start_date, end_date)
    
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
    KNN = neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform')
    KNN = KNN.fit(trainX, trainY)
    score_KNN = KNN.score(trainX, trainY)
    testY_KNN = KNN.predict(testX)
    trainY_KNN = KNN.predict(trainX)
    
    # MLP training
    MLP = neural_network.MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=3000) # hidden_layer_sizes=(first layer nodes, second layer nodes, ...) 10,5
    MLP = MLP.fit(trainX, trainY)
    score_MLP = MLP.score(trainX, trainY)
    testY_MLP = MLP.predict(testX)
    trainY_MLP = MLP.predict(trainX)
    
    # SVM training
    SVM_ = svm.SVR()
    SVM_ = SVM_.fit(trainX, trainY)
    score_SVM = SVM_.score(trainX, trainY)
    testY_SVM = SVM_.predict(testX)
    trainY_SVM = SVM_.predict(trainX)
    
    # linear regression training
    LinReg = linear_model.LinearRegression()
    LinReg = LinReg.fit(trainX, trainY)
    score_LinReg = LinReg.score(trainX, trainY)
    testY_LinReg = LinReg.predict(testX)
    trainY_LinReg = LinReg.predict(trainX)
    
    # Decision tree training
    DeciTree = tree.DecisionTreeRegressor()
    DeciTree = DeciTree.fit(trainX, trainY)
    score_Deci = DeciTree.score(trainX, trainY)
    testY_Deci = DeciTree.predict(testX)
    trainY_Deci = DeciTree.predict(trainX)
    
    # hybrid
    testY_hybrid = (testY_KNN+testY_MLP+testY_SVM+testY_LinReg+testY_Deci)/5
    trainY_hybrid = (trainY_KNN+trainY_MLP+trainY_SVM+trainY_LinReg+trainY_Deci)/5
    
    # print training accuracy
    print("Training accuracy")
    print("KNN accuracy:"+str(score_KNN))
    print("MLP accuracy:"+str(score_MLP))
    print("SVM accuracy:"+str(score_SVM))
    print("Linear Regression accuracy:"+str(score_LinReg))
    print("Decision Tree accuracy:"+str(score_Deci))
    print("")
    
    #print prediction accuracy
    print("Prediction accuracy")
    print("KNN accuracy:"+str(np.corrcoef(testY, testY_KNN)[1,0]))
    print("MLP accuracy:"+str(np.corrcoef(testY, testY_MLP)[1,0]))
    print("SVM accuracy:"+str(np.corrcoef(testY, testY_SVM)[1,0]))
    print("Linear Regression accuracy:"+str(np.corrcoef(testY, testY_LinReg)[1,0]))
    print("Decision Tree accuracy:"+str(np.corrcoef(testY, testY_Deci)[1,0]))
    print("Hybrid accuracy:"+str(np.corrcoef(testY, testY_hybrid)[1,0]))
    
    # build training data frame 
    train_data_frame = pd.DataFrame(index = trainX.index)
    train_data_frame['actual value'] = trainY
    train_data_frame['KNN predicted value'] = trainY_KNN
    train_data_frame['MLP predicted value'] = trainY_MLP
    train_data_frame['SVM predicted value'] = trainY_SVM
    train_data_frame['Linear Regression predicted value'] = trainY_LinReg
    train_data_frame['Decision Tree predicted value'] = trainY_Deci
    train_data_frame['Hybrid predicted value'] = trainY_hybrid
    
    # build test data frame
    test_data_frame = pd.DataFrame(index = testX.index)
    test_data_frame['actual value'] = testY
    test_data_frame['KNN predicted value'] = testY_KNN
    test_data_frame['MLP predicted value'] = testY_MLP
    test_data_frame['SVM predicted value'] = testY_SVM
    test_data_frame['Linear Regression predicted value'] = testY_LinReg
    test_data_frame['Decision Tree predicted value'] = testY_Deci
    test_data_frame['Hybrid predicted value'] = testY_hybrid
    
    # plot figure
    plot_figure(train_data_frame, separate=False)
    plot_figure(test_data_frame, separate=False)
    
    plt.show()
    
if __name__ == '__main__':
  test_run()
