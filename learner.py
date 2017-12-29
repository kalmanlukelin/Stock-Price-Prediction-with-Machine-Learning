from data_frame import get_self_made_data_frame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors, neural_network, svm, linear_model, tree, ensemble
import pandas as pd

def plot_figure(data, separate=False, sub_plot=False):
    if separate == True:
        for i in range(1, len(data.columns)):
            data.iloc[:,[0,i]].plot()
    else: data.plot(subplots=sub_plot)
    
def test_run():
    #get stock value
    start_date = '2009-01-01' #'2012-01-01'
    end_date = '2013-12-31' #'2014-01-01'
    stock_frame = get_self_made_data_frame('IBM', start_date, end_date)
    
    print(stock_frame)
        
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
    KNN = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance')
    KNN = KNN.fit(trainX, trainY)
    score_KNN = KNN.score(trainX, trainY)
    trainY_KNN = KNN.predict(trainX)
    testY_KNN = KNN.predict(testX)
    
    # MLP training
    MLP = neural_network.MLPRegressor(hidden_layer_sizes=(5,), max_iter=3000) # hidden_layer_sizes=(first layer nodes, second layer nodes, ...)
    MLP = MLP.fit(trainX, trainY)
    score_MLP = MLP.score(trainX, trainY)
    trainY_MLP = MLP.predict(trainX)
    testY_MLP = MLP.predict(testX)
    
    # SVM training
    SVM_ = svm.SVR()
    SVM_ = SVM_.fit(trainX, trainY)
    score_SVM = SVM_.score(trainX, trainY)
    trainY_SVM = SVM_.predict(trainX)
    testY_SVM = SVM_.predict(testX)
    
    # linear regression training
    LinReg = linear_model.LinearRegression()
    LinReg = LinReg.fit(trainX, trainY)
    score_LinReg = LinReg.score(trainX, trainY)
    trainY_LinReg = LinReg.predict(trainX)
    testY_LinReg = LinReg.predict(testX)
    
    # Decision tree training
    DeciTree = tree.DecisionTreeRegressor()
    DeciTree = DeciTree.fit(trainX, trainY)
    score_Deci = DeciTree.score(trainX, trainY)
    trainY_Deci = DeciTree.predict(trainX)
    testY_Deci = DeciTree.predict(testX)
    
    # Ransom forest training
    RanF = ensemble.RandomForestRegressor()
    RanF = RanF.fit(trainX, trainY)
    score_RanF = RanF.score(trainX, trainY)
    trainY_RanF = RanF.predict(trainX)
    testY_RanF = RanF.predict(testX)
    
    # Ensemble
    trainY_ensem_tmp = [trainY_KNN, trainY_MLP, trainY_SVM, trainY_LinReg, trainY_Deci, trainY_RanF]
    testY_ensem_tmp = [testY_KNN, testY_MLP, testY_SVM, testY_LinReg, testY_Deci, testY_RanF]
    trainY_ensem = sum(trainY_ensem_tmp)/len(trainY_ensem_tmp)
    testY_ensem = sum(testY_ensem_tmp)/len(testY_ensem_tmp)
    
    # print training data accuracy
    print("Training data accuracy")
    print("KNN accuracy:"+str(np.corrcoef(trainY, trainY_KNN)[1,0]))
    print("MLP accuracy:"+str(np.corrcoef(trainY, trainY_MLP)[1,0]))
    print("SVM accuracy:"+str(np.corrcoef(trainY, trainY_SVM)[1,0]))
    print("Linear Regression accuracy:"+str(np.corrcoef(trainY, trainY_LinReg)[1,0]))
    print("Decision Tree accuracy:"+str(np.corrcoef(trainY, trainY_Deci)[1,0]))
    print("Random Forest accuracy:"+str(np.corrcoef(trainY, trainY_RanF)[1,0]))
    print("Ensemble accuracy:"+str(np.corrcoef(trainY, trainY_ensem)[1,0]))
    print("")
    
    #print test data accuracy
    print("Test data accuracy")
    print("KNN accuracy:"+str(np.corrcoef(testY, testY_KNN)[1,0]))
    print("MLP accuracy:"+str(np.corrcoef(testY, testY_MLP)[1,0]))
    print("SVM accuracy:"+str(np.corrcoef(testY, testY_SVM)[1,0]))
    print("Linear Regression accuracy:"+str(np.corrcoef(testY, testY_LinReg)[1,0]))
    print("Decision Tree accuracy:"+str(np.corrcoef(testY, testY_Deci)[1,0]))
    print("Random Forest accuracy:"+str(np.corrcoef(testY, testY_RanF)[1,0]))
    print("Ensemble accuracy:"+str(np.corrcoef(testY, testY_ensem)[1,0]))
    
    # build training data frame 
    train_data_frame = pd.DataFrame(index = trainX.index)
    train_data_frame['Actual value'] = trainY
    train_data_frame['Predicted value - KNN'] = trainY_KNN
    train_data_frame['Predicted value - MLP'] = trainY_MLP
    train_data_frame['Predicted value - SVM'] = trainY_SVM
    train_data_frame['Predicted value - Linear Regression'] = trainY_LinReg
    train_data_frame['Predicted value - Decision Tree '] = trainY_Deci
    train_data_frame['Predicted value - Random Forest '] = trainY_RanF
    train_data_frame['Predicted value - Ensemble '] = trainY_ensem
    
    # build test data frame
    test_data_frame = pd.DataFrame(index = testX.index)
    test_data_frame['Actual value'] = testY
    test_data_frame['Predicted value - KNN'] = testY_KNN
    test_data_frame['Predicted value - MLP'] = testY_MLP
    test_data_frame['Predicted value - SVM'] = testY_SVM
    test_data_frame['Predicted value - Linear Regression'] = testY_LinReg
    test_data_frame['Predicted value - Decision Tree'] = testY_Deci
    test_data_frame['Predicted value - Random Forest'] = testY_RanF
    test_data_frame['Predicted value - Ensemble'] = testY_ensem
    
    # plot figure
    plot_figure(train_data_frame, separate=False)
    plt.title('Training data')
    
    #plot_figure(train_data_frame, separate=True)
    
    plot_figure(test_data_frame, separate=False)
    plt.title('Test data')
    
    #plot_figure(test_data_frame, separate=True)

    plt.show()
    
if __name__ == '__main__':
  test_run()
