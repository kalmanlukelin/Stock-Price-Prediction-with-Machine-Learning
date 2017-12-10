from data_frame import get_data_frame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors

def test_run():
  start_date = '2012-01-01' #'2012-01-01'
  end_date = '2014-01-01' #'2012-09-30'

  data_frame = get_data_frame('IBM', start_date, end_date, dropna=True)
  head_data_frame = data_frame.head(251)
  tail_data_frame = data_frame.tail(251)

  trainX=np.linspace(1,251,251)
  trainX=np.reshape(trainX,(-1,1))
  
  trainY=head_data_frame.iloc[:,-1]
  #print(trainY)
  
  n_neighbors = 5
  knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')

  #prediction
  y_ = knn.fit(trainX, trainY).predict(trainX)
  
  #plt.plot(trainX, trainY, 'k', trainX, y_, 'r')
  
  #data_frame.plot()
  plt.show()
    
if __name__ == '__main__':
  test_run()