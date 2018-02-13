# Intelligent Computing Algorithm Project

This project is to make prediction for stocks using supervised learning. I choose some financial indicators such as bollinger bands, volitiliy, and momentum as featuers to predict whether tomorrows's stock price will rise or fall.

## Running the project

To see the result of prediction acuracy
```
python main.py
```

To see the influence of each features on the prediction accuracy
```
python test_features.py
```

To see the prediction accuray vs time windows t. (time windows t is to see the relationship between tomorrow's stock price to that of t days ago)
```
python test_y_windows.py
```

## Supervied Learning Methods Used

* K-nearest neighbors (KNN)
* Multilayer perceptron (MLP)
* Support vector machine (SVM)
* Random forest (RanF)

## Executive summary

### Bad prediction accuracy with the sign of difference between tomorrow’s stock price and that of today. The following is the accuracy for each methods measured with by scale 0-1.

* KNN: 0.57
* MLP: 0.65
* SVM: 0.65
* RanF: 0.59

### Great prediction accuracy with the sign of difference between tomorrow’s stock price and that of 40 days ago.

* KNN: 0.87
* MLP: 0.95
* SVM: 0.90
* RanF: 0.96

## Result

### Accuracy versus Time 

![Image](https://github.com/LukeLinn/ICA_project/blob/master/result_pictures/test_y_windows.png)

### Effect of features

#### Sign of difference between tomorrow’s stock price and that of today.
![Image](https://github.com/LukeLinn/ICA_project/blob/master/result_pictures/test_features_0.png)

#### Sign of difference between tomorrow’s stock price and that of 40 days ago.
![Image](https://github.com/LukeLinn/ICA_project/blob/master/result_pictures/test_features_40.png)