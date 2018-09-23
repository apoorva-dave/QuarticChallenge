# QuarticChallenge

## Briefly describe the conceptual approach you chose! What are the trade-offs?

The steps for developing the binary classifier is as follows:

1. Load data
2. Selecting the top K features ( here 30) using SelectKBest algorithm
3. StratifiedShuffleSplit is done to handle imbalanced dataset
4. PCA is applied to reduce the dimension of the data capturing 95% variance.
5. Finally hypertuned logistic regression model is developed.
6. Accuracy measure used is AUC.

In the approach used above, the AUC obtained is 0.59 which is not that great. Here I have chosen top 30 features for the given dataset which is obtained by filter methods. Finding this value, is one of the trade offs I see. Oversampling could also be used as a method to handle imbalanced data. However, on applying SMOTE as well, approximately the same results are obtained. 

## What's the model performance? What is the complexity? Where are the bottlenecks?

The AUC measure is used to accurately predict the model. AUC of the model is 0.59. AUC is used as a measure to avoid predicting false values.  When the data set is imbalanced(as in this case) let's say it has 99% of instances in one class and only 1 % in the other. Predicting that every instance belongs to the majority class, get accuracy of 99% which is obviously not correct. 

To avoid overfitting of data, Logistic Regression technique is used rather than tree based algorithms. Complexity is O(N).
The main bottleneck is the imbalanced dataset. 

## If you had more time, what improvements would you make, and in what order of priority?

I would have tried the below steps in decreasing order of their priority:

1. I would first try different undersampling and oversampling techniques to get proper balanced dataset.
2. I would hypertune Logistic regression model more in order to get better AUC. 

## Results

Code includes 3 files -
1. SelectFeatures.py - Selects the top K features ( here 30) using SelectKBest algorithm
2. Train.py - Trains model with training data and saves model as model.sav using pickle.
3. Test.py - Tests model on test dataset and produces result.csv as output file containing two columns as 'id' and 'target'.
4. result.csv is the output csv file generated.

Run train.py to get trained model. Then run test.py to load model, predict values and create output result.csv file.
