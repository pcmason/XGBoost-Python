'''
Example code using train + test splits, k-fold CV & stratified CV to evaluate the Pimas Diabetes dataset using XGBoost
'''

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score


# Method to create k-fold CV model and output results
def createCV(method, x, y):
    model = xgb.XGBClassifier()
    results = cross_val_score(model, x, y, cv=method)
    print('\n%s Accuracy: %.3f%% (%.3f%%)' % (method, results.mean()*100.0, results.std()*100.0))


# Load data
dataset = np.loadtxt('pimas-indians-diabetes.csv', delimiter=',')
# Split into x & y
x = dataset[:, 0:8]
y = dataset[:, 8]


# Train + test split evaluation
# Split into train & test splits
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=7)
# Fit model on training data
model = xgb.XGBClassifier()
model.fit(x_train, y_train)
# Make predictions for test data
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
# Evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print('Train + Test Split Accuracy: %.3f%%' % (accuracy * 100.0))


# K-fold CV evaluation
# CV model
createCV(KFold(n_splits=10), x, y)


# Stratified CV evaluation
# CV model
createCV(StratifiedKFold(n_splits=10), x, y)


