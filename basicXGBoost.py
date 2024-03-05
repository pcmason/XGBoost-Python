'''
This is a simple example using the XGBoost algorithm on the Pimas Indians Dataset to predict early onset diabetes.
Loads in the dataset, splits into train and test splits, fits the XGBoost algorithm and then makes predictions.
'''

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load dataset
dataset = np.loadtxt('pimas-indians-diabetes.csv', delimiter=',')
# Split data into x & y
x = dataset[:, 0:8]
y = dataset[:, 8]
# Split data into train & test splits
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=7)
# Fit model on training data
model = XGBClassifier()
model.fit(x_train, y_train)
# Make predictions for the test dataset
y_pred = model.predict(x_test)
preds = [round(value) for value in y_pred]
# Evaluate predictions
accuracy = accuracy_score(y_test, preds)
print("Accuracy: %.3f%%" % (accuracy * 100.0))