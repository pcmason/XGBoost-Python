'''
Example code using a XGBoost classifier on the Otto Group Dataset that has 61,000 observations classified into 1 of 10
possible classes. The XGBoost model is then tuned with different values for number of trees and learning rate to
optimize performance of the model. The logloss for the different combinations is also output as a graph.
'''

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv('Data/train.csv')
dataset = data.values
# Split into x and y
x = dataset[:, 0:94]
y = dataset[:, 94]
# Encode string class values as integers
label_encoded_y = LabelEncoder().fit_transform(y)
# Grid search
model = xgb.XGBClassifier()
# Set number of trees & learning rates to tune
n_estimators = [100, 200, 300, 400, 500]
learning_rate = [0.0001, 0.001, 0.01, 0.1]
# Use grid search to evaluate effect on logloss of training gradient boosting model on different inputs
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring='neg_log_loss', n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(x, label_encoded_y)
# Summarize results
print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, stdev, param))
# Plot results
scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
# Create a line graph for each learning rate
for i, value in enumerate(learning_rate):
    plt.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('Log Loss')
plt.show()

# Plot performance for learning_rate=0.1 since it is not seen in the above graph
loss = [-0.007369, -0.007927, -0.008344, -0.008570, -0.008734]
plt.plot(n_estimators, loss)
plt.xlabel('n_estimators')
plt.ylabel('Log Loss')
plt.title('XGBoost learning_rate=0.1 n_estimators vs Log Loss')
plt.show()
