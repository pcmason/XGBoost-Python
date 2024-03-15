'''
Example code using 3 different methods of optimization on the Otto Group Project 10 class classification dataset. The
three methods used are: optimizing # of trees, optimizing max. depth of trees & optimizing both.
'''

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# Create method to subsample different parameters using grid search and output results
def subsample_method(grid, model, x, y):
    # Use sklearn's grid search capability
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, grid, scoring='neg_log_loss', n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(x, y)
    # Summarize results
    print('\nBest: %f using %s\n' % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('%f (%f) with: %r' % (mean, stdev, param))
    return means, stds


# Plot the logloss and subsamples on a line graph
def plot_observations(subsample, label_name, means, stds):
    plt.errorbar(subsample, means, yerr=stds)
    plt.title('XGBoost ' + label_name + ' vs Log Loss')
    plt.xlabel(label_name)
    plt.ylabel('Log Loss')
    plt.show()


# Load data
data = pd.read_csv('Data/train.csv')
dataset = data.values
# Split into x and y
x = dataset[:, 0:94]
y = dataset[:, 94]
# Encode string class values as integers
label_encoded_y = LabelEncoder().fit_transform(y)
# Define model to be used for this example
model = xgb.XGBClassifier()

# Optimization of the number of trees
n_estimators = range(50, 400, 50)
param_grid = dict(n_estimators=n_estimators)
means, stds = subsample_method(param_grid, model, x, label_encoded_y)
plot_observations(n_estimators, 'n_estimators', means, stds)

# Optimization of the maximum depth
max_depth = range(1, 11, 2)
param_grid = dict(max_depth=max_depth)
means, stds = subsample_method(param_grid, model, x, label_encoded_y)
plot_observations(max_depth, 'max_depth', means, stds)

# Optimization of both maximum depth and number of trees
n_estimators = range(50, 250, 50)
max_depth = range(2, 10, 2)
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
means, stds = subsample_method(param_grid, model, x, label_encoded_y)
# Need a different way to plot these results
scores = np.array(means).reshape(len(max_depth), len(n_estimators))
for i, value in enumerate(max_depth):
    plt.plot(n_estimators, scores[i], label='depth: ' + str(value))
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('Log Loss')
plt.show()