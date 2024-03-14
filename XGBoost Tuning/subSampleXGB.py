'''
Example code using 3 different methods of subsampling on the Otto Group Project 10 class classification dataset. The
three methods used are: subsampling of rows, subsampling of columns & subsampling of columns at each decision tree split
point.
'''

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# Create method to subsample different parameters using grid search and output results
def subsample_method(grid, label_name):
    # Use sklearn's grid search capability
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, grid, scoring='neg_log_loss', n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(x, label_encoded_y)
    # Summarize results
    print('\nBest: %f using %s\n' % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('%f (%f) with: %r' % (mean, stdev, param))
    # Plot the logloss and subsamples on a line graph
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

# Subsampling of the rows
subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
param_grid = dict(subsample=subsample)
subsample_method(param_grid, 'subsample')

# Subsampling of the columns
colsample_bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
param_grid = dict(colsample_bytree=colsample_bytree)
subsample_method(param_grid, 'colsample_bytree')

# Subsampling of columns at split points for each tree
colsample_bylevel = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
param_grid = dict(colsample_bylevel=colsample_bylevel)
subsample_method(param_grid, 'colsample_bylevel')


