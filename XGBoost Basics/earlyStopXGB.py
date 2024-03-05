'''
Implementing early stopping while evaluating the Pimas Diabetes dataset to help improve XGBoost model performancce.
Also tracks logloss & classification accuracy over training and plots as line graph.
'''

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Method to output classification accuracy
def evalModel(x_test, y_test, model):
    # Make predictions for test data
    y_pred = model.predict(x_test)
    predictions = [round(value) for value in y_pred]
    # Evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print('\nAccuracy: %.3f%%\n' % (accuracy * 100.0))


# Method to plot line graph of metrics
def plot_metric(metric):
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0'][metric], label='Train')
    ax.plot(x_axis, results['validation_1'][metric], label='Test')
    ax.legend()
    plt.ylabel(metric)
    plt.title('XGBoost %s' % metric)
    plt.show()


# Load data
dataset = np.loadtxt('pimas-indians-diabetes.csv', delimiter=',')
# Split into x & y
x = dataset[:, 0:8]
y = dataset[:, 8]
# Split into train & test splits
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=7)


# Get performance of model throughout training without early stopping
model = xgb.XGBClassifier()
eval_set1 = [(x_test, y_test)]
model.fit(x_train, y_train, eval_metric='error', eval_set=eval_set1, verbose=True)
# Make predictions for test data
evalModel(x_test, y_test, model)


# Now track logloss & classification accuracy & output as line graphs
eval_s = [(x_train, y_train), (x_test, y_test)]
model.fit(x_train, y_train, eval_metric=['error', 'logloss'], eval_set=eval_s, verbose=True)
# Evaluate model
evalModel(x_test, y_test, model)
# Retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# Plot logloss
plot_metric('logloss')
# Plot classification error
plot_metric('error')


# Use early stopping on the evaluation
model.fit(x_train, y_train, early_stopping_rounds=10, eval_metric='logloss', eval_set=eval_set1, verbose=True)
# Evaluate early stopping model
evalModel(x_test, y_test, model)

