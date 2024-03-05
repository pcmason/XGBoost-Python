'''
Example code manually and using built-in xgboost method to plot feature importance for the Pimas Diabetes dataset.
Use feature importance to evaluate model on different number of features for feature selection.
'''

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

# Load data
dataset = np.loadtxt('pimas-indians-diabetes.csv', delimiter=',')
# Split into x & y
x = dataset[:, 0:8]
y = dataset[:, 8]
# Fit model on training data
model = xgb.XGBClassifier()
model.fit(x, y)
# Feature importance
print(model.feature_importances_, '\n')


# Manually plot feature importance
# Plot feature importance as bar graph
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.show()


# Use xgboost method to plot feature importance
# Plot feature importance
xgb.plot_importance(model)
plt.show()


# Now use feature importance for feature selection
# Split data into train & test splits
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=7)
model.fit(x_train, y_train)
# Fit model using each importance as a threshold
thresholds = np.sort(model.feature_importances_)
for thresh in thresholds:
    # Select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    # Train model
    selection_model = xgb.XGBClassifier()
    selection_model.fit(select_x_train, y_train)
    # Evaluate model using 1-8 features
    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print('Thresh=%.3f, n=%d, Accuracy: %.3f%%' % (thresh, select_x_train.shape[1], accuracy*100.0))

