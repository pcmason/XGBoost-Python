'''
Example code working with LabelEncoders for the iris dataset, OneHotEncoders with the breast cancer dataset since it has
only categorical input data, and finally SimpleImputer for missing data with the horse-colic dataset.
'''

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


# Method to evaluate the model
def evalModel(x, y):
    # Split data into train & test split
    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=0.33, random_state=7)
    # Fit model on training data
    model = xgb.XGBClassifier()
    model.fit(xTrain, yTrain)
    print('\n', model)
    # Make prediction for test data
    yPred = model.predict(xTest)
    predictions = [round(value) for value in yPred]
    # Evaluate predictions
    accuracy = accuracy_score(yTest, predictions)
    print('\nAccuracy: %.3f' % (accuracy * 100.0))


# Method for Label Encoding since it occurs multiple times
def yLabelEncoder(y):
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y)
    labelEncodedY = label_encoder.transform(y)
    return labelEncodedY


# Method to load in data and return x & y variables
def load_data(file):
    # Load dataset
    data = pd.read_csv(file, header=None)
    dataset = data.values
    # Split into x & y
    xLen = len(dataset[0]) - 1
    x = dataset[:, 0:xLen]
    y = dataset[:, xLen]
    return x, y


# Use LabelEncoder on iris dataset
# Load iris dataset
irisX, irisY = load_data('iris.csv')
# Encode string class values as integers
ley = yLabelEncoder(irisY)
# Evaluate iris dataset
evalModel(irisX, ley)


# Use OneHotEncoder on breast cancer dataset
bcX, bcY = load_data('breast-cancer.csv')
bcX = bcX.astype(str)
# Encode string input values as integers
encodedX = None
for i in range(0, bcX.shape[1]):
    label_encoder = LabelEncoder()
    feature = label_encoder.fit_transform(bcX[:, i])
    # Transform feature array into 2D numpy array where each integer value is a feature vector with a length of 1
    feature = feature.reshape(bcX.shape[0], 1)
    # Create OneHotEncoder and encode feature array
    onehot_encoder = OneHotEncoder(categories='auto', sparse_output=False)
    feature = onehot_encoder.fit_transform(feature)
    # Concatenate the new encodedX value or make it the feature if this is the first loop
    if encodedX is None:
        encodedX = feature
    else:
        encodedX = np.concatenate((encodedX, feature), axis=1)

print('\nX Shape: ', encodedX.shape)
# Encode string class values as integers
ley = yLabelEncoder(bcY)
# Evaluate breast cancer dataset
evalModel(encodedX, ley)


# Impute missing values in horse-colic dataset
# Load data (cannot use method since whitespace is the delimiter)
df = pd.read_csv('horse-colic.data.csv', delim_whitespace=True, header=None)
ds = df.values
# Split into x & y
horseX = ds[:, 0:27]
horseY = ds[:, 27]
# Set missing values to 0
horseX[horseX == '?'] = np.nan
# Convert to numeric
horseX = horseX.astype('float32')
# Impute missing values as the mean
imputer = SimpleImputer()
imputedX = imputer.fit_transform(horseX)
# Encode Y class values as integers
ley = yLabelEncoder(horseY)
# Evaluate horse-colic model
evalModel(imputedX, ley)

