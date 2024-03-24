'''
Example code using the Otto Group Project 10-class classification training dataset to show off the efficacy of parallel
processing in the XGBoost library. Different number of cores will be used in the XGBoost model, the 10-fold CV and for
both with time of each being output to show impact.
'''
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
import time
import matplotlib.pyplot as plt

# Create a method for evaluating multithreading
def multithread_eval(xg, cv, x, y, kfold):
    # Track length of process
    start = time.time()
    # Create model based on number of threads (xg)
    model = xgb.XGBClassifier(nthread=xg)
    # Same for CV
    results = cross_val_score(model, x, y, cv=kfold, scoring='neg_log_loss', n_jobs=cv)
    elapsed = time.time() - start
    # Output result
    print('%d thread XGBoost, %d thread CV: %f' % (xg, cv, elapsed))


# Load data
data = pd.read_csv('Data/train.csv')
dataset = data.values
# Split into x and y
x = dataset[:, 0:94]
y = dataset[:, 94]
# Encode string class values as integers
label_encoded_y = LabelEncoder().fit_transform(y)

# Evaluate impact of number of threads
results = []
num_threads = [1, 2, 3, 4]
for n in num_threads:
    # Time how long each model takes to fit
    start = time.time()
    # Set number of threads to n
    model = xgb.XGBClassifier(nthread=n)
    model.fit(x, label_encoded_y)
    elapsed = time.time() - start
    print(n, elapsed)
    results.append(elapsed)
# Plot the results
plt.plot(num_threads, results)
plt.ylabel('Speed (seconds)')
plt.xlabel('Number of Threads')
plt.title('XGBoost Training Speed vs Number of Threads')
plt.show()

# Now implement parallelism for xgboost, CV & both and see which is fastest
# Prepare CV
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# Single thread XGBoost, parallel thread CV
multithread_eval(1, -1, x, label_encoded_y, kfold)
# Parallel thread XGBoost, single thread CV
multithread_eval(-1, 1, x, label_encoded_y, kfold)
# Parallel thread XGBoost & CV
multithread_eval(-1, -1, x, label_encoded_y, kfold)
