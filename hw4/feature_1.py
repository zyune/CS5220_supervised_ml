from sklearn.metrics import accuracy_score
from DecisionTreeClassifier import DecisionTreeClassifier
import numpy as np
import pandas as pd
import math
# df = pd.read_csv("PS4-1/PS4_1_X_Train.csv")
# y_train = pd.read_csv("PS4-1/PS4_2_Y_Train.csv")
# df['target'] = y_train['0']
# df = df.sort_values(by="-0.21961")
# X = df['-0.21961'].values
# X = X.reshape(-1, 1)
# y = df['target'].values
# y = y.reshape(-1, 1)
# num_of_test_data = len(X)
# gap = math.ceil(num_of_test_data/20)
# X = X[::gap]
# y = y[::gap]
df = pd.read_csv("PS4-1/PS4_1_X_Train.csv")
X = df.values
y_train = pd.read_csv("PS4-1/PS4_2_Y_Train.csv")
y = y_train.values
len(y)
# num_of_test_data = len(X)
# gap = math.ceil(num_of_test_data/20)
# X = X[::gap]
# y = y[::gap]
classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=6)
classifier.fit(X, y)
classifier.print_tree()
X_test = pd.read_csv("PS4-1/PS4_2_X_Test.csv")
X_test = X_test.iloc[:, :].values
Y_test = pd.read_csv("PS4-1/PS4_2_Y_Test.csv")
y = Y_test.iloc[:, -1].values.reshape(-1, 1)
Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
