from sklearn.metrics import accuracy_score
from DecisionTreeClassifier import DecisionTreeClassifier
import numpy as np
import pandas as pd
import math
import random
df = pd.read_csv("PS4-1/PS4_1_X_Train.csv")
X = df.values
y_train = pd.read_csv("PS4-1/PS4_2_Y_Train.csv")
y = y_train.values
number = len(y)
print(y)
count = 0
for i in y:
    if i == [0]:
        count += 1
print(count)
for i in range(20):
    ran_num = random.randint(0, count)
    y[ran_num] = [1]

classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=6)
classifier.fit(X, y)
classifier.print_tree()
