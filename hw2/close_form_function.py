import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
test_data = pd.read_csv("HW-1-Data-1/test_data.csv")
print(test_data.columns)
test_data['intercept'] = 1
x = test_data[['x', 'intercept']]

Y = test_data['y']

# a)
theta_and_intercept = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, Y))
theta = theta_and_intercept[0]
print(theta)
intercept = theta_and_intercept[1]
train_data = pd.read_csv("HW-1-Data-1/train_data.csv")

train_data['y_prediction'] = theta*train_data['x']+intercept
plt.scatter(train_data['x'], train_data['y'])
plt.plot(train_data['x'], train_data['y_prediction'], '-r', label='y=2x+1')
plt.savefig('a.png')
plt.close()
print(train_data)

