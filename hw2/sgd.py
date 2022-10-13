import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def stochastic_gradient_descent(X, y_true, epochs, m, learning_rate=0.01):
    # m is the bash size
    number_of_features = X.shape[1]
    # numpy array with 1 row and columns equal to number of features.
    w = np.ones(shape=(number_of_features))
    b = 0
    total_samples = X.shape[0]  # number of rows in X

    cost_list = []
    epoch_list = []

    for i in range(epochs):
        # random index from total samples
        random_index = np.random.randint(0, total_samples-1)
        sample_x = X[random_index]
        sample_y = y_true[random_index]

        y_predicted = np.dot(w, sample_x.T) + b

        # calculate the derivative, find the gradient
        w_grad = -(2/total_samples)*(sample_x.T.dot(sample_y-y_predicted))
        b_grad = -(2/total_samples)*(sample_y-y_predicted)

        w = w - learning_rate * w_grad  # Adjusted weight
        b = b - learning_rate * b_grad

        cost = np.square(sample_y-y_predicted)  # MSE (Mean Squared Error)

        if i % m == 0:  # at every 100th iteration record the cost and epoch value
            cost_list.append(cost)
            epoch_list.append(i)

    return w, b, cost, cost_list, epoch_list


test_data = pd.read_csv("HW-1-Data-1/test_data.csv")

x = test_data['x'].values
x = x.reshape(x.shape[0], 1)
Y = test_data['y'].values
Y = Y.reshape(Y.shape[0],)
w_sgd, b_sgd, cost_sgd, cost_list_sgd, epoch_list_sgd = stochastic_gradient_descent(
    x, Y.reshape(Y.shape[0],),10000,10)

plt.xlabel("epoch")
plt.ylabel("cost")
plt.plot(epoch_list_sgd, cost_list_sgd)
plt.savefig('b.png')
plt.close()
