import pandas as pd
from sgd import stochastic_gradient_descent
stochastic_gradient_descent


def predict(x, w, b):

    y = w[0] * x + b
    return y


test_data = pd.read_csv("HW-1-Data-1/test_data.csv")

x = test_data['x'].values
x = x.reshape(x.shape[0], 1)
Y = test_data['y'].values
Y = Y.reshape(Y.shape[0],)
w_sgd, b_sgd, cost_sgd, cost_list_sgd, epoch_list_sgd = stochastic_gradient_descent(
    x, Y.reshape(Y.shape[0],), 10000, 10)

print("predicted data is", predict(-2.3740563, w_sgd, b_sgd))
