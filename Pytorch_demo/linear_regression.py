# 1 design the model (input, output size, forward pass)
# 2 construct loss function and optimizer
# 3 Trading loop
#  - forward pass: compute prediction and loss
#  - back pass: gradients
#  - update weights


import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# 0 prepare the data
X_numpy, y_numpy = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
print(type(y))  # this is a tensor
y = y.view(y.shape[0], 1)  # This is similar to reshape function
print(y.shape)
n_samples, n_feathers = X.shape

# 1 model
input_size = n_feathers
output_size = 1
learning_rate = 0.01

model = nn.Linear(input_size, output_size)

# 2 construct loss function and optimizer
# you can find a loss function here https://pytorch.org/docs/stable/nn.functional.html#loss-functions
criterion = nn.MSELoss()
# you can find a optimizer here https://pytorch.org/docs/stable/optim.html
# stochastic gradient descent
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3 training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    # backward pass
    # print(loss)
    loss.backward()
    # update
    optimizer.step()
    # Sets the gradients of all optimized torch.Tensor s to zero.
    optimizer.zero_grad()
    if (epoch+1) % 10 == 0:
        print(f"epoch:{epoch+1}, loss={loss.item():.4f}")

# we want to convert it to numpy back again. but before we do we nened to detach our
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, "b")
plt.savefig("pytorch_lr_demo.png")

# for me , sklearn is more like a black box, we input the parameter, and we got the result.
# However when I use pytorch I feel like I'm doing the linear regression with the model,
# #I feel like I have deeper understanding for models when I use pytorch,
# #What's more is I feel like that pytorch has more flexibility than sklearn.
# #It's definitely a better tool module for doing homework this course
