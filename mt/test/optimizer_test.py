import numpy as np
import matplotlib.pyplot as plt


def samples(batch_size = 100):
    X = np.random.uniform(-10, 10, batch_size)
    Y = X * 2 + 3 + np.random.normal(0, 1, batch_size)
    return X, Y


def loss(true_Y, Y):
    return np.power(true_Y - Y, 2)


def f1(x, w, b):
    return x * w + b


def param_grad_w(y, x, w, b):
    return - 2 * x * (y - (x * w + b))


def param_grad_b(y, x, w, b):
    return - 2 * (y - (x * w + b))


learning_rate = 0.1
epoches = 1000
batch_size = 1000
w = np.random.uniform(0, 1)
b = 0

test_X = np.linspace(-1, 1, 1000)

losses = []
losses_X = []
X, Y = samples(batch_size=batch_size)
for epoch in range(epoches):
    grad_w = param_grad_w(Y, X, w, b)
    grad_b = param_grad_b(Y, X, w, b)
    w -= learning_rate * np.mean(grad_w, axis=0) / batch_size
    b -= learning_rate * np.mean(grad_b, axis=0) / batch_size
    print(w, b)
    losses_X.append(epoch)
    current_loss =loss(f1(test_X, 2, 3), f1(test_X, w, b))
    mean_loss = np.mean(current_loss, axis=0)
    print('loss value: ', mean_loss)
    losses.append(mean_loss)

plt.plot(losses_X, losses)
plt.show()
