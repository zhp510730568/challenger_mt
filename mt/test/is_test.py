import numpy as np

draw_num = 10000000
uniform_draws = np.random.uniform(0, 1, draw_num)

X = uniform_draws + 2


def f_x(x):
    return np.power(x, 3)


result = (3 - 2) / draw_num * np.sum(f_x(X))
print('approx value: ', result)
print('real value: ', float(65 / 4))
