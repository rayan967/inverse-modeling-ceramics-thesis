import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from skopt.learning.gaussian_process.kernels import RBF, WhiteKernel


def gpr_mean_grad(X_test, gpr):
    X_train = gpr.X_train_
    kernel = gpr.kernel_

    rbf, white = kernel.k1, kernel.k2
    l = rbf.length_scale
    alpha = gpr.alpha_

    # Compute the gradient for each test point
    gradients = []
    for x_star in X_test:
        grad_sum = 0
        for i, x_i in enumerate(X_train):
            diff = (x_star - x_i)
            k_gradient = -diff * np.exp(-np.sum(diff ** 2) / (2 * l ** 2)) / l ** 2
            grad_sum += alpha[i] * k_gradient
        gradients.append(grad_sum)

    return np.array(gradients).ravel()


def f(x):
    return np.sin(2 * np.pi * x)

def grad_f(x):
    return 2 * np.pi * np.cos(2 * np.pi * x)

# Generate sample data from the function
X_train = np.linspace(0, 1, 15)[:, np.newaxis]
y_train = f(X_train).ravel()

# Fit the GPR model to the sample data
kernel = RBF(length_scale=1) + WhiteKernel(noise_level=1)
gpr = GaussianProcessRegressor(kernel=kernel)
gpr.fit(X_train, y_train)

score = gpr.score(X_train, y_train)
print("Model R-squared score on training set:", score)

X_test = np.linspace(0, 1, 100)[:, np.newaxis]
gpr_grad = gpr_mean_grad(X_test, gpr)
actual_grad = grad_f(X_test)

# gpr_func = gpr.predict(X_test, gpr)
# actual_func = f(X_test)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(X_test, gpr_grad, label='GPR-derived Gradient', linestyle='--', color='blue')
plt.plot(X_test, actual_grad, label='Actual Gradient', color='red')
plt.xlabel('x')
plt.ylabel('Gradient Value')
plt.title('Comparison of GPR-derived Gradient and Actual Gradient')
plt.legend()
plt.grid(True)
plt.show()