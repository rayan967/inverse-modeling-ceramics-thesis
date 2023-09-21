import numpy as np
from scipy.optimize import check_grad
from sklearn.gaussian_process import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF, WhiteKernel, Matern


def gpr_mean_grad(X_test, gpr):
    X_train = gpr.X_train_
    kernel = gpr.kernel_

    kernel1, white = kernel.k1, kernel.k2
    alpha = gpr.alpha_


    gradients = []
    for x_star in X_test:
        # Compute the gradient for x_star across all training data
        k_gradient_matrix = kernel1.gradient_x(x_star, X_train)

        # Multiply the gradient matrix with alpha and sum across training data
        grad_sum = np.dot(alpha, k_gradient_matrix)

        # Adjust for normalization
        grad_sum_adjusted = gpr._y_train_std * grad_sum + gpr._y_train_mean

        gradients.append(grad_sum_adjusted)

    return np.array(gradients).ravel()


def f(x):
    return np.sin(2 * np.pi * x)

def grad_f(x):
    return 2 * np.pi * np.cos(2 * np.pi * x)

# Generate sample data from the function
X_train = np.linspace(0, 1, 15)[:, np.newaxis]
y_train = f(X_train).ravel()

# Fit the GPR model to the sample data
kernel = Matern(length_scale=1, nu=2.5) + WhiteKernel(noise_level=1)
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gpr.fit(X_train, y_train)

score = gpr.score(X_train, y_train)
print("Model R-squared score on training set:", score)

X_test = np.linspace(0, 1, 100)[:, np.newaxis]
gpr_grad = gpr_mean_grad(X_test, gpr)
actual_grad = grad_f(X_test)

# gpr_func = gpr.predict(X_test, gpr)
# actual_func = f(X_test)

error = check_grad(f, grad_f, 4)
print("Gradient error:", error)

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