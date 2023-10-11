import numpy as np
from scipy.optimize import check_grad
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from skopt.learning.gaussian_process.kernels import RBF, WhiteKernel, Matern



def gpr_mean_grad(X_test, gpr):
    X_train = gpr.X_train_
    kernel = gpr.kernel_

    #kernel_1, white = kernel.k1, kernel.k2
    alpha = gpr.alpha_
    gradients = []
    for x_star in X_test:
        # Compute the gradient for x_star across all training data
        # Only need grad of kernel_1, white is constant
        k_gradient_matrix = kernel.gradient_x(x_star, X_train)

        # Multiply the gradient matrix with alpha and sum across training data
        grad_sum = np.dot(alpha, k_gradient_matrix)

        # Adjust for normalization
        grad_sum_adjusted = grad_sum * gpr._y_train_std

        gradients.append(grad_sum_adjusted)

    return np.array(gradients).ravel()

def f(x):
    return np.sin(2 * np.pi * x)

def grad_f(x):
    return 2 * np.pi * np.cos(2 * np.pi * x)

# Generate sample data from the function
X_train = np.linspace(0, 1, 100)[:, np.newaxis]
y_train = f(X_train).ravel()

# Fit the GPR model to the sample data
kernel = RBF()
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gpr.fit(X_train, y_train)

score = gpr.score(X_train, y_train)
print("Model R-squared score on training set:", score)


X_test = np.linspace(0, 1, 100)[:, np.newaxis]
y_test = f(X_test).ravel()
gpr_grad = gpr_mean_grad(X_test, gpr)
actual_grad = grad_f(X_test)

# gpr_func = gpr.predict(X_test, gpr)
# actual_func = f(X_test)
y_pred = gpr.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)

error = check_grad(lambda x: gpr.predict(x.reshape(1, -1)), lambda x: gpr_mean_grad(x, gpr), np.array([0.4]))
error_2 = gpr_mean_grad(np.array([0.4]), gpr) - grad_f(np.array([0.4]))
print("Gradient error:", error)
#print("Gradient error2:", error_2)
"""
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
"""