import pathlib

from scipy.linalg import cho_solve
from scipy.optimize import check_grad
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skopt.learning.gaussian_process.kernels import RBF, WhiteKernel, Matern


def gpr_mean_grad(X_test, gpr):
    X_train = gpr.X_train_
    kernel = gpr.kernel_

    kernel_1, white = kernel.k1, kernel.k2
    alpha = gpr.alpha_
    gradients = []
    for x_star in X_test:
        # Compute the gradient for x_star across all training data
        # Only need grad of kernel_1, white is constant
        k_gradient_matrix = kernel_1.gradient_x(x_star, X_train)

        # Multiply the gradient matrix with alpha and sum across training data
        grad_sum = np.dot(alpha, k_gradient_matrix)

        # Adjust for normalization
        grad_sum_adjusted = grad_sum * gpr._y_train_std

        gradients.append(grad_sum_adjusted)

    return np.array(gradients).ravel()


def gpr_variance_grad(X_test, gpr):
    X_train = gpr.X_train_
    kernel = gpr.kernel_

    # Decompose the kernel into its constituent parts
    kernel_1, white = kernel.k1, kernel.k2

    # Compute the noise term from the WhiteKernel
    sigma2 = white.noise_level
    # Adjust K_inv to account for the noise term
    L_inv = np.linalg.inv(gpr.L_)
    K_inv = L_inv.T.dot(L_inv)

    gradients = []
    for x_star in X_test:
        dk_xx = kernel_1.gradient_x(x_star, x_star.reshape(1, -1))
        # Compute the kernel vector k(x_star)
        k_x_star = kernel_1(x_star.reshape(1, -1), X_train).ravel()
        # Compute the gradient of kernel vector
        dk_x_star = kernel_1.gradient_x(x_star, X_train)

        # Gradient of the variance using the product rule
        grad_variance = dk_xx - (np.dot(dk_x_star.T, np.dot(K_inv, k_x_star)) +
                                 np.dot(k_x_star.T, np.dot(K_inv, dk_x_star)))

        grad_variance_adjusted = grad_variance * gpr._y_train_std**2

        gradients.append(grad_variance_adjusted)

    return np.array(gradients).ravel()


considered_features = [
    'volume_fraction_4', 'volume_fraction_1',
]

training_data = pathlib.Path('data/training_data_rve_database.npy')
if not training_data.exists():
    print(f"Error: training data path {training_data} does not exist.")

if training_data.suffix == '.npy':
    data = np.load(training_data)
else:
    print("Invalid data")

print(f"loaded {data.shape[0]} training data pairs")

data['thermal_expansion'] *= 1e6
considered_features = [
    'volume_fraction_4', 'volume_fraction_1',
    'chord_length_mean_4', 'chord_length_mean_10', ]
considered_properties = [
    'thermal_conductivity',
    'thermal_expansion',
    'young_modulus',
    'poisson_ratio',
]
X = np.vstack(tuple(data[f] for f in considered_features)).T
Y = np.vstack(tuple(data[p] for p in considered_properties)).T
assert Y.shape[0] == X.shape[0], "number of samples does not match"



# Objective function
def objective(x):
    x = x.reshape(1, -1)  # Reshape because sklearn expects 2D array
    return gpr.predict(x)[0]

# Gradient function
def gradient(x):
    X = x.reshape(1, -1)
    return gpr_mean_grad(X_scaled, gpr)


y = Y[:, 1]
y_float = np.array([float(val) if val != b'nan' else np.nan for val in y])

# ignore NaNs in the data
clean_indices = np.argwhere(~np.isnan(y_float))
Y = y_float[clean_indices.flatten()]
X = X[clean_indices.flatten()]

kernel = Matern() + WhiteKernel()
pipe = make_pipeline(
    StandardScaler(),  # scaler for data normalization
    GaussianProcessRegressor(kernel=kernel, normalize_y=True)
)
pipe.fit(X, Y)
score = pipe.score(X, Y)
print("Model R-squared score on training set:", score)

scaler = pipe.named_steps['standardscaler']
gpr = pipe.named_steps['gaussianprocessregressor']


random_index = np.random.randint(0, X.shape[0])
x0 = X[2]
print(x0)

# Flatten x0 before passing it to check_grad
x0_flat = x0.flatten()

# Make sure that the lambda functions inside check_grad also expect a flat 1D array
error = check_grad(lambda x: gpr.predict(x.reshape(1, -1), return_std=True)[0],
                   lambda x: gpr_mean_grad(x.reshape(1, -1), gpr),
                   x0_flat)
print("Gradient error:", error)

error_variance = check_grad(lambda x: gpr.predict(x.reshape(1, -1), return_std=True)[1]**2,
                            lambda x: gpr_variance_grad(x.reshape(1, -1), gpr),
                            x0_flat)
print("Variance gradient error:", error_variance)
