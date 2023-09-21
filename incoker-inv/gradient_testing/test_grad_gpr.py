import pathlib

from scipy.optimize import check_grad
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skopt.learning.gaussian_process.kernels import RBF, WhiteKernel


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
        grad_sum_adjusted = grad_sum

        gradients.append(grad_sum_adjusted)

    return np.array(gradients).ravel()


considered_features = [
    'volume_fraction_4', 'volume_fraction_1',
]

training_data = pathlib.Path('../data/training_data_rve_database.npy')
if not training_data.exists():
    print(f"Error: training data path {training_data} does not exist.")

if training_data.suffix == '.npy':
    data = np.load(training_data)
else:
    print("Invalid data")

print(f"loaded {data.shape[0]} training data pairs")

data['thermal_expansion'] *= 1e6
considered_features = [
    'volume_fraction_4',
]
considered_properties = [
    'thermal_conductivity',
    'thermal_expansion',
    'young_modulus',
    'poisson_ratio',
]
X = np.vstack(tuple(data[f] for f in considered_features))
Y = np.vstack(tuple(data[p] for p in considered_properties)).T
assert Y.shape[0] == X.shape[0], "number of samples does not match"



# Objective function
def objective(x):
    x = x.reshape(1, -1)  # Reshape because sklearn expects 2D array
    return gpr.predict(x)[0]

# Gradient function
def gradient(x):
    X = x.reshape(1, -1)
    X_scaled = scaler.transform(X)
    return gpr_mean_grad(X_scaled, gpr)


y = Y[:, 0]
y_float = np.array([float(val) if val != b'nan' else np.nan for val in y])

# ignore NaNs in the data
clean_indices = np.argwhere(~np.isnan(y_float))
Y = y_float[clean_indices.flatten()]
X = X[clean_indices.flatten()]

kernel = RBF(length_scale=1) + WhiteKernel(noise_level=1)
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
x0 = X[random_index]

# Check the gradient at that point
error = check_grad(objective, gradient, x0)
print("Gradient error:", error)