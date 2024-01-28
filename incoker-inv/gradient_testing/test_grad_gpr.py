import pathlib

import joblib
from matplotlib import pyplot as plt
from pyDOE import lhs
from scipy.optimize import check_grad, approx_fprime
import numpy as np



def objective_function(x, desired_property, models, property_name, callback=None):
    if callback is not None:
        callback(x)
    predicted_property, uncertainty  = models[property_name]['pipe'].predict(x.reshape(1, -1), return_std=True)
    discrepancy = predicted_property - desired_property

    return (discrepancy ** 2) + uncertainty[0]*0.1


def objective_gradient(x, desired_property, models, property_name):

    predicted_property, std, gpr_grad, gpr_var_grad  = models[property_name]['pipe'].predict(x.reshape(1, -1), return_mean_grad=True,return_std=True, return_std_grad=True)
    discrepancy = predicted_property - desired_property

    # Retrieve standard deviation from the StandardScaler
    scaler = pipe.named_steps['standardscaler']
    std_dev = scaler.scale_

    # Adjust gradients
    adjusted_gpr_grad = gpr_grad / std_dev
    adjusted_gpr_var_grad = gpr_var_grad / std_dev

    return (2 * discrepancy * adjusted_gpr_grad) + adjusted_gpr_var_grad * 0.1



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


# Change next line for different feature sets from models folder
models = joblib.load("models/2d_matern.joblib")["models"]
property_name = 'thermal_conductivity'

X = models[property_name]['X_train']
X_test = models[property_name]['X_test']
Y = models[property_name]['y_train']
y_test = models[property_name]['y_test']

pipe = models[property_name]['pipe']
# Retrieve standard deviation from the StandardScaler
scaler = pipe.named_steps['standardscaler']
std_dev = scaler.scale_

random_index = np.random.randint(0, X.shape[0])
x0 = X[11]
print(x0)

# Flatten x0 before passing it to check_grad
x0_flat = x0.flatten()
prop = 5.9



print(pipe.predict(x0_flat.reshape(1, -1), return_mean_grad=True,return_std=True, return_std_grad=True))
# Make sure that the lambda functions inside check_grad also expect a flat 1D array

error = check_grad(lambda x: models[property_name]['pipe'].predict(x.reshape(1, -1), return_mean_grad=True)[0],
                   lambda x: models[property_name]['pipe'].predict(x.reshape(1, -1), return_mean_grad=True)[1]/std_dev,
                   x0_flat)
print("Gradient error:", error)

error_variance = check_grad(lambda x: models[property_name]['pipe'].predict(x.reshape(1, -1), return_mean_grad=True,return_std=True, return_std_grad=True)[1],
                            lambda x: (models[property_name]['pipe'].predict(x.reshape(1, -1), return_mean_grad=True,return_std=True, return_std_grad=True)[3]/std_dev),
                            x0_flat)
print("Variance gradient error:", error_variance)

error_obj = check_grad(
    lambda x: objective_function(x, prop, models, property_name),
    lambda x: objective_gradient(x, prop, models, property_name),
    x0_flat,
)
print("Variance gradient error:", error_obj)



property_ax_dict = {
    'thermal_conductivity':'CTC [W/(m*K)]',
    'thermal_expansion':'CTE [ppm/K]',
    'young_modulus':'Young\'s Modulus[GPa]',
    'poisson_ratio':'Poisson Ratio',
}

property_dict = {
    'thermal_conductivity':'CTC',
    'thermal_expansion':'CTE',
    'young_modulus':'Young\'s Modulus',
    'poisson_ratio':'Poisson Ratio',
}
def find_closest_point(Xt, point, selected_indices):
    distances = np.linalg.norm(Xt - point, axis=1)
    while True:
        index = np.argmin(distances)
        if index not in selected_indices:
            selected_indices.append(index)
            return Xt[index], selected_indices
        else:
            distances[index] = np.inf
# Compute the minimum and maximum values for each feature in the training data
min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)
features = models[property_name]['features']
print("Features: ", str(features))

# Define the search space dimensions based on the minimum and maximum values
bounds = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]
num_samples = 30
# LHS sampling for uniform starting points in multi-start optimization
lhs_samples = lhs(len(bounds), samples=num_samples)
for i in range(len(bounds)):
    lhs_samples[:, i] = lhs_samples[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
# Find the closest points in X to the LHS samples
selected_indices = []

Xt_initial = np.zeros((num_samples, X.shape[1]))  # Initialize closest points array
for i in range(num_samples):
    Xt_initial[i], selected_indices = find_closest_point(X, lhs_samples[i], selected_indices)
initial_points = X[selected_indices]
print(initial_points)
exact_gradients = []
approx_gradients = []
for point in initial_points:
    # Make sure that the lambda functions inside check_grad also expect a flat 1D array
    error = check_grad(lambda x: models[property_name]['pipe'].predict(x.reshape(1, -1), return_mean_grad=True)[0],
                       lambda x: models[property_name]['pipe'].predict(x.reshape(1, -1), return_mean_grad=True)[1]/std_dev,
                       point)
    print("Gradient error:", error)

    error_variance = check_grad(
        lambda x: models[property_name]['pipe'].predict(x.reshape(1, -1), return_mean_grad=True, return_std=True, return_std_grad=True)[1],
        lambda x: (models[property_name]['pipe'].predict(x.reshape(1, -1), return_mean_grad=True, return_std=True, return_std_grad=True)[3]/std_dev),
        point)
    print("Variance gradient error:", error_variance)

    error_obj = check_grad(
        lambda x: objective_function(x, prop, models, property_name),
        lambda x: objective_gradient(x, prop, models, property_name),
        point,
    )
    print("obj gradient error:", error_obj)

    # Exact gradient of variance
    _, _, mean_grad, std_grad = models[property_name]['pipe'].predict(point.reshape(1, -1), return_mean_grad=True, return_std=True, return_std_grad=True)

    exact_gradients.append(std_grad)

    # Approximated gradient using numerical differentiation
    fprime = lambda x: models[property_name]['pipe'].predict(x.reshape(1, -1), return_mean_grad=True, return_std=True, return_std_grad=True)[1]
    approx_grad = approx_fprime(point.flatten(), fprime)
    approx_gradients.append(approx_grad)

# Convert lists to numpy arrays for plotting
exact_gradients = np.array(exact_gradients)
approx_gradients = np.array(approx_gradients)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(initial_points, exact_gradients, label='Exact Gradient', color='blue')
plt.plot(initial_points, approx_gradients, label='Approximated Gradient', color='red', linestyle='dashed')
plt.xlabel('Input Point')
plt.ylabel('Gradient')
plt.title('Comparison of Exact and Approximated Variance Gradients in GPR')
plt.legend()
plt.show()
