import argparse
import pathlib

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from standard_training import extract_XY, extract_XY_3, extract_XY_2
from skopt import Optimizer
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt



def find_closest_point(Xt, yt, point, selected_indices):
    distances = np.linalg.norm(Xt - point, axis=1)
    while True:
        index = np.argmin(distances)
        if index not in selected_indices:
            selected_indices.append(index)
            return Xt[index], yt[index], selected_indices
        else:
            distances[index] = np.inf


def gradient_function(x, models, property_name):
    model = models[property_name]['pipe']
    features = models[property_name]['features']

    # Convert x to the relevant microstructure features
    microstructure_features = [x[i] for i in range(len(features))]
    X = np.array(microstructure_features).reshape(1, -1)

    # Access the scaler and GPR from the pipeline
    scaler = model.named_steps['standardscaler']
    gpr = model.named_steps['gaussianprocessregressor']

    # Scale the input data
    X_scaled = scaler.transform(X)

    # Compute the gradient using the scaled data and GPR
    return gpr_mean_grad(X_scaled, gpr)


def convert_x_to_microstructure(x, features):
    if len(features) == 9:
        volume_fraction_4 = x[0]
        volume_fraction_10 = x[1]
        volume_fraction_1 = x[2]
        chord_length_mean_4 = x[3]
        chord_length_mean_10 = x[4]
        chord_length_mean_1 = x[5]
        chord_length_variance_4 = x[6]
        chord_length_variance_10 = x[7]
        chord_length_variance_1 = x[8]

        # Construct the microstructure representation
        microstructure = {
            'volume_fraction_4': volume_fraction_4,
            'volume_fraction_10': volume_fraction_10,
            'volume_fraction_1': volume_fraction_1,
            'chord_length_mean_4': chord_length_mean_4,
            'chord_length_mean_10': chord_length_mean_10,
            'chord_length_mean_1': chord_length_mean_1,
            'chord_length_variance_4': chord_length_variance_4,
            'chord_length_variance_10': chord_length_variance_10,
            'chord_length_variance_1': chord_length_variance_1,
        }
    elif len(features) == 8:
        volume_fraction_4 = x[0]
        volume_fraction_1 = x[1]
        chord_length_mean_4 = x[2]
        chord_length_mean_10 = x[3]
        chord_length_mean_1 = x[4]
        chord_length_variance_4 = x[5]
        chord_length_variance_10 = x[6]
        chord_length_variance_1 = x[7]

        # Construct the microstructure representation
        microstructure = {
            'volume_fraction_4': volume_fraction_4,
            'volume_fraction_1': volume_fraction_1,
            'chord_length_mean_4': chord_length_mean_4,
            'chord_length_mean_10': chord_length_mean_10,
            'chord_length_mean_1': chord_length_mean_1,
            'chord_length_variance_4': chord_length_variance_4,
            'chord_length_variance_10': chord_length_variance_10,
            'chord_length_variance_1': chord_length_variance_1,
        }
    elif len(features) == 3:
        volume_fraction_4 = x[0]
        volume_fraction_1 = x[1]
        chord_length_ratio = x[2]

        # Construct the microstructure representation
        microstructure = {
            'volume_fraction_4': volume_fraction_4,
            'volume_fraction_1': volume_fraction_1,
            'chord_length_ratio': chord_length_ratio,
        }
    elif len(features) == 2:
        volume_fraction_4 = x[0]
        chord_length_ratio = x[1]

        # Construct the microstructure representation
        microstructure = {
            'volume_fraction_4': volume_fraction_4,
            'chord_length_ratio': chord_length_ratio,
        }

    return microstructure


def objective_function(x, desired_property, models, property_name):
    features = models[property_name]['features']
    microstructure = convert_x_to_microstructure(x, features)
    predicted_property = predict_property(property_name, microstructure, models)
    discrepancy = predicted_property - desired_property
    return (discrepancy ** 2)  # Return the squared discrepancy for minimization


def objective_gradient(x, desired_property, models, property_name):
    features = models[property_name]['features']
    gpr_grad = gradient_function(x, models, property_name)
    predicted_property = predict_property(property_name, convert_x_to_microstructure(x, features), models)
    discrepancy = predicted_property - desired_property
    return (2 * discrepancy * gpr_grad)


def predict_property(property_name, microstructure, models):
    model = models[property_name]['pipe']
    features = models[property_name]['features']


    microstructure_features = [microstructure[feature] for feature in features]

    X = np.array(microstructure_features).reshape(1, -1)

    # Use the GPR model to predict the property value
    predicted_value = model.predict(X)

    return predicted_value[0]


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
        grad_sum_adjusted = gpr._y_train_std * grad_sum + gpr._y_train_mean

        gradients.append(grad_sum_adjusted)

    return np.array(gradients).ravel()


def optimise_for_value(prop, X):
    # Random sample starting points
    initial_points = X[np.random.choice(X.shape[0], NUM_STARTS, replace=False)]

    best_result = None
    best_value = float('inf')

    for initial_point in initial_points:
        res = minimize(
            fun=lambda x: objective_function(x, prop, models, property_name),
            jac=lambda x: objective_gradient(x, prop, models, property_name),
            x0=initial_point,
            bounds=bounds,
            method="L-BFGS-B"
        )
        if res.fun < best_value:
            best_value = res.fun
            best_result = res

    optimal_x = best_result.x
    optimal_microstructure = convert_x_to_microstructure(optimal_x, features)
    optimal_property_value = predict_property(property_name, optimal_microstructure, models)

    print(optimal_microstructure)
    print("Error in optimisation: " + str(np.abs(prop - optimal_property_value)))


print("starting opt")

training_data = pathlib.Path("training_data_rve_database.npy")
if not training_data.exists():
    print(f"Error: training data path {training_data} does not exist.")

if training_data.suffix == '.npy':
    data = np.load(training_data)
else:
    print("Invalid data")

print(f"loaded {data.shape[0]} training data pairs")

data['thermal_expansion'] *= 1e6


# Change next two lines for different feature sets or models
models = joblib.load("models/3d-features.joblib")["models"]
X, Y = extract_XY_3(data)

clean_indices = np.argwhere(~np.isnan(Y))
Y = Y[clean_indices.flatten()]
X = X[clean_indices.flatten()]


property_name = 'thermal_expansion'
# Compute the minimum and maximum values for each feature in the training data
min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)
max_values = np.max(X, axis=0)
features = models[property_name]['features']
print("Features: ", str(features))

# Define the search space dimensions based on the minimum and maximum values
bounds = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]

cons = [{'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 1},
        {'type': 'ineq', 'fun': lambda x: -x[2] + 0.01},]

dimensions = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]

NUM_STARTS = 10  # Number of starting points for multi-start
initial_points = X[np.random.choice(X.shape[0], NUM_STARTS, replace=False)]

property_name = 'thermal_expansion'
prop_bounds = (min(Y[:, 1]), max(Y[:, 1]))  

num_points = 100
prop_values = np.linspace(prop_bounds[0], prop_bounds[1], num_points)

optimal_microstructures = []
optimal_volume_fractions_4 = []
optimal_thermal_expansions = []

actual_volume_fractions_4 = X[:, 0]
actual_thermal_expansions = Y[:, 1]
count = 0

for prop in prop_values:
    count += 1
    best_result = None
    best_value = float('inf')

    for initial_point in initial_points:
        res = minimize(
            fun=lambda x: objective_function(x, prop, models, property_name),
            jac=lambda x: objective_gradient(x, prop, models, property_name),
            x0=initial_point,
            bounds=bounds,
            method="L-BFGS-B"
        )
        if res.fun < best_value:
            best_value = res.fun
            best_result = res

    optimal_x = best_result.x
    optimal_microstructure = convert_x_to_microstructure(optimal_x, features)

    optimal_microstructures.append(optimal_microstructure)
    # Store the optimal volume fraction and thermal expansion value
    optimal_volume_fractions_4.append(optimal_microstructure['volume_fraction_4'])
    optimal_thermal_expansions.append(prop)


predicted_optimal_thermal_expansions = []
predicted_actual_thermal_expansions = []

predicted_thermal_expansions = []

for optimal_microstructure in optimal_microstructures:
    predicted_value = predict_property('thermal_expansion', optimal_microstructure, models)
    predicted_thermal_expansions.append(predicted_value)

# Convert the list to a NumPy array for easier manipulation
predicted_thermal_expansions = np.array(predicted_thermal_expansions)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(prop_values, predicted_thermal_expansions))
print(f"Root Mean Square Error (RMSE) between predicted and desired values: {rmse}")

# Calculate R2 score for optimized and actual feature sets
r2 = r2_score(prop_values, predicted_thermal_expansions)
print(f"R2 score between predicted and desired values: {r2*100}")


plt.figure()
plt.scatter(actual_volume_fractions_4, actual_thermal_expansions, label="Actual",  color='blue', marker='o')
plt.scatter(optimal_volume_fractions_4, optimal_thermal_expansions, label="Optimized", color='red', marker='x')
plt.xlabel("Volume Fraction Zirconia")
plt.ylabel("Thermal Expansion")
plt.legend()
plt.show()
