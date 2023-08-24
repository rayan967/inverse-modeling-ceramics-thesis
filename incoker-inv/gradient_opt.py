import argparse
import pathlib

import joblib
import numpy as np
from standard_training import extract_XY
from skopt import Optimizer
from scipy.optimize import minimize, LinearConstraint


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


def convert_x_to_microstructure(x):
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

    return microstructure


def objective_function(x, desired_property, models, property_name):
    microstructure = convert_x_to_microstructure(x)
    predicted_property = predict_property(property_name, microstructure, models)
    discrepancy = predicted_property - desired_property
    return discrepancy ** 2  # Return the squared discrepancy for minimization


def objective_gradient(x, desired_property, models, property_name):
    gpr_grad = gradient_function(x, models, property_name)
    predicted_property = predict_property(property_name, convert_x_to_microstructure(x), models)
    discrepancy = predicted_property - desired_property
    return 2 * discrepancy * gpr_grad


def predict_property(property_name, microstructure, models):
    model = models[property_name]['pipe']
    features = [
        'volume_fraction_4', 'volume_fraction_10', 'volume_fraction_1',
        'chord_length_mean_4', 'chord_length_mean_10', 'chord_length_mean_1',
        'chord_length_variance_4', 'chord_length_variance_10', 'chord_length_variance_1'
    ]

    microstructure_features = [microstructure[feature] for feature in features]

    X = np.array(microstructure_features).reshape(1, -1)

    # Use the GPR model to predict the property value
    predicted_value = model.predict(X)
    return predicted_value[0]


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

models = joblib.load("models/tuned-hyperparameters.joblib")["models"]

X, Y = extract_XY(data)

prop = 5.9
property_name = 'thermal_expansion'
# Compute the minimum and maximum values for each feature in the training data
min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)

# Define the search space dimensions based on the minimum and maximum values
bounds = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]

cons = [{'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 1},
        {'type': 'ineq', 'fun': lambda x: -x[2] + 0.01},]

dimensions = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]

NUM_STARTS = 10  # Number of starting points for multi-start

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
        constraints=cons,
        method="L-BFGS-B"
    )
    if res.fun < best_value:
        best_value = res.fun
        best_result = res

optimal_x = best_result.x
optimal_microstructure = convert_x_to_microstructure(optimal_x)
optimal_property_value = predict_property(property_name, optimal_microstructure, models)

print(optimal_microstructure)
print("Error in optimisation: "+str(np.abs(prop - optimal_property_value)))