import argparse
import pathlib

import joblib
import numpy as np
from standard_training import extract_XY_
from skopt import Optimizer
from scipy.optimize import minimize, LinearConstraint


def gradient_descent_optimization(initial_guess, objective_function, gradient_function, lr=0.01,
                                  max_iterations=100):
    x = initial_guess
    prev_obj_val = float('inf')

    for _ in range(max_iterations):
        # Compute objective function and gradient
        obj_val = objective_function(x)
        grad = gradient_function(x)

        # Gradient descent step
        x -= lr * grad
        # Check for early stopping
        if abs(prev_obj_val - obj_val) < 1e-6:
            print("Early stopping triggered.")
            break
        prev_obj_val = obj_val

    return x, obj_val


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
    microstructure_features = [x[i] for i in range(features)]
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
    volume_fraction_1 = x[1]
    chord_length_ratio = x[2]

    # Construct the microstructure representation
    microstructure = {
        'volume_fraction_4': volume_fraction_4,
        'volume_fraction_1': volume_fraction_1,
        'chord_length_ratio': chord_length_ratio,
    }

    return microstructure


def objective_function(x, desired_property, models, property_name):
    microstructure = convert_x_to_microstructure(x)

    predicted_property = predict_property(property_name, microstructure, models)

    # Calculate the difference between the predicted property and the desired property
    discrepancy = predicted_property - desired_property

    return abs(discrepancy)  # Return the discrepancy (for minimization)


def predict_property(property_name, microstructure, models):
    model = models[property_name]['pipe']
    features = [
        'volume_fraction_4', 'volume_fraction_1',
        'chord_length_ratio'
    ]

    microstructure_features = [microstructure[feature] for feature in features]

    X = np.array(microstructure_features).reshape(1, -1)

    # Use the GPR model to predict the property value
    predicted_value = model.predict(X)
    return predicted_value[0]


def gpr_mean_grad(X_test, gpr):
    X_train = gpr.X_train_
    y_train = gpr.y_train_
    kernel = gpr.kernel_

    rbf, white = kernel.k1, kernel.k2
    l = rbf.length_scale
    sigma_n = np.sqrt(white.noise_level)  # The noise level

    K = kernel(X_train, X_train)  # Covariance matrix of training points
    I = np.eye(K.shape[0])  # Identity matrix of the same shape as K
    alpha = np.linalg.solve(K + sigma_n**2 * I, y_train)

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

training_data = pathlib.Path('models/3d-features.npy')
if not training_data.exists():
    print(f"Error: training data path {training_data} does not exist.")

if training_data.suffix == '.npy':
    data = np.load(training_data)
else:
    print("Invalid data")

print(f"loaded {data.shape[0]} training data pairs")

data['thermal_expansion'] *= 1e6

models = joblib.load("models/3d-features.joblib")["models"]

X, Y = extract_XY_(data)

# Compute the minimum and maximum values for each feature in the training data
min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)

# Define the search space dimensions based on the minimum and maximum values
dimensions = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]


property_name = "thermal_expansion"
prop = 5.9
# Define learning rate and max iterations
learning_rate = 0.01
max_iterations = 500

# initial_guess = np.mean(dimensions, axis=1)
initial_guess, closest_point_value, selected_indices = find_closest_point(X, Y, np.mean(dimensions, axis=1), [])


print(initial_guess)

# Perform gradient descent optimization
optimal_x, optimal_value = gradient_descent_optimization(initial_guess,
                                                         lambda x: objective_function(x, prop, models,
                                                                                      property_name),
                                                         lambda x: gradient_function(x, models, property_name),
                                                         lr=learning_rate,
                                                         max_iterations=max_iterations)
optimal_microstructure = convert_x_to_microstructure(optimal_x)

# Evaluate the desired property using the optimal microstructure
optimal_property_value = predict_property("thermal_expansion", optimal_microstructure, models)

print(optimal_microstructure)
print("Error in optimisation: " + str(np.abs(prop - optimal_property_value)))