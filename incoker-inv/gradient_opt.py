import argparse
import pathlib

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from simlopt.basicfunctions.utils.creategrid import createPD
from pyDOE import lhs
from standard_training import extract_XY, extract_XY_3, extract_XY_2
from skopt import Optimizer
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt



def find_closest_point(Xt, point, selected_indices):
    distances = np.linalg.norm(Xt - point, axis=1)
    while True:
        index = np.argmin(distances)
        if index not in selected_indices:
            selected_indices.append(index)
            return Xt[index], selected_indices
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
    #print("Current microstructure:", str(microstructure))
    predicted_property = predict_property(property_name, microstructure, models)
    discrepancy = predicted_property - desired_property
    #print("Objective value: ", str(discrepancy ** 2))
    return (discrepancy ** 2)  # Return the squared discrepancy for minimization


def objective_gradient(x, desired_property, models, property_name):
    features = models[property_name]['features']
    gpr_grad = gradient_function(x, models, property_name)
    predicted_property = predict_property(property_name, convert_x_to_microstructure(x, features), models)
    discrepancy = predicted_property - desired_property
    #print("Objective Gradient: ", str(2 * discrepancy * gpr_grad))
    #print("\n")

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


def optimise_for_value(prop, X, property_name):
    # Random sample starting points
    #initial_points = X[np.random.choice(X.shape[0], NUM_STARTS, replace=False)]
    # LHS sampling for uniform starting points in multi-start optimization
    num_samples = 30
    lhs_samples = lhs(len(bounds), samples=num_samples)
    for i in range(len(bounds)):
        lhs_samples[:, i] = lhs_samples[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    # Find the closest points in X to the LHS samples
    selected_indices = []
    Xt_initial = np.zeros((num_samples, X.shape[1]))  # Initialize closest points array
    for i in range(num_samples):
        Xt_initial[i], selected_indices = find_closest_point(X, lhs_samples[i], selected_indices)
    initial_points = X[selected_indices]

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


def find_best_starting_point(X, models, property_name, bounds, features, NUM_STARTS=1000):
    # Exhaustive search to find the average absolute error for each starting point
    starting_point_scores = {}
    prop_bounds = (min(Y[:, 1])+1, max(Y[:, 1])-1)

    num_points = 100
    prop_values = np.linspace(prop_bounds[0], prop_bounds[1], num_points)

    # Get random starting points
    random_indices = np.random.choice(X.shape[0], NUM_STARTS, replace=False)
    starting_points = X[random_indices]

    num_props = len(prop_values)

    for idx, initial_point in zip(random_indices, starting_points):
        total_absolute_error = 0
        for prop in prop_values:
            best_result = None
            best_value = float('inf')

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

            # Predict the thermal expansion
            predicted_value = predict_property(property_name, optimal_microstructure, models)

            # Calculate the absolute error for this prop value
            absolute_error = abs(predicted_value - prop)

            total_absolute_error += absolute_error

        # Calculate the average absolute error for this starting point
        avg_absolute_error = total_absolute_error / num_props

        starting_point_scores[idx] = avg_absolute_error

    # Sort the starting points by average absolute error
    sorted_indices = sorted(starting_point_scores, key=starting_point_scores.get)[:10]

    # Find the best 10 starting points and their average absolute errors
    best_starting_points = [X[idx] for idx in sorted_indices]
    best_avg_absolute_errors = [starting_point_scores[idx] for idx in sorted_indices]

    # Print the best 10 starting points and their corresponding average absolute errors
    for i in range(10):
        print(f"Best Starting Point {i+1}: {best_starting_points[i]}")
        print(f"Best Average Absolute Error {i+1}: {best_avg_absolute_errors[i]}")
    print(sorted_indices)
    return best_starting_points, sorted_indices  # Return the best 10 starting points and their indices in X


print("starting opt")

property_name = 'thermal_conductivity'

# Change next line for different feature sets from models folder
models = joblib.load("models/2d_model.joblib")["models"]

X = models[property_name]['X_train']
X_test = models[property_name]['X_test']
Y = models[property_name]['y_train']
y_test = models[property_name]['y_test']

# Compute the minimum and maximum values for each feature in the training data
min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)
features = models[property_name]['features']
print("Features: ", str(features))

# Define the search space dimensions based on the minimum and maximum values
bounds = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]

cons = [{'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 1},
        {'type': 'ineq', 'fun': lambda x: -x[2] + 0.01},]

dimensions = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]

NUM_STARTS = 30  # Number of starting points for multi-start

prop_bounds = (min(Y), max(Y))

# LHS sampling for uniform starting points in multi-start optimization
num_samples = 30
lhs_samples = lhs(len(bounds), samples=num_samples)
for i in range(len(bounds)):
    lhs_samples[:, i] = lhs_samples[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

# Find the closest points in X to the LHS samples
selected_indices = []
Xt_initial = np.zeros((num_samples, X.shape[1]))  # Initialize closest points array
for i in range(num_samples):
    Xt_initial[i], selected_indices = find_closest_point(X, lhs_samples[i], selected_indices)
initial_points = X[selected_indices]

# or random choice
# initial_points = X[np.random.choice(X.shape[0], NUM_STARTS, replace=False)]


# Grid of 100 points across property bounds for plot
num_points = 100
prop_values = np.linspace(prop_bounds[0], prop_bounds[1], num_points)

optimal_microstructures = []
optimal_volume_fractions_4 = []
optimal_properties = []

actual_volume_fractions_4 = X[:, 0]
actual_properties = Y
count = 0
error_bars_min = []
error_bars_max = []

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
    optimal_properties.append(prop)


predicted_optimal_properties = []
predicted_actual_properties = []

predicted_properties = []

for optimal_microstructure in optimal_microstructures:
    predicted_value = predict_property(property_name, optimal_microstructure, models)
    predicted_properties.append(predicted_value)

predicted_properties = np.array(predicted_properties)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(prop_values, predicted_properties))
print(f"Root Mean Square Error (RMSE) between predicted and desired values: {rmse}")

# Calculate R2 score for optimized and actual feature sets
r2 = r2_score(prop_values, predicted_properties)
print(f"R2 score between predicted and desired values: {r2*100}")


plt.figure()
plt.scatter(actual_volume_fractions_4, actual_properties, label="Actual",  color='blue', marker='o')
plt.scatter(optimal_volume_fractions_4, optimal_properties, label="Optimized", color='red', marker='x')
plt.xlabel("Volume Fraction Zirconia")
plt.ylabel(property_name)
plt.legend()
plt.show()

# x0 = np.array([[0.92994, 0.0097084, 1.8811808208744365]])
# best_starting_point, sorted_indices = find_best_starting_point(X, models, property_name, bounds, features)
# optimise_for_value(34, X, property_name)

