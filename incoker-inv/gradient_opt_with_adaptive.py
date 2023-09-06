import argparse
import pathlib
import argparse

from joblib import dump
from adapt import *
from simlopt.gpr.gaussianprocess import *
from simlopt.hyperparameter.utils.crossvalidation import *
from simlopt.basicfunctions.utils.creategrid import createPD
from pyDOE import lhs
from sklearn.model_selection import train_test_split
from sklearn import metrics

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

    # Compute the gradient using the scaled data and GPR
    return gpr_mean_grad(X, model)


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
    predicted_value = model.predictmean(X)

    return predicted_value[0]


def gpr_mean_grad(X_test, gpr):
    gradients = gpr.predictderivative(X_test, asmatrix=True)
    return np.array(gradients).ravel()


def optimise_for_value(prop, X, property_name):
    # Random sample starting points
    initial_points = X[np.random.choice(X.shape[0], NUM_STARTS, replace=False)]
    #initial_points = X[sorted_indices[:NUM_STARTS]]

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
models = joblib.load("models/3d_agp_model.joblib")["models"]
X, Y = extract_XY_3(data)

clean_indices = np.argwhere(~np.isnan(Y))
Y = Y[clean_indices.flatten()]
X = X[clean_indices.flatten()]

property_name = 'thermal_expansion'
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

NUM_STARTS = 10  # Number of starting points for multi-start

property_name = 'thermal_expansion'
prop_bounds = (min(Y[:, 1]), max(Y[:, 1]))


x0 = np.array([[0.92994, 0.0097084, 1.8811808208744365]])
#best_starting_point, sorted_indices = find_best_starting_point(X, models, property_name, bounds, features)
#optimise_for_value(5.9, X, property_name)
# LHS sampling for uniform starting points in multi-start optimization
num_samples = 1
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
optimal_thermal_expansions = []

actual_volume_fractions_4 = X[:, 0]
actual_thermal_expansions = Y[:, 1]
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
    optimal_thermal_expansions.append(prop)


predicted_optimal_thermal_expansions = []
predicted_actual_thermal_expansions = []

predicted_thermal_expansions = []

for optimal_microstructure in optimal_microstructures:
    predicted_value = predict_property('thermal_expansion', optimal_microstructure, models)
    predicted_thermal_expansions.append(predicted_value)

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