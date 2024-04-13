"""Deprecated: use gradient_opt with adaptive GPR refactored to skopt instead."""

import os
import statistics
import sys
import time

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
import argparse
import pathlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
from adapt import *
from pyDOE import lhs
from scipy.optimize import minimize
from simlopt.basicfunctions.utils.createfolderstructure import *
from simlopt.basicfunctions.utils.creategrid import createPD
from simlopt.gpr.gaussianprocess import *
from simlopt.hyperparameter.utils.crossvalidation import *
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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
    model = models

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
            "volume_fraction_4": volume_fraction_4,
            "volume_fraction_10": volume_fraction_10,
            "volume_fraction_1": volume_fraction_1,
            "chord_length_mean_4": chord_length_mean_4,
            "chord_length_mean_10": chord_length_mean_10,
            "chord_length_mean_1": chord_length_mean_1,
            "chord_length_variance_4": chord_length_variance_4,
            "chord_length_variance_10": chord_length_variance_10,
            "chord_length_variance_1": chord_length_variance_1,
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
            "volume_fraction_4": volume_fraction_4,
            "volume_fraction_1": volume_fraction_1,
            "chord_length_mean_4": chord_length_mean_4,
            "chord_length_mean_10": chord_length_mean_10,
            "chord_length_mean_1": chord_length_mean_1,
            "chord_length_variance_4": chord_length_variance_4,
            "chord_length_variance_10": chord_length_variance_10,
            "chord_length_variance_1": chord_length_variance_1,
        }
    elif len(features) == 3:
        volume_fraction_4 = x[0]
        volume_fraction_1 = x[1]
        chord_length_ratio = x[2]

        # Construct the microstructure representation
        microstructure = {
            "volume_fraction_4": volume_fraction_4,
            "volume_fraction_1": volume_fraction_1,
            "chord_length_ratio": chord_length_ratio,
        }
    elif len(features) == 2:
        volume_fraction_4 = x[0]
        chord_length_ratio = x[1]

        # Construct the microstructure representation
        microstructure = {
            "volume_fraction_4": volume_fraction_4,
            "chord_length_ratio": chord_length_ratio,
        }

    return microstructure


def objective_function(x, desired_property, models, property_name):
    microstructure = convert_x_to_microstructure(x, features)
    predicted_property = predict_property(property_name, microstructure, models)
    discrepancy = predicted_property - desired_property
    return discrepancy**2  # Return the squared discrepancy for minimization


def objective_gradient(x, desired_property, models, property_name):
    gpr_grad = gradient_function(x, models, property_name)
    predicted_property = predict_property(property_name, convert_x_to_microstructure(x, features), models)
    discrepancy = predicted_property - desired_property
    return 2 * discrepancy * gpr_grad


def predict_property(property_name, microstructure, models):
    model = models
    microstructure_features = [microstructure[feature] for feature in features]
    X = np.array(microstructure_features).reshape(1, -1)
    # Use the GPR model to predict the property value
    predicted_value = model.predictmean(X)

    return predicted_value[0]


def gpr_mean_grad(X_test, gpr):
    gradients = gpr.predictderivative(X_test, asmatrix=True)
    uncer = gpr.predictderivativevariance(X_test)
    return np.array(gradients).ravel()


def optimise_for_value(prop, X, property_name):
    # Random sample starting points
    initial_points = X[np.random.choice(X.shape[0], NUM_STARTS, replace=False)]
    # initial_points = X[sorted_indices[:NUM_STARTS]]

    best_result = None
    best_value = float("inf")

    for initial_point in initial_points:
        res = minimize(
            fun=lambda x: objective_function(x, prop, models, property_name),
            jac=lambda x: objective_gradient(x, prop, models, property_name),
            x0=initial_point,
            bounds=bounds,
            method="L-BFGS-B",
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
property_name = "thermal_conductivity"

property_ax_dict = {
    "thermal_conductivity": "CTC [W/(m*K)]",
    "thermal_expansion": "CTE [ppm/K]",
    "young_modulus": "Young's Modulus[GPa]",
    "poisson_ratio": "Poisson Ratio",
}

property_dict = {
    "thermal_conductivity": "CTC",
    "thermal_expansion": "CTE",
    "young_modulus": "Young's Modulus",
    "poisson_ratio": "Poisson Ratio",
}
training_data = pathlib.Path("data/training_data_rve_database.npy")
if not training_data.exists():
    print(f"Error: training data path {training_data} does not exist.")

if training_data.suffix == ".npy":
    data = np.load(training_data)
else:
    print("Invalid data")

print(f"loaded {data.shape[0]} training data pairs")

data["thermal_expansion"] *= 1e6


# Change next two lines for different feature sets or models
gp = joblib.load("models/CTC_adapt.joblib")
models = gp
X, Y = gp.X, gp.yt
print(len(X), len(Y))

property_name = "thermal_conductivity"
# Compute the minimum and maximum values for each feature in the training data
min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)
features = ["volume_fraction_4", "chord_length_ratio"]

print("Features: ", str(features))

# Define the search space dimensions based on the minimum and maximum values
bounds = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]

NUM_STARTS = 30  # Number of starting points for multi-start

property_name = "thermal_conductivity"
prop_bounds = (min(Y[:,]), max(Y[:,]))
print(prop_bounds)

# best_starting_point, sorted_indices = find_best_starting_point(X, models, property_name, bounds, features)
# optimise_for_value(5.9, X, property_name)
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
num_points = 20
prop_values = np.linspace(prop_bounds[0], prop_bounds[1], num_points)

optimal_microstructures = []
optimal_volume_fractions_4 = []
optimal_thermal_expansions = []
actual_rho = X[:, 1]
actual_properties = Y
actual_volume_fractions_4 = X[:, 0]
actual_thermal_expansions = Y[:,]
count = 0
error_bars_min = []
error_bars_max = []
all_solutions = []
optimal_rho = []
optimal_properties = []
all_properties = []
obj_fun = []
start_time = time.time()

for prop in prop_values:
    best_result = None
    best_value = float("inf")

    for initial_point in initial_points:
        res = minimize(
            fun=lambda x: objective_function(x, prop, gp, property_name),
            # jac=lambda x: objective_gradient(x, prop, gp, property_name),
            x0=initial_point,
            bounds=bounds,
            method="L-BFGS-B",
        )
        if res.fun < 0.1:
            all_solutions.append(res.x)
            all_properties.append(prop)
            obj_fun.append(res.fun)
            count += 1

        if res.fun < best_value:
            best_value = res.fun
            best_result = res

    optimal_x = best_result.x
    optimal_microstructure = convert_x_to_microstructure(optimal_x, features)

    optimal_microstructures.append(optimal_microstructure)
    # Store the optimal volume fraction and thermal expansion value
    optimal_volume_fractions_4.append(optimal_microstructure["volume_fraction_4"])
    optimal_rho.append(optimal_microstructure["chord_length_ratio"])
    optimal_properties.append(prop)

end_time = time.time()
computation_time = end_time - start_time

print(f"Optimization took {computation_time}")
num_sol = count / 20
print(f"Avg no of sols {num_sol} ")

predicted_properties = []
all_volume_fractions_4 = []
all_rho = []

for solution in all_solutions:
    sol_microstructure = convert_x_to_microstructure(solution, features)

    predicted_value = predict_property(property_name, sol_microstructure, gp)
    predicted_properties.append(predicted_value)

    all_volume_fractions_4.append(sol_microstructure["volume_fraction_4"])
    all_rho.append(sol_microstructure["chord_length_ratio"])
predicted_properties = np.array(predicted_properties)

plt.figure()
plt.scatter(all_properties, predicted_properties, label="", color="blue", marker="o")
plt.xlabel(f"Optimised value for {property_dict[property_name]}")
plt.ylabel(f"Desired value for {property_dict[property_name]}")
plt.legend()
plt.show()


# Calculate RMSE
rmse = np.sqrt(mean_squared_error(all_properties, predicted_properties))
print(f"Root Mean Square Error (RMSE) between predicted and desired values: {rmse}")

# Calculate R2 score for optimized and actual feature sets
r2 = r2_score(all_properties, predicted_properties)
print(f"R2 score between predicted and desired values: {r2*100}")

print(statistics.mean(obj_fun))

plt.figure()

plt.scatter(actual_volume_fractions_4, actual_properties, label="Ground truth", color="blue", marker="o", alpha=0.5)
plt.scatter(
    all_volume_fractions_4, all_properties, label="Observed optimized structures", color="red", marker="x", alpha=0.5
)

plt.xlabel("Volume Fraction Zirconia")
plt.ylabel(property_ax_dict[property_name])
plt.legend()
plt.show()
##################
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


# Scatter plot for actual points
ax.scatter(actual_volume_fractions_4, actual_rho, actual_properties, label="Ground truth", color="blue", marker="o")

# Scatter plot for optimized points
ax.scatter(
    all_volume_fractions_4, all_rho, all_properties, label="Observed optimized structures", color="red", marker="x"
)


# Set labels
ax.set_xlabel("Volume Fraction Zirconia")
ax.set_ylabel("Particle Size Ratio")
ax.set_zlabel(property_ax_dict[property_name])

v2_values = np.linspace(min(X[:, 0]), max(X[:, 0]), num=50)
rho_values = np.linspace(min(X[:, 1]), max(X[:, 1]), num=50)
v2_grid, rho_grid = np.meshgrid(v2_values, rho_values)

feature_grid = np.vstack([v2_grid.ravel(), rho_grid.ravel()]).T

predictions = gp.predictmean(feature_grid)

predictions_grid = predictions.reshape(v2_grid.shape)
ax.plot_surface(
    v2_grid,
    rho_grid,
    predictions_grid,
    rstride=1,
    cstride=1,
    color="b",
    alpha=0.1,
)  # Set color and transparency)

# Show legend
ax.legend()

plt.show()
###############
