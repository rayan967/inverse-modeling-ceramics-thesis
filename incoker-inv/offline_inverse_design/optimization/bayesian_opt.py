import argparse
import pathlib

import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from skopt import Optimizer, gp_minimize
from scipy.optimize import minimize, LinearConstraint
import statistics

import joblib
import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.metrics import mean_squared_error, r2_score
from pyDOE import lhs
from scipy.optimize import minimize, brute
import matplotlib.pyplot as plt
import time

def find_closest_point(Xt, point, selected_indices):
    distances = np.linalg.norm(Xt - point, axis=1)
    while True:
        index = np.argmin(distances)
        if index not in selected_indices:
            selected_indices.append(index)
            return Xt[index], selected_indices
        else:
            distances[index] = np.inf

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


def objective_function(x, desired_property, pipe, property_name, callback=None):
    if callback is not None:
        callback(x)
    x = np.array(x)
    predicted_property, uncertainty  = pipe.predict(x.reshape(1, -1), return_std=True)
    discrepancy = predicted_property - desired_property

    return (discrepancy ** 2)[0] #+ uncertainty[0]*0.01


def objective_gradient(x, desired_property, pipe, property_name):

    predicted_property, std, gpr_grad, gpr_var_grad  = pipe.predict(x.reshape(1, -1), return_mean_grad=True,return_std=True, return_std_grad=True)
    discrepancy = predicted_property - desired_property

    #print("Objective Gradient: ", str(2 * discrepancy * gpr_grad))
    #print("\n")
    # Retrieve standard deviation from the StandardScaler
    scaler = pipe.named_steps['standardscaler']
    std_dev = scaler.scale_

    # Adjust gradients
    adjusted_gpr_grad = gpr_grad / std_dev
    adjusted_gpr_var_grad = gpr_var_grad / std_dev

    return (2 * discrepancy * adjusted_gpr_grad) #+ adjusted_gpr_var_grad*0.01
    #0.0008 for TC
    #0.01 for YM, TE, PR

def optimise_for_value(prop, X, property_name):
    # Random sample starting points
    #initial_points = X[np.random.choice(X.shape[0], NUM_STARTS, replace=False)]
    # LHS sampling for25 uniform starting points in multi-start optimization
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

    #indices = np.where(np.isclose(X[:, 0], 0.0070801))
    #initial_points = X[selected_indices]
    print(bounds)
    best_result = None
    best_value = float('inf')
    best_iterates = []
    best_f_values = []

    all_solutions = []

    for initial_point in initial_points:
        current_iterates = []
        current_f_values = []

        def callback(x):
            current_iterates.append(np.copy(x))
            f_val = objective_function(x, prop, pipe, property_name, None)
            current_f_values.append(f_val)

        res = gp_minimize(lambda x: objective_function(x, prop, pipe, property_name, callback),
                          dimensions,
                          x0=initial_point.tolist(),
                          )

        if res.fun < 1e-01:
            all_solutions.append(res.x)
        if res.fun < best_value:
            best_value = res.fun
            best_result = res
            best_iterates = current_iterates.copy()
            best_f_values = current_f_values.copy()


    optimal_x = best_result.x
    optimal_microstructure = convert_x_to_microstructure(optimal_x, features)
    optimal_property_value, uncertainty  = models[property_name]['pipe'].predict(optimal_x.reshape(1, -1), return_std=True)

    print(optimal_microstructure)
    print("Error in optimisation: " + str(np.abs(prop - optimal_property_value)))

    print("Iter\tX1\t\t\tX2\t\t\tf(X)")

    for i, (iterate, f_val) in enumerate(zip(best_iterates, best_f_values)):
        print(f"{i+1}\t{iterate[0]:.6f}\t{iterate[1]:.6f}\t{f_val[0]:.6f}")


    if len(features) <= 2:
        X = models[property_name]['X_train']
        # Dense grid
        v2_values = np.linspace(min(X[:, 0]), max(X[:, 0]), num=100)
        rho_values = np.linspace(min(X[:, 1]), max(X[:, 1]), num=100)
        v2_grid, rho_grid = np.meshgrid(v2_values, rho_values)

        feature_grid = np.vstack([v2_grid.ravel(), rho_grid.ravel()]).T

        predictions = []
        for feature in feature_grid:
            value = objective_function(feature, prop, pipe, property_name)
            predictions.append(value)

        predictions = np.array(predictions)

        predictions_grid = predictions.reshape(v2_grid.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(v2_grid, rho_grid, predictions_grid, alpha=0.5)  # Reduced alpha to see points clearly

        # Labels and title
        ax.set_xlabel('Volume Fraction (zirconia)')
        ax.set_ylabel('Particle Size Ratio')
        ax.set_zlabel("J(x) for "+property_ax_dict[property_name] + " = " + str(prop))

        """        # Plotting the iterates
        iterates = np.array(best_iterates)
        num_iterates = len(iterates)
        colors = plt.cm.coolwarm(np.linspace(0, 1, num_iterates))

        for i in range(num_iterates):
            x_val, y_val = iterates[i, 0], iterates[i, 1]
            z_val = (models[property_name]['pipe'].predict([[x_val, y_val]]) - prop) ** 2
            if i>num_iterates-2 or i<2:
                ax.scatter(x_val, y_val, z_val, c=[colors[i]], s=20)
        """

        for solution in all_solutions:
            x_val, y_val = solution[0], solution[1]
            predicted_value, std_dev = models[property_name]['pipe'].predict([[x_val, y_val]], return_std=True)
            z_val = objective_function(solution.reshape(1,-1), prop, pipe, property_name)
            ax.scatter(x_val, y_val, z_val, color='red', s=20)  # Red color for the solutions
        # Show the plot
        plt.show()


print("starting opt")

property_name = 'thermal_conductivity'

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
# Change next line for different feature sets from models folder
models = joblib.load("models/2d_rbf_grad.joblib")["models"]
pipe = models[property_name]['pipe']

X = models[property_name]['X_train']
X_test = models[property_name]['X_test']
Y = models[property_name]['y_train']
y_test = models[property_name]['y_test']

# Compute the minimum and maximum values for each feature in the training data
min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)
features = models[property_name]['features']
print("Features: ", str(features))
print(str(len(X_test)+len(X)))
# Define the search space dimensions based on the minimum and maximum values
bounds = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]
print(bounds)
cons = [{'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 1},
        {'type': 'ineq', 'fun': lambda x: -x[2] + 0.01},]

dimensions = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]

NUM_STARTS = 30  # Number of starting points for multi-start
num_samples = 30

prop_bounds = (min(Y), max(Y))
print(prop_bounds)
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


# Try optimising one value
optimise_for_value(5.9, X, property_name)
#########
# or random choice
# initial_points = X[np.random.choice(X.shape[0], NUM_STARTS, replace=False)]


## Now optimizing subspace
# Grid of 100 points across property bounds for plot
num_points = 20
prop_values = np.linspace(prop_bounds[0], prop_bounds[1], num_points)


optimal_microstructures = []
optimal_volume_fractions_4 = []
optimal_properties = []
all_properties = []

optimal_rho = []

actual_volume_fractions_4 = X[:, 0]
actual_rho = X[:, 1]
actual_properties = Y
count = 0
error_bars_min = []
error_bars_max = []
all_solutions = []
start_time = time.time()
obj_fun = []

for prop in prop_values:
    best_result = None
    best_value = float('inf')

    for initial_point in initial_points:
        """
        m = scipy.optimize.check_grad(
            lambda x: objective_function(x, prop, pipe, property_name),
            lambda x: objective_gradient(x, prop, pipe, property_name),
            x0=initial_point,
        )
        print("Grad error", str(m))
        """
        res = gp_minimize(lambda x: objective_function(x, prop, models, property_name),
                          dimensions,
                          x0=initial_point,
                          n_calls=100,
                          )
        if res.fun < 1:
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
    optimal_volume_fractions_4.append(optimal_microstructure['volume_fraction_4'])
    optimal_rho.append(optimal_microstructure['chord_length_ratio'])
    optimal_properties.append(prop)
end_time = time.time()
computation_time = end_time - start_time

print(f"Optimization took {computation_time}")
num_sol = count/20

def count_unique_elements_rounded(solutions):
    # Rounding each element to three decimal places
    rounded_solutions = [tuple(np.round(element, 5)) for element in solutions]
    # Counting unique elements
    unique_elements = len(set(rounded_solutions))
    return unique_elements


# Count pairs that are farther than the threshold
num_sol = count_unique_elements_rounded(all_solutions)/20


print(f"Avg no of sols {num_sol} ")


predicted_properties = []
all_volume_fractions_4 = []
all_rho = []

for solution in all_solutions:
    predicted_value, uncertainty  = models[property_name]['pipe'].predict(solution.reshape(1, -1), return_std=True)

    predicted_properties.append(predicted_value)

    all_volume_fractions_4.append(solution[0])
    all_rho.append(solution[1])
predicted_properties = np.array(predicted_properties)

plt.figure()
plt.scatter(all_properties, predicted_properties, label="",  color='blue', marker='o')
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

plt.scatter(actual_volume_fractions_4, actual_properties, label="Ground truth",  color='blue', marker='o', alpha=0.5)
plt.scatter(all_volume_fractions_4, all_properties, label="Observed optimized structures", color='red', marker='x', alpha=0.5 )

plt.xlabel("Volume Fraction Zirconia")
plt.ylabel(property_ax_dict[property_name])
plt.legend()
plt.show()
##################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Scatter plot for actual points
ax.scatter(actual_volume_fractions_4, actual_rho, actual_properties, label="Ground truth", color='blue', marker='o')

# Scatter plot for optimized points
ax.scatter(all_volume_fractions_4, all_rho, all_properties, label="Observed optimized structures", color='red', marker='x')




# Set labels
ax.set_xlabel('Volume Fraction Zirconia')
ax.set_ylabel('Particle Size Ratio')
ax.set_zlabel(property_ax_dict[property_name])

v2_values = np.linspace(min(X[:, 0]), max(X[:, 0]), num=50)
rho_values = np.linspace(min(X[:, 1]), max(X[:, 1]), num=50)
v2_grid, rho_grid = np.meshgrid(v2_values, rho_values)

feature_grid = np.vstack([v2_grid.ravel(), rho_grid.ravel()]).T

predictions, uncertainty = (models[property_name]['pipe'].predict(feature_grid, return_std=True))

predictions_grid = predictions.reshape(v2_grid.shape)
ax.plot_surface(v2_grid, rho_grid, predictions_grid, rstride=1, cstride=1,
                       color='b', alpha=0.1, ) # Set color and transparency)

# Show legend
ax.legend()

plt.show()
###############
