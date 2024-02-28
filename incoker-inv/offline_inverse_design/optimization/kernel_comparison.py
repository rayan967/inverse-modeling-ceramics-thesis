import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from pyDOE import lhs
from scipy.optimize import minimize, brute
import matplotlib.pyplot as plt
import time

K_inv = None


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


def objective_function(x, desired_property, pipe, property_name, rho, callback=None):
    if callback is not None:
        callback(x)
    # predicted_property, std  = pipe.predict(x.reshape(1, -1), return_std=True)
    global predicted_property, std, gpr_grad, gpr_var_grad
    predicted_property, std, gpr_grad, gpr_var_grad = pipe.predict(
        x.reshape(1, -1), return_mean_grad=True, return_std=True, return_std_grad=True
    )
    discrepancy = predicted_property - desired_property

    return (discrepancy**2) + std[0] * rho


def objective_gradient(x, desired_property, pipe, property_name, rho):
    global predicted_property, std, gpr_grad, gpr_var_grad
    discrepancy = predicted_property - desired_property
    # print("Objective Gradient: ", str(2 * discrepancy * gpr_grad))
    # print("\n")
    # Retrieve standard deviation from the StandardScaler
    scaler = pipe.named_steps["standardscaler"]
    std_dev = scaler.scale_
    # Adjust gradients
    adjusted_gpr_grad = gpr_grad / std_dev
    adjusted_gpr_var_grad = gpr_var_grad / std_dev
    return (2 * discrepancy * adjusted_gpr_grad) + adjusted_gpr_var_grad * rho
    # 0.0008 for TC


print("starting opt")

property_name = "thermal_expansion"

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


# Initialize a dictionary to hold the results
results = {}

modelnames = [
    "models/2d_dp.joblib",
    "models/2d_rq.joblib",
    "models/2d_rbf.joblib",
    "models/2d_matern.joblib",
    "models/2d_rdw.joblib",
]

# Change next line for different feature sets from models folder
# Properties and kernels to iterate over
properties = ["thermal_conductivity", "thermal_expansion", "young_modulus", "poisson_ratio"]
kernels = ["2d_dp", "2d_rq", "2d_rbf_grad", "2d_matern", "2d_rdw"]
property_reults = {
    property: pd.DataFrame(columns=["Kernel", "R2", "RMSE", "Computation Time", "Iters", "Sols"])
    for property in properties
}


for kernel_name in modelnames:
    for property_name in properties:
        models = joblib.load(kernel_name)["models"]

        pipe = models[property_name]["pipe"]

        X = models[property_name]["X_train"]
        X_test = models[property_name]["X_test"]
        Y = models[property_name]["y_train"]
        y_test = models[property_name]["y_test"]

        # Compute the minimum and maximum values for each feature in the training data
        min_values = np.min(X, axis=0)
        max_values = np.max(X, axis=0)
        features = models[property_name]["features"]
        print("Features: ", str(features))

        # Define the search space dimensions based on the minimum and maximum values
        bounds = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]
        print(bounds)
        cons = [
            {"type": "eq", "fun": lambda x: x[0] + x[1] + x[2] - 1},
            {"type": "ineq", "fun": lambda x: -x[2] + 0.01},
        ]

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

        ## Now optimizing subspace
        # Grid of 100 points across property bounds for plot
        num_points = 20
        prop_values = np.linspace(prop_bounds[0], prop_bounds[1], num_points)
        print("#########################")
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

        # Global variable to store iteration counts
        iteration_counts = []

        def iteration_callback(x):
            """Callback function to track iterations."""
            if "current_count" not in iteration_callback.__dict__:
                iteration_callback.current_count = 0
            iteration_callback.current_count += 1

        for prop in prop_values:
            best_result = None
            best_value = float("inf")

            for initial_point in initial_points:
                iteration_callback.current_count = 0  # Reset count for this run
                if property_name == "thermal_conductivity":
                    res = minimize(
                        fun=lambda x: objective_function(x, prop, pipe, property_name, 0.01),
                        jac=lambda x: objective_gradient(x, prop, pipe, property_name, 0.01),
                        x0=initial_point,
                        bounds=bounds,
                        method="L-BFGS-B",
                        callback=iteration_callback,
                    )
                else:
                    res = minimize(
                        fun=lambda x: objective_function(x, prop, pipe, property_name, 0.01),
                        jac=lambda x: objective_gradient(x, prop, pipe, property_name, 0.01),
                        x0=initial_point,
                        bounds=bounds,
                        method="L-BFGS-B",
                        callback=iteration_callback,
                    )

                iteration_counts.append(iteration_callback.current_count)
                if res.fun < 1:
                    count += 1
                    all_solutions.append(res.x)
                    all_properties.append(prop)
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
        num_sol = count / 20

        predicted_properties = []
        all_volume_fractions_4 = []
        all_rho = []

        for solution in all_solutions:
            predicted_value, uncertainty = models[property_name]["pipe"].predict(
                solution.reshape(1, -1), return_std=True
            )
            predicted_properties.append(predicted_value)
            all_volume_fractions_4.append(solution[0])
            all_rho.append(solution[1])
        predicted_properties = np.array(predicted_properties)
        num_sol = count / 20

        def count_unique_elements_rounded(solutions):
            # Rounding each element to three decimal places
            rounded_solutions = [tuple(np.round(element, 5)) for element in solutions]
            # Counting unique elements
            unique_elements = len(set(rounded_solutions))
            return unique_elements

        # Count pairs that are farther than the threshold
        num_sol = count_unique_elements_rounded(all_solutions) / 20

        print(f"Optimization took {computation_time} seconds")
        print(f"Avg no of sols {num_sol} ")
        # After the optimization is complete
        average_iterations = np.mean(iteration_counts)
        print(f"Average number of iterations per initial point: {average_iterations}")
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(all_properties, predicted_properties))
        print(f"Root Mean Square Error (RMSE) between predicted and desired values: {rmse}")

        # Calculate R2 score for optimized and actual feature sets
        r2 = r2_score(all_properties, predicted_properties)
        print(f"R2 score between predicted and desired values: {r2 * 100}")

        r2, rmse, comp_time, average_iterations = r2, rmse, computation_time, average_iterations
        new_row = pd.DataFrame(
            {
                "Kernel": [kernel_name],
                "R2": [r2],
                "RMSE": [rmse],
                "Computation Time": [f"{comp_time}s"],
                "Iters": [average_iterations],
                "Sols": [num_sol],
            }
        )
        property_reults[property_name] = pd.concat([property_reults[property_name], new_row], ignore_index=True)

for property, df in property_reults.items():
    latex_subtable = df.to_latex(
        index=False,
        header=True,
        float_format="%.3e",
        caption=f"Results for {property} property",
        label=f"tab:{property}",
    )
    print(f"{property} property Results:")
    print(latex_subtable)
    print("\n")
