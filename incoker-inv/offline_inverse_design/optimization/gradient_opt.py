"""Perform inverse optimization for material properties using gradient-based methods.

This script utilizes a pre-trained machine learning surrogate model to predict material properties
from microstructural features and iteratively adjusts these features to meet a target property value.
It supports optimization for properties such as thermal conductivity, thermal expansion, Young's
modulus, and Poisson's ratio.

Usage:
  Run this script from the command line, specifying the model file, property name, and target
  property value as arguments. Property name can be one of thermal_conductivity, thermal_expansion,
  young_modulus and poisson_ratio.

Example:
  python offline_inverse_training/gradient_opt.py --model_file <path_to_surrogate_model.joblib>
  --property_name <property_name> --property_value <desired_value>
"""

import argparse
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score


def find_closest_point(Xt, point, selected_indices):
    """
    Find the closest point in Xt to a given point that has not been selected yet.

    Parameters:
    - Xt (numpy.ndarray): The array of points to search through.
    - point (numpy.ndarray): The target point to find the closest point to.
    - selected_indices (list): Indices of points that have already been selected.

    Returns:
    - tuple: The closest point to the given point and the updated list of selected indices.
    """
    distances = np.linalg.norm(Xt - point, axis=1)
    while True:
        index = np.argmin(distances)
        if index not in selected_indices:
            selected_indices.append(index)
            return Xt[index], selected_indices
        else:
            distances[index] = np.inf


def objective_function(x, desired_property, pipe, scale=0.01, callback=None):
    """
    Define the objective function for the optimization problem.

    Parameters:
    - x (numpy.ndarray): Current solution vector.
    - desired_property (float): The target value for the property being optimized.
    - pipe (sklearn.pipeline.Pipeline): The prediction pipeline.
    - scale: A scaling factor for the uncertainty
    - callback (function, optional): A callback function to execute additional procedures at each iteration.

    Returns:
    - float: The squared discrepancy between the predicted property value and the desired property value added with the
    uncertainty at that point.
    """
    if callback is not None:
        callback(x)
    predicted_property, uncertainty = pipe.predict(x.reshape(1, -1), return_std=True)
    discrepancy = predicted_property - desired_property

    return (discrepancy**2) + (uncertainty[0] * scale)


def objective_gradient(x, desired_property, pipe, scale=0.01):
    """
    Compute the gradient of the objective function.

    Parameters:
    - x (numpy.ndarray): Current solution vector.
    - desired_property (float): The target value for the property being optimized.
    - pipe (sklearn.pipeline.Pipeline): The prediction pipeline.
    - scale: A scaling factor for the uncertainty

    Returns:
    - numpy.ndarray: The gradient of the objective function with respect to the solution vector.
    """
    predicted_property, std, gpr_grad, gpr_var_grad = pipe.predict(
        x.reshape(1, -1), return_mean_grad=True, return_std=True, return_std_grad=True
    )
    discrepancy = predicted_property - desired_property

    # Retrieve standard deviation from the StandardScaler
    scaler = pipe.named_steps["standardscaler"]
    std_dev = scaler.scale_

    # Adjust gradients
    adjusted_gpr_grad = gpr_grad / std_dev
    adjusted_gpr_var_grad = gpr_var_grad / std_dev

    return (2 * discrepancy * adjusted_gpr_grad) + (adjusted_gpr_var_grad * scale)

    # s = 0.0008 for TC
    # s = 0.01 for YM, TE, PR


def optimise_for_value(
    prop, X, property_name, pipe, bounds, features, property_ax_dict, initial_points, minima_threshold
):
    """
    Optimize microstructure for a given property value using gradient-based optimization.

    Parameters:
    - prop (float): Target property value.
    - X (numpy.ndarray): Dataset of microstructural features.
    - property_name (str): Name of the property to be optimized.
    - pipe (sklearn.pipeline.Pipeline): Prediction pipeline.
    - bounds (list of tuples): Bounds for the optimization variables.
    - features (list): List of features used in the model.
    - property_ax_dict (dict): Dictionary mapping property names to labels for plotting.
    - initial_points (numpy.ndarray): Initial points for the optimization.
    - minima_threshold (float): Acceptance threshold for considering solutions as valid minima.

    Optimizes the microstructure to achieve a target property value and visualizes the optimization process.
    """
    best_result = None
    best_value = float("inf")
    best_iterates = []
    best_f_values = []

    all_solutions = []

    if property_name == "thermal_conductivity":
        scale = 0.0008
    else:
        scale = 0.01

    for initial_point in initial_points:
        current_iterates = []
        current_f_values = []

        def callback(x):
            current_iterates.append(np.copy(x))
            f_val = objective_function(x, prop, pipe, scale)
            current_f_values.append(f_val)

        res = minimize(
            fun=lambda x: objective_function(x, prop, pipe, scale, callback),
            jac=lambda x: objective_gradient(x, prop, pipe, scale),
            x0=initial_point,
            bounds=bounds,
            method="L-BFGS-B",
        )

        # All solutions
        if res.fun < minima_threshold:
            all_solutions.append(res.x)

        # Single best solution
        if res.fun < best_value:
            best_value = res.fun
            best_result = res
            best_iterates = current_iterates.copy()
            best_f_values = current_f_values.copy()

    optimal_x = best_result.x
    optimal_property_value, uncertainty = pipe.predict(optimal_x.reshape(1, -1), return_std=True)

    print("\nError in optimisation for best solution: " + str(np.abs(prop - optimal_property_value)))

    print("\n\nIterates for best solution")
    print("Iter\tX1\t\t\tX2\t\t\tf(X)")

    for i, (iterate, f_val) in enumerate(zip(best_iterates, best_f_values)):
        print(f"{i+1}\t{iterate[0]:.6f}\t{iterate[1]:.6f}\t{f_val[0]:.6f}")

    if len(features) <= 2:
        print("\n\nAll solutions")
        print(f"No. \t{features[0]}\t{features[1]}\t{property_ax_dict[property_name]}")
        for i, solution in enumerate(all_solutions):
            value = pipe.predict(solution.reshape(1, -1))
            print(f"{i}\t{solution[0]}\t{solution[1]}\t{value}")
        # Dense grid
        v2_values = np.linspace(min(X[:, 0]), max(X[:, 0]), num=100)
        rho_values = np.linspace(min(X[:, 1]), max(X[:, 1]), num=100)
        v2_grid, rho_grid = np.meshgrid(v2_values, rho_values)

        feature_grid = np.vstack([v2_grid.ravel(), rho_grid.ravel()]).T

        predictions = []
        for feature in feature_grid:
            value = objective_function(feature, prop, pipe, scale)
            predictions.append(value)

        predictions = np.array(predictions)

        predictions_grid = predictions.reshape(v2_grid.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(v2_grid, rho_grid, predictions_grid, alpha=0.5)  # Reduced alpha to see points clearly

        # Labels and title
        ax.set_xlabel("Volume Fraction (zirconia)")
        ax.set_ylabel("Particle Size Ratio")
        ax.set_zlabel("J(x) for " + property_ax_dict[property_name] + " = " + str(prop))

        """        # Plot the iterates
        iterates = np.array(best_iterates)
        num_iterates = len(iterates)
        colors = plt.cm.coolwarm(np.linspace(0, 1, num_iterates))

        for i in range(num_iterates):
            x_val, y_val = iterates[i, 0], iterates[i, 1]
            z_val = (pipe.predict([[x_val, y_val]]) - prop) ** 2
            if i>num_iterates-2 or i<2:
                ax.scatter(x_val, y_val, z_val, c=[colors[i]], s=20)
        plt.show()

        """

        # Plot the minima
        for solution in all_solutions:
            x_val, y_val = solution[0], solution[1]
            predicted_value, std_dev = pipe.predict([[x_val, y_val]], return_std=True)
            z_val = objective_function(solution.reshape(1, -1), prop, pipe, scale)
            ax.scatter(x_val, y_val, z_val, color="red", s=20)  # Red color for the solutions
        # Show the plot
        plt.show()


def inverse_validate(
    X,
    Y,
    property_name,
    pipe,
    bounds,
    prop_bounds,
    features,
    property_dict,
    property_ax_dict,
    initial_points,
    minima_threshold,
):
    """
    Validate and optimizes the inverse problem for a given material property across a range of target values.

    Parameters:
    - X (numpy.ndarray): Dataset of microstructural features.
    - Y (numpy.ndarray): Dataset of property values corresponding to X.
    - property_name (str): Name of the property being optimized.
    - pipe (sklearn.pipeline.Pipeline): Prediction pipeline.
    - bounds (list of tuples): Bounds for the optimization variables.
    - prop_bounds (tuple): Bounds for the property values to be optimized.
    - features (list): List of features used in the model.
    - property_dict (dict): Dictionary mapping property names to more readable labels.
    - property_ax_dict (dict): Dictionary mapping property names to axis labels for plotting.
    - initial_points (numpy.ndarray): Initial points for the optimization.
    - minima_threshold (float): Acceptance threshold for considering solutions as valid minima.

    Runs the optimization for a range of target property values and visualizes the results.
    """
    # Now optimizing for range of values

    # Grid of 100 points across property bounds for plot
    num_points = 20
    prop_values = np.linspace(prop_bounds[0], prop_bounds[1], num_points)
    all_properties = []
    actual_volume_fractions_4 = X[:, 0]
    actual_rho = X[:, 1]
    actual_properties = Y
    count = 0
    all_solutions = []
    start_time = time.time()
    obj_fun = []

    if property_name == "thermal_conductivity":
        scale = 0.0008
    else:
        scale = 0.01

    for prop in prop_values:

        for initial_point in initial_points:

            res = minimize(
                fun=lambda x: objective_function(x, prop, pipe, scale),
                jac=lambda x: objective_gradient(x, prop, pipe, scale),
                x0=initial_point,
                bounds=bounds,
                method="L-BFGS-B",
            )
            if res.fun < minima_threshold:
                all_solutions.append(res.x)
                all_properties.append(prop)
                obj_fun.append(res.fun)
                count += 1

    end_time = time.time()
    computation_time = end_time - start_time

    print(f"Optimization took {computation_time}")

    predicted_properties = []
    all_volume_fractions_4 = []
    all_rho = []

    for solution in all_solutions:
        predicted_value, uncertainty = pipe.predict(solution.reshape(1, -1), return_std=True)

        predicted_properties.append(predicted_value)

        all_volume_fractions_4.append(solution[0])
        all_rho.append(solution[1])
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

    # 2D plot
    plt.figure()
    # Scatter plot for actual points
    plt.scatter(actual_volume_fractions_4, actual_properties, label="Ground truth", color="blue", marker="o", alpha=0.5)
    # Scatter plot for optimized points
    plt.scatter(
        all_volume_fractions_4,
        all_properties,
        label="Observed optimized structures",
        color="red",
        marker="x",
        alpha=0.5,
    )
    plt.xlabel("Volume Fraction Zirconia")
    plt.ylabel(property_ax_dict[property_name])
    plt.legend()
    plt.show()

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Scatter plot for actual points
    ax.scatter(actual_volume_fractions_4, actual_rho, actual_properties, label="Ground truth", color="blue", marker="o")
    # Scatter plot for optimized points
    ax.scatter(
        all_volume_fractions_4, all_rho, all_properties, label="Observed optimized structures", color="red", marker="x"
    )
    ax.set_xlabel("Volume Fraction Zirconia")
    ax.set_ylabel("Particle Size Ratio")
    ax.set_zlabel(property_ax_dict[property_name])

    # Define the prediction surface with a grid in 3D plot
    v2_values = np.linspace(min(X[:, 0]), max(X[:, 0]), num=50)
    rho_values = np.linspace(min(X[:, 1]), max(X[:, 1]), num=50)
    v2_grid, rho_grid = np.meshgrid(v2_values, rho_values)
    feature_grid = np.vstack([v2_grid.ravel(), rho_grid.ravel()]).T
    # Predict the property values for the grid
    predictions, uncertainty = pipe.predict(feature_grid, return_std=True)
    predictions_grid = predictions.reshape(v2_grid.shape)
    ax.plot_surface(
        v2_grid,
        rho_grid,
        predictions_grid,
        rstride=1,
        cstride=1,
        color="b",
        alpha=0.1,
    )
    ax.legend()
    plt.show()


def main():
    """Execute the main functionality of the script that performs inverse optimization for material properties."""
    parser = argparse.ArgumentParser(description="Inverse Validation and Optimization for Material Properties")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the model file")
    parser.add_argument(
        "--property_name",
        type=str,
        choices=["thermal_conductivity", "thermal_expansion", "young_modulus", "poisson_ratio"],
        required=True,
        help="Name of the property to optimize",
    )
    parser.add_argument("--property_value", type=float, required=True, help="Target value for the property")

    args = parser.parse_args()

    print("Starting optimization")

    property_name = args.property_name
    property_value = args.property_value
    model_file = args.model_file

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

    models = joblib.load(model_file)["models"]
    pipe = models[property_name]["pipe"]

    X = models[property_name]["X_train"]
    Y = models[property_name]["y_train"]

    # Preprocess parameters for optimization

    # Compute the minimum and maximum values for each feature in the training data
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    features = models[property_name]["features"]
    print("Features: ", str(features))

    # Define the search space dimensions based on the minimum and maximum values
    bounds = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]
    print("X Bounds: ", bounds)

    # Number of starting points for multi-start
    num_samples = 30

    prop_bounds = (min(Y), max(Y))
    print("Y Bounds", prop_bounds)
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

    # Acceptance threshold for minima
    minima_threshold = 1

    # Optimize desired value
    optimise_for_value(
        property_value, X, property_name, pipe, bounds, features, property_ax_dict, initial_points, minima_threshold
    )

    # Validate inverse design over a range of values
    inverse_validate(
        X,
        Y,
        property_name,
        pipe,
        bounds,
        prop_bounds,
        features,
        property_dict,
        property_ax_dict,
        initial_points,
        minima_threshold,
    )


if __name__ == "__main__":
    main()
