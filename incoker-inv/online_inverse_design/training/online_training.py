"""
online_training.py

This script is designed for the inverse design of ZTA ceramic microstructures using Gaussian Process Regression (GPR) models. It includes functionalities for loading and preprocessing data, performing hyperparameter optimization, adaptive sampling, and model evaluation.

Modules:
- pathlib, sys, os, json, joblib: For file operations and system interactions.
- pandas (pd), numpy (np): For data manipulation and numerical operations.
- sklearn.metrics: For model evaluation metrics.
- Custom modules (online_adapt, simlopt): For adaptive GPR.
"""

import sys
import json
import joblib
from pathlib import Path
from sklearn import metrics

# Add incoker-inv to sys path
current_file = Path(__file__).resolve()
run_directory = current_file.parent.parent.parent
sys.path.append(str(run_directory))
import numpy as np
import argparse
from simlopt.gpr.gaussianprocess import *
from simlopt.hyperparameter.utils.crossvalidation import *
from simlopt.basicfunctions.utils.creategrid import createPD
from pyDOE import lhs
from sklearn.model_selection import train_test_split
from simlopt.basicfunctions.utils.createfolderstructure import *
from online_adapt import *
import yaml

plt.close("all")
plt.ioff()

# material properties to consider in training
considered_properties = [
    "thermal_conductivity",
    "thermal_expansion",
    "young_modulus",
    "poisson_ratio",
]

# For result indexing
property_dict = {
    "thermal_conductivity": "Thermal conductivity",
    "thermal_expansion": "Thermal expansion",
    "young_modulus": "Young modulus",
    "poisson_ratio": "Poisson ratio",
}

# For simulation options
property_dict_category = {
    "thermal_conductivity": "thermal_conductivity",
    "thermal_expansion": "thermal_expansion",
    "young_modulus": "elasticity",
    "poisson_ratio": "elasticity",
}
phase_zirconia = 11
phase_alumina = 4


def load_config(config_path):
    """
    Load configuration parameters from a YAML file.

    Parameters:
    - config_path (str): Path to the configuration YAML file.

    Returns:
    - dict: A dictionary containing configuration parameters.
    """
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def load_test_data(base_path, prop_name):
    """
    Load test data from JSON files within a specified directory.

    Parameters:
    - base_path (pathlib.Path): Directory path to load data from.
    - prop_name (str): Material property considered.

    Returns:
    - tuple: A tuple containing two numpy arrays, X (features) and y (target values).
    """
    info_files = list(base_path.glob("**/info.json"))

    X = []
    y = []
    for file in info_files:
        data = json.loads(file.read_text())
        if not (("homogenization" in data or "mean" in data) and "v_phase" in data):
            continue

        vf = data["v_phase"][str(phase_zirconia)]
        clr = data["chord_length_ratio"]
        X.append([vf, clr])
        y.append(get_output(data, prop_name))

    return np.array(X), np.array(y)


def generate_candidate_point(input, simulation_options, property_name, output_stream, runpath, run_phase):
    output_stream.error_detected = False  # Set error flag
    output_path = pathlib.Path(runpath, run_phase, f"v={input[0]},r={input[1]}")
    output_path.mkdir(parents=True, exist_ok=True)
    simulation_options["output_path"] = output_path

    result = prediction_pipeline.generate_and_predict(input, simulation_options)
    output_value = get_output(result, property_name)

    vf = result["v_phase"][str(phase_zirconia)]
    cl_11 = result["chord_length_analysis"]["phase_chord_lengths"][phase_zirconia]["mean_chord_length"]
    cl_4 = result["chord_length_analysis"]["phase_chord_lengths"][phase_alumina]["mean_chord_length"]
    clr = cl_11 / cl_4

    # Check for Mapdl error
    if output_stream.error_detected:
        # Reset error flag
        output_stream.error_detected = False
        raise Exception("Error detected during operation: Mapdl")

    return [vf, clr], output_value


def createerror(Xt, random=False, graddata=False):
    """
    Create error estimates for the input data.

    Parameters:
    - Xt (numpy.ndarray): Input data.
    - random (bool, optional): Flag to use random error. Defaults to False.
    - graddata (bool, optional): Flag to include gradient data in error calculation. Defaults to False.

    Returns:
    - tuple: A tuple of error estimates.
    """
    N = Xt.shape[0]
    dim = Xt.shape[1]
    epsXgrad = None
    if random:
        epsXt = np.random.rand(N) * (0.1 - 0.025) + 0.025
        epsXt = epsXt.reshape((1, N))
        if graddata:
            epsXgrad = np.random.rand(N * dim) * (0.1 - 0.025) + 0.025
    else:
        variances = np.var(Xt, axis=0)
        avg_variance = np.mean(variances)
        epsXt = avg_variance * np.ones((1, N))
        vardata = 1e-4
        epsXt = vardata * np.ones((1, N))  # 1E-1 for basic setup
        if graddata:
            vargraddata = 1e-1
            epsXgrad = vargraddata * np.ones((1, N * dim))
    return epsXt, epsXgrad


def accuracy_test(model, X_test, y_test, tolerance=1e-2):
    """
    Calculate the accuracy of the model on test data.

    Parameters:
    - model: Trained GPR model.
    - X_test (numpy.ndarray): Test data features.
    - y_test (numpy.ndarray): Test data target values.
    - tolerance (float, optional): Tolerance for accuracy. Defaults to 1E-2.

    Returns:
    - float: Accuracy score.
    """
    # Predict mean for test data
    y_pred = model.predictmean(X_test)

    # Calculate whether each prediction is within the tolerance of the true value
    score = metrics.r2_score(y_true=y_test, y_pred=y_pred) * 100

    return score


def create_initial_design_points(parameterranges):
    """
    Creates 9 design points in a 2D parameter space including corners,
    midpoints of boundaries, and the center.

    Parameters:
    parameterranges (numpy array): Array of parameter ranges [[min_x, max_x], [min_y, max_y]]

    Returns:
    numpy array: Array of 9 design points.
    """
    min_x, max_x = parameterranges[0]
    min_y, max_y = parameterranges[1]
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    # Define corner points, midpoints, and center point
    points = np.array(
        [
            [min_x, min_y],
            [min_x, max_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, mid_y],
            [max_x, mid_y],
            [mid_x, min_y],
            [mid_x, max_y],
            [mid_x, mid_y],
        ]
    )
    return points


def get_output(result, property_name):
    # For CTE
    if property_name == "thermal_expansion":
        output_value = result["mean"]

    # For the rest
    else:
        output_value = result["homogenization"][property_dict[property_name]]["value"]
    return output_value


class DualOutputStream:
    """
    A class to handle dual output streams, allowing simultaneous writing to both the terminal and a log file.

    Methods:
    - __init__(filename): Constructor to initialize the dual output stream.
    - write(message): Writes a message to both the terminal and the log file.
    - flush(): Flush method for Python 3 compatibility.
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

        msg = "intersection(s) found between triangle"
        msg2 = "ansys.mapdl.core.errors.MapdlRuntimeError:"
        if msg in message or msg2 in message:
            self.error_detected = True

    def flush(self):
        # This flush method is needed for Python 3 compatibility.
        # This handles the flush command by doing nothing.
        # You might want to extend this method in the future.
        pass


def main(config_path):
    """
    This function manages the workflow of the script, including loading data,
    setting up the GPR model, performing adaptive sampling, and evaluating model performance.
    """
    # Load the configuration file
    config = load_config(config_path)

    # Use the loaded configuration
    property_name = config["property_name"]
    simulation_options = config["simulation_options"]
    simulation_options["material_property"] = property_dict_category[property_name]
    adaptive_phase_parameters = config["adaptive_phase_parameters"]
    validation_data_path = Path(config["validation_data_path"])
    output_freq = config["output_freq"]
    max_samples = config["max_samples"]

    # Convert the parameter ranges to a numpy array if needed
    parameterranges = np.array(
        [config["parameterranges"]["VolumeFractionZirconia"], config["parameterranges"]["ChordLengthRatio"]]
    )

    execpath = Path(config["execpath"])

    " Adaptive phase folders"
    foldername = createfoldername("ZTA-adaptive", "2D")
    runpath = createfolders(execpath, foldername)

    # Print terminal output to text
    output_stream = DualOutputStream(execpath / "terminal_output.txt")
    sys.stdout = output_stream

    """
    # Load validation data
    if property_name == 'thermal_expansion':
        X_test, y_test = load_test_data_CTE(
            f'/data/ray29582/adaptive_gp_InCoKer/validation_data/test_data_32_{property_dict_category[property_name]}')
    elif property_name == 'thermal_conductivity':
        X_test, y_test = load_test_data(
            f'/data/pirkelma/adaptive_gp_InCoKer/thermal_conductivity/20231215/validation_data/mean/test_data_32_{property_dict_category[property_name]}', property_dict[property_name])
    else:
        X_test, y_test = load_test_data(
            f'/data/ray29582/adaptive_gp_InCoKer/validation_data/test_data_32_{property_dict_category[property_name]}', property_dict[property_name])
    """

    X_test, y_test = load_test_data(validation_data_path, property_name)
    assert y_test.shape[0] == X_test.shape[0], "number of samples does not match"
    y_test = y_test.reshape(-1, 1)
    # Initial problem constants
    dim = X_test.shape[1]

    # Clean validation data of nan values
    clean_indices = np.argwhere(~np.isnan(y_test))
    y_test = y_test[clean_indices.flatten()]
    X_test = X_test[clean_indices.flatten()]

    # Custom parameter space boundaries for each feature
    print(f"Parameter ranges: {parameterranges}")

    # Parameters for adaptive phase from config file
    totalbudget = adaptive_phase_parameters["totalbudget"]
    incrementalbudget = adaptive_phase_parameters["incrementalbudget"]
    TOLFEM = adaptive_phase_parameters["TOLFEM"]
    TOLAcqui = adaptive_phase_parameters["TOLAcqui"]
    TOLrelchange = adaptive_phase_parameters["TOLrelchange"]

    # Overall desired reconstruction tolerance
    TOL = adaptive_phase_parameters["TOL"]
    epsphys = np.var(y_test)  # Assumed or known variance of physical measurement!

    initial_design_points = create_initial_design_points(parameterranges)
    Xt_initial = []
    yt_initial = []

    # If initial points are available or restarting a failed run, set compute = False
    compute = config["compute"]

    if compute:
        # Generate initial design points (border points) as training data
        for i, point in enumerate(initial_design_points):
            try:
                print(f"--- Initial Iteration {i}")
                X, Y = generate_candidate_point(
                    point, simulation_options, property_name, output_stream, runpath, "initial_points"
                )
                Xt_initial.append(X)
                yt_initial.append(Y)
                print(f"Initial point: {str(point)}")
                print(f"Found point: {str(X)}")
                print(f"Found value: {str(Y)}")
            except Exception as e:
                print("Skipping")
                print(e)
                continue

        Xt_initial = np.array(Xt_initial)
        yt_initial = np.array(yt_initial).reshape(-1, 1)

    # Restart failed run
    else:
        Xt_initial, yt_initial = load_test_data(Path(runpath) / "initial_points", property_name)
        yt_initial = np.array(yt_initial).reshape(-1, 1)
        # Adjust iteration number for failed run
        iter_count = Xt_initial.shape[0] - 8
        print(f"Row count in Xt_initial: {iter_count}")

    # Initial hyperparameter parameters
    region = [(0.01, 2) for _ in range(dim)]
    assert len(region) == dim, "Too much or fewer hyperparameters for the given problem dimension"

    # Create expected error for each initial point, constant error is passed but true error has to be implemented
    epsXt, epsXgrad = createerror(Xt_initial, random=False, graddata=False)

    print("Initial X")
    print(Xt_initial)
    print("Initial Y")
    print(yt_initial)

    # Train initial GPR
    gp = GPR(Xt_initial, yt_initial, None, None, epsXt, None)
    gp.optimizehyperparameter(region, "mean", False)

    print("\n")

    print("---------------------------------- Adaptive parameters")
    print("Number of initial data points:       {}".format(len(gp.yt)))
    print("Overall stopping tolerance:          {}".format(TOL))
    print("Hyperparameter bounds:               {}".format(region))
    print("\n")

    # Validate initial GPR
    y_pred = gp.predictmean(X_test)

    # calculate MSE
    mse = np.mean((y_pred - y_test) ** 2)

    # calculate RMSE
    rmse = np.sqrt(mse)

    print(property_name)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("Accuracy: ", accuracy_test(gp, X_test, y_test))

    if compute:
        GP_adapted = adapt_inc(
            gp,
            parameterranges,
            TOL,
            TOLAcqui,
            TOLrelchange,
            epsphys,
            X_test,
            y_test,
            execpath,
            runpath,
            output_stream,
            property_name,
            simulation_options,
            output_freq,
            max_samples
        )

    # Pass iteration number for failed run
    else:
        GP_adapted = adapt_inc(
            gp,
            parameterranges,
            TOL,
            TOLAcqui,
            TOLrelchange,
            epsphys,
            X_test,
            y_test,
            execpath,
            runpath,
            output_stream,
            property_name,
            simulation_options,
            output_freq,
            max_samples,
            iter_count,
        )

    print("-----Adaptive run complete:-----")

    y_pred = GP_adapted.predictmean(X_test)

    # calculate MSE
    mse = np.mean((y_pred - y_test) ** 2)

    # calculate RMSE
    rmse = np.sqrt(mse)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("Accuracy: ", accuracy_test(GP_adapted, X_test, y_test))
    # Store model
    joblib.dump(gp, execpath / f"final_gp_{property_name}.joblib")


if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    file_directory = current_file.parent
    config_path = file_directory / "config.yaml"
    main(config_path)
