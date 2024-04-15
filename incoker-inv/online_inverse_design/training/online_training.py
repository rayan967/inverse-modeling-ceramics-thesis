"""
Train an adaptive GP using online simulations.

This script is designed for the inverse design of ZTA ceramic microstructures using Gaussian Process Regression (GPR)
models. It includes functionalities for loading and preprocessing data, performing hyperparameter optimization, adaptive
sampling, and model evaluation.

Modules:
- pathlib, sys, os, json, joblib: For file operations and system interactions.
- pandas (pd), numpy (np): For data manipulation and numerical operations.
- sklearn.metrics: For model evaluation metrics.
- Custom modules (online_adapt, simlopt): For adaptive GPR.
"""

import json
import sys
from pathlib import Path

import joblib

current_file = Path(__file__).resolve()
run_directory = current_file.parent.parent.parent
sys.path.append(str(run_directory))
import matplotlib.pyplot as plt
import numpy as np
import yaml
from generate_predict_utils import (
    accuracy_test,
    generate_candidate_point,
    get_output,
    phase_zirconia,
    property_dict_category,
)
from online_adapt import adapt_inc, calculate_weights, weighted_distance
from simlopt.basicfunctions.utils.createfolderstructure import (
    createfoldername,
    createfolders,
)
from simlopt.gpr.gaussianprocess import GPR

plt.close("all")
plt.ioff()


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
    info_files = list(base_path.rglob("info.json"))
    X = []
    y = []
    for file in info_files:
        data = json.loads(file.read_text())
        if not (("homogenization" in data or "mean" in data) and "v_phase" in data):
            continue

        vf = data["v_phase"][str(phase_zirconia)]
        main_cl_11 = data["chord_length_analysis"]["phase_chord_lengths"]["11"]["mean_chord_length"]
        main_cl_4 = data["chord_length_analysis"]["phase_chord_lengths"]["4"]["mean_chord_length"]
        clr = main_cl_11 / main_cl_4

        X.append([vf, clr])
        y.append(get_output(data, prop_name))

    return np.array(X), np.array(y)


def load_data_for_restart(base_path, prop_name):
    """
    Load restart data from JSON files within a specified directory.

    Load data for restarting a failed run. Iterates through each parameter set directory
    within `base_path`, then iterates through RVE subdirectories to compile y values
    and compute variance. Uses data from the first RVE subdirectory for X values.

    Parameters:
    - base_path (Path): Base directory path containing parameter set directories.
    - phase_zirconia (int): ID for zirconia phase.
    - phase_alumina (int): ID for alumina phase.

    Returns:
    - Xt (numpy.ndarray): Loaded X values (features).
    - yt (numpy.ndarray): Loaded y values (target values).
    - epsXt (numpy.ndarray): Estimated variances of y values for each parameter set.
    """
    parameter_set_dirs = [x for x in base_path.iterdir() if x.is_dir()]
    Xt = []
    yt = []
    epsXt = []
    for param_dir in parameter_set_dirs:
        y_values = []
        X_values = []
        for i, rve_dir in enumerate(param_dir.iterdir()):
            if rve_dir.is_dir():
                info_path = rve_dir / "info.json"
                if info_path.exists():
                    with open(info_path, "r") as f:
                        data = json.load(f)
                        if not (("homogenization" in data or "mean" in data) and "v_phase" in data):
                            continue
                        y = get_output(data, prop_name)
                        y_values.append(y)

                        vf = data["v_phase"][str(phase_zirconia)]
                        main_cl_11 = data["chord_length_analysis"]["phase_chord_lengths"]["11"]["mean_chord_length"]
                        main_cl_4 = data["chord_length_analysis"]["phase_chord_lengths"]["4"]["mean_chord_length"]
                        clr = main_cl_11 / main_cl_4
                        X_values.append([vf, clr])

        # Compute variance of y_values if not empty
        if y_values and X_values:
            variance = np.var(y_values, ddof=1) if len(y_values) > 1 else 1e-4
            epsXt.append(variance)

            # Use first RVE y_value for yt_initial
            yt.append(y_values[0])
            Xt.append(X_values[0])

    return np.array(Xt), np.array(yt).reshape(-1, 1), np.array(epsXt).reshape(1, -1)


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


def create_initial_design_points(parameterranges):
    """
    Create design points in the initial phase for adaptive GP.

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


class DualOutputStream:
    """
    A class to handle dual output streams, allowing simultaneous writing to both the terminal and a log file.

    Methods:
    - __init__(filename): Constructor to initialize the dual output stream.
    - write(message): Writes a message to both the terminal and the log file.
    - flush(): Flush method for Python 3 compatibility.
    """

    def __init__(self, filename):
        """Initiate the output stream variables."""
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        """Write output error when certain error strings are read."""
        self.terminal.write(message)
        self.log.write(message)

        msg = "intersection(s) found between triangle"
        msg2 = "ansys.mapdl.core.errors.MapdlRuntimeError:"
        if msg in message or msg2 in message:
            self.error_detected = True

    def flush(self):
        """Flush the output stream."""
        pass


def main(config_path):
    """
    Set up and begin the adaptive GP training.

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
    # totalbudget = adaptive_phase_parameters["totalbudget"]
    # incrementalbudget = adaptive_phase_parameters["incrementalbudget"]
    # TOLFEM = adaptive_phase_parameters["TOLFEM"]
    TOLAcqui = adaptive_phase_parameters["TOLAcqui"]
    TOLrelchange = adaptive_phase_parameters["TOLrelchange"]

    # Overall desired reconstruction tolerance
    TOL = adaptive_phase_parameters["TOL"]
    epsphys = np.var(y_test)  # Assumed or known variance of physical measurement!

    initial_design_points = create_initial_design_points(parameterranges)
    Xt_initial = []
    yt_initial = []
    epsXt = []
    # If initial points are available or restarting a failed run, set compute = False
    compute = config["compute"]
    mul_generate_options = config["multiple_generation_options"]

    if compute:
        # Generate initial design points (border points) as training data
        for i, point in enumerate(initial_design_points):
            print(f"--- Initial Iteration {i} ---")
            try:
                best_X, best_y, variance = generate_candidate_point(
                    point,
                    simulation_options,
                    property_name,
                    output_stream,
                    runpath,
                    "initial_points",
                    mul_generate_options,
                    parameterranges,
                )

                print(f"Initial point: {str(point)}")
                print(f"Found point: {str(best_X)}")
                print(f"Found value: {str(best_y)}")
                print(f"Found epsXt: {str(variance)}")

            except Exception as e:
                print("Error generating candidate point:", e)
                continue

        Xt_initial = np.array(Xt_initial)
        yt_initial = np.array(yt_initial).reshape(-1, 1)
        epsXt = np.array(epsXt).reshape(1, -1)
        iter_count = i

    # Restart failed run
    else:
        Xt_initial, yt_initial, epsXt = load_data_for_restart(Path(runpath) / "initial_points", property_name)
        # Adjust iteration number for failed run
        # iter_count = Xt_initial.shape[0] - 8
        # TODO: recheck
        iter_count = Xt_initial.shape[0]
        print(f"Row count in Xt_initial: {iter_count}")

    # Initial hyperparameter parameters
    region = [(0.01, 2) for _ in range(dim)]
    assert len(region) == dim, "Too much or fewer hyperparameters for the given problem dimension"

    # Create expected error for each initial point, constant error is passed but true error has to be implemented
    # epsXt, epsXgrad = createerror(Xt_initial, random=False, graddata=False)

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
        mul_generate_options,
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
