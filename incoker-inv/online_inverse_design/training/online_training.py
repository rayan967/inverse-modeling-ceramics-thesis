"""
online_training.py

This script is designed for the inverse design of ZTA ceramic microstructures using Gaussian Process Regression (GPR) models. It includes functionalities for loading and preprocessing data, performing hyperparameter optimization, adaptive sampling, and model evaluation.

Modules:
- pathlib, sys, os, json, joblib: For file operations and system interactions.
- pandas (pd), numpy (np): For data manipulation and numerical operations.
- sklearn.metrics: For model evaluation metrics.
- Custom modules (online_adapt, simlopt): For adaptive GPR.
"""

import pathlib
import sys
import os
import json
import joblib
import pandas as pd
from sklearn import metrics
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(os.path.dirname(current_directory))
sys.path.append(parent_directory)
import numpy as np
import argparse
from simlopt.gpr.gaussianprocess import *
from simlopt.hyperparameter.utils.crossvalidation import *
from simlopt.basicfunctions.utils.creategrid import createPD
from pyDOE import lhs
from sklearn.model_selection import train_test_split
from simlopt.basicfunctions.utils.createfolderstructure import *
from online_adapt import *

plt.close('all')
plt.ioff()


def load_test_data(base_path, prop_name, prop='homogenization'):
    """
    Load test data from JSON files within a specified directory.

    Parameters:
    - base_path (str): Directory path to load data from.
    - prop (str, optional): Property to extract from the data. Defaults to 'homogenization'.

    Returns:
    - tuple: A tuple containing two numpy arrays, X (features) and y (target values).
    """
    base_path = pathlib.Path(base_path)
    info_files = list(base_path.glob('**/info.json'))

    X = []
    y = []
    for file in info_files:
        data = json.loads(file.read_text())
        if not (prop in data and "v_phase" in data):
            continue
        vf = data["v_phase"]['11']
        clr = data["chord_length_ratio"]
        X.append([vf, clr])
        y.append(data[prop][prop_name]["value"])

    return np.array(X), np.array(y)

def load_test_data_CTE(base_path, prop='mean'):
    """
    Load test data for CTE from JSON files within a specified directory.

    Parameters:
    - base_path (str): Directory path to load data from.
    - prop (str, optional): Property to extract from the data. Defaults to 'homogenization'.

    Returns:
    - tuple: A tuple containing two numpy arrays, X (features) and y (target values).
    """
    base_path = pathlib.Path(base_path)
    info_files = list(base_path.glob('**/info.json'))

    X = []
    y = []
    for file in info_files:
        data = json.loads(file.read_text())
        if not (prop in data and "v_phase" in data):
            continue
        vf = data["v_phase"]['11']
        clr = data["chord_length_ratio"]
        X.append([vf, clr])
        y.append(data[prop])

    return np.array(X), np.array(y)


def main():
    """
    This function manages the workflow of the script, including loading data,
    setting up the GPR model, performing adaptive sampling, and evaluating model performance.
    """

    # material properties to consider in training
    considered_properties = [
        'thermal_conductivity',
        #'thermal_expansion',
        # 'young_modulus',
        # 'poisson_ratio',
    ]

    # For result indexing
    property_dict = {
        'thermal_conductivity':'Thermal conductivity',
        'thermal_expansion':'Thermal expansion',
        'young_modulus': 'Young modulus',
        'poisson_ratio': 'Poisson ratio',
                     }

    # For simulation options
    property_dict_category = {
        'thermal_conductivity':'thermal_conductivity',
        'thermal_expansion':'thermal_expansion',
        'young_modulus': 'elasticity',
        'poisson_ratio': 'elasticity',
                             }

    # Print terminal output to text
    output_stream = DualOutputStream("terminal_output.txt")
    sys.stdout = output_stream

    execpath = './adapt'
    execname = None

    if not os.path.exists(execpath):
        os.makedirs(execpath)

    ' Adaptive phase folders'
    foldername = createfoldername("ZTA-adaptive", "2D", "1E5")
    runpath = createfolders(execpath, foldername)


    for i, property_name in enumerate(considered_properties):

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

        assert y_test.shape[0] == X_test.shape[0], "number of samples does not match"
        y_test = y_test.reshape(-1,1)
        # Initial problem constants
        dim = X_test.shape[1]

        # Clean validation data of nan values
        clean_indices = np.argwhere(~np.isnan(y_test))
        y_test = y_test[clean_indices.flatten()]
        X_test = X_test[clean_indices.flatten()]

        # Custom parameter space boundaries for each feature
        parameterranges = np.array([[0.15, 0.85],[0.3, 4.0]])
        print(f"Parameter ranges: {parameterranges}")


        # Parameters for adaptive phase
        totalbudget         = 1E20          # Total budget to spend, to be implemented (TBI)
        incrementalbudget   = 1E5           # Incremental budget, TBI
        TOLFEM              = 0.0           # Reevaluation tolerance, TBI
        TOLAcqui            = 1.0           # Acquisition tolerance
        TOLrelchange        = 0             # Tolerance for relative change of global error estimation

        # Overall desired reconstruction tolerance
        TOL                 = 0             # Needs to be set based on property, but set to 0 for full evaluation

        epsphys             = np.var(y_test)    # Assumed or known variance of physical measurement!

        initial_design_points = create_initial_design_points(parameterranges)
        Xt_initial = []
        yt_initial = []

        # If initial points are available or restarting a failed run, set compute = False
        compute = True

        if compute:
            # Generate initial design points (border points) as training data
            for i, point in enumerate(initial_design_points):
                print(f"--- Initial Iteration {i}")
                try:
                    # Set error flag for Mapdl error to False
                    output_stream.error_detected = False

                    # Output path for generated structures
                    output_path = pathlib.Path(runpath, "initial_points", f"v={point[0]},r={point[1]}")
                    output_path.mkdir(parents=True, exist_ok=True)

                    print(f"Initial point: {str(point)}")
                    input = (point[0], point[1])

                    # Generate design point
                    options = {
                        "material_property": property_dict_category[property_name],
                        "particle_quantity": 200,
                        "dim": 32,
                        "max_vertices": 25000,
                        "output_path": output_path

                    }
                    result = prediction_pipeline.generate_and_predict(input, options)

                    # For CTE
                    if property_name == 'thermal_expansion':
                        output_value = result["mean"]

                    # For the rest
                    else:
                        output_value = result["homogenization"][property_dict[property_name]]["value"]

                    vf = result["v_phase"]["11"]
                    cl_11 = result["chord_length_analysis"]["phase_chord_lengths"][11]["mean_chord_length"]
                    cl_4 = result["chord_length_analysis"]["phase_chord_lengths"][4]["mean_chord_length"]
                    clr = cl_11 / cl_4

                    # Check for Mapdl error
                    if output_stream.error_detected:
                        # Reset error flag
                        output_stream.error_detected = False
                        raise Exception("Error detected during operation: Mapdl")

                    # Store the design points and corresponding output
                    Xt_initial.append([vf, clr])
                    yt_initial.append(output_value)
                    print(f"Initial point: {str(point)}")
                    print(f"Found point: {str([vf, clr])}")
                    print(f"Found value: {str(output_value)}")

                except Exception as e:
                    print("Skipping")
                    print(e)
                    continue

            Xt_initial = np.array(Xt_initial)
            yt_initial = np.array(yt_initial).reshape(-1, 1)

        # Restart failed run
        else:
            Xt_initial, yt_initial = load_test_data('adaptZTA-adaptive_2D_1E5/initial_points')
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
            GP_adapted = adapt_inc(gp, parameterranges, TOL, TOLAcqui, TOLrelchange, epsphys, X_test, y_test, runpath, output_stream, property_name)

        # Pass iteration number for failed run
        else:
            GP_adapted = adapt_inc(gp, parameterranges, TOL, TOLAcqui, TOLrelchange, epsphys, X_test, y_test, runpath, output_stream, property_name, iter_count)



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
        joblib.dump(gp, f"adapt/final_gp_{property_name}.joblib")


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
    N   = Xt.shape[0]
    dim = Xt.shape[1]
    epsXgrad = None
    if random:
        epsXt = np.random.rand(N) * (0.1-0.025) + 0.025
        epsXt = epsXt.reshape((1,N))
        if graddata:
            epsXgrad = np.random.rand(N*dim) * (0.1-0.025) + 0.025
    else:
        variances = np.var(Xt, axis=0)
        avg_variance = np.mean(variances)
        epsXt = avg_variance*np.ones((1,N))
        vardata = 1E-4
        epsXt = vardata*np.ones((1,N)) #1E-1 for basic setup
        if graddata:
            vargraddata = 1E-1
            epsXgrad = vargraddata*np.ones((1,N*dim))
    return epsXt, epsXgrad


def accuracy_test(model, X_test, y_test, tolerance=1E-2):
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
    score = metrics.r2_score(y_true=y_test, y_pred=y_pred)*100

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
    points = np.array([[min_x, min_y],
                       [min_x, max_y],
                       [max_x, min_y],
                       [max_x, max_y],
                       [min_x, mid_y],
                       [max_x, mid_y],
                       [mid_x, min_y],
                       [mid_x, max_y],
                       [mid_x, mid_y]])
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


if __name__ == "__main__":
    main()


