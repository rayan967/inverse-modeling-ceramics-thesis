import pathlib
import sys
import os
import json
import joblib
import pandas as pd
from sklearn import metrics
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
import numpy as np
import argparse
from online_adapt import *
from simlopt.gpr.gaussianprocess import *
from simlopt.hyperparameter.utils.crossvalidation import *
from simlopt.basicfunctions.utils.creategrid import createPD
from pyDOE import lhs
from sklearn.model_selection import train_test_split
from simlopt.basicfunctions.utils.createfolderstructure import *
plt.close('all')
plt.ioff()

considered_features = [
    'volume_fraction_4', 'volume_fraction_1',
    'chord_length_mean_4', 'chord_length_mean_10',
]

# material properties to consider in training
considered_properties = [
    'thermal_conductivity',
    #'thermal_expansion',
    #'young_modulus',
    #'poisson_ratio',
]

def load_test_data(base_path, prop='homogenization'):
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
        y.append(data[prop]["Thermal conductivity"]["value"])

    return np.array(X), np.array(y)

def load_test_data(base_path, prop='homogenization'):
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
        y.append(data[prop]["Thermal conductivity"]["value"])

    return np.array(X), np.array(y)



def main():

    output_stream = DualOutputStream("terminal_output.txt")
    sys.stdout = output_stream
    Xt, Y = load_test_data('/data/pirkelma/adaptive_gp_InCoKer/thermal_conductivity/20231215/validation_data/mean/test_data_32_thermal_conductivity')

    assert Y.shape[0] == Xt.shape[0], "number of samples does not match"

    execpath = './/adapt'
    execname = None

    ' Adaptive phase '
    foldername = createfoldername("ZTA-adaptive", "2D", "1E5")
    runpath = createfolders(execpath, foldername)

    print(foldername)
    print(runpath)


    for i, property_name in enumerate(considered_properties):

        yt = Y
        yt = yt.reshape(-1,1)
        # Initial problem constants
        N   = Xt.shape[0]
        dim = Xt.shape[1]
        m   = yt.shape[1]

        clean_indices = np.argwhere(~np.isnan(yt))
        yt = yt[clean_indices.flatten()]
        Xt = Xt[clean_indices.flatten()]

        # Calculate parameter space boundaries for each feature
        parameterranges = np.array([[np.min(Xt[:, i]), np.max(Xt[:, i])] for i in range(dim)])
        print(f"DB Parameter ranges: {parameterranges}")

        parameterranges = np.array([[0.15, 0.85],[0.3, 5.0]])
        print(f"Parameter ranges: {parameterranges}")


        # Parameters for adaptive phase
        totalbudget         = 1E20          # Total budget to spend
        incrementalbudget   = 1E5           # Incremental budget
        TOL                 = 1E-2          # Overall desired reconstruction tolerance
        TOLFEM              = 0.0           # Reevaluation tolerance
        TOLAcqui            = 1.0           # Acquisition tolerance
        TOLrelchange        = 0             # Tolerance for relative change of global error estimation

        epsphys             = np.var(yt)    # Assumed or known variance of physical measurement!
        adaptgrad           = False         # Toggle if gradient data should be adapted

        X_test, y_test = Xt, yt

        initial_design_points = create_initial_design_points(parameterranges)
        Xt_initial = []
        yt_initial = []


        # If initial points are available or restarting a failed run, set compute = True
        compute = False

        if compute:

            for i, point in enumerate(initial_design_points):
                print(f"--- Initial Iteration {i}")
                try:
                    output_stream.error_detected = False

                    output_path = pathlib.Path(runpath, "initial_points", f"v={input[0]},r={input[1]}")
                    output_path.mkdir(parents=True, exist_ok=True)

                    print(f"Initial point: {str(point)}")
                    input = (point[0], point[1])
                    options = {
                        #"material_property": "elasticity",
                        "material_property": "thermal_conductivity",
                        "particle_quantity": 200,
                        "dim": 32,
                        "max_vertices": 25000,
                        "output_path": output_path

                    }

                    result = prediction_pipeline.generate_and_predict(input, options)
                    output_value = result["homogenization"]["Thermal conductivity"]["value"]

                    vf = result["v_phase"]["11"]
                    cl_11 = result["chord_length_analysis"]["phase_chord_lengths"][11]["mean_chord_length"]
                    cl_4 = result["chord_length_analysis"]["phase_chord_lengths"][4]["mean_chord_length"]
                    clr = cl_11 / cl_4

                    if output_stream.error_detected:
                        output_stream.error_detected = False

                    # Store the design points and corresponding output
                    Xt_initial.append([vf, clr])
                    yt_initial.append(output_value)
                    print(f"Initial point: {str(point)}")
                    print(f"Found point: {str([vf, clr])}")
                    print(f"Found value: {str(output_value)}")

                except Exception as e:
                    if str(e) == "list index out of range":
                        print(e)
                        print("Skipping")
                        continue
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

        epsXt, epsXgrad = createerror(Xt_initial, random=False, graddata=False)

        epsphys = np.var(yt)
        print("Initial X")
        print(Xt_initial)
        print("Initial Y")
        print(yt_initial)

        gp = GPR(Xt_initial, yt_initial, None, None, epsXt, None)
        gp.optimizehyperparameter(region, "mean", False)

        print("\n")

        print("---------------------------------- Adaptive parameters")
        print("Number of initial data points:       {}".format(len(gp.yt)))
        print("Overall stopping tolerance:          {}".format(TOL))
        print("Hyperparameter bounds:               {}".format(region))
        print("\n")
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
            GP_adapted = adapt_inc(gp, parameterranges, TOL, TOLAcqui, TOLrelchange, epsphys, X_test, y_test, runpath, output_stream)

        # Pass iteration number for failed run
        else:
            GP_adapted = adapt_inc(gp, parameterranges, TOL, TOLAcqui, TOLrelchange, epsphys, X_test, y_test, runpath, output_stream, iter_count)



        print("-----Adaptive run complete:-----")

        y_pred = GP_adapted.predictmean(X_test)

        # calculate MSE
        mse = np.mean((y_pred - y_test) ** 2)

        # calculate RMSE
        rmse = np.sqrt(mse)
        print("MSE: ", mse)
        print("RMSE: ", rmse)
        print("Accuracy: ", accuracy_test(GP_adapted, X_test, y_test))
        joblib.dump(gp, "adapt/final_gp.joblib")



def createerror(Xt, random=False, graddata=False):
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
    Parameters
    ----------
    model : GPR model
    X_test : np.array
        Test data (features).
    y_test : np.array
        Test data (true values).
    tolerance : float
        Tolerance for the accuracy score.

    Returns
    -------
    score : float
        Accuracy score between 0 and 100.
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

import sys

class DualOutputStream:
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


