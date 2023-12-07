import pathlib
import sys
import os

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
import signal

class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")
signal.signal(signal.SIGALRM, timeout_handler)


plt.close('all')
plt.ioff()

considered_features = [
    'volume_fraction_4', 'volume_fraction_1',
    'chord_length_mean_4', 'chord_length_mean_10',
]

# material properties to consider in training
considered_properties = [
    'thermal_conductivity_composite',
    #'thermal_expansion',
    #'young_modulus',
    #'poisson_ratio',
]


def extract_XY(data):
    X = np.vstack(tuple(data[f] for f in considered_features)).T
    Y = np.vstack(tuple(data[p] for p in considered_properties)).T

    return X, Y

def extract_XY_2(data):
    """Use for 2 features."""

    filtered_indices = np.where(data['volume_fraction_1'] == 0.0)

    chord_length_ratio = data['chord_length_mean_4'][filtered_indices] / data['chord_length_mean_10'][filtered_indices]

    volume_fraction_4 = data['volume_fraction_4'][filtered_indices]

    X = np.vstack((volume_fraction_4, chord_length_ratio)).T

    Y = np.vstack(tuple(data[p][filtered_indices] for p in considered_properties)).T

    global considered_features

    considered_features = [
    'volume_fraction_4',
    'chord_length_ratio'
]

    return X, Y


def extract_XY_3(data):
    """Use for 3 features."""

    chord_length_ratio = data['chord_length_mean_4'] / data['chord_length_mean_10']
    X = np.vstack((data['volume_fraction_4'], data['volume_fraction_1'], chord_length_ratio)).T
    Y = np.vstack(tuple(data[p] for p in considered_properties)).T

    global considered_features

    considered_features = [
    'volume_fraction_4', 'volume_fraction_1',
    'chord_length_ratio'
]
    return X, Y



def main():

    training_data = pathlib.Path("data/training_data_rve_database.npy")
    if not training_data.exists():
        print(f"Error: training data path {training_data} does not exist.")

    output_stream = DualOutputStream("terminal_output.txt")
    sys.stdout = output_stream
    #if training_data.suffix == '.npy':
    #    data = np.load(training_data)
    #else:
    #    print("Invalid data")

    #print(f"loaded {data.shape[0]} training data pairs")

    #data['thermal_expansion'] *= 1e6

    # Training data
    #Xt, Y = extract_XY_2(data)

    df = pd.read_csv("data/thermal_conductivity_tcc_64.csv")
    models = {}
    # Calculate chord_length_mean_ratio
    df['chord_length_mean_ratio'] = df['chord_length_mean_zro2'] / df['chord_length_mean_al2o3']
    df.dropna(inplace=True)

    features = ['volume_fraction_zro2', 'chord_length_mean_ratio']

    Xt = df[features].to_numpy()
    Y = df['thermal_conductivity_composite'].to_numpy()
    property_name = ['thermal_conductivity_composite']

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

        parameterranges = np.array([[0.15, 0.85],[0.3, 2.0]])
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

        Xt, X_test, yt, y_test = train_test_split(Xt, yt, random_state=0)

        initial_design_points = create_initial_design_points(parameterranges)
        Xt_initial = []
        yt_initial = []
        for i, point in enumerate(initial_design_points):
            print(f"--- Initial Iteration {i}")
            try:
                signal.alarm(600)
                output_stream.error_detected = False

                print(f"Initial point: {str(point)}")
                input = (point[0], point[1])
                options = {
                    "material_property": "thermal_conductivity",
                    "particle_quantity": 200,
                    "dim": 16,
                    "max_vertices": 10000
                }
                result = prediction_pipeline.generate_and_predict(input, options)
                output_value = result["homogenization"]["Thermal conductivity"]["value"]
                vf = result["v_phase"]["11"]
                cl_11 = result["chord_length_analysis"]["phase_chord_lengths"][11]["mean_chord_length"]
                cl_4 = result["chord_length_analysis"]["phase_chord_lengths"][4]["mean_chord_length"]
                clr = cl_11 / cl_4

                if output_stream.error_detected:
                    output_stream.error_detected = False
                    raise Exception("Error detected during operation: Mapdl")

                # Store the design points and corresponding output
                Xt_initial.append([vf, clr])
                yt_initial.append(output_value)
                print(f"Initial point: {str(point)}")
                print(f"Found point: {str(Xt_initial[i])}")
                print(f"Found value: {str(output_value)}")
                signal.alarm(0)

            except TimeoutException as te:
                print(f"Timeout occurred for initial iteration {i}: {te}")
                signal.alarm(0)
                continue
            except Exception as e:
                print(e)
                continue

        Xt_initial = np.array(Xt_initial)
        yt_initial = np.array(yt_initial).reshape(-1, 1)
        # Initial hyperparameter parameters
        region = [(0.01, 2) for _ in range(dim)]
        assert len(region) == dim, "Too much or fewer hyperparameters for the given problem dimension"

        epsXt, epsXgrad = createerror(Xt_initial, random=False, graddata=False)

        epsphys = np.var(yt)

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

        GP_adapted = adapt_inc(gp, parameterranges, TOL, TOLAcqui, TOLrelchange, epsphys, Xt, yt, X_test, y_test, runpath, output_stream)

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


