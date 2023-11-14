import pathlib
import sys
import os

from sklearn import metrics

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
import numpy as np
import argparse
from adapt import *
from simlopt.gpr.gaussianprocess import *
from simlopt.hyperparameter.utils.crossvalidation import *
from simlopt.basicfunctions.utils.creategrid import createPD
from pyDOE import lhs
from sklearn.model_selection import train_test_split


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



def main(train_data_file, export_model_file):

    training_data = pathlib.Path(train_data_file)
    if not training_data.exists():
        print(f"Error: training data path {training_data} does not exist.")

    if training_data.suffix == '.npy':
        data = np.load(training_data)
    else:
        print("Invalid data")

    print(f"loaded {data.shape[0]} training data pairs")

    data['thermal_expansion'] *= 1e6

    # Training data
    Xt, Y = extract_XY_2(data)

    assert Y.shape[0] == Xt.shape[0], "number of samples does not match"

    for i, property_name in enumerate(considered_properties):

        yt = Y[:, i]
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
        Xt_initial = np.zeros((9, 2))
        yt_initial = np.zeros((9, 1))

        for i, point in enumerate(initial_design_points):
            # Your FEM simulation call
            input = (point[0], point[1])
            Xt_initial[i], yt_initial[i], Xt, yt = find_closest_point(Xt, yt, point, None)


        # Initial hyperparameter parameters
        region = [(0.01, 2) for _ in range(dim)]
        assert len(region) == dim, "Too much or fewer hyperparameters for the given problem dimension"

        epsXt, epsXgrad = createerror(Xt_initial, random=False, graddata=False)

        epsphys = np.var(yt)

        gp = GPR(Xt_initial, yt_initial, None, None, epsXt, None)
        gp.optimizehyperparameter(region, "mean", False)

        print("\n")

        print("---------------------------------- Adaptive parameters")
        print("Number of initial data points:       {}".format(N))
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

        GP_adapted = adapt_inc(gp, parameterranges, TOL, TOLAcqui, TOLrelchange, epsphys, Xt, yt, X_test, y_test)

        print("-----Adaptive run complete:-----")

        y_pred = GP_adapted.predictmean(X_test)

        # calculate MSE
        mse = np.mean((y_pred - y_test) ** 2)

        # calculate RMSE
        rmse = np.sqrt(mse)
        print("MSE: ", mse)
        print("RMSE: ", rmse)
        print("Accuracy: ", accuracy_test(GP_adapted, X_test, y_test))


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ML models using structure-property data of '
                                                 'RVEs.')
    parser.add_argument('train_data_file', type=pathlib.Path,
                        help='Path to the database of RVE structures and simulation results or '
                             'numpy file with training data already loaded')
    parser.add_argument('--export_model_file', type=pathlib.Path, required=False,
                        help='Path to a file where the trained models will be exported to.')
    args = parser.parse_args()

    main(args.train_data_file, args.export_model_file)


