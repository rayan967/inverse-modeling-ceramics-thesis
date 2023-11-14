import pathlib
import sys
import os
from time import sleep

import joblib
import numpy as np
from sklearn.metrics import r2_score

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

import argparse
from online_adapt import *
from simlopt.gpr.gaussianprocess import *
from simlopt.hyperparameter.utils.crossvalidation import *
from simlopt.basicfunctions.utils.creategrid import createPD
from pyDOE import lhs
from sklearn.model_selection import train_test_split
import incoker_micro_sims
from incoker_micro_sims import prediction_pipeline
import shutil


plt.close('all')
plt.ioff()

considered_features = [
    'volume_fraction_4', 'volume_fraction_10', 'volume_fraction_1',
    'chord_length_mean_4', 'chord_length_mean_10', 'chord_length_mean_1',
    'chord_length_variance_4', 'chord_length_variance_10', 'chord_length_variance_1'
]

# material properties to consider in training
considered_properties = [
    #'thermal_conductivity',
    'thermal_expansion',
    #'young_modulus',
    #'poisson_ratio',
]
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

def extract_XY(data):
    X = np.vstack(tuple(data[f] for f in considered_features)).T
    Y = np.vstack(tuple(data[p] for p in considered_properties)).T

    return X, Y


def main():

    training_data = pathlib.Path("data/training_data_rve_database.npy")
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
        parameterranges = np.array([[np.min(Xt[:, i]), np.max(Xt[:, i])],[np.min(Xt[:, i]), 3.0]])

        print(parameterranges)

        # Parameters for adaptive phase
        totalbudget         = 1E20          # Total budget to spend
        incrementalbudget   = 1E5           # Incremental budget
        TOL                 = 1E-2          # Overall desired reconstruction tolerance
        TOLFEM              = 0.0           # Reevaluation tolerance
        TOLAcqui            = 1.0           # Acquisition tolerance
        TOLrelchange        = 0             # Tolerance for relative change of global error estimation

        epsphys             = np.var(yt)    # Assumed or known variance of physical measurement!
        adaptgrad           = False         # Toggle if gradient data should be adapted

        selected_indices = []
        XGLEE = createPD(30, dim, "latin", parameterranges)
        Xt_initial = np.zeros_like(XGLEE)
        yt_initial = np.zeros((30, yt.shape[1]))
        for i in range(30):

            shutil.rmtree("output")
            sleep(30)
            input = (XGLEE[i][0],XGLEE[i][1])
            options = {
                # "material_property": "elasticity"
                "material_property": "thermal_expansion",
                # "material_property": "thermal_conductivity",
                "particle_quantity": 60,
                "dim": 16,
                "max_vertices": 10000
            }
            result = prediction_pipeline.generate_and_predict(input, options)

            vf = result["volume_fractions"][11]
            cl_11 = result["chord_length_analysis"]["phase_chord_lengths"][11]["mean_chord_length"]
            cl_4 = result["chord_length_analysis"]["phase_chord_lengths"][4]["mean_chord_length"]
            clr = cl_11 / cl_4

            Xt_initial[i] = np.array([vf,clr]).reshape(1,-1)
            yt_initial[i] = np.array([result["mean"]]).reshape(1,-1)




        N   = Xt_initial.shape[0]

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


        GP_adapted = adapt_inc(gp, parameterranges, TOL, TOLAcqui, TOLrelchange, epsphys, Xt, yt)
        gp.optimizehyperparameter(region, "mean", False)

        print("-----Adaptive run complete:-----")



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


    r2 = r2_score(y_test, y_pred)

    return r2


if __name__ == "__main__":
    main()


