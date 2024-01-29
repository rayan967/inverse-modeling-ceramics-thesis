"""
online_adapt.py

This script contains functions for adaptive sampling and optimization
using a Gaussian Process surrogate model. It is designed to iteratively
select new candidate points for evaluation and update the surrogate model
to optimize a given property.
"""


import joblib
import numpy as np
import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
import pathlib
from incoker_micro_sims import prediction_pipeline
from simlopt.gpr.gaussianprocess import *
from simlopt.basicfunctions.utils.creategrid import createPD
from simlopt.optimization.errormodel_new import MCGlobalEstimate, acquisitionfunction, estiamteweightfactors
from adaptive_training import accuracy_test
from simlopt.optimization.utilities import *
import matplotlib.pyplot as plt

property_dict = {
    'thermal_conductivity': 'Thermal conductivity',
    'thermal_expansion': 'Thermal expansion',
    'young_modulus': 'Young modulus',
    'poisson_ratio': 'Poisson ratio',
}

property_dict_category = {
    'thermal_conductivity': 'thermal_conductivity',
    'thermal_expansion': 'thermal_expansion',
    'young_modulus': 'elasticity',
    'poisson_ratio': 'elasticity',
}

def adapt_inc(gp, parameterranges, TOL, TOLAcqui, TOLrelchange, epsphys, X_test, y_test, runpath, output_stream, property_name, iter_count=None):
    """
    Perform the adaptive sampling and optimization process.

    Args:
        gp (GaussianProcess): The Gaussian Process surrogate model.
        parameterranges (numpy.ndarray): Array of parameter ranges.
        TOL (float): Desired tolerance for the global error estimate.
        TOLAcqui (float): Tolerance for the acquisition function.
        TOLrelchange (float): Tolerance for relative change in global error estimate.
        epsphys (float): Physical variance in property.
        X_test (numpy.ndarray): Test data features.
        y_test (numpy.ndarray): Test data targets.
        runpath (str): Path to the output directory.
        output_stream: Output stream for logging.
        property_name (str): Name of the property being optimized.
        iter_count (int): Current iteration (for restarted runs).

    Returns:
        GaussianProcess: The updated Gaussian Process model.
    """
    # If restarted run, use iter_count
    if iter_count is None:
        counter = 0
    else:
        counter = iter_count
    # Size of current data
    N = gp.getdata[0]

    # Dimensions of data
    dim = gp.getdata[2]

    # Size of grid
    NMC = 600

    # List for storing values for plots
    global_errors = []
    accuracies = []
    cases = {1:"Case 1: Gradient data is not available.",
             2:"Case 1: Gradient data is available."}

    # Calculate weights to normalize parameter ranges for finding fair euclidean distance
    weights = calculate_weights(parameterranges)

    # Initialize zones to exclude that cannot be generated
    exclusion_zones = []
    generated_points_history = np.empty((0, len(parameterranges)))
    'Check for which cases are set.'
    if gp.getXgrad is None:
        Ngrad = gp.getdata[1] #Is None, when Xgrad is None
        case = 1
    elif gp.getXgrad is not None:
        Ngrad = gp.getdata[1]
        case = 2
    figurepath = os.path.join(runpath+"/", "iteration_plots/")
    print("---------------------------------- Start adaptive phase")
    print(cases[case])
    print("Number of initial points:          "+str(len(gp.yt)))
    print("Desired tolerance:                 "+str(TOL))
    print("\n")


    # Main adaptive loop
    while True:
        print(f"--- Iteration {counter}")
        print("Number of points:          " + str(len(gp.yt)))

        # Generate candidate points
        XGLEE = createPD(NMC, dim, "grid", parameterranges, exclusion_zones)
        dfGLEE = gp.predictderivative(XGLEE, True)
        varGLEE = gp.predictvariance(XGLEE, True)
        normvar = np.linalg.norm(np.sqrt(np.abs(varGLEE)), 2, axis=0) ** 2

        # Estimate weight factors
        w = estiamteweightfactors(dfGLEE, epsphys)


        # MC global error estimation
        mcglobalerrorbefore = MCGlobalEstimate(w, normvar, NMC,
                                               parameterranges)
        print("Global error estimate before optimization:   {:1.5f}".format(mcglobalerrorbefore))

        """ ------------------------------Acquisition phase ------------------------------ """
        'Add new candidate points'

        XC = np.array([])
        Xc = np.array([])
        Yc = np.array([])

        normvar_TEST    = np.linalg.norm(np.sqrt(np.abs(varGLEE)),2,axis=0)
        # Acquisition phase
        XC, index, value = acquisitionfunction(gp, dfGLEE, normvar, w, XGLEE, epsphys,
                                               TOLAcqui, generated_points_history)  # Use your acquisition function

        if XC.size != 0:
            print(" Number of possible candidate points: {}".format(XC.shape[0]))
            print(" Found canditate point(s):            {}".format(XC[0]))
            print(" Use ith highest value   :            {}".format(index))
            print(" Value at index          :            {}".format(value))
            print("\n")

            # Try to generate candidate XC add it to the GP model

            try:
                # Set error flag for Mapdl error to False
                output_stream.error_detected = False

                input = (XC[0][0], XC[0][1])
                output_path = pathlib.Path(runpath, "adaptive_points", f"v={input[0]:.2f},r={input[1]:.2f}")
                output_path.mkdir(parents=True, exist_ok=True)

                # Generate candidate point
                options = {
                    "material_property": property_dict_category[property_name],
                    "particle_quantity": 200,
                    "dim": 32,
                    "max_vertices": 25000,
                    "output_path": output_path
                }
                result = prediction_pipeline.generate_and_predict(input, options)
                print(result.keys())

                vf = result["v_phase"]["11"]
                cl_11 = result["chord_length_analysis"]["phase_chord_lengths"][11]["mean_chord_length"]
                cl_4 = result["chord_length_analysis"]["phase_chord_lengths"][4]["mean_chord_length"]
                clr = cl_11 / cl_4

                # For CTE
                if property_name == 'thermal_expansion':
                    output_value = result["mean"]

                # For the rest
                else:
                    output_value = result["homogenization"][property_dict[property_name]]["value"]

                Xc = np.array([vf,clr]).reshape(1,-1)
                Yc = np.array([output_value]).reshape(1,-1)

                # Check for Mapdl error
                if output_stream.error_detected:
                    # Reset error flag
                    output_stream.error_detected = False
                    raise Exception("Error detected during operation: Mapdl")

                # Find distance between requested candidate XC and generated point Xc
                dist = weighted_distance(XC[0], Xc[0], weights)
                distance_threshold = 0.04
                print(f"Distance between generated and requested point: {dist}")

                # If distance too large, add as an exclusion zone
                if dist > 0.1:
                    print(f"Distance to generated point {str(Xc[0])} is larger than threshold: {dist}")
                    print(f"Excluding point from future sampling: {str(XC[0])}")
                    exclusion_zone = (XC[0], distance_threshold)
                    exclusion_zones.append(exclusion_zone)
                    print(f"Distance is larger than threshold: {distance_threshold}")

                # Plot iterations to adaptZTA-adaptive_2D_1E5
                plotiteration(gp, w, normvar_TEST, N, Ngrad, XGLEE, XC, mcglobalerrorbefore, parameterranges, figurepath,
                          counter, Xc)

            except Exception as e:
                if str(e) == "list index out of range":
                    print(e)
                    print(f"Excluding point from future sampling: {str(XC[0])}")
                    exclusion_zone = (XC[0], 0.05)
                    exclusion_zones.append(exclusion_zone)
                    continue

                print(f"Error at {str(counter)} iteration at size {str(len(gp.yt))}")
                print(f"Error: {e}")
                continue
        else:
            print("Something went wrong, no candidate point was found.")
            print("\n")
            continue

        generated_points_history = np.vstack([generated_points_history, XC[0]])

        if Yc.size == 0:
            continue

        # Use constant variance for current point
        epsXc = 1E-4 * np.ones((1, XC.shape[0]))  # eps**2

        # Add point to GP
        gp.adddatapoint(Xc)
        gp.adddatapointvalue(Yc)
        gp.addaccuracy(epsXc)
        print(" Found canditate point(s):            {}".format(XC[0]))
        print(f"Found Xc: {str(Xc)}, Yc: {str(Yc)}")
        print("Size of data: ", str(len(gp.yt)))

        # A posteriori MC global error estimation
        dfGLEE = gp.predictderivative(XGLEE, True)
        varGLEE = gp.predictvariance(XGLEE, True)
        wpost = estiamteweightfactors(dfGLEE, epsphys)
        normvar = np.linalg.norm(np.sqrt(np.abs(varGLEE)), 2, axis=0) ** 2
        mcglobalerrorafter = MCGlobalEstimate(wpost, normvar, NMC, parameterranges)
        global_errors.append(mcglobalerrorafter)

        acc = accuracy_test(gp, X_test, y_test)
        print(" Current accuracy:            {}".format(str(acc)))
        accuracies.append(acc)

        # Check convergence
        if mcglobalerrorafter <= TOL:
            print("--- Convergence")
            print(" Desired tolerance is reached, adaptive phase is done.")
            plot_global_errors(global_errors)
            plot_accuracy(accuracies)
            return gp

        # Adjust budget
        relchange = np.abs(mcglobalerrorbefore - mcglobalerrorafter) / mcglobalerrorbefore * 100
        if relchange < TOLrelchange:
            TOLAcqui *= 0.9999
            print("Relative change is below set threshold. Adjusting TOLAcqui.")

        # Check number of points

        if len(gp.yt) >= 150:
            print("--- Maximum number of points reached")
            plot_global_errors(global_errors)
            plot_accuracy(accuracies)
            return gp

        if len(gp.yt) % 5 == 0:
            plot_global_errors(global_errors)
            plot_accuracy(accuracies)
            joblib.dump(gp, f"adapt/{property_name}_{str(len(gp.yt))}_gp.joblib")

        Nmax = 50
        N = gp.getdata[0]
        if N < Nmax:
            print("--- A priori hyperparameter adjustment")
            region = [(0.01, 2) for _ in range(dim)]
            gp.optimizehyperparameter(region, "mean", False)
        else:
            print("--- A priori hyperparameter adjustment")
            print("Number of points is higher then "+str(Nmax))
            print("No optimization is performed")
        print("\n")
        counter += 1


def plot_global_errors(global_errors):
    """
    Plot the global error estimate per iteration.

    Args:
        global_errors (list): List of global error estimates.

    Returns:
        None

    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(global_errors) + 1), global_errors, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('MC Global Error Estimate')
    plt.title('MC Global Error Estimate per Iteration')
    plt.grid(True)
    plt.savefig('mc_global_error_plot.png')


def plot_accuracy(accuracies):
    """
    Plot the accuracy per iteration.

    Args:
        accuracies (list): List of accuracy values.

    Returns:
        None

    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Iteration')
    plt.grid(True)
    plt.savefig('accuracy_plot.png')


def calculate_weights(parameterranges):
    """
    Calculate weights for each parameter inversely proportional to their range.

    :param parameterranges: Array of parameter ranges.
    :return: Array of weights.
    """
    ranges = parameterranges[:, 1] - parameterranges[:, 0]
    weights = 1 / ranges
    return weights

def weighted_distance(point_a, point_b, weights):
    """
    Calculate the weighted Euclidean distance between two points.

    :param point_a: First point (array-like).
    :param point_b: Second point (array-like).
    :param weights: Weights for each dimension (array-like).
    :return: Weighted distance.
    """
    diff = np.array(point_a) - np.array(point_b)
    weighted_diff = diff * weights
    return np.sqrt(np.sum(weighted_diff ** 2))