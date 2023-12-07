import shutil
from time import sleep

import joblib
import numpy as np
from simlopt.gpr.gaussianprocess import *

import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from incoker_micro_sims import prediction_pipeline

from simlopt.gpr.gaussianprocess import *

from simlopt.basicfunctions.utils.creategrid import createPD
from simlopt.optimization.errormodel_new import MCGlobalEstimate, acquisitionfunction, estiamteweightfactors
import matplotlib.pyplot as plt
from adaptive_training import accuracy_test
from simlopt.optimization.utilities import *

from sklearn import metrics
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")

# Set the signal handler for the SIGALRM signal
signal.signal(signal.SIGALRM, timeout_handler)


def adapt_inc(gp, parameterranges, TOL, TOLAcqui, TOLrelchange, epsphys, Xt, yt, X_test, y_test, runpath, output_stream):
    # Initialization
    err = 0
    flag = 0
    counter = 0
    N = gp.getdata[0]
    dim = gp.getdata[2]
    m = yt.shape[1]
    NMC = 600
    totaltime = 0
    totalFEM = 0
    global_errors = []
    accuracies = []
    cases = {1:"Case 1: Gradient data is not available.",
             2:"Case 1: Gradient data is available."}

    weights = calculate_weights(parameterranges)

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

    XGLEE_f = X_test

    # Main adaptive loop
    while True:
        print(f"--- Iteration {counter}")
        print("Number of points:          " + str(len(gp.yt)))

        # Generate candidate points
        XGLEE = createPD(NMC, dim, "grid", parameterranges, exclusion_zones)
        dfGLEE = gp.predictderivative(XGLEE, True)
        varGLEE = gp.predictvariance(XGLEE, True)
        normvar = np.linalg.norm(np.sqrt(np.abs(varGLEE)), 2, axis=0) ** 2
        w = estiamteweightfactors(dfGLEE, epsphys)


        # MC global error estimation
        dfGLEE_f = gp.predictderivative(XGLEE_f, True)
        varGLEE_f = gp.predictvariance(XGLEE_f, True)
        normvar_f = np.linalg.norm(np.sqrt(np.abs(varGLEE_f)), 2, axis=0) ** 2
        w_f = estiamteweightfactors(dfGLEE_f, epsphys)
        mcglobalerrorbefore = MCGlobalEstimate(w_f, normvar_f, NMC,
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

        # Find closest point in the training data and add it to the GP model
        if XC.size != 0:
            print(" Number of possible candidate points: {}".format(XC.shape[0]))
            print(" Found canditate point(s):            {}".format(XC[0]))
            print(" Use ith highest value   :            {}".format(index))
            print(" Value at index          :            {}".format(value))
            print("\n")


            try:
                signal.alarm(600)
                output_stream.error_detected = False
                # y = generate_and_predict(XC[0], 'thermal_expansion')
                # XC[0] = [-0.31546006  0.27063561  1.00704592], y = [5.5]
                input = (XC[0][0], XC[0][1])
                options = {
                    #"material_property": "elasticity"
                    #"material_property": "thermal_expansion",
                     "material_property": "thermal_conductivity",
                    "particle_quantity": 200,
                    "dim": 16,
                    "max_vertices": 10000
                }
                result = prediction_pipeline.generate_and_predict(input, options)
                print(result.keys())

                vf = result["v_phase"]["11"]
                cl_11 = result["chord_length_analysis"]["phase_chord_lengths"][11]["mean_chord_length"]
                cl_4 = result["chord_length_analysis"]["phase_chord_lengths"][4]["mean_chord_length"]
                clr = cl_11 / cl_4

                output_value = result["homogenization"]["Thermal conductivity"]["value"]

                Xc = np.array([vf,clr]).reshape(1,-1)
                Yc = np.array([output_value]).reshape(1,-1)

                if output_stream.error_detected:
                    output_stream.error_detected = False
                    raise Exception("Error detected during operation: Mapdl")

                dist = weighted_distance(XC[0], Xc[0], weights)
                distance_threshold = 0.05
                if dist > 0.2:
                    print(f"Distance to generated point {str(Xc[0])} is larger than threshold: {dist}")
                    print(f"Excluding point from future sampling: {str(XC[0])}")
                    exclusion_zone = (XC[0], distance_threshold)
                    exclusion_zones.append(exclusion_zone)
                    print(f"Distance is larger than threshold: {dist}")

                print(f"distance between generated and requested point: {dist}")
                signal.alarm(0)

            except TimeoutException as te:
                print(f"Timeout occurred for iteration {counter}: {te}")
                signal.alarm(0)
                continue
            except Exception as e:
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
        epsXc = 1E-4 * np.ones((1, XC.shape[0]))  # eps**2
        gp.adddatapoint(Xc)
        gp.adddatapointvalue(Yc)
        gp.addaccuracy(epsXc)
        print(" Found canditate point(s):            {}".format(XC[0]))
        print(f"Found Xc: {str(Xc)}, Yc: {str(Yc)}")
        print("Size of data: ", str(len(gp.yt)))

        # A posteriori MC global error estimation
        dfGLEE_f = gp.predictderivative(XGLEE_f, True)
        varGLEE_f = gp.predictvariance(XGLEE_f, True)
        wpost_f = estiamteweightfactors(dfGLEE_f, epsphys)
        normvar_f = np.linalg.norm(np.sqrt(np.abs(varGLEE_f)), 2, axis=0) ** 2
        mcglobalerrorafter = MCGlobalEstimate(wpost_f, normvar_f, NMC, parameterranges)
        global_errors.append(mcglobalerrorafter)

        acc = accuracy_test(gp, X_test, y_test)
        print(" Current accuracy:            {}".format(str(acc)))
        plotiteration(gp,w,normvar_TEST,N,Ngrad,XGLEE,XC,mcglobalerrorbefore,parameterranges, figurepath,counter, Xc)
        accuracies.append(acc)

        # Check convergence
        #  if mcglobalerrorafter <= TOL:
        #     print("--- Convergence")
        #    print(" Desired tolerance is reached, adaptive phase is done.")
        #    plot_global_errors(global_errors)
        #    plot_accuracy(accuracies)
        #    return gp

        # Adjust budget
        relchange = np.abs(mcglobalerrorbefore - mcglobalerrorafter) / mcglobalerrorbefore * 100
        if relchange < TOLrelchange:
            TOLAcqui *= 0.9999
            print("Relative change is below set threshold. Adjusting TOLAcqui.")

        # Check number of points

        if len(gp.yt) >= 600:
            print("--- Maximum number of points reached")
            plot_global_errors(global_errors)
            plot_accuracy(accuracies)
            return gp

        if len(gp.yt) % 100 == 0:
            plot_global_errors(global_errors)
            plot_accuracy(accuracies)
            joblib.dump(gp, f"adapt/{str(len(gp.yt))}_gp.joblib")

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


def find_closest_point(Xt, yt, point, gp=None):
    distances = np.linalg.norm(Xt - point, axis=1)

    index = np.argmin(distances)
    selected_x = Xt[index].reshape(1, -1)
    selected_y = yt[index].reshape(1, -1)

    # Removing the selected data point from Xt and yt
    Xt = np.delete(Xt, index, axis=0)
    yt = np.delete(yt, index, axis=0)

    if gp is not None:
        y = gp.yt
        common_elements = len(set(y.flatten()) & set(yt.flatten()))
        print(f"Number of unique elements in y: {len(set(y.flatten()))}")
        print(f"Number of common unique elements between y and yt: {common_elements}")

    return selected_x, selected_y, Xt, yt


def plot_global_errors(global_errors):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(global_errors) + 1), global_errors, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('MC Global Error Estimate')
    plt.title('MC Global Error Estimate per Iteration')
    plt.grid(True)
    plt.savefig('mc_global_error_plot.png')


def plot_accuracy(accuracies):
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