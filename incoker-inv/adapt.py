import numpy as np

import time
from timeit import default_timer as timer

import scipy

from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint

from simlopt.basicfunctions.covariance.cov import *
from simlopt.basicfunctions.utils.creategrid import *
from simlopt.basicfunctions.kaskade.kaskadeio import *

from simlopt.optimization.errormodel_new import *
from simlopt.optimization.workmodel import *
from simlopt.optimization.utilities import *

from simlopt.gpr.gaussianprocess import *

from simlopt.IOlogging.iotofile import *

from simlopt.basicfunctions.utils.creategrid import createPD
from simlopt.optimization.errormodel_new import MCGlobalEstimate, acquisitionfunction, estiamteweightfactors


def adapt_inc(gp, parameterranges, TOL, TOLAcqui, TOLrelchange, epsphys, Xt, yt):
    # Initialization
    counter = 0
    N = gp.getdata[0]
    dim = gp.getdata[2]
    m = yt.shape[1]
    NMC = 100
    totaltime = 0
    totalFEM = 0
    selected_indices = []


    print("---------------------------------- Start adaptive phase")
    print("Number of initial points:          "+str(N))
    print("Desired tolerance:                 "+str(TOL))
    print("\n")

    # Main adaptive loop
    while True:
        counter += 1
        print(f"--- Iteration {counter}")

        # Generate candidate points
        XGLEE = createPD(NMC, dim, "latin", parameterranges)

        dfGLEE = gp.predictderivative(XGLEE, True)
        varGLEE = gp.predictvariance(XGLEE, True)
        normvar = np.linalg.norm(np.sqrt(np.abs(varGLEE)), 2, axis=0) ** 2
        w = estiamteweightfactors(dfGLEE, epsphys)
        # MC global error estimation
        mcglobalerrorbefore = MCGlobalEstimate(w, normvar, NMC,
                                               parameterranges)

        """ ------------------------------Acquisition phase ------------------------------ """
        'Add new candidate points'
        XC = np.array([])
        # Acquisition phase
        XC, index, value = acquisitionfunction(gp, dfGLEE, normvar, w, XGLEE, epsphys,
                                               TOLAcqui)  # Use your acquisition function

        # Find closest point in the training data and add it to the GP model
        if XC.size != 0:
            print(" Number of possible candidate points: {}".format(XC.shape[0]))
            print(" Found canditate point(s):            {}".format(XC[0]))
            print(" Use ith highest value   :            {}".format(index))
            print(" Value at index          :            {}".format(value))
            print("\n")

            closest_point, closest_point_value, selected_indices = find_closest_point(Xt, yt, XC[0], selected_indices)
            print("Point closest to Xc:", str(closest_point))
            print("Yc:", str(closest_point_value))

            epsXc = 1E-4 * np.ones((1, XC.shape[0]))  # eps**2
            gp.adddatapoint(closest_point)
            gp.adddatapointvalue(closest_point_value)
            gp.addaccuracy(epsXc)
            print("Size of data: ", str(gp.getdata[0]))
        else:
            print("Something went wrong, no candidate point was found.")
            print("\n")


        # A posteriori MC global error estimation
        dfGLEE = gp.predictderivative(XGLEE, True)
        varGLEE = gp.predictvariance(XGLEE, True)
        wpost = estiamteweightfactors(dfGLEE, epsphys)
        normvar = np.linalg.norm(np.sqrt(np.abs(varGLEE)), 2, axis=0) ** 2
        mcglobalerrorafter = MCGlobalEstimate(wpost, normvar, NMC, parameterranges)

        # Check convergence
        if mcglobalerrorafter <= TOL:
            print("--- Convergence")
            print(" Desired tolerance is reached, adaptive phase is done.")
            return gp

        # Adjust budget
        relchange = np.abs(mcglobalerrorbefore - mcglobalerrorafter) / mcglobalerrorbefore * 100
        if relchange < TOLrelchange:
            TOLAcqui *= 0.9999
            print("Relative change is below set threshold. Adjusting TOLAcqui.")

        # Check number of points
        if counter >= 1715:
            print("--- Maximum number of points reached")
            return gp


def find_closest_point(Xt, yt, point, selected_indices):
    distances = np.linalg.norm(Xt - point, axis=1)
    while True:
        index = np.argmin(distances)
        if index not in selected_indices:
            selected_indices.append(index)
            return Xt[index].reshape(1,-1), yt[index].reshape(1,-1), selected_indices
        else:
            distances[index] = np.inf
