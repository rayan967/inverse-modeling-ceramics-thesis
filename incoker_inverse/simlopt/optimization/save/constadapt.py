import numpy as np

import copy
import time
from timeit import default_timer as timer

import scipy
import scipy.optimize as optimize
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.optimize import SR1

import matplotlib.pyplot as plt

from basicfunctions.covariance.cov import *
from basicfunctions.utils.creategrid import *
from basicfunctions.kaskade.kaskadeio import *

from optimization.errormodel_new import *
from optimization.workmodel import *
from optimization.utilities import *

from gpr.gaussianprocess import *


def adapt(gp, totalbudget, incrementalbudget, parameterranges,
          TOL, TOLFEM, TOLFILTER, TOLAcqui,TOLrelchange, epsphys,
          execpath, execname, adaptgrad, fun):

    'Problem dimension'
    dim = gp.getdata[2]

    'Counter variables'
    counter = 0
    totaltime = 0
    totalFEM = 0
    nosolutioncounter = 0
    itercounter = 0

    'Solver options'
    xtol = 1*1E-5
    gtol = 1*1E-5
    itermax = 100000
    s = 1

    """
    If withgradient is True, we check if there is already gradient data.
    otherwise we need to at as much gradientdata as trainig data is already available """
    N = gp.getdata[0]

    cases = {1:"Case 1: Gradient data is not available.",
             2:"Case 1: Gradient data is available."}

    'Check for which cases are set.'
    if gp.getXgrad is None:
        Ngrad = gp.getdata[1] #Is None, when Xgrad is None
        case = 1
    elif gp.getXgrad is not None:
        Ngrad = gp.getdata[1]
        case = 2

    print("---------------------------------- Start adaptive phase")
    print(cases[case])
    print("Number of initial points:          "+str(N))
    print("Total budget:                      "+str(totalbudget))
    print("Desired tolerance:                 "+str(TOL))
    print("Workmodel exponent:                "+str(s))
    print("\n")

    "Open logs"
    logpath = os.path.join(execpath+"/", "logs/")
    logpath_general = os.path.join(execpath+"/")
    figurepath = os.path.join(execpath+"/", "iteration_plots/")

    ' Empty array for possible dummy points - for printing'
    XCdummy= np.empty((0,dim))

    ' Initial acquisition phase '
    NMC = 25
    XGLEE = createPD(NMC, dim, "grid", parameterranges)
    dfXC = gp.predictderivative(XGLEE, True)
    varXC = gp.predictvariance(XGLEE)
    w = estiamteweightfactors(dfXC, epsphys)
    NGLEE = XGLEE.shape[0]
    mcglobalinitial = MCGlobalEstimate(w,varXC,NGLEE,parameterranges)

    'Epsilon^2 at this points'
    epsilon = np.squeeze(gp.getaccuracy)
    currentcost = totalcompwork(epsilon**(-1),s)

    ' Logging '
    try:
        costerrorlog = open(logpath+"costerror.txt","a")
    except IOError:
        print ("Error: File does not appear to exist.")
        return 0

    costerrorlog.write(str(currentcost)+" "+str(mcglobalinitial[0]))
    costerrorlog.write("\n")
    costerrorlog.close()

    NC = 0
    while currentcost < totalbudget:

        N = gp.getX.shape[0]
        ' Logging '
        try:
            costerrorlog = open(logpath+"costerror.txt","a")
            file = open(logpath_general+"log.txt","a")
        except IOError:
          print ("Error: File does not appear to exist.")
          return 0
        if counter == 0:
            file.write("Error estimate before"+" "+"Used budget"+" "+"Error estimate after")
            file.write("\n")

        if counter == 0:
            file.write("Error estimate before"+" "+"Used budget"+" "+"Error estimate after")
            file.write("\n")

        t0design = time.perf_counter()
        print("---------------------------------- Iteration / Design: {}".format(counter))
        print("Current number of points: {} ".format(N))
        print("Current number of candidate points: {} ".format(NC))
        print("\n")

        """ ------------------------------ MC GLOBAL ERROR ESTIMATION ------------------------------ """
        dfGLEE  = gp.predictderivative(XGLEE, True)
        varGLEE = gp.predictvariance(XGLEE)
        w = estiamteweightfactors(dfGLEE, epsphys)
        mcglobalerrorbefore = MCGlobalEstimate(w,varGLEE,NGLEE,parameterranges)
        file.write( str(mcglobalerrorbefore[0]) + str(" "))

        XCdummy= np.empty((0,dim))
        print("Global error estimate before optimization:   {:1.5f}".format(mcglobalerrorbefore[0]))
        print("Computational cost before optimization:      {:0.0f}".format(currentcost))
        print("\n")

        """ ------------------------------Acquisition phase ------------------------------ """
        'Add new candidate points'
        print("--- Acquisition phase")
        NMC = 25
        #XGLEE = createPD(NMC, dim, "grid", parameterranges)

        XC = np.array([])
        while XC.size == 0:
            #print(" Adjusting acquisition tolerance")
            XC,Xdummy = acquisitionfunction(gp,dfGLEE,varGLEE,w,XGLEE,epsphys,TOLAcqui,XCdummy)
            TOLAcqui*=0.9999
            if TOLAcqui < 0.01:
                print("No new candidate points were found. Use current data points.")
                print(" Current tolerance {}".format(TOLAcqui))
                XC = np.array([])
                NC = 0
                break
        print(" Current tolerance {}".format(TOLAcqui))
        print(" Number of possible candidate points {}".format(XC.shape[0]))

        TOLAcqui = 1.0
        print("Reset tolerance to {} for next design.".format(TOLAcqui))
        print("\n")

        ' Add found candidate point '
        accuracy = 1E-6
        if XC.size != 0:
            NC = XC.shape[0]
            epsXc = accuracy*np.ones((1, XC.shape[0]))  # eps**2
            meanXc = gp.predictmean(XC)
            gp.adddatapoint(XC)

            params = [1.0,0.95,0.9]
            ytnew = np.zeros((len(params)))
            for i,param in enumerate(params):
                ytnew[i]=fun["function"](XC,param)
            
            #gp.adddatapointvalue(meanXc)
            gp.adddatapointvalue(np.atleast_2d(ytnew))
            gp.addaccuracy(epsXc)

        plotiteration(gp,w,varGLEE,N,Ngrad,XGLEE,XC,XCdummy,mcglobalerrorbefore,figurepath,counter)

        """ ------------------------------ Solve minimization problem ------------------------------ """
        epsilon = np.squeeze(gp.getaccuracy)
        currentcost = totalcompwork(epsilon**(-1),s)
        
        """ ------------------------------ MC GLOBAL ERROR ESTIMATION ------------------------------ """
        dfGLEE  = gp.predictderivative(XGLEE, True)
        varGLEE = gp.predictvariance(XGLEE)
        w = estiamteweightfactors(dfGLEE, epsphys)
        mcglobalerrorafter = MCGlobalEstimate(w,varGLEE,NGLEE,parameterranges)
                
        
        if mcglobalerrorafter[0] <= TOL:
            print("\n")
            print("Desired tolerance is still reached, adaptive phase is done.")
            print(" Final error estimate: {:1.6f}".format( mcglobalerrorafter[0]))
            print(" Total used time: {:0.4f} seconds".format(totaltime))
            #gp.addaccuracy(vsol**(-1), [0, None])
            print("Save everything...")
            gp.savedata(execpath+'/saved_data')
            return gp
        
        costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter[0]))
        costerrorlog.write("\n")
        counter += 1

        Nmax = 20
        if N < Nmax:
            print("--- A priori hyperparameter adjustment")
            region = ((0.01, 3),(0.01, 3))
            gp.optimizehyperparameter(region, "mean", False)
        else:
            print("--- A priori hyperparameter adjustment")
            print("Number of points is higher then "+str(Nmax))
            print("No optimization is performed")
        print("\n")

     
        t1design=time.perf_counter()
        print("Time used for complete design iteration: {:0.2f} seconds".format(t1design-t0design))
        print("\n")

        file.write("\n")
        file.close()
        costerrorlog.close()
