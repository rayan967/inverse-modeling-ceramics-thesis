import numpy as np

import time
from timeit import default_timer as timer

import scipy
from scipy import optimize

from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.optimize import SR1

import matplotlib.pyplot as plt

from basicfunctions.covariance.cov import *
from basicfunctions.utils.creategrid import *
from basicfunctions.kaskade.kaskadeio import *

from optimization.errormodel import *
from optimization.workmodel import *
from optimization.utilities import *

from gpr.gaussianprocess import *



def adapt(gp, totalbudget, incrementalbudget, parameterranges,
          TOL, TOLFEM, TOLAcqui,TOLrelchange, epsphys,
          execpath, execname, adaptgrad, fun):

    'Problem dimension'
    dim = gp.getdata[2]

    'Counter variables'
    counter = 0
    totaltime = 0
    totalFEM = 0
    nosolutioncounter = 0

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

    ' Initial acquisition phase '
    NMC = 25
    XGLEE = createPD(NMC, dim, "grid", parameterranges)
    dfXC  = gp.predictderivative(XGLEE, True)
    varXC = gp.predictvariance(XGLEE)
    w     = estiamteweightfactors(dfXC, epsphys)
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
        varGLEE = gp.predictvariance(XGLEE,False)
        w       = estiamteweightfactors(dfGLEE, epsphys)
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
        #varGLEE = np.linalg.norm(gp.predictvariance(XGLEE,True),axis=0)
        XC = np.array([])
        while XC.size == 0:
            #print(" Adjusting acquisition tolerance")
            XC,Xdummy = acquisitionfunction(gp,dfGLEE,np.sqrt(np.abs(varGLEE)),w,XGLEE,epsphys,TOLAcqui,XCdummy)
            TOLAcqui*=0.999
            if TOLAcqui < 0.1:
                print("No new candidate points were found. Use current data points.")
                print(" Current tolerance {}".format(TOLAcqui))
                XC = np.array([])
                NC = 0
                break
        print(" Current tolerance {}".format(TOLAcqui))
        print(" Number of possible candidate points: {}".format(XC.shape[0]))
        print(" Found canditate point(s): {}".format(XC[0]))

        TOLAcqui = 1.0
        print("Reset tolerance to {} for next design.".format(TOLAcqui))
        print("\n")

        ' Add found candidate point '
        if XC.size != 0:
            NC = XC.shape[0]
            epsXc = 1E20*np.ones((1, XC.shape[0]))  # eps**2
            meanXc = gp.predictmean(XC,True)
            gp.adddatapoint(XC)
            gp.adddatapointvalue(meanXc)
            gp.addaccuracy(epsXc)

        plotiteration(gp,w,np.sqrt(np.abs(varGLEE)),N,Ngrad,XGLEE,XC,XCdummy,mcglobalerrorbefore,parameterranges, figurepath,counter)
        gp.savedata(execpath+'/saved_data/',str(counter))        
        
        """ ------------------------------ Solve minimization problem ------------------------------ """
        print("--- Solve minimization problem")

        'Turn epsilon^2 into v'
        epsilon = np.squeeze(gp.getaccuracy)
        v = epsilon**(-1)

        ' Current cost by adding initial values is added to the overall budget '
        currentcost= totalcompwork(v, s)
        file.write(str(currentcost) + str(" "))

        ' Keep track of all points '
        Nall =  N + NC

        'Set start values'
        if counter == 0:
            v[0:]= 10
            print(" Used start value for all points: {}".format(10))
            print("\n")
        else:
            'Adapt initial values by taking the max value of the new solution'
            v[N:] = 10

        'Bounds on v'
        lowerbound= v.tolist()
        upperbound= [np.inf]*Nall
        bounds= Bounds(lowerbound, upperbound)
        bounds.lb[N:] = 0.0

        total_n= 0
        nonlinear_constraint= NonlinearConstraint(lambda x: compworkconstrain(x,s), currentcost+incrementalbudget, currentcost+incrementalbudget,
                                                   jac=lambda x: compworkconstrainjac(x,s),
                                                   hess=compworkconstrainhess)
        t0=time.perf_counter()
        #Case 1 and 2
        X = gp.getX
        hyperparameter = gp.gethyperparameter
        #var = gp.predictvariance(X)
        df = gp.predictderivative(gp.getX, True)
        wmin = estiamteweightfactors(df, epsphys)

        if gp.getXgrad is not None:
            K = kernelmatrixsgrad(X, gp.getXgrad, hyperparameter[0,:], gp.getaccuracy*0.0, gp.getgradientaccuracy*0.0)
            Ngrad = gp.getXgrad.shape[0]
            tensor = np.zeros((Nall+Ngrad*dim, Nall+Ngrad*dim, Nall))
            for kk in range(Nall):
                tensor[kk, kk, kk] = 1
        else:
            K = kernelmatrix(X, X, hyperparameter[0,:])
            tensor = np.zeros((Nall, Nall, Nall))
            for kk in range(Nall):
                tensor[kk, kk, kk] = 1

        args = (wmin, X, K, Nall, tensor, parameterranges, adaptgrad)

        #
        sol=scipy.optimize.minimize(targetfunction, v,
                                          args=args,
                                          method='trust-constr',
                                          jac=gradientoftargetfunction,bounds = bounds,
                                          hess=BFGS(),
                                          constraints=[nonlinear_constraint],
                                          options={'verbose': 1, 'maxiter': itermax, 'xtol': xtol, 'gtol': gtol})
        t1=time.perf_counter()
        total_n=t1-t0

        totaltime += total_n
        #nrofdeletedpoints=0

        if sol.success == True:

            print("Solution found within {} iterations".format(sol.nit))
            print("Used time:            {:0.2f} seconds".format(total_n))
            print("Last function value:  {}".format(sol.fun))
            print("\n")

            'Solution for v'
            vsol=sol.x
            currentcost=totalcompwork(vsol, s)
            print("Found solution with:")
            prettyprintvector(vsol, dim, False)

            'Solution for epsilon'
            currentepsilonsol=1/np.sqrt(vsol)
            'Turn eps^2 to eps for comparing'
            epsilon=np.sqrt(epsilon)

            """ ---------- Block for adapting output (y) values ---------- """
            ' Check which point changed in its accuracy. Only if the change is significant a new simulation is done '
            ' since only then the outout value really changed. Otherwise the solution is just set as a new solution.'

            indicesofchangedpoints=np.where(np.abs(np.atleast_2d(epsilon-currentepsilonsol[:Nall])) > TOLFEM)
            if indicesofchangedpoints[1].size == 0:
                print("\n")
                print("No sufficient change between the solutions.")
                print("Solution is set as new optimal design.")
                gp.addaccuracy(currentepsilonsol**2, [0, N+NC])
            else:
                print("\n")
                print("--- Start simulation block")
                print("Sufficient change in the solutions is detected, optain new simulation values")
                print("for point(s): {}".format(indicesofchangedpoints[1]))

                t0FEM=time.perf_counter()
                print("\n")

                for jj in range(indicesofchangedpoints[1].shape[0]):
                    currentFEMindex=indicesofchangedpoints[1][jj]

                    ' Get new values for calculated solution '
                    epsXtnew=currentepsilonsol[currentFEMindex].reshape((1, -1))
                    
                    #params = [1.0,0.95,0.9,0.85]
                    #params = [1E-2,1E-2,1E-4,1E-4,1E-3,1E-2,1E-3,1E-3]
                    params = [1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65]
                    #params = [1.0]
                    ytnew = np.zeros((len(params)))
                    for i,param in enumerate(params):
                        ytnew[i]=fun["function"](np.atleast_2d(gp.getX[currentFEMindex]),param)

                    ' Add new value to GP '
                    gp.addaccuracy(epsXtnew**2, currentFEMindex)
                    gp.adddatapointvalue(ytnew, currentFEMindex)

                t1FEM=time.perf_counter()
                totalFEM=t1FEM-t0FEM
                print("Simulation block done within: {:1.4f} s".format(totalFEM))
                print("\n")
            #N += NC

            """ ------------------------------ MC GLOBAL ERROR ESTIMATION ------------------------------ """
            print("--- A posteriori error estimate")
            dfGLEE = gp.predictderivative(XGLEE, True)
            varGLEE = gp.predictvariance(XGLEE)

            mcglobalerrorafter = MCGlobalEstimate(w,varGLEE,NGLEE,parameterranges)
            file.write( str(mcglobalerrorafter[0]))
            print("Global error estimate after optimization:   {:1.8f}".format(mcglobalerrorafter[0]))
            print("Computational cost after optimization:      {:0.0f}".format(currentcost))
            print("\n")
            vsol=currentepsilonsol**(-2)  #Filtered solution
           
            costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter[0]))
            costerrorlog.write("\n")

            if mcglobalerrorafter[0] < TOL:
                print("--- Convergence")
                print(" Desired tolerance is reached, adaptive phase is done.")
                print(" Final error estimate: {:1.8f}".format(mcglobalerrorafter[0]))
                print(" Total used time: {:0.2f} seconds".format(totaltime+totalFEM))
                print(" Save everything !")
                gp.savedata(execpath+'/saved_data')
                costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter[0]))
                costerrorlog.write("\n")
                file.close()
                costerrorlog.close()
                return gp

            'If the error descreases too slow we add more budget to spend'
            print("--- Adjust budget")
            relchange = np.abs(mcglobalerrorbefore[0]-mcglobalerrorafter[0]) / mcglobalerrorbefore[0]*100
            if relchange < TOLrelchange:
                print("Relative change: "+str(relchange))
                print(" Relative change is below set threshold.")
                print(" Adjusting TOLrelchange !")
                TOLAcqui*=0.9999
            else:
                print("Relative change is sufficient - no necessary budget adjustments.")
                print("\n")

            print("--- New parameters")
            print("Error estimate after optimization and filtering: {:1.5f}".format(mcglobalerrorafter[0]))
            print("Computational cost after optimization:           {:0.0f}".format(currentcost))
            print("New budget for next design:                      {:0.0f}".format(currentcost+incrementalbudget))
            print("Used time:                                       {:0.2f} seconds".format(total_n))
            print("\n")

        else:

            ' Set new start value for next design '
            vsol=sol.x

            ' Set new cummulated cost '
            currentcost=totalcompwork(vsol, s)

# =============================================================================
#             if globalerrorafter < TOL:
#                 print("\n")
#                 print("Desired tolerance is still reached, adaptive phase is done.")
#                 print(" Final error estimate: {:1.6f}".format(globalerrorafter))
#                 print(" Total used time: {:0.4f} seconds".format(totaltime))
#                 gp.addaccuracy(vsol**(-1), [0, None])
#                 print("Save everything...")
#                 gp.savedata(execpath+'/saved_data')
#                 return gp
# =============================================================================

            print("\n")
            print("No solution found.")
            print(" " + sol.message)
            print("Total used time: {:0.4f} seconds".format(totaltime))

            incrementalbudget += 1E7
            N += NC  # All candidates go core
            NC=0  # There are no furhter candidate points

            print("Adjust budget to spend")
            print("  New budget to spend: {:0.4f}".format(incrementalbudget))
            print("\n")

        counter += 1

        print("--- Adjust hyperparameter")
        region = ((0.1, 10),   (0.1, 10))    
        gp.optimizehyperparameter(region, "mean", False)
        print("\n")

        print("--- Adapt solver accuracy and budet")
        if sol.nit == 1:
            print("Solving was done within one iteration.")
            print("Increase accuracy to")
            xtol = xtol*0.5
            gtol = gtol*0.5
            print(" xtol: {}".format(xtol))
            print(" gtol: {}".format(gtol))
        else:
            print(" No necessary accuracy adjustments.")
            print("\n")

        t1design=time.perf_counter()
        print("Time used for complete design iteration: {:0.2f} seconds".format(t1design-t0design))
        print("\n")

        file.write("\n")
        file.close()
        costerrorlog.close()
