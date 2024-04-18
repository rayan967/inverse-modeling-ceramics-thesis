import numpy as np

import copy
import time
from timeit import default_timer as timer

import scipy
import scipy.optimize as optimize
from scipy.optimize import minimize

from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint

import matplotlib.pyplot as plt

from basicfunctions.covariance.cov import *
from basicfunctions.utils.creategrid import *
from basicfunctions.kaskade.kaskadeio import *

from scipy.optimize import SR1
import matplotlib.tri as tri

from optimization.errormodel import *
from optimization.workmodel import *
from optimization.utilities import *

from gpr.gaussianprocess import *


def adapt(gp, totalbudget, incrementalbudget, parameterranges,
          TOL, TOLFEM, TOLFILTER, TOLAcqui,TOLrelchange, epsphys,
          execpath, execname, adaptgrad, fun):

    'Problem dimension'
    dim= gp.getdata[2]

    'Counter variables'
    counter= 0
    totaltime= 0
    totalFEM= 0
    nosolutioncounter= 0
    itercounter= 0

    'Solver options'
    xtol = 1*1E-4
    gtol = 1*1E-4
    itermax = 100000
    s = 2

    """ If withgradient is True, we check if there is already gradient data.
    otherwise we need to at as much gradientdata as trainig data is already available """
    N = gp.getdata[0]
      
    cases = {1:"Case 3: Gradients should be adapted, and are available.",
             2:"Case 4: Gradients should be adapted, but none are available."}

    'Check for which cases are set.'
    if gp.getXgrad is not None:
        case = 1     
    else:
        'Extend gp with estiamted gradient data, if none were provided'
        case = 2
        Xt = gp.getX
        nX = Xt.shape[0]
        epsXgrad = 1E20*np.ones((1,nX*dim))
        dy = gp.predictderivative(Xt)
        
        gp.addgradientdatapoint(Xt)
        gp.adddgradientdatapointvalue(dy)
        gp.addgradaccuracy(epsXgrad)

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
    w = estiamteweightfactors(varXC, XGLEE, dfXC, epsphys)
    NGLEE = XGLEE.shape[0]
    mcglobalinitial = MCGlobalEstimate(w,varXC,NGLEE,parameterranges)

    'Epsilon^2 at this points'
    epsilon = np.squeeze(gp.getaccuracy)
    epsilongrad = np.squeeze(gp.getgradientaccuracy)
    
    currentcost = totalcompwork(epsilon**(-1),s)
    currentcost += totalcompwork(epsilongrad**(-1),s)
        
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
        Ngrad = gp.getXgrad.shape[0]
        
        try:
            costerrorlog = open(logpath+"costerror.txt","a")
            file = open(logpath_general+"log.txt","a")
            solutionlog = open(logpath+"solution.txt","a")
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
        dfGLEE = gp.predictderivative(XGLEE, True)
        varGLEE = gp.predictvariance(XGLEE)
        w = estiamteweightfactors(dfGLEE, epsphys)
        mcglobalerrorbefore = MCGlobalEstimate(w,varGLEE,NGLEE,parameterranges)
        file.write( str(mcglobalerrorbefore[0]) + str(" "))

        XCdummy= np.empty((0,dim))
        print("Global error estimate before optimization:   {:1.8f}".format(mcglobalerrorbefore[0]))
        print("Computational cost before optimization:      {:0.0f}".format(currentcost))
        print("\n")
        
        """ ------------------------------Acquisition phase ------------------------------ """
        'Add new candidate points'
        print("--- Acquisition phase")
        NMC = 25
        
        XC = np.array([])
        while XC.size == 0:
            #print(" Adjusting acquisition tolerance")
            XC,Xdummy = acquisitionfunction(gp,dfGLEE,varGLEE,w,XGLEE,epsphys,TOLAcqui,XCdummy)
            TOLAcqui*=0.99
            if TOLAcqui < 0.001:
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
        if XC.size != 0:
            NC = XC.shape[0]
            epsXc = 1E20*np.ones((1, XC.shape[0]))  # eps**2
            meanXc = gp.predictmean(XC,True)
            gp.adddatapoint(XC)
            gp.adddatapointvalue(meanXc)
            gp.addaccuracy(epsXc)

            epsXgrad = 1E20*np.ones((1,XC.shape[0]*dim))
            dyXC = gp.predictderivative(XC)

            gp.addgradientdatapoint(XC)
            gp.adddgradientdatapointvalue(dyXC)
            gp.addgradaccuracy(epsXgrad)

        plotiteration(gp,w,varGLEE,N,Ngrad,XGLEE,XC,XCdummy,mcglobalerrorbefore,figurepath,counter)

        """ ------------------------------ Solve minimization problem ------------------------------ """

        print("--- Solve minimization problem")
        
        'Turn epsilon^2 into v'
        epsilon = np.squeeze(gp.getaccuracy)
        epsilongrad = np.squeeze(gp.getgradientaccuracy)
        
        v = epsilon**(-1)
        vgrad = epsilongrad**(-1)
        
        ' Keep track of all points '
        Nall =  N + NC
        Nallgrad = Ngrad + NC 

        'Set start values'
        if counter == 0:
            #print("Used start value: {}".format( (incrementalbudget/Nall)**(1/s)))
            v[0:]= 10
            print(" Used start value for all points: {}".format(10))
            vgrad[0:] = 10
            print(" Used start value for all gradient points: {}".format(10))
            print("\n")
        else:
            v[N:] = 0.0
            vgrad[Ngrad*dim:] = 0.0

        'Bounds on v and vgrad'
        lowerbound= v.tolist()
        upperbound= [np.inf]*Nall
        bounds= Bounds(lowerbound, upperbound)
        
        lowerboundgrad = vgrad.tolist()
        upperboundgrad = [np.inf]*Nallgrad*dim
        boundsgrad = Bounds(lowerboundgrad,upperboundgrad)

        'Connect bounds'
        lower = np.concatenate((bounds.lb,boundsgrad.lb))
        upper = np.concatenate((bounds.ub,boundsgrad.ub))

        'Build final bound object'
        bounds = Bounds(lower, upper)

        'Combine vs'
        v = np.concatenate((v,vgrad))
        
        'Current cost '
        currentcost= totalcompwork(v, s)
        file.write(str(currentcost) + str(" ") )

        'Create nonlinear constraints'
        total_n= 0
        nonlinear_constraint= NonlinearConstraint(lambda x: compworkconstrain(x,s), currentcost+incrementalbudget, currentcost+incrementalbudget,
                                                  jac=lambda x: compworkconstrainjac(x,s),
                                                  hess=compworkconstrainhess)
        t0=time.perf_counter()

        X,Xgrad = gp.getX,gp.getXgrad
        hyperparameter = gp.gethyperparameter
        df = gp.predictderivative(gp.getX, True)
        var = gp.predictvariance(X)
        wmin = estiamteweightfactors(var, X, df, epsphys)

        K = kernelmatrixsgrad(X, Xgrad, hyperparameter, gp.getaccuracy*0.0, gp.getgradientaccuracy*0.0)

        tensor = np.zeros((Nall+Nallgrad*dim, Nall+Nallgrad*dim, Nall+Nallgrad*dim))
        for kk in range(Nall+Nallgrad*dim):
            tensor[kk, kk, kk] = 1

        args = (wmin, X, K, Nall,tensor, parameterranges, adaptgrad)
          
        sol=scipy.optimize.minimize(targetfunction, v,
                                          args=args,
                                          method='trust-constr',
                                          jac=gradientoftargetfunction,
                                          bounds = bounds,hess=BFGS(),
                                          constraints=[nonlinear_constraint],
                                          options={'verbose': 1, 'maxiter': itermax, 'xtol': xtol, 'gtol': gtol})
        t1=time.perf_counter()
        total_n=t1-t0

        totaltime += total_n
        nrofdeletedpoints=0

        if sol.success == True:

            print("Solution found within {} iterations".format(sol.nit))
            print("Used time:            {:0.2f} seconds".format(total_n))
            print("Last function value:  {}".format(sol.fun))
            print("\n")

            'Solution for v'
            vsol=sol.x
            currentcost=totalcompwork(vsol, s)
            print("Point solution: ")
            #print(vsol[:Nall])
            prettyprintvector(vsol[:Nall], dim, False)
            solutionlog.write("Point solution: ")
            solutionlog.write(str(vsol[:Nall]))
            solutionlog.write("\n")
            
            print("Gradient solution: ")
            #print(vsol[Nall:])
            solutionlog.write("Gradient solution: ")
            prettyprintvector(vsol[Nall:], dim, True)
            solutionlog.write("\n")
            solutionlog.write("\n")
            solutionlog.close()
            
            'Solution for epsilon'
            currentepsilonsol=vsol**(-1/2)
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
                gp.addaccuracy(currentepsilonsol[:Nall]**2,[0,None])
                gp.addgradaccuracy(currentepsilonsol[Nall:]**2,[0,None])
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
                    ytnew=fun["function"](np.atleast_2d(gp.getX[currentFEMindex])).reshape((1, -1))

                    ' Add new value to GP '
                    gp.addaccuracy(epsXtnew**2, currentFEMindex)
                    gp.adddatapointvalue(ytnew, currentFEMindex)

                t1FEM=time.perf_counter()
                totalFEM=t1FEM-t0FEM
                print("Simulation block done within: {:1.4f} s".format(totalFEM))
                
                indicesofchangedgrad=np.where(np.abs(np.atleast_2d(epsilongrad-currentepsilonsol[Nall:])) > TOLFEM)
                
                gradsol = currentepsilonsol[Nall:]
                if indicesofchangedgrad[1].size == 0:
                    print("\n")
                    print("No sufficient change between the solutions.")
                    print("Solution is set as new optimal design.")
                    gp.addaccuracy(gradsol**2, [0, N+NC])
                else:
                    
                    print("\n")
                    print("--- Start gradient simulation block")
                    print("Sufficient change in the solutions is detected, optain new simulation values")
                    print("for components(s): {}".format(indicesofchangedgrad[1]))

                    t0FEM=time.perf_counter()
                    
                    print("\n")
                    for jj in range(indicesofchangedgrad[1].shape[0]):
                        
                        'Point and componentindex'
                        pointindex = int(np.floor(indicesofchangedgrad[1]/dim)[jj])
                        componentindex = (indicesofchangedgrad[1]%dim)[jj]
                        
                        'List of components that change'
                        currentgradindex=indicesofchangedgrad[1][jj] 
                        
                        'Get gradients for calculated solution'
                        epsXgradnew=gradsol[currentgradindex].reshape((1, -1))
                        ygradnew = np.squeeze(fun["gradient"](np.atleast_2d(gp.getXgrad[pointindex,:])).reshape((1, -1)))
                        
                        'Add new value to GP'
                        gp.addgradaccuracy(epsXgradnew**2,currentgradindex)
                        gp.adddgradientdatapointvalue(ygradnew[componentindex],currentgradindex)
                        
                    t1FEM=time.perf_counter()
                    totalFEM=t1FEM-t0FEM
                    print("Gradient simulation block done within: {:1.4f} s".format(totalFEM))

            """ ------------------------------ MC GLOBAL ERROR ESTIMATION ------------------------------ """
            print("--- A posteriori error estimate")
            print("Estimate derivatives for a posteriori estimation")
            dfGLEE = gp.predictderivative(XGLEE, True)
            print("...done")
            varGLEE = gp.predictvariance(XGLEE)

            mcglobalerrorafter = MCGlobalEstimate(w,varGLEE,NGLEE,parameterranges)
            file.write(str(mcglobalerrorafter[0]))
            print("Global error estimate after optimization:   {:1.8f}".format(mcglobalerrorafter[0]))
            print("Computational cost after optimization:      {:0.0f}".format(currentcost))
            print("\n")

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
                print(" Adjusting TOLrelchange !".format())
                TOLAcqui*=0.99
            else:
                print("Relative change is sufficient - no necessary budget adjustments.")
                print("\n")

            print("--- New parameters")
            print("Error estimate after optimization and filtering: {:1.8f}".format(mcglobalerrorafter[0]))
            print("Computational cost after optimization:           {:0.0f}".format(currentcost))
            print("New budget for next design:                      {:0.0f}".format(currentcost+incrementalbudget))
            print("Used time:                                       {:0.2f} seconds".format(total_n))
            print("\n")

        else:

            ' Set new start value for next design '
            vsol=sol.x

            ' Set new cummulated cost '
            currentcost=totalcompwork(vsol, s)

            if globalerrorafter < TOL:
                print("\n")
                print("Desired tolerance is still reached, adaptive phase is done.")
                print(" Final error estimate: {:1.6f}".format(globalerrorafter))
                print(" Total used time: {:0.4f} seconds".format(totaltime))
                gp.addaccuracy(vsol**(-1), [0, None])
                print("Save everything...")
                gp.savedata(execpath+'/saved_data')
                return gp

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
               
        print("--- Adjust hyperparameter")
        #region = ((1, 20), (1, 10), (1, 10))
        region = ( (1, 10),   (1, 10))
        gp.optimizehyperparameter(region, "mean", False)
        print("\n")
        
        t1design=time.perf_counter()
        print("Time used for complete design iteration: {:0.2f} seconds".format(t1design-t0design))
        print("\n")

        file.write("\n")
        file.close()
        costerrorlog.close()
