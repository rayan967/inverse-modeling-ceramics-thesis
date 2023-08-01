import time

import numpy as np
import scipy

from scipy.optimize import minimize

from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint

import matplotlib.pyplot as plt

from basicfunctions.covariance.cov import *
from basicfunctions.utils.creategrid import *
from basicfunctions.kaskade.kaskadeio import *

from scipy.optimize import SR1
import matplotlib.tri as tri

from optimization.errormodel_new import *
from optimization.workmodel import *
from optimization.utilities import *

from gpr.gaussianprocess import *

from IOlogging.iotofile import *

def adapt(gp, totalbudget, incrementalbudget, parameterranges,
          TOL, TOLFEM, TOLAcqui,TOLrelchange, epsphys,
          runpath, execname, adaptgrad, fun):

    'Problem dimension'
    dim= gp.getdata[2]

    'Counter variables'
    counter= 0
    totaltime= 0
    totalFEM= 0

    'Solver options'
    xtol = 1*1E-4
    gtol = 1*1E-4
    itermax = 100000
    s = 1

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
        gp.addgradientdatapointvalue(dy)
        gp.addgradaccuracy(epsXgrad)
    print("\n")

    print("---------------------------------- Start adaptive phase")
    print(cases[case])
    print("Number of initial points:          "+str(N))
    print("Total budget:                      "+str(totalbudget))
    print("Desired tolerance:                 "+str(TOL))
    print("Workmodel exponent:                "+str(s))
    print("\n")

    "Open logs"
    logpath = os.path.join(runpath+"/", "logs/")
    logpath_general = os.path.join(runpath+"/")
    figurepath = os.path.join(runpath+"/", "iteration_plots/")

    ' Initial acquisition phase '
    NMC = 30
    XGLEE = createPD(NMC, dim, "grid", parameterranges)
    NGLEE = XGLEE.shape[0]

    dfXC  = gp.predictderivative(XGLEE, True)
    varXC = gp.predictvariance(XGLEE,True)
    normvar = np.linalg.norm(np.sqrt(varXC),2,axis=0)**2
    w       = estiamteweightfactors(dfXC, epsphys)

    mcglobalinitial = MCGlobalEstimate(w,normvar,NGLEE,parameterranges)

    'Epsilon^2 at this points'
    epsilon = np.squeeze(gp.getaccuracy)
    epsilongrad = np.squeeze(gp.getgradientaccuracy)
   
    ' Cost of current accuracies '
    ' Since we are using gradient data and assume the gradient cost as much as the FEM simulation and get it for "free" since'
    ' we already did the simulation in the first place, we dont have to adjust the costs'
    currentcost = totalcompwork(epsilon**(-1),s)
    #currentcost += totalcompwork(epsilongrad**(-1),s)/dim #Devide by dim for get cost of grad
    currentcost += totalcompwork(epsilongrad**(-1),s)

    ' Logging '
    try:
        costerrorlog = open(logpath+"costerror.txt","a")
    except IOError:
        print ("Error: File does not appear to exist.")
        return 0

    costerrorlog.write(str(currentcost)+" "+str(mcglobalinitial))
    costerrorlog.write("\n")
    costerrorlog.close()

    NC = 0
    while currentcost < totalbudget:

        N = gp.getX.shape[0]
        Ngrad = gp.getXgrad.shape[0]

        ' Logging '
        logger = IOToLog(os.path.join(runpath+"/","iteration_log"),"iteration_"+str(counter))

        try:
            costerrorlog = open(logpath+"costerror.txt","a")
            file = open(logpath_general+"log.txt","a")
            vlog = open(logpath_general+"vlog.txt","a")
        except IOError:
            print ("Error: File does not appear to exist.")
            return 0
        if counter == 0:
            file.write("Error estimate before"+" "+"Used budget"+" "+"Error estimate after")
            file.write("\n")

        t0design = time.perf_counter()
        print("---------------------------------- Iteration / Design: {}".format(counter))
        print("Current number of points: {} ".format(N))
        print("Current number of candidate points: {} ".format(NC))
        print("\n")

        logger.addToFile("---------------------------------- Iteration / Design: {}".format(counter))
        logger.addToFile("Current number of points: {} ".format(N))
        logger.addToFile("Current number of candidate points: {} ".format(NC))
        logger.addToFile("\n")

        'Log current GP hyperparameter'
        logger.addToFile("--- Used hyperparameters")
        for i in range(gp.m):
            hp = gp.gethyperparameter[i,:]
            hpstring = np.array2string(hp, precision=2, separator=',',suppress_small=True)
            logger.addToFile("Hyperparameter experiment "+str(i) +": " +hpstring)
        logger.addToFile("\n")

        """ ------------------------------ MC GLOBAL ERROR ESTIMATION ------------------------------ """
        print("--- A posteriori error estimate")
        logger.addToFile("--- A posteriori error estimate")

        t0apriori=time.perf_counter()
        dfGLEE  = gp.predictderivative(XGLEE, True)
        if np.any(np.isnan(dfGLEE)):
            print("STOP")
        varGLEE = gp.predictvariance(XGLEE,True)

        normvar = np.linalg.norm(np.sqrt(np.abs(varGLEE)),2,axis=0)**2
        w       = estiamteweightfactors(dfGLEE, epsphys)
        mcglobalerrorbefore = MCGlobalEstimate(w,normvar,NGLEE,parameterranges)
        file.write( str(mcglobalerrorbefore) + str(" "))

        epsprior = gp.getaccuracy
        epspriorgrad = gp.getgradientaccuracy
        vprior = 1/epsprior
        vpriorgrad = 1/epspriorgrad
        
        currentcostprior=totalcompwork(vprior, s)
        currentcostprior+=totalcompwork(vpriorgrad, s)

        print("Global error estimate before optimization:   {:1.5f}".format(mcglobalerrorbefore))
        print("Computational cost before optimization:      {:0.0f}".format(currentcostprior))
        print("\n")

        logger.addToFile("Global error estimate before optimization:   {:1.5f}".format(mcglobalerrorbefore))
        logger.addToFile("Computational cost before optimization:      {:0.0f}".format(currentcostprior))
        logger.addToFile("\n")

        t1apriori=time.perf_counter()
        tapriori = t1apriori-t0apriori

        """ ------------------------------Acquisition phase ------------------------------ """
        print("--- Acquisition phase")

        t0acqui = time.perf_counter()
        logger.addToFile("--- Acquisition phase")
        XC = np.array([])
        normvar_TEST = np.linalg.norm(np.sqrt(np.abs(varGLEE)),2,axis=0)
        while XC.size == 0:
            XC = acquisitionfunction(gp,dfGLEE,normvar_TEST,w,XGLEE,epsphys,TOLAcqui)
            TOLAcqui*=0.999
            if TOLAcqui < 0.1:
                print("No new candidate points were found. Use current data points.")
                print(" Current tolerance {}".format(TOLAcqui))

                logger.addToFile("No new candidate points were found. Use current data points.")
                logger.addToFile(" Current tolerance {}".format(TOLAcqui))

                XC = np.array([])
                NC = 0
                break
        print(" Current tolerance                    {}".format(TOLAcqui))
        print(" Number of possible candidate points: {}".format(XC.shape[0]))
        print(" Found canditate point(s):            {}".format(XC[0]))

        logger.addToFile(" Current tolerance                    {}".format(TOLAcqui))
        logger.addToFile(" Number of possible candidate points: {}".format(XC.shape[0]))
        logger.addToFile(" Found canditate point(s):            {}".format(XC[0]))

        TOLAcqui = 1.0
        print("Reset tolerance to {} for next design.".format(TOLAcqui))
        print("\n")

        logger.addToFile("Reset tolerance to {} for next design.".format(TOLAcqui))
        logger.addToFile("\n")

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
            gp.addgradientdatapointvalue(dyXC)
            gp.addgradaccuracy(epsXgrad)

        plotiteration(gp,w,normvar_TEST,N,Ngrad,XGLEE,XC,mcglobalerrorbefore,parameterranges, figurepath,counter)
        gp.savedata(runpath+'/saved_data/',str(counter))

        t1acqui = time.perf_counter()
        tacqui = t1acqui-t0acqui

        """ ------------------------------ Solve minimization problem ------------------------------ """
        print("--- Solve minimization problem")

        'Turn epsilon^2 into v'
        epsilon = np.squeeze(gp.getaccuracy)
        epsilongrad = np.squeeze(gp.getgradientaccuracy)

        v = epsilon**(-1)
        vgrad = epsilongrad**(-1)

        'Log current v'
        #vlog.write(v.reshape(1, v.shape[0]))
        vlog.write(" ".join(map(str,v)))
        vlog.write("\n")
        vlog.close()

        ' Keep track of all points '
        Nall =  N + NC
        Nallgrad = Ngrad + NC

        'Set start values'
        if counter == 0:
            #print("Used start value: {}".format( (incrementalbudget/Nall)**(1/s)))
            #v[0:]= 10
            #print(" Used start value for all points: {}".format(10))
            #vgrad[0:] = 10
            #print(" Used start value for all gradient points: {}".format(10))
            #print("\n")
            print(" Use initial values as start values")
            print("\n")
            logger.addToFile(" Used start value for all points: {}".format(10))
            logger.addToFile("\n")
        else:
            v[N:] = 10.0
            vgrad[Ngrad*dim:] = 10.0

        'Bounds on v and vgrad'
        lowerbound = v.tolist()
        upperbound = [np.inf]*Nall
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
        currentcost = totalcompwork(v,s)
        file.write(str(currentcost) + str(" ") )
        
        'Create nonlinear constraints'
        total_n= 0
        nonlinear_constraint= NonlinearConstraint(lambda x: compworkconstrain(x,s), currentcost+incrementalbudget, currentcost+incrementalbudget,
                                                  jac=lambda x: compworkconstrainjac(x,s),
                                                  hess=compworkconstrainhess)
        t0minimization=time.perf_counter()

        X,Xgrad = gp.getX,gp.getXgrad
        hyperparameter = gp.gethyperparameter
        df = gp.predictderivative(gp.getX, True)
        wmin = estiamteweightfactors(df, epsphys)

        m = gp.m
        K = np.zeros((Nall+Nallgrad*dim, Nall+Nallgrad*dim,m))
        Ngrad = gp.getXgrad.shape[0]
        for i in range(m):
            K[:,:,i] = kernelmatrixsgrad(X, Xgrad, hyperparameter[i,:],
                                         gp.getaccuracy*0.0, gp.getgradientaccuracy*0.0)

        tensor = np.zeros((Nall+Nallgrad*dim, Nall+Nallgrad*dim, Nall+Nallgrad*dim))
        tensor[np.diag_indices(Nall+Nallgrad*dim,ndim=3)] = np.ones((Nall+Nallgrad*dim))

        # X is deprecated
        args = (wmin, X, K, Nall,tensor, parameterranges, adaptgrad)

        sol=scipy.optimize.minimize(targetfunction, v,
                                          args=args,
                                          method='trust-constr',
                                          jac=gradientoftargetfunction,
                                          bounds = bounds,hess=BFGS(),
                                          constraints=[nonlinear_constraint],
                                          options={'verbose': 1, 'maxiter': itermax, 'xtol': xtol, 'gtol': gtol})
        t1minimization = time.perf_counter()
        total_n=t1minimization-t0minimization
        totaltime = total_n

        if sol.success == True:

            print("Solution found within {} iterations".format(sol.nit))
            print("Used time:            {:0.2f} seconds".format(total_n))
            print("Last function value:  {}".format(sol.fun))
            print("\n")

            logger.addToFile("Solution found within {} iterations".format(sol.nit))
            logger.addToFile("Used time:            {:0.2f} seconds".format(total_n))
            logger.addToFile("Last function value:  {}".format(sol.fun))
            logger.addToFile("\n")


            'Solution for v'
            vsol=sol.x
            #currentcost = totalcompwork(vsol, s)
            currentcost =  totalcompwork(vsol[:Nall],s)
            currentcost += totalcompwork(vsol[Nall:],s) #Devide by dim for get cost of grad
            
            print("Point solution: ")
            prettyprintvector(vsol[:Nall], dim, False)
            print("\n")

            print("Gradient solution: ")
            prettyprintvector(vsol[Nall:], dim, True)

            logger.addToFile("Found solution with:")
            logger.addToFile(str(vsol[:Nall]))
            logger.addToFile("\n")

            logger.addToFile("Found gradient solution with:")
            logger.addToFile(str(vsol[Nall:]))
            logger.addToFile("\n")

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

                logger.addToFile("\n")
                logger.addToFile("--- Start simulation block")
                logger.addToFile("Sufficient change in the solutions is detected, optain new simulation values")
                logger.addToFile("for point(s): {}".format(indicesofchangedpoints[1]))

                t0FEM=time.perf_counter()
                print("\n")

                for jj in range(indicesofchangedpoints[1].shape[0]):
                    currentFEMindex=indicesofchangedpoints[1][jj]

                    ' Get new values for calculated solution '
                    epsXtnew=currentepsilonsol[currentFEMindex].reshape((1, -1))

                    #params = [1.0,0.95,0.9,0.85,0.8]
                    params =  [1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65]
                    #params =  [1.0]
                    ytnew = np.zeros((len(params)))
                    for i,param in enumerate(params):
                        ytnew[i]=fun["function"](np.atleast_2d(gp.getX[currentFEMindex]),param)

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
                        #params =  [1.0,0.95,0.9,0.85,0.8]
                        params =  [1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65]
                        #params =  [1.0]
                        ygradnew = np.zeros((dim,len(params)))
                        for i,param in enumerate(params):
                            ygradnew[:,i] = np.squeeze(fun["gradient"](np.atleast_2d(gp.getXgrad[pointindex,:]),param).reshape((1, -1)))

                        'Add new value to GP'
                        gp.addgradaccuracy(epsXgradnew**2,currentgradindex)

                        gp.addgradientdatapointvalue(ygradnew[componentindex],currentgradindex)

                    t1FEM=time.perf_counter()
                    tFEM=t1FEM-t0FEM
                    print("Gradient simulation block done within: {:1.4f} s".format(totalFEM))

                    logger.addToFile("Simulation block done within: {:1.4f} s".format(totalFEM))
                    logger.addToFile("\n")

            """ ------------------------------ MC GLOBAL ERROR ESTIMATION ------------------------------ """
            print("--- A posteriori error estimate")
            logger.addToFile("--- A posteriori error estimate")

            t0post = time.perf_counter()

            dfGLEE = gp.predictderivative(XGLEE, True)
            varGLEE = gp.predictvariance(XGLEE,True)
            wpost = estiamteweightfactors(dfGLEE, epsphys)
            normvar = np.linalg.norm(np.sqrt(np.abs(varGLEE)),2,axis=0)**2

            mcglobalerrorafter = MCGlobalEstimate(wpost,normvar,NGLEE,parameterranges)
            file.write( str(mcglobalerrorafter))
            print("Global error estimate after optimization:   {:1.8f}".format(mcglobalerrorafter))
            print("Computational cost after optimization:      {:0.0f}".format(currentcost))
            print("\n")

            logger.addToFile("Global error estimate after optimization:   {:1.8f}".format(mcglobalerrorafter))
            logger.addToFile("Computational cost after optimization:      {:0.0f}".format(currentcost))
            logger.addToFile("\n")
            #vsol=currentepsilonsol**(-2)  #Filtered solution

            costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter))
            costerrorlog.write("\n")

            t1post = time.perf_counter()
            tpost = t1post - t0post

            if mcglobalerrorafter <= TOL:
                print("--- Convergence")
                print(" Desired tolerance is reached, adaptive phase is done.")
                print(" Final error estimate: {:1.8f}".format(mcglobalerrorafter))
                print(" Total used time: {:0.2f} seconds".format(totaltime+totalFEM))
                print(" Save everything !")

                logger.addToFile("--- Convergence")
                logger.addToFile(" Desired tolerance is reached, adaptive phase is done.")
                logger.addToFile(" Final error estimate: {:1.8f}".format(mcglobalerrorafter))
                logger.addToFile(" Total used time: {:0.2f} seconds".format(totaltime+totalFEM))
                logger.addToFile(" Save everything !")

                gp.savedata(runpath+'/saved_data')

                costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter))
                costerrorlog.write("\n")

                file.close()
                costerrorlog.close()
                logger.closeOutputLog()

                return gp

            'If the error descreases too slow we add more budget to spend'
            print("--- Adjust budget")
            relchange = np.abs(mcglobalerrorbefore-mcglobalerrorafter) / mcglobalerrorbefore*100
            if relchange < TOLrelchange:
                print("Relative change: "+str(relchange))
                print(" Relative change is below set threshold.")
                print(" Adjusting TOLrelchange !")
                logger.addToFile("Relative change: "+str(relchange))
                logger.addToFile(" Relative change is below set threshold.")
                logger.addToFile(" Adjusting TOLrelchange !")
                TOLAcqui*=0.9999
            else:
                print("Relative change is sufficient - no necessary budget adjustments.")
                print("\n")
                logger.addToFile("Relative change is sufficient - no necessary budget adjustments.")
                logger.addToFile("\n")

            print("--- New parameters")
            print("Error estimate after optimization and filtering: {:1.5f}".format(mcglobalerrorafter))
            print("Computational cost after optimization:           {:0.0f}".format(currentcost))
            print("New budget for next design:                      {:0.0f}".format(currentcost+incrementalbudget))
            print("Used time:                                       {:0.2f} seconds".format(total_n))
            print("\n")

            logger.addToFile("--- New parameters")
            logger.addToFile("Error estimate after optimization and filtering: {:1.5f}".format(mcglobalerrorafter))
            logger.addToFile("Computational cost after optimization:           {:0.0f}".format(currentcost))
            logger.addToFile("New budget for next design:                      {:0.0f}".format(currentcost+incrementalbudget))
            logger.addToFile("Used time:                                       {:0.2f} seconds".format(total_n))
            logger.addToFile("\n")

        else:

            ' Set new start value for next design '
            vsol=sol.x

            ' Set new cummulated cost '
            currentcost=totalcompwork(vsol[:Nall], s)

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
        #region = ((1, 20), (1, 10), (1, 10))
        #region = ((0.01, 3),   (0.01, 3))
        region = ((0.001, 3), (0.001, 3))

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

        logger.addToFile("Times used for a priori error estimate       : {:0.2f} seconds".format(tapriori))
        logger.addToFile("Times used for acquisition phase             : {:0.2f} seconds".format(tacqui))
        logger.addToFile("Times used for solving minimization problem  : {:0.2f} seconds".format(totaltime))
        logger.addToFile("Times used for solving fem simulation        : {:0.2f} seconds".format(tFEM))
        logger.addToFile("Times used for a posteriori error estimate   : {:0.2f} seconds".format(tpost))
        logger.addToFile("Time used for complete design iteration      : {:0.2f} seconds".format(t1design-t0design))
        logger.addToFile("\n")
        logger.closeOutputLog()
