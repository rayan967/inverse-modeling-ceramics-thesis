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

from optimization.errormodel_new import *
from optimization.workmodel import *
from optimization.utilities import *

from gpr.gaussianprocess import *
from IOlogging.iotofile import *

def updateDataPointValues(gp,epsilonsolutionbefore,epsilonsolcurrent,NofXt,TOLFEM,adaptgrad,fun):

    ' Create subvectors '
    currentpointsolutions = epsilonsolcurrent[:NofXt]
    if adaptgrad:
        currentgradsolutions = epsilonsolcurrent[NofXt:]

    epsilon = epsilonsolutionbefore

    dim = gp.getdata[2]

    indicesofchangedpoints=np.where(np.abs(np.atleast_2d(epsilon[:NofXt]-currentpointsolutions)) > TOLFEM)
    if indicesofchangedpoints[1].size == 0:
        print("\n")
        print("No sufficient change between the solutions.")
        print("Solution is set as new optimal design.")
        gp.addaccuracy(currentpointsolutions,[0,None])
        if adaptgrad:
            gp.addgradaccuracy(currentgradsolutions,[0,None])

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
            epsXtnew=currentpointsolutions[currentFEMindex].reshape((1, -1))
            params = [1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65]
            ytnew = np.zeros((len(params)))
            for i,param in enumerate(params):
                ytnew[i]=fun["function"](np.atleast_2d(gp.getX[currentFEMindex]),param)

            ' Add new value to GP '
            gp.addaccuracy(epsXtnew, currentFEMindex)
            gp.adddatapointvalue(ytnew, currentFEMindex)
            
        t1FEM=time.perf_counter()
        totalFEM=t1FEM-t0FEM
        print("Simulation block done within: {:1.4f} s".format(totalFEM))
        print("\n")
        
        if adaptgrad:
            indicesofchangedgrad=np.where(np.abs(np.atleast_2d(epsilonsolutionbefore[NofXt:]-currentgradsolutions)) > TOLFEM)

            if indicesofchangedgrad[1].size == 0:

                print("No sufficient change between the solutions.")
                print("Solution is set as new optimal design.")
                gp.addgradaccuracy(currentgradsolutions,[0,None])
            else:

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
                    epsXgradnew=currentgradsolutions[currentgradindex].reshape((1, -1))
                    #params =  [1.0,0.95,0.9,0.85,0.8]
                    params = [1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65]
                    #params =  [1.0]
                    ygradnew = np.zeros((dim,len(params)))
                    for i,param in enumerate(params):
                        ygradnew[:,i] = np.squeeze(fun["gradient"](np.atleast_2d(gp.getXgrad[pointindex,:]),param).reshape((1, -1)))
    
                    'Add new value to GP'
                    gp.addgradaccuracy(epsXgradnew,currentgradindex)
                    gp.addgradientdatapointvalue(ygradnew[componentindex],currentgradindex)


                t1FEM=time.perf_counter()
                totalFEM=t1FEM-t0FEM
                print("Gradient simulation block done within: {:1.4f} s".format(totalFEM))
                print("\n")
    return gp

def adapt(gp, totalbudget,incrementalbudget,parameterranges,
            TOL,TOLFEM,TOLAcqui,TOLrelchange,epsphys,
            runpath, execname, adaptgrad , fun):

    'Problem dimension'
    dim= gp.getdata[2]

    'Counter variables'
    counter= 0
    totaltime= 0
    totalFEM= 0
    graddataavailable = False

    'Solver options'
    xtol = 1*1E-4
    gtol = 1*1E-4
    itermax = 100000
    s = 2

    cases = {1:"Case 1: Gradient data is not available.",
             2:"Case 2: Gradient data is available."}

    'Check for which cases are set.'
    if gp.getXgrad is None:
        case = 1
        N = gp.getX.shape[0]
        Ngrad = 0
    else:
        case = 2
        N = gp.getX.shape[0]
        Ngrad = gp.getXgrad.shape[0]
        graddataavailable = True

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
    currentcost = totalcompwork(epsilon**(-1),s)
    if gp.getXgrad is not None:
        currentcost+= totalcompwork(gp.getgradientaccuracy**(-1),s)

    print("Initial point accurcies")
    prettyprintvector(np.squeeze(gp.getaccuracy**(-1)), dim, False)
    print("\n")

    ' If gradient data is available add the costs to the current cost'
    if graddataavailable:
        epsilongrad = np.squeeze(gp.getgradientaccuracy)
        currentcost += totalcompwork(epsilongrad**(-1),s)
        print("Initial point accurcies")
        prettyprintvector(np.squeeze(gp.getgradientaccuracy**(-1)), dim, True)

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

        'Update the number of data points'
        N = gp.getX.shape[0]
        if gp.getXgrad is not None:
            Ngrad = gp.getXgrad.shape[0]
            graddataavailable = True
        else:
            Ngrad = 0
            graddataavailable = False

        'Open logs'
        
        ' Logging '
        logger = IOToLog(os.path.join(runpath+"/","iteration_log"),"iteration_"+str(counter))

        try:
            costerrorlog = open(logpath+"costerror.txt","a")
            file = open(logpath_general+"log.txt","a")
            #olutionlog = open(logpath+"solution.txt","a")
        except IOError:
          print ("Error: File does not appear to exist.")
          return 0
        if counter == 0:
            file.write("Error estimate before"+" "+"Used budget"+" "+"Error estimate after")
            file.write("\n")

        print("---------------------------------- Iteration / Design: {}".format(counter))
        print("Current number of points: {} ".format(N))
        print("Current number of gradient points: {} ".format(Ngrad))
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
        #dfGLEEDEBUG = gp.predictderivative(XGLEE, True)
        if np.any(np.isnan(dfGLEE)):
            print("STOP")
        varGLEE = gp.predictvariance(XGLEE,True)

        normvar = np.linalg.norm(np.sqrt(np.abs(varGLEE)),2,axis=0)**2
        w       = estiamteweightfactors(dfGLEE, epsphys)
        mcglobalerrorbefore = MCGlobalEstimate(w,normvar,NGLEE,parameterranges)
        file.write( str(mcglobalerrorbefore) + str(" "))

        epsprior = gp.getaccuracy
        vprior = 1/epsprior     
        currentcostprior=totalcompwork(vprior, s)
        if gp.getXgrad is not None:
            print("Assuming gradient data was obtained by adjoing gradients - no furhter costs need to be added")
            print("\n")
            logger.addToFile("Assuming gradient data was obtained by adjoing gradients - no furhter costs need to be added")
            logger.addToFile("\n")


        print("Global error estimate before optimization:   {:1.5f}".format(mcglobalerrorbefore))
        print("Computational cost before optimization:      {:0.0f}".format(currentcostprior))
        print("\n")

        logger.addToFile("Global error estimate before optimization:   {:1.5f}".format(mcglobalerrorbefore))
        logger.addToFile("Computational cost before optimization:      {:0.0f}".format(currentcostprior))
        logger.addToFile("\n")

        t1apriori=time.perf_counter()
        tapriori = t1apriori-t0apriori

        """ ------------------------------Acquisition phase ------------------------------ """
        'Add new candidate points'
        print("--- Acquisition phase")
        
        breaked = False
        t0acqui = time.perf_counter()
        logger.addToFile("--- Acquisition phase")
        XC = np.array([])
        normvar_TEST = np.linalg.norm(np.sqrt(np.abs(varGLEE)),2,axis=0)
        XC,index,value = acquisitionfunction(gp,dfGLEE,normvar_TEST,w,XGLEE,epsphys,TOLAcqui)
        
        if XC.size  == 0:
            print("Something went wrong, no candidate point was found.")
            print("\n")
            logger.addToFile("Something went wrong, no candidate point was found.")
            logger.addToFile("Number of possible candidate points: {}".format(XC.shape[0]))
            logger.addToFile("\n")

        ' Add found candidate point '
        if XC.size != 0:

            print(" Number of possible candidate points: {}".format(XC.shape[0]))
            print(" Found canditate point(s):            {}".format(XC[0]))
            print(" Use ith highest value   :            {}".format(index))
            print(" Value at index          :            {}".format(value))
            print("\n")
            
            logger.addToFile(" Number of possible candidate points: {}".format(XC.shape[0]))
            logger.addToFile(" Found canditate point(s):            {}".format(XC[0]))
            logger.addToFile(" Use ith highest value   :            {}".format(index))
            logger.addToFile(" Value at index          :            {}".format(value))
            logger.addToFile("\n")
            
# =============================================================================
#             NC = XC.shape[0]
#             epsXc = 1E20*np.ones((1, XC.shape[0]))  # eps**2
#             meanXc = gp.predictmean(XC,True)
#             gp.adddatapoint(XC)
#             gp.adddatapointvalue(meanXc)
#             gp.addaccuracy(epsXc)
# =============================================================================

        plotiteration(gp,w,normvar_TEST,N,Ngrad,XGLEE,XC,mcglobalerrorbefore,parameterranges, figurepath,counter)
        gp.savedata(runpath+'/saved_data/',str(counter))        
        
        t1acqui = time.perf_counter()
        tacqui = t1acqui-t0acqui

        """ ------------------------------ Solve minimization problem without gradient info at candidate point ------------------------------ """
        cases = {0: "1: Minimize without graddata at candidate point.",
                 1: "2: Minimize with graddata at candidate point."}

        foundcases = {0: "Add point without gradient data.",
                      1: "Add point with gradient data."}

        if graddataavailable is False:

            print("--- Solve minimization problem")
            print("Solver parameters:")
            print(" Tolerance: {}".format(xtol))
            print(" Max. iterations: {}".format(itermax))
            print(" Workmodel exponent: {}".format(s))
            print("\n")

            if XC.size != 0:
                NC = XC.shape[0]

                'Add candidate point to current GPR'
                epsXc = 1E20*np.ones((1, XC.shape[0]))
                meanXc = gp.predictmean(XC)
                gp.adddatapoint(XC)
                gp.adddatapointvalue(meanXc)
                gp.addaccuracy(epsXc)

            'Turn epsilon^2 into v'
            epsilon = np.squeeze(gp.getaccuracy)
            v = epsilon**(-1)

            ' Current cost by adding initial values is added to the overall budget '
            currentcost= totalcompwork(v, s)
            file.write(str(currentcost) + str(" "))

            for i in range(2):
                print(cases[i])
                
                logger.addToFile("------------ " + cases[i])

                'Set start value for candidate point'
                v[N:] = 10.0
                
                'Bounds on v'
                lowerbound= v.tolist()
                upperbound= [np.inf]*(N+NC)
                bounds= Bounds(lowerbound, upperbound)
                bounds.lb[N:] = 10.0 #TODO

                X = gp.getX
                hyperparameter = gp.gethyperparameter
                df = gp.predictderivative(gp.getX, True)
                wmin = estiamteweightfactors(df, epsphys)

                if i == 0:
  
                    m = gp.m
                    K = np.zeros((X.shape[0], X.shape[0],m))
                    for jj in range(m):
                        K[:,:,jj] = kernelmatrix(X, X, hyperparameter[jj,:])
                        
                    # Tensor of ones for the gradient
                    tensor = np.zeros((N+NC, N+NC, N+NC))
                    tensor[np.diag_indices(N+NC,ndim=3)] = np.ones((N+NC))

                    optimizegradientdata = False

                else:
                    if breaked:
                        break

                    'Add candidate gradient data to current GPR'
                    epsXgrad = 1E20*np.ones((1,XC.shape[0]*dim))
                    dyXC = gp.predictderivative(XC)

                    gp.addgradientdatapoint(XC)
                    gp.addgradientdatapointvalue(dyXC)
                    gp.addgradaccuracy(epsXgrad)

                    epsilongrad = np.squeeze(gp.getgradientaccuracy)
                    vgrad = epsilongrad**(-1)
                    currentcost= totalcompwork(np.concatenate((v,vgrad)), s)

                    vgrad[:] = 10.0                   

                    'Bounds on vgrad'
                    lowerboundgrad = vgrad.tolist()
                    upperboundgrad = [np.inf]*NC*dim
                    boundsgrad = Bounds(lowerboundgrad,upperboundgrad)

                    'Connect bounds'
                    lower = np.concatenate((bounds.lb,boundsgrad.lb))
                    upper = np.concatenate((bounds.ub,boundsgrad.ub))

                    'Build final bound object'
                    bounds = Bounds(lower, upper)

                    'Combine vs'
                    v = np.concatenate((v,vgrad))
                    
                    'Create unit tensor'
                    X = gp.getX
                    Xgrad = gp.getXgrad
                    
                    m = gp.m
                    K = np.zeros((N+NC+NC*dim, N+NC+NC*dim,m))
                    for jj in range(m):
                        K[:,:,jj] = kernelmatrixsgrad(X, Xgrad, hyperparameter[jj,:], gp.getaccuracy*0.0, gp.getgradientaccuracy*0.0)
                    
                    tensor = np.zeros((N+NC+NC*dim, N+NC+NC*dim, N+NC+NC*dim))
                    tensor[np.diag_indices(N+NC+NC*dim,ndim=3)] = np.ones((N+NC+NC*dim))

                    optimizegradientdata = True

                'Create nonlinear constraints'
                nonlinear_constraint= NonlinearConstraint(lambda x: compworkconstrain(x,s), currentcost+incrementalbudget, currentcost+incrementalbudget,
                                                          jac=lambda x: compworkconstrainjac(x,s),
                                                          hess=compworkconstrainhess)

                args = (wmin, X, K, N+NC, tensor, parameterranges, optimizegradientdata)
                sol=scipy.optimize.minimize(targetfunction, v,
                                            args=args,
                                            method='trust-constr',
                                            jac=gradientoftargetfunction,
                                            bounds = bounds, hess=BFGS(),
                                            constraints=[nonlinear_constraint],
                                            options={'verbose': 1, 'maxiter': itermax, 'xtol': xtol, 'gtol': gtol})

                if sol.success == True:
                    print("\n")
                    print(" Solution found within {} iterations".format(sol.nit))
                    print(" Last function value:  {}".format(sol.fun))
                    print("\n")

                    logger.addToFile("Solution found within {} iterations".format(sol.nit))
                    logger.addToFile("Last function value:  {}".format(sol.fun))
                    logger.addToFile("\n")

                    'Solution for v'
                    vsol=sol.x
                    #currentcost=totalcompwork(vsol, s)

                    'Create temporary GP for a priori estimate'
                    if i == 0:
                        epsXt = 1/vsol
                        epsXgrad = None
                        print("Found point solution:")
                        prettyprintvector(vsol, dim, False)
                        print("\n")
                        
                        logger.addToFile("Found solution with:")
                        logger.addToFile(str(vsol))
                        logger.addToFile("\n")

                    else:
                        epsXt =    1/vsol[:N+NC]
                        epsXgrad = 1/vsol[N+NC:]

                        print("Found point solution:")
                        prettyprintvector(vsol[:N+NC], dim, False)
                        print("\n")

                        print("Found gradient solution:")
                        prettyprintvector(vsol[N+NC:], dim, True)
                        print("\n")
                        
                        logger.addToFile("Found solution with:")
                        logger.addToFile(str(vsol[:N+NC]))
                        logger.addToFile("\n")
                        
                        logger.addToFile("Found gradient solution with:")
                        logger.addToFile(str(vsol[N+NC:]))
                        logger.addToFile("\n")



                    """ ------------------------------ A POST MC GLOBAL ERROR ESTIMATION ------------------------------ """
                    print("--- A posteriori error estimate")
                    
                    gptmp   = GPR(gp.getX, gp.gety, gp.getXgrad, gp.getygrad, epsXt, epsXgrad, gp.gethyperparameter)
                    dfGLEE  = gptmp.predictderivative(XGLEE, True)
                    varGLEE = gptmp.predictvariance(XGLEE,True)
                    normvar = np.linalg.norm(np.sqrt(np.abs(varGLEE)),2,axis=0)**2
                    w       = estiamteweightfactors(dfGLEE, epsphys)
                   
                    mcglobalerrorafter = MCGlobalEstimate(w,normvar,NGLEE,parameterranges)
                    
                    if mcglobalerrorafter <= TOL:
                        print("--- Convergence")
                        print(" Desired tolerance is reached, adaptive phase is done.")
                        print(" Final error estimate: {:1.8f}".format(mcglobalerrorafter))
                        #print(" Total used time: {:0.2f} seconds".format(totaltime+totalFEM))
                        print(" Save everything !")
                    
                        logger.addToFile("--- Convergence")
                        logger.addToFile(" Desired tolerance is reached, adaptive phase is done.")
                        logger.addToFile(" Final error estimate: {:1.8f}".format(mcglobalerrorafter))
                        #logger.addToFile(" Total used time: {:0.2f} seconds".format(totaltime+totalFEM))
                        logger.addToFile(" Save everything !")
                    
                        gp.savedata(runpath+'/saved_data')
                    
                        costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter))
                        costerrorlog.write("\n")
                    
                        file.close()
                        costerrorlog.close()
                        logger.closeOutputLog()
                    
                        return gp


                    file.write( str(mcglobalerrorafter))

                    if i == 0:
                        currentbesterror = mcglobalerrorafter
                        bestcase = i
                        bestsolution = vsol
                    else:
                        if mcglobalerrorafter < currentbesterror:
                            bestcase = i
                            bestsolution = vsol
                    
                    currentcost=totalcompwork(bestsolution, s)

                    file.write( str(mcglobalerrorafter))
                    print("Global error estimate after optimization:   {:1.8f}".format(mcglobalerrorafter))
                    print("Computational cost after optimization:      {:0.0f}".format(currentcost))
                    print("\n")
                    
                    logger.addToFile("Global error estimate after optimization:   {:1.8f}".format(mcglobalerrorafter))
                    logger.addToFile("Computational cost after optimization:      {:0.0f}".format(currentcost))
                    logger.addToFile("\n")

                else:
                    print("\n")
                    print("No solution found.")
                    print(" " + sol.message)
                    break

            print("--- Optimization summary")
            print(foundcases[bestcase])
            print(" Add point: {}".format(XC[0,:]))
            print(" Point accuracy: {}".format(bestsolution[:N+NC][-1]**(-1)))
            if bestcase == 1:
                print(" Gradient accuracy: {}".format(bestsolution[N+NC:][-2:]**(-1)))
            costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter))
            costerrorlog.write("\n")
            costerrorlog.close()
            print("\n")
            
            logger.addToFile("--- Optimization summary")
            logger.addToFile(str(foundcases[bestcase]))
            logger.addToFile(" Add point: {}".format(XC[0,:]))
            logger.addToFile(" Point accuracy: {}".format(bestsolution[:N+NC][-1]**(-1)))
            if bestcase == 1:
                logger.addToFile(" Gradient accuracy: {}".format(bestsolution[N+NC:][-2:]**(-1)))
            logger.addToFile(str(currentcost)+" "+str(mcglobalerrorafter))
            logger.addToFile("\n")

            'Add (best) solution to GPR'
            if bestcase == 0:
                optimizegradientdata = False
                gp.deletegradientdatapoint()

            epsilon = v**(-1)
            epsilonsolcurrent = bestsolution**(-1)
            gp = updateDataPointValues(gp,epsilon,epsilonsolcurrent,N+NC,TOLFEM,optimizegradientdata,fun)

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
            
            print("--- Adapt solver accuracy")
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
    
            print("--- Adapt buget")
            print("Prior incremental budget: {}".format(incrementalbudget))
            logger.addToFile("Prior incremental budget: {}".format(incrementalbudget))
            
            incrementalbudget *= 1.1
            print("New incremental budget:   {}".format(incrementalbudget))
            print("\n")
            logger.addToFile("New incremental budget:   {}".format(incrementalbudget))
            logger.addToFile("\n")
            
            counter += 1
            
            logger.closeOutputLog()

        elif graddataavailable:
            
            print("--- Solve minimization problem")
            print("Solver parameters:")
            print(" Tolerance: {}".format(xtol))
            print(" Max. iterations: {}".format(itermax))
            print(" Workmodel exponent: {}".format(s))
            print("\n")

            optimizegradientdata = True

            if XC.size != 0:
                NC = XC.shape[0]

                epsXc = 1E20*np.ones((1, XC.shape[0]))  # eps**2
                meanXc = gp.predictmean(XC,True)
                gp.adddatapoint(XC)
                gp.adddatapointvalue(meanXc)
                gp.addaccuracy(epsXc)

            'Current cost'
            epsilon = np.squeeze(gp.getaccuracy)
            epsilongrad = np.squeeze(gp.getgradientaccuracy)

            'v from eps'
            v = epsilon**(-1)
            vgrad = epsilongrad**(-1)

            #currentcost = totalcompwork(np.concatenate((v,vgrad)), s)
            #file.write(str(currentcost) + str(" ") )

            for i in range(2):
                print(cases[i])

                'Turn epsilon^2 into v'
                epsilon = np.squeeze(gp.getaccuracy)
                epsilongrad = np.squeeze(gp.getgradientaccuracy)

                'v from eps'
                v = epsilon**(-1)
                vgrad = epsilongrad**(-1)

                'Create arguments for minimization'
                X = gp.getX
                Xgrad = gp.getXgrad #Neu 17:15
                
                hyperparameter = gp.gethyperparameter
                df = gp.predictderivative(gp.getX, True)
                wmin = estiamteweightfactors(df, epsphys)

                'Set start values'
                v[N:] = 10.0

                'Bounds on v'
                lowerbound= v.tolist()
                upperbound= [np.inf]*(N+NC)
                bounds= Bounds(lowerbound, upperbound)

                if i == 0:

                    'Set new bounds'
                    lowerboundgrad = vgrad.tolist()
                    upperboundgrad = [np.inf]*Ngrad*dim
                    boundsgrad = Bounds(lowerboundgrad,upperboundgrad)
                    
                    currentcost = totalcompwork(np.concatenate((v,vgrad)), s)

                    m = gp.m
                    K = np.zeros((N+NC+Ngrad*dim,N+NC+Ngrad*dim,m))
                    for jj in range(m):
                        K[:,:,jj] =kernelmatrixsgrad(X, Xgrad, hyperparameter[jj,:], gp.getaccuracy*0.0, gp.getgradientaccuracy*0.0)
                        
                    # Tensor of ones for the gradient
                    tensor = np.zeros((N+NC+Ngrad*dim,N+NC+Ngrad*dim, N+NC+Ngrad*dim))
                    tensor[np.diag_indices(N+NC+Ngrad*dim,ndim=3)] = np.ones((N+NC+Ngrad*dim))

                else:

                    if breaked:
                        'Propagate solutions down, system is optimized with just existing data'
                        bestcase = 1
                        v = np.concatenate((v,vgrad))
                        break

                    'Add data gradient data'
                    if XC.size != 0:
                        epsXgrad = 1E20*np.ones((1,XC.shape[0]*dim))
                        dyXC = gp.predictderivative(XC)

                        gp.addgradientdatapoint(XC)
                        gp.addgradientdatapointvalue(dyXC)
                        gp.addgradaccuracy(epsXgrad)

                    'Get gradient data, since data is added'
                    epsilongrad = np.squeeze(gp.getgradientaccuracy)
                    vgrad = epsilongrad**(-1)

                    currentcost = totalcompwork(np.concatenate((v,vgrad)), s)
                    
                    Xgrad = gp.getXgrad
                   
                    'Set new bounds'
                    vgrad[Ngrad*dim:] = 10.0
                    lowerboundgrad = vgrad.tolist()
                    upperboundgrad = [np.inf]*((Ngrad+NC)*dim)
                    boundsgrad = Bounds(lowerboundgrad,upperboundgrad)
                    
                    m = gp.m
                    K = np.zeros((N+NC+(Ngrad+NC)*dim, N+NC+(Ngrad+NC)*dim,m))
                    for jj in range(m):
                        K[:,:,jj] = kernelmatrixsgrad(X, Xgrad, hyperparameter[jj,:], gp.getaccuracy*0.0, gp.getgradientaccuracy*0.0)
                    
                    tensor = np.zeros((N+NC+(Ngrad+NC)*dim, N+NC+(Ngrad+NC)*dim, N+NC+(Ngrad+NC)*dim))
                    tensor[np.diag_indices(N+NC+(Ngrad+NC)*dim,ndim=3)] = np.ones((N+NC+(Ngrad+NC)*dim))

                'Connect bounds'
                lower = np.concatenate((bounds.lb,boundsgrad.lb))
                upper = np.concatenate((bounds.ub,boundsgrad.ub))

                'Build final bound object'
                bounds = Bounds(lower, upper)

                'Combine vs'
                v = np.concatenate((v,vgrad))

                'Create nonlinear constraints'
                nonlinear_constraint= NonlinearConstraint(lambda x: compworkconstrain(x,s), currentcost+incrementalbudget, currentcost+incrementalbudget,
                                                          jac=lambda x: compworkconstrainjac(x,s),
                                                          hess=compworkconstrainhess)
                args = (wmin, X, K, N+NC, tensor, parameterranges, optimizegradientdata)

                'Solve minimisation'
                sol=scipy.optimize.minimize(targetfunction, v,
                                          args=args,
                                          method='trust-constr',
                                          jac=gradientoftargetfunction,
                                          bounds = bounds,hess=BFGS(),
                                          constraints=[nonlinear_constraint],
                                          options={'verbose': 1, 'maxiter': itermax, 'xtol': xtol, 'gtol': gtol})

                if sol.success == True:

                    print("\n")
                    print(" Solution found within {} iterations".format(sol.nit))
                    print(" Last function value:  {}".format(sol.fun))
                    print("\n")

                    'Solution for v'
                    vsol=sol.x
                    #currentcost=totalcompwork(vsol, s)

                    'Create temporary GP for a priori estimate'
                    epsXt = 1/vsol[:(N+NC)]
                    epsXgrad = 1/vsol[(N+NC):]

                    print("Found point solution:")
                    prettyprintvector(vsol[:(N+NC)], dim, False)
                    print("\n")

                    print("Found gradient solution:")
                    prettyprintvector(vsol[(N+NC):], dim, True)
                    print("\n")
                    
                    
                    logger.addToFile("Found solution with:")
                    logger.addToFile(str(vsol[:(N+NC)]))
                    logger.addToFile("\n")

                    logger.addToFile("Found gradient solution with:")
                    logger.addToFile(str(vsol[(N+NC):]))
                    logger.addToFile("\n")

                    """ ------------------------------ A POSTERIOIRI MC GLOBAL ERROR ESTIMATION ------------------------------ """
                    print("--- A posteriori error estimate")
                    gptmp = GPR(gp.getX, gp.gety, gp.getXgrad, gp.getygrad, epsXt, epsXgrad, gp.gethyperparameter)
                    dfGLEE = gptmp.predictderivative(XGLEE, True)
                    varGLEE = gptmp.predictvariance(XGLEE,True)
                    normvar = np.linalg.norm(np.sqrt(np.abs(varGLEE)),2,axis=0)**2
                    w = estiamteweightfactors(dfGLEE, epsphys)

                    mcglobalerrorafter = MCGlobalEstimate(w,normvar,NGLEE,parameterranges)

                    file.write( str(mcglobalerrorafter))
                    
                    if mcglobalerrorafter <= TOL:
                        print("--- Convergence")
                        print(" Desired tolerance is reached, adaptive phase is done.")
                        print(" Final error estimate: {:1.8f}".format(mcglobalerrorafter))
                        #print(" Total used time: {:0.2f} seconds".format(totaltime+totalFEM))
                        print(" Save everything !")
                    
                        logger.addToFile("--- Convergence")
                        logger.addToFile(" Desired tolerance is reached, adaptive phase is done.")
                        logger.addToFile(" Final error estimate: {:1.8f}".format(mcglobalerrorafter))
                        #logger.addToFile(" Total used time: {:0.2f} seconds".format(totaltime+totalFEM))
                        logger.addToFile(" Save everything !")
                    
                        gp.savedata(runpath+'/saved_data')
                    
                        costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter))
                        costerrorlog.write("\n")
                    
                        file.close()
                        costerrorlog.close()
                        logger.closeOutputLog()
                    
                        return gp

                    'Save solution without optimized gradient data as current best solution'
                    if i == 0:
                        currentbesterror = mcglobalerrorafter
                        bestcase = i
                        bestsolution = vsol
                        currentcost=totalcompwork(bestsolution, s)
                    else:
                        
                        #relativechange = np.abs(mcglobalerrorafter-currentbesterror) / currentbesterror * 100
                        #relchangeTOL = 1
                        #if relativechange < relchangeTOL:
                        #    print("Relative change is less then {} percent. Set regular data point!".format(relchangeTOL))
                        #    print("\n")
                        #    bestcase = 0
                        
                        if mcglobalerrorafter < currentbesterror:
                            bestcase = i
                            bestsolution = vsol
                            currentcost=totalcompwork(bestsolution, s)
                            
                    print("Global error estimate after optimization:   {:1.8f}".format(mcglobalerrorafter))
                    print("Computational cost after optimization:      {:0.0f}".format(currentcost))
                    print("\n")

                    logger.addToFile("Global error estimate after optimization:   {:1.8f}".format(mcglobalerrorafter))
                    logger.addToFile("Computational cost after optimization:      {:0.0f}".format(currentcost))
                    logger.addToFile("\n")

                    #costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter[0]))
                    #costerrorlog.write("\n")

                else:
                    print("\n")
                    print("No solution found.")
                    print(" " + sol.message)
                    break

            print("--- Optimization summary")
            print(foundcases[bestcase])
            
            'Only breakes of no candidate point was found'
            if not breaked:
                print(" Add point: {}".format(XC[0,:]))
                print(" Point accuracy: {}".format(bestsolution[:N+NC][-1]**(-1)))
                if bestcase == 1:
                    print(" Gradient accuracy: {}".format(bestsolution[N+NC:][-2:]**(-1)))
                costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter))
                costerrorlog.write("\n")
                costerrorlog.close()
                
                logger.addToFile("--- Optimization summary")
                logger.addToFile(str(foundcases[bestcase]))
                logger.addToFile(" Add point: {}".format(XC[0,:]))
                logger.addToFile(" Point accuracy: {}".format(bestsolution[:N+NC][-1]**(-1)))
                if bestcase == 1:
                    logger.addToFile(" Gradient accuracy: {}".format(bestsolution[N+NC:][-2:]**(-1)))
                logger.addToFile(str(currentcost)+" "+str(mcglobalerrorafter))
                logger.addToFile("\n")
                
            else:
                print("No further candidate point was found. Optimisation was performed with current data points.")
            
            'v contains veps as well as vepsgrad  with dimension Xt+NC + (Xgrad+NC)*dim'
            epsilon = v**(-1)
                        
            if bestcase == 0:
                'Delete the added gradient point, since only a data point is added'
                gp.deletegradientdatapoint()
                'Delete last to entries - since its gradient data'
                epsilon = epsilon[:epsilon.shape[0]-dim]
            
            epsilonsolcurrent = bestsolution**(-1)
            gp = updateDataPointValues(gp,epsilon,epsilonsolcurrent,N+NC,TOLFEM,optimizegradientdata,fun)

            print("--- Adjust hyperparameter")
            region = ((0.01, 5),   (0.01, 5))
            gp.optimizehyperparameter(region, "mean", False)
            print("\n")

            counter += 1

            logger.closeOutputLog()