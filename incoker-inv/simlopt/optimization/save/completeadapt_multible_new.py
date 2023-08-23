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

from copy import copy
from copy import deepcopy



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
            #epsXtnew=currentpointsolutions[currentFEMindex].reshape((1, -1))
            params = [1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65]
            params =  [0,2,4]
            ytnew = np.zeros((len(params)))
            for i,param in enumerate(params):
                ytnew[i]=fun["function"](np.atleast_2d(gp.getX[currentFEMindex]),param)

            ' Add new value to GP '
            #gp.addaccuracy(epsXtnew, currentFEMindex)
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
                    #epsXgradnew=currentgradsolutions[currentgradindex].reshape((1, -1))
                    #params =  [1.0,0.95,0.9,0.85,0.8]
                    #params = [1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65]
                    params =  [0,2,4]
                    ygradnew = np.zeros((dim,len(params)))
                    for i,param in enumerate(params):
                        ygradnew[:,i] = np.squeeze(fun["gradient"](np.atleast_2d(gp.getXgrad[pointindex,:]),param).reshape((1, -1)))
    
                    'Add new value to GP'
                    #gp.addgradaccuracy(epsXgradnew,currentgradindex)
                    gp.addgradientdatapointvalue(ygradnew[componentindex],currentgradindex)


                t1FEM=time.perf_counter()
                totalFEM=t1FEM-t0FEM
                print("Gradient simulation block done within: {:1.4f} s".format(totalFEM))
                print("\n")
    return gp

def optGPRWithoutGradData(gp,N,XC,s,epsphys,incrementalbudget,parameterranges,logger):
    
    xtol = 1*1E-4
    gtol = 1*1E-4
    itermax = 100000
    dim = gp.getdata[2]
        
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
  
    solutionlist = list()    
    originallist = list()    
    
    cases = {0: "------ 1: Minimize without graddata at candidate point.",
             1: "------ 2: Minimize with graddata at candidate point."}
    for i in range(2):
        
        print(cases[i])
        logger.addToFile("------------" + cases[i])

        'Set start value for candidate point'
        v[N:] = 10.0
        
        'Bounds on v'
        lowerbound= v.tolist()
        upperbound= [np.inf]*(N+NC)
        bounds= Bounds(lowerbound, upperbound)
        bounds.lb[N:] = 1.0 #TODO

        X = gp.getX
        hyperparameter = gp.gethyperparameter
        df = gp.predictderivative(gp.getX, True)
        wmin = estiamteweightfactors(df, epsphys)
        
        if i == 0:

            m = gp.m
            K = np.zeros((X.shape[0], X.shape[0],m))
            for jj in range(m):
                K[:,:,jj] = kernelmatrix(X, X, hyperparameter[jj,:])
            tensor = np.zeros((N+NC, N+NC, N+NC))
            tensor[np.diag_indices(N+NC,ndim=3)] = np.ones((N+NC))
            optimizegradientdata = False
            
        elif i==1:

            'Add candidate gradient data to current GPR'
            epsXgrad = 1E20*np.ones((1,XC.shape[0]*dim))
            dyXC = gp.predictderivative(XC)

            gp.addgradientdatapoint(XC)
            gp.addgradientdatapointvalue(dyXC)
            gp.addgradaccuracy(epsXgrad)

            epsilongrad = np.squeeze(gp.getgradientaccuracy)
            vgrad = epsilongrad**(-1)

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

        print("Max. usabale computational budget: {}".format(currentcost+incrementalbudget))
        print("\n")
        logger.addToFile("Max. usabale computational budget: {}".format(currentcost+incrementalbudget))
        logger.addToFile("\n")

        #hess=compworkconstrainhess
        nonlinear_constraint= NonlinearConstraint(lambda x: compworkconstrain(x,s), currentcost+incrementalbudget, currentcost+incrementalbudget,
                                                  jac=lambda x: compworkconstrainjac(x,s)
                                                  )
        args = (wmin, X, K, N+NC, tensor, parameterranges, optimizegradientdata)
        
        #print(v)
        method = 'SLSQP'
        #hess=BFGS(),
        sol=scipy.optimize.minimize(targetfunction, v,
                                    args=args,
                                    method=method,
                                    jac=gradientoftargetfunction,
                                    bounds = bounds, 
                                    constraints=[nonlinear_constraint],
                                    options={'verbose': 1, 'maxiter': itermax, 'xtol': xtol, 'gtol': gtol})

        if sol.success == True:
            print("\n")
            print(" Solution found within {} iterations".format(sol.nit))
            print(" Last function value:  {}".format(sol.fun))
            print("\n")

            vsol = sol.x

            logger.addToFile("Solution found within {} iterations".format(sol.nit))
            logger.addToFile("Last function value:  {}".format(sol.fun))
            logger.addToFile("\n")

            if i == 0:
                print("Found point solution:")
                prettyprintvector(vsol, dim, False)
                print("\n")
                
                logger.addToFile("Found solution with:")
                logger.addToFile(str(vsol))
                logger.addToFile("\n")
                
                tmpGPR1 = deepcopy(gp)
                epsXt = 1/vsol
                tmpGPR1.addaccuracy(epsXt,[0,None])
                
                solutionlist.append(tmpGPR1)
                originallist.append(v)
                
            else:
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
                
                
                tmpGPR2 = deepcopy(gp)
                epsXt =1/vsol[:N+NC]
                epsXgrad = 1/vsol[N+NC:]              
                tmpGPR2.addaccuracy(epsXt,[0,None])                               
                tmpGPR2.addgradaccuracy(epsXgrad,[0,None])
            
                solutionlist.append(tmpGPR2)
                originallist.append(v)

    return solutionlist,originallist

def optGPRWithGradData(gp,N,Ngrad,XC,s,epsphys,incrementalbudget,parameterranges,logger):
    
    xtol = 1*1E-4
    gtol = 1*1E-4
    itermax = 100000
    dim = gp.getdata[2]
    m = gp.m
        
    if XC.size != 0:
        NC = XC.shape[0]

        'Add candidate point to current GPR'
        epsXc = 1E20*np.ones((1, XC.shape[0]))
        meanXc = gp.predictmean(XC)
        
        gp.adddatapoint(XC)
        gp.adddatapointvalue(meanXc)
        gp.addaccuracy(epsXc)

    'Current cost'
    epsilon = np.squeeze(gp.getaccuracy)
    
    'v from eps'
    v = epsilon**(-1)
    #vgrad = epsilongrad**(-1)

    ' Current cost by adding initial values is added to the overall budget '
    currentcost= totalcompwork(v, s)

    solutionlist = list()    
    originallist = list()    
    
    cases = {0: "------ 1: Minimize without graddata at candidate point.",
             1: "------ 2: Minimize with graddata at candidate point."}
    for i in range(2):
        
        print(cases[i])
        logger.addToFile("------------ " + cases[i])

       
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
        bounds.lb[N:] = 1

        if i == 0:

            'Set new bounds'
            lowerboundgrad = vgrad.tolist()
            upperboundgrad = [np.inf]*Ngrad*dim
            boundsgrad = Bounds(lowerboundgrad,upperboundgrad)
            
            #currentcost = totalcompwork(np.concatenate((v,vgrad)), s)

            
            K = np.zeros((N+NC+Ngrad*dim,N+NC+Ngrad*dim,m))
            for jj in range(m):
                K[:,:,jj] =kernelmatrixsgrad(X, Xgrad, hyperparameter[jj,:], gp.getaccuracy*0.0, gp.getgradientaccuracy*0.0)
                
            # Tensor of ones for the gradient
            tensor = np.zeros((N+NC+Ngrad*dim,N+NC+Ngrad*dim, N+NC+Ngrad*dim))
            tensor[np.diag_indices(N+NC+Ngrad*dim,ndim=3)] = np.ones((N+NC+Ngrad*dim))
            
            optimizegradientdata = False

        else:

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

            #currentcost = totalcompwork(np.concatenate((v,vgrad)), s)
            
            Xgrad = gp.getXgrad
           
            'Set new bounds'
            vgrad[Ngrad*dim:] = 10.0
            lowerboundgrad = vgrad.tolist()
            upperboundgrad = [np.inf]*((Ngrad+NC)*dim)
            boundsgrad = Bounds(lowerboundgrad,upperboundgrad)
            
            
            K = np.zeros((N+NC+(Ngrad+NC)*dim, N+NC+(Ngrad+NC)*dim,m))
            for jj in range(m):
                K[:,:,jj] = kernelmatrixsgrad(X, Xgrad, hyperparameter[jj,:], gp.getaccuracy*0.0, gp.getgradientaccuracy*0.0)
            
            tensor = np.zeros((N+NC+(Ngrad+NC)*dim, N+NC+(Ngrad+NC)*dim, N+NC+(Ngrad+NC)*dim))
            tensor[np.diag_indices(N+NC+(Ngrad+NC)*dim,ndim=3)] = np.ones((N+NC+(Ngrad+NC)*dim))
            
            optimizegradientdata = True

        'Connect bounds'
        lower = np.concatenate((bounds.lb,boundsgrad.lb))
        upper = np.concatenate((bounds.ub,boundsgrad.ub))

        'Build final bound object'
        bounds = Bounds(lower, upper)

        'Combine vs'
        v = np.concatenate((v,vgrad))

        'Create nonlinear constraints'
        #,
        hess=compworkconstrainhess
        nonlinear_constraint= NonlinearConstraint(lambda x: compworkconstrain(x,s), currentcost+incrementalbudget, currentcost+incrementalbudget,
                                                  jac=lambda x: compworkconstrainjac(x,s))
        
        args = (wmin, X, K, N+NC, tensor, parameterranges, optimizegradientdata)

        'Solve minimisation'
        method = 'SLSQP'
        #,hess=BFGS(),
        sol=scipy.optimize.minimize(targetfunction, v,
                                  args=args,
                                  method=method,
                                  jac=gradientoftargetfunction,
                                  bounds = bounds,
                                  constraints=[nonlinear_constraint],
                                  options={'verbose': 1, 'maxiter': itermax, 'xtol': xtol, 'gtol': gtol})

        if sol.success == True:
            print("\n")
            print(" Solution found within {} iterations".format(sol.nit))
            print(" Last function value:  {}".format(sol.fun))
            print("\n")

            vsol = sol.x

            logger.addToFile("Solution found within {} iterations".format(sol.nit))
            logger.addToFile("Last function value:  {}".format(sol.fun))
            logger.addToFile("\n")

            if i == 0:
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
                
                
                tmpGPR1 = deepcopy(gp)
                epsXt =1/vsol[:N+NC]
                epsXgrad = 1/vsol[N+NC:]              
                tmpGPR1.addaccuracy(epsXt,[0,None])                               
                tmpGPR1.addgradaccuracy(epsXgrad,[0,None])
            
                solutionlist.append(tmpGPR1)
                originallist.append(v)
                
            else:
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
                
                
                tmpGPR2 = deepcopy(gp)
                epsXt =1/vsol[:N+NC]
                epsXgrad = 1/vsol[N+NC:]              
                tmpGPR2.addaccuracy(epsXt,[0,None])                               
                tmpGPR2.addgradaccuracy(epsXgrad,[0,None])
            
                solutionlist.append(tmpGPR2)
                originallist.append(v)

    return solutionlist,originallist

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
    s = 0.5

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

    while currentcost < totalbudget:
        NC = 0
        t0design=time.perf_counter()
        
        'Update the number of data points'
        N = gp.getX.shape[0]
        Ngrad = 0
        graddataavailable = False

        if gp.getXgrad is not None:
            Ngrad = gp.getXgrad.shape[0]
            graddataavailable = True
        
        ' Logging '
        logger = IOToLog(os.path.join(runpath+"/","iteration_log"),"iteration_"+str(counter))

        try:
            costerrorlog = open(logpath+"costerror.txt","a")
        except IOError:
          print ("Error: File does not appear to exist.")
          return 0

        print("---------------------------------- Iteration / Design: {}".format(counter))
        print("Current number of points: {} ".format(N))
        print("Current number of gradient points: {} ".format(Ngrad))
        print("Current number of candidate points: {} ".format(NC))
        print("\n")
        
        logger.addToFile("---------------------------------- Iteration / Design: {}".format(counter))
        logger.addToFile("Current number of points: {} ".format(N))
        logger.addToFile("Current number of candidate points: {} ".format(NC))
        logger.addToFile("\n")
        
        'Log current GP hyperparameter for debugging'
        logger.addToFile("--- Used hyperparameters")
        for i in range(gp.m):
            hp = gp.gethyperparameter[i,:]
            hpstring = np.array2string(hp, precision=2, separator=',',suppress_small=True)
            logger.addToFile("Hyperparameter experiment "+str(i) +": " +hpstring)
        logger.addToFile("\n")

        """ ------------------------------ MC GLOBAL ERROR ESTIMATION ------------------------------ """
        print("--- A priori error estimate")
        logger.addToFile("--- A priori error estimate")

        t0apriori=time.perf_counter()
        dfGLEE  = gp.predictderivative(XGLEE, True)
        if np.any(np.isnan(dfGLEE)):
            print("STOP")
        varGLEE = gp.predictvariance(XGLEE,True)

        normvar = np.linalg.norm(np.sqrt(np.abs(varGLEE)),2,axis=0)**2
        w       = estiamteweightfactors(dfGLEE, epsphys)
        mcglobalerrorbefore = MCGlobalEstimate(w,normvar,NGLEE,parameterranges)

        epsprior = gp.getaccuracy
        vprior = 1/epsprior     
        currentcostprior = totalcompwork(vprior, s)
        
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
        print("--- Acquisition phase")
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
            
            NC = 1

        plotiteration(gp,w,normvar_TEST,N,Ngrad,XGLEE,XC,mcglobalerrorbefore,parameterranges, figurepath,counter)
        gp.savedata(runpath+'/saved_data/',str(counter))        
        
        t1acqui = time.perf_counter()
        tacqui = t1acqui-t0acqui

        """ ------------------------------ Solve minimization problem ------------------------------ """
        print("--- Solve minimization problem")
        print("Solver parameters:")
        print(" Tolerance: {}".format(xtol))
        print(" Max. iterations: {}".format(itermax))
        print(" Workmodel exponent: {}".format(s))
        print("\n")
        
        t0opt = time.perf_counter()
        if graddataavailable:
            solutionlist,initialvalues = optGPRWithGradData(gp,N,Ngrad,XC,s,epsphys,
                                        incrementalbudget,parameterranges,
                                        logger)
        else:
            solutionlist,initialvalues = optGPRWithoutGradData(gp,N,XC,s,epsphys,
                                        incrementalbudget,parameterranges,
                                        logger)
        t1opt = time.perf_counter()
        totalopttime = t1opt-t0opt
        
        """ ------------------------------ A posteriori error estimate  ------------------------------ """
               
        solcases = {0: "1: A posteriori estimate without gradient data.",
                    1: "2: A posteriori estimate with gradient data."}
        print("--- A posteriori error estimate")
        print("\n")
        
        logger.addToFile("--- A posteriori error estimate")
        logger.addToFile("\n")
        
        t0post = time.perf_counter()
        for i in range(len(solutionlist)):
        
            print(solcases[i])
            logger.addToFile(solcases[i])
            logger.addToFile("\n")
            
            gptemp = solutionlist[i]
            
            'Calculate actual used work'
            epspost = gptemp.getaccuracy
            vpost = 1/epspost
            currentcostposteriori=totalcompwork(vpost, s)
            
            dfGLEE = gptemp.predictderivative(XGLEE, True)
            varGLEE = gptemp.predictvariance(XGLEE,True)
            wpost = estiamteweightfactors(dfGLEE, epsphys)
            normvar = np.linalg.norm(np.sqrt(np.abs(varGLEE)),2,axis=0)**2

            mcglobalerrorafter = MCGlobalEstimate(wpost,normvar,NGLEE,parameterranges)

            print("Error estimate after optimization and filtering: {:1.5f}".format(mcglobalerrorafter))
            print("Computational cost after optimization:           {:0.0f}".format(currentcostposteriori))
            print("\n")
            
            logger.addToFile("Error estimate after optimization and filtering: {:1.5f}".format(mcglobalerrorafter))
            logger.addToFile("Computational cost after optimization:           {:0.0f}".format(currentcostposteriori))
            logger.addToFile("\n")
            
            breaked = False
            if i == 0:
                currentbesterror = mcglobalerrorafter
                bestcase = i
                epspost = gptemp.getaccuracy
                bestsolution = 1/epspost
                optimizegradientdata = False
                gp = solutionlist[0]
                epsilon = 1/initialvalues[0]
                
            else:
                relativechange = np.abs(mcglobalerrorafter - currentbesterror) / currentbesterror
                TOLRELCHANGE = 0.1
                if  relativechange < TOLRELCHANGE:
                    
                    print("Relative change between errors is to small. Neglect optimized gradients.")
                    print("Relative change: {}".format(relativechange))
                    print("\n")
                    logger.addToFile("Relative change between errors is to small. Neglect optimized gradients.")
                    logger.addToFile("Relative change: {}".format(relativechange))
                    logger.addToFile("\n")                   
                    
                    currentbesterror = mcglobalerrorafter
                    bestcase = 0
                    epspost = solutionlist[0].getaccuracies
                    bestsolution = 1/epspost
                    optimizegradientdata = False
                    gp = solutionlist[0]
                    epsilon = 1/initialvalues[0]
                    breaked = True
                
                elif mcglobalerrorafter < currentbesterror and breaked == False:
                    bestcase = i
                    epspost = gptemp.getaccuracies
                    bestsolution = 1/epspost
                    optimizegradientdata = True
                    gp = solutionlist[1]
                    epsilon = 1/initialvalues[1]
                    
                else:
                    bestcase = 0
                    epspost = solutionlist[0].getaccuracy
                    bestsolution = 1/epspost
                    optimizegradientdata = False
                    gp = solutionlist[0]
                    epsilon = 1/initialvalues[0]
                    
        t1post = time.perf_counter()
        tpost = t1post - t0post
                
        currentcost=totalcompwork(bestsolution, s)
        
        foundcases = {0: "Add point without gradient data.",
                      1: "Add point with gradient data."}
        
        print("--- Optimization summary")
        print(foundcases[bestcase])
        print(" Add point: {}".format(XC[0,:]))
        print(" Point accuracies")
        prettyprintvector(bestsolution[0,:N+NC]**(-1), dim, False)
        if bestcase == 1:
            print(" Gradient accuracy")
            prettyprintvector(bestsolution[0,N+NC:]**(-1), dim, True)
        costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter))
        costerrorlog.write("\n")
        print("\n")

        logger.addToFile("--- Optimization summary")
        logger.addToFile(str(foundcases[bestcase]))
        logger.addToFile(" Add point: {}".format(XC[0,:]))
        logger.addToFile(" Point accuracy: {}".format(bestsolution[0,:N+NC]**(-1)))
        if bestcase == 1:
            logger.addToFile(" Gradient accuracy: {}".format(bestsolution[0,N+NC:]**(-1)))
        #logger.addToFile(str(currentcost)+" "+str(mcglobalerrorafter))
        logger.addToFile("\n")
                

        """ ------------------------------ Set data ------------------------------ """
        epspost = np.squeeze(epspost)
        gp = updateDataPointValues(gp, epsilon, epspost, 
                                   N+NC , TOLFEM, 
                                   optimizegradientdata, fun)

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
            costerrorlog.close()
            logger.closeOutputLog()
        
            gp.savedata(runpath+'/saved_data')
            return gp
            
        epsilon = epspost
        print("New budget for next design:                      {:0.0f}".format(currentcost+incrementalbudget))
        print("\n")
        logger.addToFile("New budget for next design:                      {:0.0f}".format(currentcost+incrementalbudget))
        logger.addToFile("\n")

        """ ------------------------------ Adjusting ------------------------------ """
        Nmax = 100
        if N < Nmax:
            print("--- A posteriori hyperparameter adjustment")
            region = ((0.01, 3),   (0.01, 3))
            gp.optimizehyperparameter(region, "mean", False)
        else:
            print("--- A posteriori hyperparameter adjustment")
            print("Number of points is higher then "+str(Nmax))
            print("No optimization is performed")
        print("\n")

        print("--- Adapt budget")
        print("Prior incremental budget: {}".format(incrementalbudget))
        logger.addToFile("Prior incremental budget: {}".format(incrementalbudget))
        incrementalbudget *= 1.1
        print("New incremental budget:   {}".format(incrementalbudget))
        print("\n")
        logger.addToFile("New incremental budget:   {}".format(incrementalbudget))
        logger.addToFile("\n")

        t1design=time.perf_counter()
        print("Time used for complete design iteration: {:0.2f} seconds".format(t1design-t0design))
        print("\n")

        costerrorlog.close()

        logger.addToFile("Times used for a priori error estimate       : {:0.2f} seconds".format(tapriori))
        logger.addToFile("Times used for acquisition phase             : {:0.2f} seconds".format(tacqui))
        logger.addToFile("Times used for solving minimization problem  : {:0.2f} seconds".format(totalopttime))
        #logger.addToFile("Times used for solving fem simulation        : {:0.2f} seconds".format(tFEM))
        logger.addToFile("Times used for a posteriori error estimate   : {:0.2f} seconds".format(tpost))
        logger.addToFile("Time used for complete design iteration      : {:0.2f} seconds".format(t1design-t0design))
        logger.addToFile("\n")
        logger.closeOutputLog()
        
        counter +=1