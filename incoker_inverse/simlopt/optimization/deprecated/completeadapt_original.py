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


def DEBUGGRADIENDTS(X):
    N = X.shape[0]
    dim = X.shape[1]

    a = [1.0]
    dy = np.zeros((N,dim,len(a)))

    for j in range(len(a)):
        for i in range(N):
            x = X[i,:]
            dy[i,:,j]  = np.array([np.cos(x[0])-a[j]*x[1]*np.sin(x[0]*x[1]), -a[j]*x[0]*np.sin(x[0]*x[1])])
    return dy


def estiamteweightfactors(var, X, dy, epsphys):

    dim = X.shape[1]
    delta = 1E-6

    m = 1
    w = np.zeros((X.shape[0]))
    #dy = DEBUGGRADIENDTS(X)

    'Check if epsphys might be a matrix'
    if isinstance(epsphys, (np.floating, float)):
        SigmaLL = epsphys*np.eye(m)
        SigmaLL = np.linalg.inv(SigmaLL)

    for i, x in enumerate(X):
        Jprime = dy[i, :]
        w[i] = np.linalg.norm(( np.linalg.inv((SigmaLL*np.outer(Jprime.T,Jprime)+delta*np.eye((dim)) )) @ (SigmaLL*Jprime.reshape((-1, 1))))  , 2)

    return w


def MCGlobalEstimate(w,var,Nall,parameterranges):
    volofparameterspace = np.prod(parameterranges[:, 1])
    return volofparameterspace/Nall * np.dot(w,var)


def targetfunction(v, w, X, hyperparameter, K, Nall,tensor, parameterranges,adaptgrad=False):

    v = v.reshape((1, -1))

    sigma = hyperparameter[0]
    L = hyperparameter[1:]
    volofparameterspace = np.prod(parameterranges[:, 1])

    errorsum = 0

    'Inverse of KXX'
    invK = np.linalg.inv(K+1E-6*np.eye((K.shape[0])))

    'Inverse of KXX-1+V'
    if adaptgrad:
        invKV = np.linalg.inv(invK+np.diagflat(v))
    else:
        tmpzero = np.zeros((K.shape[0],K.shape[0]))
        tmpzero[:v.shape[1],:v.shape[1]] += np.diagflat(v)
        invKV = np.linalg.inv(invK+tmpzero)

    globalerrorestimate = (volofparameterspace/Nall) * np.dot(np.diag(invKV)[:w.shape[0]],w)

    return globalerrorestimate

def gradientoftargetfunction(v, w, X, hyperparameter, K, Nall,tensor, parameterranges,adaptgrad=False):

    v = v.reshape((1, -1))

    sigma = hyperparameter[0]
    Lhyper = hyperparameter[1:]
    volofparameterspace = np.prod(parameterranges[:, 1])

    errorgradsum = 0

    'Inverse of KXX'
    invK = np.linalg.inv(K+1E-6*np.eye((K.shape[0])))

    'Inverse of KXX-1+V'
    if adaptgrad:
        invKV = np.linalg.inv(invK+np.diagflat(v))
    else:
        tmpzero = np.zeros((K.shape[0],K.shape[0]))
        tmpzero[:v.shape[1],:v.shape[1]] += np.diagflat(v)
        invKV = np.linalg.inv(invK+tmpzero)

    t0grad = time.perf_counter()
    for i in range(X.shape[0]):
        gradvar = np.diag(invKV@tensor[:,:,i]@invKV)
        if adaptgrad == False:
            errorgradsum += w[i] * gradvar[:w.shape[0]]
        else:
            errorgradsum += w[i] * gradvar
    grad = volofparameterspace/Nall * errorgradsum

    t1grad= time.perf_counter()
    return np.squeeze(grad)

def hessianoftargetfunction(v, w, X, yt, hyperparameter, KXX, Nall, parameterranges, logtransform=False):

    v= v.reshape((1, -1))

    sigma= hyperparameter[0]
    Lhyper= hyperparameter[1:]
    volofparameterspace= np.prod(parameterranges[:, 1])

    errorhesssum= 0
    errorsum= 0

    'Inverse in eps for df'
    KXXdf= KXX+np.diagflat(v**(-1))
    alpha= np.linalg.solve(KXXdf, yt)

    'Inverse of KXX'
    invKXX= np.linalg.inv(KXX)
    # invKXX[np.abs(invKXX) < 1E-6] = 0.0

    'Inverse of KXX-1+V'
    invKV= np.linalg.inv(invKXX+np.diagflat(v))
    # invKV[np.abs(invKV) < 1E-6] = 0.0

    'Unit matrix from euclidean vector'
    unitmatrix= np.eye(X.shape[0])

    t0hess= time.perf_counter()
    for i, x in enumerate(X):

        ei =  unitmatrix[i, :]
        hessvar= np.zeros((Nall, Nall))

        for ii in range(Nall):

            ei_ii =  unitmatrix[ii, :]
            dvi_V = np.outer(ei_ii.T, ei_ii)

            for jj in range(ii+1):

                ei_jj =  unitmatrix[jj, :]
                dvj_V = np.outer(ei_jj.T, ei_jj)

                hessvar[ii, jj] = ei.T@(invKV@dvi_V@invKV@dvj_V@invKV+invKV@dvj_V@invKV@dvi_V@invKV)@ei

        diag= np.diagflat(np.diag(hessvar))
        hessvar += hessvar.T
        hessvar -= diag

        errorhesssum += w[i] * hessvar #w[i]**2 * hessvar

    hessian= volofparameterspace/Nall * errorhesssum

    t1hess= time.perf_counter()
    #print("Time in hess: "+str(t1hess-t0hess))
    return hessian




""" Computational work evaluation """
def totalcompwork(v, s=1):
    return np.sum(v**(s))
def totalcompworkeps(epsilon):
    return np.sum(1/2*epsilon**(-2))

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
    currentcost = totalcompwork(epsilon**(-1),s)

    ' If gradient data is available add the costs to the current cost'
    if graddataavailable:
        epsilongrad = np.squeeze(gp.getgradientaccuracy)
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

        'Update the number of data points'
        N = gp.getX.shape[0]
        if gp.getXgrad is not None:
            Ngrad = gp.getXgrad.shape[0]
            graddataavailable = True    
        else:
            Ngrad = 0
            graddataavailable = False
            
        'Open logs'
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
        print("Current number of gradient points: {} ".format(Ngrad))
        print("Current number of candidate points: {} ".format(NC))
        print("\n")

        """ ------------------------------ MC GLOBAL ERROR ESTIMATION ------------------------------ """
        print("--- A priori error estimate")
        dfGLEE = gp.predictderivative(XGLEE, True)
        varGLEE = gp.predictvariance(XGLEE)
        w = estiamteweightfactors(varGLEE, XGLEE, dfGLEE, epsphys)
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

        plotiteration(gp,w,varGLEE,N,Ngrad,XGLEE,XC,XCdummy,mcglobalerrorbefore,figurepath,counter)

        """ If XC is empty, take the current GPR and perfom a minimization just with the tiven data """
        

        """ ------------------------------ Solve minimization problem without gradient info at candidate point ------------------------------ """
        cases = {0: "1: Solve without graddata at candidate point.",
                 1: "2: Solve with graddata at candidate point."}
        
        foundcases = {0: "Add point without gradient data.",
                      1: "Add point with gradient data."}
        
        if graddataavailable is False:
            print("--- Solve minimization problem")
        
            for i in range(2):
                print(cases[i])

                if i == 0:
                    
                    NC = XC.shape[0]
                    epsXc = 1E20*np.ones((1, XC.shape[0]))  # eps**2
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
                                        
                    'Set start value for candidate point'
                    v[N:] = 10.0

                    'Bounds on v'
                    lowerbound= v.tolist()
                    upperbound= [np.inf]*(N + NC)
                    bounds= Bounds(lowerbound, upperbound)
                    bounds.lb[N:] = 0.0

                    total_n= 0
                    nonlinear_constraint= NonlinearConstraint(lambda x: compworkconstrain(x,s), currentcost+incrementalbudget, currentcost+incrementalbudget,
                                                               jac=lambda x: compworkconstrainjac(x,s),
                                                               hess=compworkconstrainhess)
                    ' Calculate arguments for '
                    X = gp.getX
                    hyperparameter = gp.gethyperparameter
                    var = gp.predictvariance(X)
                    df = gp.predictderivative(gp.getX, True)
                    wmin = estiamteweightfactors(var, X, df, epsphys)

                    K = kernelmatrix(X, X, hyperparameter)
                    tensor = np.zeros((N + NC, N + NC, N + NC))
                    for kk in range(N + NC):
                        tensor[kk, kk, kk] = 1

                    args = (wmin, X, hyperparameter, K, N+NC, tensor, parameterranges, False)

                else:

                    epsXgrad = 1E20*np.ones((1,XC.shape[0]*dim))
                    dyXC = gp.predictderivative(XC)

                    gp.addgradientdatapoint(XC)
                    gp.adddgradientdatapointvalue(dyXC)
                    gp.addgradaccuracy(epsXgrad)

                    'Turn epsilon^2 into v'
                    epsilon = np.squeeze(gp.getaccuracy)
                    v = epsilon**(-1)
                    epsilongrad = np.squeeze(gp.getgradientaccuracy)
                    vgrad = epsilongrad**(-1)

                    'Set start value for candidate point and its gradioent value'
                    v[N:]    = 10.0
                    vgrad[:] = 10.0

                    'Bounds on v and vgrad'
                    lowerbound= v.tolist()
                    upperbound= [np.inf]*(N+NC)
                    bounds= Bounds(lowerbound, upperbound)
                    bounds.lb[N:] = 0.0

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

                    'Create nonlinear constraints'
                    nonlinear_constraint= NonlinearConstraint(lambda x: compworkconstrain(x,s), currentcost+incrementalbudget, currentcost+incrementalbudget,
                                                              jac=lambda x: compworkconstrainjac(x,s),
                                                              hess=compworkconstrainhess)
                    X,Xgrad = gp.getX,gp.getXgrad
                    hyperparameter = gp.gethyperparameter
                    df = gp.predictderivative(gp.getX, True)
                    var = gp.predictvariance(X)
                    wmin = estiamteweightfactors(var, X, df, epsphys)

                    K = kernelmatrixsgrad(X, Xgrad, hyperparameter, gp.getaccuracy*0.0, gp.getgradientaccuracy*0.0)

                    tensor = np.zeros((N+NC+NC*dim,N+NC+NC*dim,N+NC+NC*dim))
                    for kk in range(N+NC+NC*dim):
                        tensor[kk, kk, kk] = 1

                    args = (wmin, X, hyperparameter, K, N+NC,tensor, parameterranges, True)


                sol=scipy.optimize.minimize(targetfunction, v,
                                          args=args,
                                          method='trust-constr',
                                          jac=gradientoftargetfunction,
                                          bounds = bounds,hess=BFGS(),
                                          constraints=[nonlinear_constraint],
                                          options={'verbose': 1, 'maxiter': itermax, 'xtol': xtol, 'gtol': gtol})

                if sol.success == True:

                    print("Solution found within {} iterations".format(sol.nit))
                    print("Used time:            {:0.2f} seconds".format(total_n))
                    print("Last function value:  {}".format(sol.fun))
                    print("\n")

                    'Solution for v'
                    vsol=sol.x
                    costofsolution=totalcompwork(vsol, s)

                    'Create temporary GP for a priori estimate'
                    if i == 0:
                        epsXt = 1/vsol
                        epsXgrad = None
                        currentestimate = 0
                        print("Found point solution:")
                        prettyprintvector(vsol, dim, False)
                        print("\n")
                        
                    else:
                        epsXt = 1/vsol[:N+NC]
                        epsXgrad = 1/vsol[N+NC:]
                        
                        print("Found point solution:")
                        prettyprintvector(vsol[:N+NC], dim, False)
                        print("\n")

                        print("Found gradient solution:")
                        prettyprintvector(vsol[N+NC:], dim, True)
                        print("\n")

                    gptmp = GPR(gp.getX, gp.gety, gp.getXgrad, gp.getygrad, epsXt, epsXgrad, gp.gethyperparameter)

                    """ ------------------------------ A PRIORI MC GLOBAL ERROR ESTIMATION ------------------------------ """
                    print("--- A posteriori error estimate")
                    dfGLEE = gptmp.predictderivative(XGLEE, True)
                    varGLEE = gptmp.predictvariance(XGLEE)

                    mcglobalerrorafter = MCGlobalEstimate(w,varGLEE,NGLEE,parameterranges)

                    file.write( str(mcglobalerrorafter[0]))
                    print("Global error estimate after optimization:   {:1.8f}".format(mcglobalerrorafter[0]))
                    print("Computational cost after optimization:      {:0.0f}".format(costofsolution))
                    print("\n")

                    if i == 0:
                        currentbesterror = mcglobalerrorafter
                        bestcase = i
                        bestsolution = vsol
                    else:
                        if mcglobalerrorafter < currentbesterror:
                            bestcase = i
                            bestsolution = vsol

                    costerrorlog.write(str(costofsolution)+" "+str(mcglobalerrorafter[0]))
                    costerrorlog.write("\n")
                    

            print(foundcases[bestcase])
            print("\n")
            
            'Add (best) solution to GPR'
            if bestcase == 0:
                'Lösche letzten Gradientenpunkt'
                gp.deletegradientdatapoint()
                'Füge eps als beste Lösung hinzu'
                gp.addaccuracy(bestsolution**(-1),[0,None])

            else:              
                'Add data accuracy'
                gp.addaccuracy(bestsolution[:N+NC]**(-1),[0,None])
                'Add gradient data accuracy'
                gp.addgradaccuracy(bestsolution[N+NC:]**(-1),[0,None])
            counter += 1
            
        elif graddataavailable:
            print("--- Solve minimization problem")
            
            for i in range(2):
                print(cases[i])

                if i == 0:
                   
                    NC = XC.shape[0]
                    epsXc = 1E20*np.ones((1, XC.shape[0]))  # eps**2
                    meanXc = gp.predictmean(XC)
                    gp.adddatapoint(XC)
                    gp.adddatapointvalue(meanXc)
                    gp.addaccuracy(epsXc)

                    'Turn epsilon^2 into v'
                    epsilon = np.squeeze(gp.getaccuracy)
                    epsilongrad = np.squeeze(gp.getgradientaccuracy)
                    
                    v = epsilon**(-1)
                    vgrad = epsilongrad**(-1)
                    
                    'Set start values'
                    v[N:] = 10.0
                   
                    'Bounds on v and vgrad'
                    lowerbound= v.tolist()
                    upperbound= [np.inf]*(N+NC)
                    bounds= Bounds(lowerbound, upperbound)
                    
                    lowerboundgrad = vgrad.tolist()
                    upperboundgrad = [np.inf]*Ngrad*dim
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
                    nonlinear_constraint= NonlinearConstraint(lambda x: compworkconstrain(x,s), currentcost+incrementalbudget, currentcost+incrementalbudget,
                                                              jac=lambda x: compworkconstrainjac(x,s),
                                                              hess=compworkconstrainhess)

                    X,Xgrad = gp.getX,gp.getXgrad
                    hyperparameter = gp.gethyperparameter
                    df = gp.predictderivative(gp.getX, True)
                    var = gp.predictvariance(X)
                    wmin = estiamteweightfactors(var, X, df, epsphys)

                    K = kernelmatrixsgrad(X, Xgrad, hyperparameter, gp.getaccuracy*0.0, gp.getgradientaccuracy*0.0)

                    tensor = np.zeros((N+NC+Ngrad*dim, N+NC+Ngrad*dim, N+NC+Ngrad*dim))
                    for kk in range(N+NC+Ngrad*dim):
                        tensor[kk, kk, kk] = 1

                    args = (wmin, X, hyperparameter, K, N+NC, tensor, parameterranges, True)
                   
                else:
                    
                    'Add data gradient data'
                    epsXgrad = 1E20*np.ones((1,XC.shape[0]*dim))
                    dyXC = gp.predictderivative(XC)

                    gp.addgradientdatapoint(XC)
                    gp.adddgradientdatapointvalue(dyXC)
                    gp.addgradaccuracy(epsXgrad)

                    'Turn epsilon^2 into v'
                    epsilon = np.squeeze(gp.getaccuracy)
                    epsilongrad = np.squeeze(gp.getgradientaccuracy)
                    
                    v = epsilon**(-1)
                    vgrad = epsilongrad**(-1)
                    
                    'Set start values'
                    v[N:] = 10.0
                    vgrad[Ngrad+NC*dim:] = 10.0

                    'Bounds on v and vgrad'
                    lowerbound= v.tolist()
                    upperbound= [np.inf]*(N+NC)
                    bounds= Bounds(lowerbound, upperbound)
                    
                    lowerboundgrad = vgrad.tolist()
                    upperboundgrad = [np.inf]*((Ngrad+NC)*dim)
                    boundsgrad = Bounds(lowerboundgrad,upperboundgrad)

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
                    
                    'Create arguments for minimization'
                    X,Xgrad = gp.getX,gp.getXgrad
                    hyperparameter = gp.gethyperparameter
                    df = gp.predictderivative(gp.getX, True)
                    var = gp.predictvariance(X)
                    wmin = estiamteweightfactors(var, X, df, epsphys)

                    K = kernelmatrixsgrad(X, Xgrad, hyperparameter, gp.getaccuracy*0.0, gp.getgradientaccuracy*0.0)

                    tensor = np.zeros((N+NC+(Ngrad+NC)*dim, N+NC+(Ngrad+NC)*dim, N+NC+(Ngrad+NC)*dim))
                    for kk in range(N+NC+(Ngrad+NC)*dim):
                        tensor[kk, kk, kk] = 1

                    args = (wmin, X, hyperparameter, K, N+NC, tensor, parameterranges, True)

                sol=scipy.optimize.minimize(targetfunction, v,
                                          args=args,
                                          method='trust-constr',
                                          jac=gradientoftargetfunction,
                                          bounds = bounds,hess=BFGS(),
                                          constraints=[nonlinear_constraint],
                                          options={'verbose': 1, 'maxiter': itermax, 'xtol': xtol, 'gtol': gtol})
                
                if sol.success == True:

                    print("Solution found within {} iterations".format(sol.nit))
                    print("Used time:            {:0.2f} seconds".format(total_n))
                    print("Last function value:  {}".format(sol.fun))
                    print("\n")

                    'Solution for v'
                    vsol=sol.x
                    costofsolution=totalcompwork(vsol, s)

                    'Create temporary GP for a priori estimate'
                    if i == 0:
                        epsXt = 1/vsol[:(N+NC)]
                        epsXgrad = 1/vsol[(N+NC):]
                        currentestimate = 0
                        print("Found point solution:")
                        prettyprintvector(vsol[:(N+NC)], dim, False)
                        print("\n")
                        
                        print("Found gradient solution:")
                        prettyprintvector(vsol[(N+NC):], dim, True)
                        print("\n")
                        
                    else:
                        epsXt = 1/vsol[:(N+NC)]
                        epsXgrad = 1/vsol[(N+NC):]
                        
                        print("Found point solution:")
                        prettyprintvector(vsol[:(N+NC)], dim, False)
                        print("\n")

                        print("Found gradient solution:")
                        prettyprintvector(vsol[(N+NC):], dim, True)
                        print("\n")
                        
                    gptmp = GPR(gp.getX, gp.gety, gp.getXgrad, gp.getygrad, epsXt, epsXgrad, gp.gethyperparameter)

                    """ ------------------------------ A PRIORI MC GLOBAL ERROR ESTIMATION ------------------------------ """
                    print("--- A posteriori error estimate")
                    dfGLEE = gptmp.predictderivative(XGLEE, True)
                    varGLEE = gptmp.predictvariance(XGLEE)

                    mcglobalerrorafter = MCGlobalEstimate(w,varGLEE,NGLEE,parameterranges)

                    file.write( str(mcglobalerrorafter[0]))
                    print("Global error estimate after optimization:   {:1.8f}".format(mcglobalerrorafter[0]))
                    print("Computational cost after optimization:      {:0.0f}".format(costofsolution))
                    print("\n")

                    if i == 0:
                        currentbesterror = mcglobalerrorafter
                        bestcase = i
                        bestsolution = vsol
                    else:
                        if mcglobalerrorafter < currentbesterror:
                            bestcase = i
                            bestsolution = vsol

                    costerrorlog.write(str(costofsolution)+" "+str(mcglobalerrorafter[0]))
                    costerrorlog.write("\n")
            
            print(foundcases[bestcase])
            print("\n")
            
            'Add (best) solution to GPR'
            if bestcase == 0:
                'Lösche letzten Gradientenpunkt'
                gp.deletegradientdatapoint()
                'Füge eps als beste Lösung hinzu'
                gp.addaccuracy(bestsolution**(-1),[0,None])
            else:              
                'Add data accuracy'
                gp.addaccuracy(bestsolution[:(N+NC)]**(-1),[0,None])
                'Add gradient data accuracy'
                gp.addgradaccuracy(bestsolution[(N+NC):]**(-1),[0,None])

            counter += 1