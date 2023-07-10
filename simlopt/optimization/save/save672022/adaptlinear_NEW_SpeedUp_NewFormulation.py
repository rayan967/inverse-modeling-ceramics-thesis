import numpy as np

import copy
import scipy
import scipy.optimize as optimize
import time
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from basicfunctions.covariance.cov import *
from basicfunctions.utils.creategrid import *

from basicfunctions.kaskade.kaskadeio import *

from scipy.optimize import minimize
from gpr.gaussianprocess import *

from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint

def maxpointwiseerror(C1, C2, df, SigmaLL, SigmaP, variance, norm, scale=1.0):

    invsigma = np.linalg.inv(SigmaLL)
    invnorm = np.linalg.norm(invsigma, norm)
    N = df.shape[0]
    currentmax = 0
    dim = df.shape[1]

    for i in range(N):
        L = np.min(np.linalg.eig(np.outer(df[i, :], invsigma*df[i, :]) +
                   np.linalg.inv(SigmaP)+1E-7*np.eye(dim))[0])  # float

        if isinstance(L, complex):
            L = L.real
        if L <= 1E-4:
            print("Warning: L is too small, set value to: "+str(1))
            L = 1

        pointwiseerror = (12*C1*invnorm/L + 1/C2) * scale * variance[i, :]
        if pointwiseerror > currentmax:
            currentmax = pointwiseerror

    return currentmax


def pointwiseerror(C1, C2, df, SigmaLL, SigmaP, variance, norm):

    dim = df.shape[0]

    invsigma = np.linalg.inv(SigmaLL)
    invnorm = np.linalg.norm(invsigma, norm)

    L = np.min(np.linalg.eig(np.outer(df, invsigma*df) +
                             np.linalg.inv(SigmaP)+1E-7*np.eye(dim))[0])  # float

    if isinstance(L, complex):
        L = L.real

    if L <= 1E-4:
        print("Warning: L is too small, set value to: "+str(1))
        L = 1

    pointwiseerror = (12*C1*invnorm/L + 1/C2) *  variance
    return pointwiseerror


def globalerrorestimate(v, X, yt, hyperparameter, KXX, C1, C2, Nall, SigmaLL, SigmaP, parameterranges):

    v = v.reshape((1, -1))

    sigma = hyperparameter[0]
    L = hyperparameter[1:]
    volofparameterspace = np.prod(parameterranges[:, 1])
    
    errorsum = 0

    'Inverse in eps for df'
    KXXdf = KXX+np.diagflat(v**(-1))
    alpha = np.linalg.solve(KXXdf, yt)
    
    'Inverse of KXX'
    invKXX = np.linalg.inv(KXX)
    invKXX[np.abs(invKXX) < 1E-6] = 0.0

    'Inverse of KXX-1+V'
    invKV = np.linalg.inv(invKXX+np.diagflat(v))
    invKV[np.abs(invKV) < 1E-6] = 0.0

    'Unit matrix from euclidean vector'
    unitmatrix = np.eye(X.shape[0])

    for i, x in enumerate(X):
        'Local variance'
        var = unitmatrix[:, i].T@invKV@unitmatrix[:, i]

        'Local derivative estiamte'
        KxX = KXX[i, :].reshape((1, -1))
        df = dGPR(x.reshape((1, -1)), X, KxX, L)@alpha

        'Local error estimation at x \in X'
        pwee = pointwiseerror(C1, C2, df.T, SigmaLL, SigmaP, var, np.inf)
        errorsum += pwee**2

    globalerrorestimate = (volofparameterspace/Nall) * \
        errorsum  # Squared ! Thats why no root
    return globalerrorestimate  # 14.4


def gradientofglobalerrorestimate(v, X, yt, hyperparameter, KXX, C1, C2, Nall, SigmaLL, SigmaP, parameterranges):

    dim = X.shape[1]

    v = v.reshape((1, -1))

    sigma = hyperparameter[0]
    Lhyper = hyperparameter[1:]
    volofparameterspace = np.prod(parameterranges[:, 1])
    
    errorsum = 0

    invsigma = np.linalg.inv(SigmaLL)
    invnorm = np.linalg.norm(invsigma, np.inf)

    'Inverse in eps for df'
    KXXdf = KXX+np.diagflat(v**(-1))
    alpha = np.linalg.solve(KXXdf, yt)

    'Inverse of KXX'
    invKXX = np.linalg.inv(KXX)
    invKXX[np.abs(invKXX) < 1E-6] = 0.0

    'Inverse of KXX-1+V'
    invKV = np.linalg.inv(invKXX+np.diagflat(v))
    invKV[np.abs(invKV) < 1E-6] = 0.0

    'Unit matrix from euclidean vector'
    unitmatrix = np.eye(X.shape[0])
    
    for i, x in enumerate(X):
        
        'Euclidean unit vector'
        ei =  unitmatrix[i,:]
                       
        'Local variance'
        var = ei.T@invKV@ei
 
        'Local derivative estiamte'
        KxX = KXX[i, :].reshape((1, -1))
        df = dGPR(x.reshape((1, -1)), X, KxX, Lhyper)@alpha

        L = np.min(np.linalg.eig(np.outer(df, invsigma*df) +
                   np.linalg.inv(SigmaP)+1E-4*np.eye(dim))[0])  # float

        if isinstance(L, complex):
            L = L.real
        if L <= 1E-4:
            print("Warning: L is too small, set value to: "+str(1))
            L = 1
                    
        KR = 12/L*invnorm*C1 + (1/C2)

        gradvar = np.zeros((Nall, 1))
        
        for kk in range(Nall):
            ei_kk =  unitmatrix[kk,:]
            gradvar[kk, 0] = -ei.T@ invKV @ np.outer(ei_kk.T,ei_kk) @ invKV @ ei #Checked with Mathematica
           
        errorsum += 2 * KR * var * KR * gradvar
        
    grad = volofparameterspace/Nall * errorsum
    
    return np.squeeze( grad)

def estimateconstants(gp,df):
    X = gp.getX
    
    C1 = np.max(np.linalg.norm(df, np.inf, axis=1))  # float
    
    C2 = 0
    for ii in range(X.shape[0]):
        hess = gp.predicthessian(np.array([X[ii, :]]))
        maxhess = np.linalg.norm(hess, np.inf)
        if maxhess > C2:
            C2 = maxhess
    return C1,C2


""" Inequality functions """
def totalcompwork(v, s=1):
    return np.sum(v**(s))
def totalcompworkeps(epsilon):
    return np.sum(1/2*epsilon**(-2))



""" Data functions """
def fun(x):
    return np.sin(x[0])+np.cos(x[0]*x[1])

def dfun(x):
    if x.shape[0] > 1:
        return np.array([np.cos(x[:, 0])-x[:, 1]*np.sin(x[:, 0]*x[:, 1]), -x[:, 0]*np.sin(x[:, 0]*x[:, 1])]).T
    else:
        return np.array([np.cos(x[0])-x[1]*np.sin(x[0]*x[1])],
                        [-x[0]*np.sin(x[0]*x[1])])


def adapt(gp, N, NC, totalbudget, budgettospend,
          SigmaLL, SigmaP, parameterranges,
          TOL, TOLe, TOLFEM, TOLFILTER, loweraccuracybound, nrofnewpoints, execpath, execname,
          beta=1.4, sqpiter=1000, sqptol=1E-5):
    """

    Parameters
    ----------
    gp : Gaussian process
        Gaussian process which is getting adapted
    N : int
        Number is base data points
    NC : int
        Number of initial candidate points.
    totalbudget : float
        Total available computational budget
    budgettospend : float
        Delta W to used in every design
    SigmaLL : np.arry([[mxm]])
        Likelihood-covariance matrix
    SigmaP : np.arry([[dimxdim]])
        Parameter prior
    parameterranges : np.array([]) dimxdim
        Ranges of parameterspace
    TOL : float
        Desired tolerance for the estimated global parameter error
    TOLe : float
        Tolerance...
    loweraccuracybound : float
        lower accuracy bound for the simulated data.
    nrofnewpoints : int
        Number of new points which are set if th error isnt changing any more
    beta : float, optional
        Amplification factor for the variance. Helps not to underestimate the variance. The default is 1.2.

    Returns
    -------
    gp :  Gaussian process
        Adapted gaussian process
    errorlist : list
        List of global estimated errors.
    costlist : list
        List of costs per design.

    """
    errorlist = []
    realerrorlist = []
    costlist = []
    dim = gp.getdata[2]

    counter = 0
    totaltime = 0
    totalFEM = 0
    nosolutioncounter = 0

    'Epsilon^2 at this points'
    epsilon = np.squeeze(gp.getaccuracy) 
    
    'Solver options'
    s = 2
    method = "SLSQP"

    currentcost = totalcompworkeps(epsilon)
    costlist.append(currentcost)
    
    print("\n")
    print("---------------------------------- Start optimization")
    print("Number of initial points:          "+str(N))
    print("Total budget:                      "+str(totalbudget))
    print("Desired tolerance:                 "+str(TOL))
    print("Lower accuracy bound:              "+str(loweraccuracybound))
    print("Number of adaptively added points: "+str(nrofnewpoints))
    print("\n")

    while currentcost < totalbudget:

        t0design = time.perf_counter()
        print("---------------------------------- Iteration / Design: {}".format(counter))
        print("Current number of points: {} ".format(N))
        print("Current number of candidate points: {} ".format(NC))
        print("Estimate derivatives")
        df = gp.predictderivative(gp.getX, True)
        print("Estimate constants")
        C1,C2 = estimateconstants(gp,df)
        print("  C1: {:10.2f}".format(C1))
        print("  C2: {:10.2f}".format(C2))
       

        'Calculate KXX once, since this matrix is not changing'
        X, yt, Nall = gp.getX, gp.gety, gp.getdata[0]
        hyperparameter = gp.gethyperparameter
        KXX = kernelmatrix(X, X, hyperparameter)
        
        'Turn epsilon^2 into v'
        v = epsilon**(-1)
        
        """ NEW FORMULATION STARTS HERE """
        
        globarerrorbefore = globalerrorestimate(v, X, yt, hyperparameter, KXX, C1, C2, Nall,
                                          SigmaLL, SigmaP, parameterranges)
        N = Nall-NC  # N - Core points
        
        print("Global error estimate before optimization:   {:1.8f}".format(globarerrorbefore))
        print("Computational cost before optimization:      {:0.0f}".format(currentcost))
        print("\n")
        print("--- Solve minimization problem")

        """ Set start values here. Since the candidate points are added with 1E20 to simulate them not beeing active
        and threfore not alter the error, we have to lower the accuracies to a reasonable startvalue """
        epssquaredinitial = 0.1 #eps^2
        epsilon[N:] = epssquaredinitial
        v[N:] = 1/epssquaredinitial
        
# =============================================================================
#         if counter == 0:
#             costlist.append(currentcost)
#             errorlist.append(globarerrorbefore)
#         else:
#             deltaeps = min(epsilon)*0.5
#             epsilon = epsilon - deltaeps
#             v = epsilon**(-1)
# =============================================================================
        
        currentcost = totalcompwork(v,s)

        total_n = 0 
        if method == "SLSQP":
            
            def lowerbound(v,vbar):
                #print("v-vbar: " + str(v-vbar))
                return v-vbar
            def lowerboundjac(v,vbar):
                Nv = v.shape[0]
                return np.eye((Nv))

            def compworkconstrain(v, currentcost, budgettospend, s=1):
                #print("Current solution: "+str(v))
                print("Current cost of sol. "+ str(np.sum(v**s)))
                print("Current Budget: "+str(currentcost+budgettospend))
                print("KKT: "+str(currentcost+budgettospend-np.sum(v**s)))
                print("\n")
                return currentcost+budgettospend-np.sum(v**s)
            def compworkconstrainjac(v, currentcost, budgettospend, s=1):
                return -s*v**(s-1)
            
            arguments = (currentcost, budgettospend,s,)
            argumentslowerbound = (v,)

            
# =============================================================================
#             tmp = np.concatenate((v.reshape((-1,1)),np.ones((Nall,1))*1E10),axis=1)
#             bounds = tuple(map(tuple, tmp))
#             
# =============================================================================
            con = [{'type': 'ineq',
                   'fun': compworkconstrain,
                    'jac': compworkconstrainjac,
                    'args': arguments},
                   {'type': 'ineq',
                    'fun': lowerbound,
                    'jac': lowerboundjac,
                    'args': argumentslowerbound}
                   ]
            
            """ Add little variation to the solution after the first design"""
            t0 = time.perf_counter()
            sol = scipy.optimize.minimize(globalerrorestimate, v,
                                          args=(X, yt, hyperparameter, KXX, C1, C2,
                                                Nall,  SigmaLL, SigmaP, parameterranges),
                                          method=method,
                                          jac=gradientofglobalerrorestimate,
                                          constraints=con, 
                                          options={'maxiter': sqpiter, 'ftol': sqptol, 'disp': False})
            t1 = time.perf_counter()
            total_n = t1-t0
            
        elif method == "trust-constr":
            
            'Bounds on v'
            upperbound = [np.inf]*Nall
            lowerbound = v.tolist()
            bounds = Bounds(lowerbound, upperbound)
            
            'Nonlinear constraints'
            def compworkconstrain(v,s):
                return np.array([np.sum(v**s)])
            def compworkconstrainjac(v,s):
                return np.array([s*v**(s-1)])
            def compworkconstrainhess(x,v):
                s  = 2
                return s*(s-1)*v[0]*np.diagflat(x**(s-2))    

            nonlinear_constraint = NonlinearConstraint(lambda x: compworkconstrain(x,s), -np.inf, currentcost+budgettospend, 
                                                       jac=lambda x: compworkconstrainjac(x,s), 
                                                       hess=compworkconstrainhess)
            t0 = time.perf_counter()
            sol = scipy.optimize.minimize(globalerrorestimate, v,
                                          args=(X, yt, hyperparameter, KXX, C1, C2,
                                                Nall,  SigmaLL, SigmaP, parameterranges),
                                          method=method,bounds = bounds,
                                          jac=gradientofglobalerrorestimate,
                                          constraints=[nonlinear_constraint],
                                          options={'verbose': 1})
            t1 = time.perf_counter()
            total_n = t1-t0

        totaltime += total_n
        
        nrofdeletedpoints = 0
        if sol.success == True:

            print("Solution found within {} iterations".format(sol.nit))
            print("Used time: {:0.2f} seconds".format(total_n))
            print("Last function value: {}".format(sol.fun))
           
            'Solution for v'
            vsol = sol.x 

            'Solution for epsilon'
            currentepsilonsol = vsol**(-1/2) 
            'Turn eps^2 to eps for comparing'
            epsilon = np.sqrt(epsilon)

            """ ---------- Block for adapting output (y) values ---------- """
            ' Check which point changed in its accuracy. Only if the change is significant a new simulation is done '
            ' since only then the outout value really changed. Otherwise the solution is just set as a new solution.'

            indicesofchangedpoints = np.where(np.abs(np.atleast_2d(epsilon-currentepsilonsol)) > TOLFEM)
            if indicesofchangedpoints[1].size == 0:
                print("\n")
                print("No sufficient change between the solutions.")
                print("Solution is set as new optimal design.")
                gp.addaccuracy(currentepsilonsol**2, [0, N+NC])
            else:
                print("\n")
                print(
                    "Sufficient change in the solutions is detected, optain new simulation values")
                print("for point(s): {}".format(indicesofchangedpoints[1]))

                t0FEM = time.perf_counter()
                print("\n")
                print("--- Start simulation block")
                for jj in range(indicesofchangedpoints[1].shape[0]):
                    currentFEMindex = indicesofchangedpoints[1][jj]
                    
                    ' Get new values for calcualted solution'
                    epsXtnew = currentepsilonsol[currentFEMindex].reshape((1, -1))
                    ytnew = fun(gp.getX[currentFEMindex, :]).reshape((1, -1))

                    ' Add new value to GP '
                    gp.addaccuracy(epsXtnew**2, currentFEMindex)
                    gp.adddatapointvalue(ytnew, currentFEMindex)

                t1FEM = time.perf_counter()
                totalFEM = t1FEM-t0FEM
                print("Simulation block done within: {:1.4f} s".format(totalFEM))

            ' Filter all points which seemingly have no influence on the global parameter error '
            ' We preemptively filter the list beginning at N since we dont change core points'

            #epsilon = np.squeeze(epsilon)
            indicesofchangedpoints = np.where(np.abs(np.atleast_2d(np.abs(vsol[N:]-v[N:])) < 10))
            print("\n")
            print("--- Point filtering")

            if indicesofchangedpoints[1].size != 0:
                nrofdeletedpoints = indicesofchangedpoints[1].size
                nrofdeletedpoints = nrofnewpoints
                print("Points of no work are detected.")
                print("Delete points with index: {}".format(indicesofchangedpoints[1]+(N)))
                print("Add {} new candidate points.".format(nrofdeletedpoints))

                ' Problem - the indices do not correspond to the acutal indices within the GP anymore'
                idx = indicesofchangedpoints[1]+(N)
                gp.deletedatapoint(idx)

                ' Delete points from solution vector'
                epsilon = np.delete(epsilon, idx)
                currentepsilonsol = np.delete(currentepsilonsol, idx)
                'Core candiates need to be added as gradient info with high error'
                # Adapt only NC , since initial data is not deleted
                NC -= idx.shape[0]
                print("Set {} as core points.".format(NC))
                if gp.getXgrad is not None:
                    Xcgrad = gp.getX[N:N+NC]
                    epsXcgrad = 1E10 * np.ones((1, Xcgrad.shape[0]*dim))  # eps**2
                    dfXcgrad = gp.predictderivative(Xcgrad)
                    gp.addgradientdatapoint(Xcgrad)
                    gp.adddgradientdatapointvalue(dfXcgrad)
                    gp.addgradaccuracy(epsXcgrad)
                N += NC  # Candidates go core
                NC = 0
            else:
                print(
                    "All points may have sufficient influence on the global parameter error.")
                print("Transfer all candidate points to core points.")
                print("Set {} as core points.".format(NC))
                nrofdeletedpoints = nrofnewpoints
                print("Add {} new candidate points.".format(nrofdeletedpoints))
                if gp.getXgrad is not None and NC > 0:
                    Xcgrad = gp.getX[N:N+NC]
                    epsXcgrad = 1E10 * \
                        np.ones((1, Xcgrad.shape[0]*dim))  # eps**2
                    dfXcgrad = gp.predictderivative(Xcgrad)
                    gp.addgradientdatapoint(Xcgrad)
                    gp.adddgradientdatapointvalue(dfXcgrad)
                    gp.addgradaccuracy(epsXcgrad)
                N += NC
                NC = 0

            ' Global parameter estimate after optimisation and filtering.'

            'Calculate KXX once, since this this matrix is not changing, ugly but speeds up'
            # All data inluding candiate points
            X, yt, Nall = gp.getX, gp.gety, gp.getdata[0]
            KXX = kernelmatrix(X, X, hyperparameter)

            'Transform solution back from epsilon to v'
            vsol = currentepsilonsol**(-2)
            globalerrorafter = globalerrorestimate(vsol, X, yt, hyperparameter, KXX, C1, C2, Nall,
                                                   SigmaLL, SigmaP, parameterranges)
            errorlist.append(globalerrorafter)

            ' Set new cummulated cost '
            currentcost = totalcompwork(vsol,s)
            costlist.append(currentcost)

            if globalerrorafter < TOL:
                print("\n")
                print("--- Convergence ---")
                print("Desired tolerance is reached, adaptive phase is done.")
                print("Final error estimate: {:1.8f}".format(globalerrorafter))
                print("Total used time: {:0.2f} seconds".format(totaltime+totalFEM))
                # errorlist.append(globalerrorafter)
                print("Save everything !")
                gp.savedata(execpath+'/saved_data')
                return gp, errorlist, costlist, realerrorlist

            ' Add new candidate points when points are deleted '
            if nrofdeletedpoints > 0:
                NC = nrofdeletedpoints
                XC = createPD(NC, dim, "random", parameterranges)
                epsXc = 1E10*np.ones((1, XC.shape[0]))  # eps**2
                meanXc = gp.predictmean(XC)
                gp.adddatapoint(XC)
                gp.adddatapointvalue(meanXc)
                gp.addaccuracy(epsXc)

# =============================================================================
#             'Plot for debug'
#             fig, axs = plt.subplots(1, 1)
#             axs.scatter(X[0:N, 0], X[0:N, 1], zorder=2)  # Initial points
#             axs.scatter(XC[:, 0], XC[:, 1], color='red', marker='x', zorder=2)  # Added points
#             fig.savefig(execpath+'/'+str(counter)+'_'+'.png')
# =============================================================================

            """ ---------------------------------------------------------- """
            print("\n")
            print("--- New parameters")
            print("Error estimate after optimization and filtering: {:1.8f}".format(globalerrorafter))
            print("Computational cost after optimization:           {:0.0f}".format(currentcost))
            print("New budget for next design:                      {:0.0f}".format(currentcost+budgettospend))
            print("Used time:                                       {:0.2f} seconds".format(total_n))
            print("\n")

            if np.linalg.norm(globalerrorafter-globarerrorbefore) < TOLe:
                print("No change in error estimate detected.")
                if currentcost < totalbudget:

                    ' Get maximum error, calcualte comp work to be at x percent of max error '
                    variance = gp.predictvariance(gp.getX)
                    maxpwe = maxpointwiseerror(
                        C1, C2, df, SigmaLL, SigmaP, variance, np.inf)

                    ebeta = beta*maxpwe
                    deltaw = totalcompwork(ebeta,s)
                    budgettospend += deltaw
                    N += NC  # All candidates go core
                    NC = 0  # There are no furhter candidate points

                    print("Adjust budget to spend")
                    print("  Maximum error estiamte: {:1.8f}".format(maxpwe[0]))
                    print("  New \u0394W: {:0.1f}".format(deltaw))
                    print("  New budget to spend: {:0.1f}".format(budgettospend))
                    print("\n")
                else:
                    print("  Budget is not sufficient, stopt adaptive phase.")
                    print("\n")
                    print("Save everything...")
                    gp.savedata(execpath+'/saved_data')
                    return gp, errorlist, costlist
        else:

            ' Set new start value for next design '
            vsol = sol.x #v

            'Calculate KXX once, since this thisd matrix is not changing, ugly but speeds up'
            X = gp.getX
            yt = gp.gety
            Nall = gp.getdata[0]  # All data inluding candiate points
            hyperparameter = gp.gethyperparameter
            KXX = kernelmatrix(X, X, hyperparameter)
            globalerrorafter = globalerrorestimate(vsol, X, yt, hyperparameter, KXX, C1, C2, Nall,
                                                   SigmaLL, SigmaP, parameterranges)
            ' Set new cummulated cost '
            currentcost = totalcompwork(vsol,s)
            costlist.append(currentcost)

            if globalerrorafter < TOL:
                print("\n")
                print("Desired tolerance is still reached, adaptive phase is done.")
                print(" Final error estimate: {:1.6f}".format(globalerrorafter))
                print(" Total used time: {:0.4f} seconds".format(totaltime))
                gp.addaccuracy(vsol**(-1), [0, None])
                errorlist.append(globalerrorafter)
                print("Save everything...")
                gp.savedata(execpath+'/saved_data')
                return gp, errorlist, costlist

            print("\n")
            print("No solution found.")
            print(" " + sol.message)
            print("Total used time: {:0.4f} seconds".format(totaltime))

            ' Get maximum error, calcualte comp work to be at x percent of max error '
            variance = gp.predictvariance(gp.getX)
            maxpwe = maxpointwiseerror(C1, C2, df, SigmaLL, SigmaP, variance, np.inf)

            ebeta = beta*maxpwe
            deltaw = totalcompwork(ebeta,s)
            #budgettospend += deltaw
            budgettospend += 1E5
            N += NC  # All candidates go core
            NC = 0  # There are no furhter candidate points

            print("Adjust budget to spend")
            print("  Maximum error estiamte: {:1.7f}".format(maxpwe[0]))
            print("  New \u0394W: {:0.4f}".format(deltaw))
            print("  New budget to spend: {:0.4f}".format(budgettospend))
            print("\n")
            
        counter += 1
        epsilon = vsol**(-1/2) #epsilon

        if nrofdeletedpoints > 0:
            'Only added, when a solution was found.'
            epsilon = np.concatenate((epsilon, np.squeeze(epsXc)))
        t1design = time.perf_counter()
        epsilon = epsilon**2
        

        
        print("Time used for complete design iteration: {:0.2f} seconds".format(t1design-t0design))
        print("\n")
