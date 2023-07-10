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

from gpr.gaussianprocess import *

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


def pointwiseerror(C1, C2, df, SigmaLL, SigmaP, variance, norm, scale=1.0):

    dim = df.shape[0]

    invsigma = np.linalg.inv(SigmaLL)
    invnorm = np.linalg.norm(invsigma, norm)

    L = np.min(np.linalg.eig(np.outer(df, invsigma*df) +
                             np.linalg.inv(SigmaP)+1E-4*np.eye(dim))[0])  # float

    if isinstance(L, complex):
        L = L.real

    if L <= 1E-4:
        print("Warning: L is too small, set value to: "+str(1))
        L = 1
# =============================================================================
#     print("Estimate")
#     print(str(12*C1*invnorm/L) +" "+ str(L))
#     print("C2: "+ str(C2))
# =============================================================================
    
    pointwiseerror = (12*C1*invnorm/L + 1/C2) * scale * variance

    #print(variance)
# =============================================================================
#     print("Pre 1: "+str( 12*C1*invnorm/L))
#     print("Pre 2: "+str(1/C2))
#     print("Pre 1 mult: "+str(12*C1*invnorm/L*variance))
#     print("Pointwise: "+str(pointwiseerror))
# =============================================================================
    #print("\n")

    return pointwiseerror

def realpointwiseerror(C1, C2, df, SigmaLL, SigmaP, variance,dferror, norm):

    dim = df.shape[0]

    invsigma = np.linalg.inv(SigmaLL)
    invnorm = np.linalg.norm(invsigma, norm)

    L = np.min(np.linalg.eig(np.outer(df, invsigma*df) +
                             np.linalg.inv(SigmaP)+1E-3*np.eye(dim))[0])  # float

    if isinstance(L, complex):
        L = L.real

    if L <= 1E-4:
        print("Warning: L is too small, set value to: "+str(1))
        L = 1
# =============================================================================
#     print("Estimate")
#     print(str(12*C1*invnorm/L) +" "+ str(L))
#     print("C2: "+ str(C2))
# =============================================================================
    
    pointwiseerror = 12*C1*invnorm/L * variance + 1/C2 * dferror

    #print(variance)
# =============================================================================
#     print("Pre 1: "+str( 12*C1*invnorm/L))
#     print("Pre 2: "+str(1/C2))
#     print("Pre 1 mult: "+str(12*C1*invnorm/L*variance))
#     print("Pointwise: "+str(pointwiseerror))
# =============================================================================
    #print("\n")

    return pointwiseerror


def globalerrorestimate(epsilon, gp, C1, C2, NC, SigmaLL, SigmaP, ranges, scale=1.0):

    # Recast
    epsilon = epsilon.reshape((1, -1))

    X = gp.getX  # All data , including the candidate points  !!
    N = gp.getdata[0]

    # 1.2 Add reasonable error to candidate points (start value)
    epscandiate = (epsilon[0, (N-NC):].reshape((1, -1)))**2
    gp.addaccuracy(epscandiate, [(N-NC), None])

    # 1.3 Adapt the accuracy
    epsXt = (epsilon[0, 0:(N-NC)].reshape((1, -1)))**2
    gp.addaccuracy(epsXt, [0, (N-NC)])

    # 2. MC integrate everything
    volofparameterspace = np.prod(ranges[:, 1])

    errorsum = 0

    df = gp.predictderivative(X, True)
    variance = gp.predictvariance(X)
    #print("variance: "+str(variance))

    for i in range(N):
        pwee = pointwiseerror(
            C1, C2, df[i, :], SigmaLL, SigmaP, variance[i, :], np.inf, scale)
        
        #print(pwee)
        errorsum += pwee**2
    #print("estimate errorsum:" + str(errorsum))
    globalerrorestimate = (volofparameterspace/N) * \
        errorsum  # Squared ! Thats why no root
    #print("estimate: " + str(globalerrorestimate))
    return globalerrorestimate # 14.4


def realglobalerrorestimate(epsilon, gp, fun, dfun, C1, C2, NC, SigmaLL, SigmaP, ranges, scale=1.0):

    
    # Recast
    epsilon = epsilon.reshape((1, -1))

    X = gp.getX  # All data , including the candidate points  !!
    N = gp.getdata[0]
    dim = 2
    
    # 1.2 Add reasonable error to candidate points (start value)
    epscandiate = (epsilon[0, (N-NC):].reshape((1, -1)))**2
    gp.addaccuracy(epscandiate, [(N-NC), None])

    # 1.3 Adapt the accuracy
    epsXt = (epsilon[0, 0:(N-NC)].reshape((1, -1)))**2
    gp.addaccuracy(epsXt, [0, (N-NC)])
    
    # 2. MC integrate everything
    volofparameterspace = np.prod(ranges[:, 1])

    errorsum = 0

    dfreal = dfun(X)
    dfgp = gp.predictderivative(X,True)
    
    eps = np.abs(gp.predictmean(X)-fun(X).reshape((-1,1)))
   # print("eps: "+str(eps))
    epsdf = np.linalg.norm(dfgp-dfreal,np.inf,axis=1)
    
    invsigma = np.linalg.inv(SigmaLL)
    invnorm = np.linalg.norm(invsigma, np.inf)

    for i in range(N):
    
        L = np.min(np.linalg.eig(np.outer(dfreal[i, :].reshape((2)), invsigma*dfreal[i, :].reshape((2))) + np.linalg.inv(SigmaP)+1E-3*np.eye(dim))[0])  # float
        if isinstance(L, complex):
            L = L.real
        if L <= 1E-4:
            print("Warning: L is too small, set value to: "+str(1))
            L = 1
        pointwiseerror = 12*C1*invnorm/L*eps[i,:] + 1/C2*epsdf[i]
        errorsum += pointwiseerror**2

    globalerrorestimate = (volofparameterspace/N) * errorsum 
    return globalerrorestimate


def gradientofglobalerrorestimate(epsilon, gp, C1, C2, NC, SigmaLL, SigmaP, ranges, scale=1.0):

    # Recast
    epsilon = epsilon.reshape((1, -1))

    X = gp.getX
    N = gp.getdata[0]
    Ngrad = gp.getdata[1]
    dim = gp.getdata[2]

    # 1.2 Add reasonable error to candidate points (start value)
    epscandiate = (epsilon[0, (N-NC):].reshape((1, -1)))**2
    gp.addaccuracy(epscandiate, [(N-NC), None])

    # 1.3 Adapt the accuracy
    epsXt = (epsilon[0, 0:(N-NC)].reshape((1, -1)))**2
    gp.addaccuracy(epsXt, [0, (N-NC)])

    # 2. MC integrate everything
    volofparameterspace = np.prod(ranges[:, 1])

    tmpgradsum = 0
    invsigma = np.linalg.inv(SigmaLL)
    invnorm = np.linalg.norm(invsigma, np.inf)

    df = gp.predictderivative(X, True)
    variance = gp.predictvariance(X)

    hyperparameter = gp.gethyperparameter
    accuracies = gp.getaccuracy

    # Calculate KXX once !
    if gp.getXgrad is None:

        matricestest = kernelmatrices(
            X, X, hyperparameter, accuracies)
        KXX = matricestest[2]
        tmp = KXX - np.diagflat(epsilon**2)
        invK = np.linalg.inv(KXX)

        matsize = N

    else:
        matrices = kernelmatricesgrad(X, X, gp.getXgrad, hyperparameter,
                                      accuracies, gp.getgradientaccuracy)
        K = matrices[5]
        tmp = K - np.diagflat(np.concatenate((epsilon**2,
                              np.zeros((1, (N-NC)*dim))), axis=1))
        invK = np.linalg.inv(K)

        matsize = N+Ngrad*dim

    # For every point within X
    for i in range(N):

        pwee = pointwiseerror(
            C1, C2, df[i, :], SigmaLL, SigmaP, variance[i, :], np.inf, scale)

        """ For every point , calculate the derivative of the variance in respect to epsilon \in R^n """
        L = np.min(np.linalg.eig(np.outer(df[i, :], invsigma*df[i, :]) +
                   np.linalg.inv(SigmaP)+1E-7*np.eye(dim))[0])  # float

        if isinstance(L, complex):
            L = L.real

        if L <= 1E-4:
            print("Warning: L is too small, set value to: "+str(1))
            L = 1

        KR = 12/L*invnorm*C1 + (1/C2)

        ' All necessary kernel matrices '
        dvardeps = np.zeros((N, 1))

        KxX = tmp[i, :].reshape((1, -1))
        # Inner loop for calculating d/d(eps1,...epsN) var(eps1,...,epsN)|p_p'
        for kk in range(N):
            ' Derivative dE / deps'
            dEdepsi = np.zeros((matsize, matsize))
            dEdepsi[kk, kk] = 2*(epsilon[0, kk])
            ' Derivative dsigma/deps'
            dvardeps[kk, 0] = KxX @ invK @ dEdepsi @ invK @ KxX.T

        tmpgradsum += pwee * KR * scale * dvardeps

    gradient = 2 * volofparameterspace/N * tmpgradsum

    return gradient


def estimateconstants(gp):
    X = gp.getX
    C2 = 0
    for ii in range(X.shape[0]):
        hess = gp.predicthessian(np.array([X[ii, :]]))
        maxhess = np.linalg.norm(hess, np.inf)
        if maxhess > C2:
            C2 = maxhess
    return C2


def calculaterealerror():
    pass

def totalcompwork(epsilon):
    return np.sum(1/2*epsilon**(-2))

def budgetconstrain(x, budget):
    return budget-np.sum(1/2*x**(-2))

def budgetconstrainjac(x, budget):
    return x**(-3)

def compworkconstrain(x, currentcost, budgettospend):
    return currentcost+budgettospend-np.sum(1/2*x**(-2))

def compworkconstrainjac(x, currentcost, budgettospend):
    return x**(-3)

def fun(x):
        return np.sin(x[0])+np.cos(x[0]*x[1])
  
def funerror(x):
        return np.array([np.sin(x[:,0])+np.cos(x[:,0]*x[:,1])])

        
def dfun(x):
    
    if x.shape[0]>1:
        return np.array([np.cos(x[:,0])-x[:,1]*np.sin(x[:,0]*x[:,1]),-x[:,0]*np.sin(x[:,0]*x[:,1])]).T
    else:
        return np.array([np.cos(x[0])-x[1]*np.sin(x[0]*x[1])],
                        [-x[0]*np.sin(x[0]*x[1])])


def adapt(gp, N: int, NC, totalbudget, budgettospend,
          SigmaLL, SigmaP, parameterranges,
          TOL, TOLe, TOLFEM, loweraccuracybound, nrofnewpoints, execpath, execname,
          beta=1.4, sqpiter=800, sqptol=1E-5):
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

    epsilon = np.squeeze(np.sqrt(gp.getaccuracy)) # Square root since we minimize epsilon. Squeeze to keep data format consistent.


    currentcost = totalcompwork(epsilon)
    costlist.append(currentcost)
    print("\n")
    print("---------------------------------- Start optimization")
    print("Number of initial points:      "+str(N))
    print("Total budget:                  "+str(totalbudget))
    print("Desired tolerance:             "+str(TOL))
    print("Lower accuracy bound:          "+str(loweraccuracybound))
    print("Number of adaptive new points: "+str(nrofnewpoints))
    print("\n")

    while currentcost < totalbudget:

        '1. Predict constants C1 / C2. If counter > 0 those are predicted with new information !'
        print("---------------------------------- Iteration / Design: {}".format(counter))
        print("Current number of points: {} ".format(N))
        print("Estimate derivatives")
        df = gp.predictderivative(gp.getX, True)
        print("Calculate constants")
        C1 = np.linalg.norm(df, np.inf)  # float
        C2 = estimateconstants(gp)
        print("  C1: {:10.2f}".format(C1))
        print("  C2: {:10.2f}".format(C2))
        scale = 1.0

        globarerror = globalerrorestimate(epsilon, gp, C1, C2, NC, SigmaLL, SigmaP, parameterranges)
        #rgee    = realglobalerrorestimate(epsilon, gp, funerror, dfun, C1, C2, NC, SigmaLL, SigmaP, parameterranges)
        
        if counter == 0:
            errorlist.append(globarerror[0])
            #realerrorlist.append(rgee[0])
        print("Current error estimate E(D(e)): {:1.8f}".format(globarerror[0]))
        print("Current computational cost:     {:0.0f}".format(currentcost))

        """ The bounds have to be adapted to the current data errors at x """
        if totalcompwork(loweraccuracybound*np.ones((1, N+NC))) > totalbudget:
            print("Allowed lower bound exceeds comp. budget.")
            print("\n")
            return gp, errorlist, costlist
        else:
            tmp = np.concatenate(
                (loweraccuracybound*np.ones((1, N+NC)), epsilon.reshape((1, -1)))).T
            bounds = tuple(map(tuple, tmp))

        """ ---------- Minimization block ---------- """
        arguments = (currentcost, budgettospend,)
        argumentsbudget = (totalbudget,)
        con = [{'type': 'ineq',
               'fun': compworkconstrain,
                'jac': compworkconstrainjac,
                'args': arguments},
               {'type': 'ineq',
                'fun': budgetconstrain,
                'jac': budgetconstrainjac,
                'args': argumentsbudget}
               ]
        print("\n")
        print("--- Solve minimization problem")
        t0 = time.perf_counter()
        sol = scipy.optimize.minimize(globalerrorestimate, epsilon,
                                      args=(gp, C1, C2, NC, SigmaLL,
                                            SigmaP, parameterranges, scale),
                                      method='SLSQP',
                                      jac=gradientofglobalerrorestimate,
                                      bounds=bounds,
                                      constraints=con,
                                      options={'maxiter': sqpiter, 'ftol': sqptol, 'disp': False})
        t1 = time.perf_counter()
        total_n = t1-t0
        totaltime += total_n


        nrofdeletedpoints = 0
        if sol.success == True:

            print("Solution found within {} iterations".format(sol.nit))
            ' Set new start value for next design '
            currentsol = sol.x

            if currentcost > totalbudget:
                print("Something went wront. Too much budget spend for this solution.")
                print("Save everything...")
                print("\n")
                gp.savedata(execpath+'/saved_data')
                return gp, errorlist, costlist

            """ ---------- Block for adapting output (y) values ---------- """
            ' Check which point changed in its accuracy. Only if the change is significant a new simulation is done '
            ' since only then the outout value really changed. Otherwise the solution is just set as a new solution.'
            indicesofchangedpoints = np.where(
                np.abs(np.atleast_2d(epsilon-currentsol)) > TOLFEM)
            if indicesofchangedpoints[1].size == 0:
                print("\n")
                print("No sufficient change between the solutions.")
                print("Solution is set as new optimal design.")
                gp.addaccuracy(currentsol**2, [0, N+NC])
            else:
                print("\n")
                print("Sufficient change in the solutions is detected, optain new simulation values")
                print("for point(s): {}".format(indicesofchangedpoints[1]))

                t0FEM = time.perf_counter()
                print("\n")
                print("--- Start simulation block")
                for jj in range(indicesofchangedpoints[1].shape[0]):
                    currentFEMindex = indicesofchangedpoints[1][jj]
# =============================================================================
#                     parameter = {"--x1": gp.getX[currentFEMindex,0],
#                                  "--x2": gp.getX[currentFEMindex,1],
#                                  "--eps": currentsol[currentFEMindex]}
#                     runkaskade(execpath, execname, parameter)
#                     print("\n")
#                     'Read simulation data and get function value'
#                     simulationdata = readtodict(execpath, "dump.log")
#                     reached = np.asarray(simulationdata["flag"])
#                     epsXtnew = np.asarray(simulationdata["accuracy"])
# =============================================================================
                    epsXtnew = currentsol[currentFEMindex].reshape((1,-1))
                    ytnew = fun(gp.getX[currentFEMindex, :]).reshape((1,-1))

                    ' Add new value to GP '
                    gp.addaccuracy(epsXtnew**2, currentFEMindex)
                    gp.adddatapointvalue(ytnew, currentFEMindex)

                t1FEM = time.perf_counter()
                totalFEM = t1FEM-t0FEM
                print(
                    "Simulation block done within: {:1.4f} s".format(totalFEM))

            ' Filter all points which seemingly has no influence on the global parameter error '
            ' We preemptively filter the list beginning at N '

            TOLFILTER = 1E-5
            epsilon = np.squeeze(epsilon)
            indicesofchangedpoints = np.where(
                np.abs(np.atleast_2d(epsilon[N:]-currentsol[N:])) < TOLFILTER)
            print("\n")
            print("--- Point filtering")
            
            if indicesofchangedpoints[1].size != 0:
                nrofdeletedpoints = indicesofchangedpoints[1].size

                print("Points of no work are detected.")
                print("Delete points with index: {}".format(indicesofchangedpoints[1]+(N)))
                print("Add {} new candidate points.".format(nrofdeletedpoints))

                ' Problem - the indices do not correspond to the acutal indices within the GP anymore'
                idx = indicesofchangedpoints[1]+(N)
                gp.deletedatapoint(idx)

                ' Delete points from solution vector'
                epsilon = np.delete(epsilon, idx)
                currentsol = np.delete(currentsol, idx)
                NC -= idx.shape[0] # Adapt only NC , since initial data is not deleted
                N += NC  # Candidates go core
            else:
                print("All points may have sufficient influence on the global parameter error.")

            ' Global parameter estimate after optimisation and filtering.'
            newerrorestimate = np.sqrt(globalerrorestimate(currentsol, gp, C1, C2, NC, SigmaLL, SigmaP, parameterranges)[0]) # 19.4 - 13:14
            errorlist.append(newerrorestimate)
            
            #newrealerrorestimate = realglobalerrorestimate(currentsol,gp,funerror, dfun, C1, C2, NC, SigmaLL, SigmaP, parameterranges)[0]
            #realerrorlist.append(newrealerrorestimate)

            ' Set new cummulated cost '
            currentcost = totalcompwork(currentsol)
            costlist.append(currentcost)

            if newerrorestimate < TOL:
                print("\n")
                print("--- Convergence ---")
                print("Desired tolerance is reached, adaptive phase is done.")
                print("Final error estimate: {:1.6f}".format(
                    newerrorestimate))
                print("Total used time: {:0.2f} seconds".format(
                    totaltime+totalFEM))
                #errorlist.append(newerrorestimate)
                print("Save everything !")
                gp.savedata(execpath+'/saved_data')
                return gp, errorlist, costlist,realerrorlist

            ' Add new candidate points when points are deleted '
            if nrofdeletedpoints > 0:
                NC = nrofdeletedpoints
                XC = createPD(NC, dim, "sobol", parameterranges)
                epsXc = 1E-1*np.ones((1, XC.shape[0]))  # eps**2
                meanXc = gp.predictmean(XC)
                gp.adddatapoint(XC)
                gp.adddatapointvalue(meanXc)
                gp.addaccuracy(epsXc)


            """ ---------------------------------------------------------- """
            print("\n")
            print("--- New parameters")
            print("Error estimate after optimization and filtering: {:1.5f}".format(newerrorestimate))
            print("Current computational cost of found solution:    {:0.0f}".format(currentcost))
            print("New budget for next design:                      {:0.0f}".format(currentcost+budgettospend))
            print("Used time:                                       {:0.2f} seconds".format(total_n))
            print("\n")

            if np.linalg.norm(newerrorestimate-globarerror) < TOLe:
                print("---No change in error estimate detected.")
                if currentcost < totalbudget:
                    nosolutioncounter += 1
                    print("-No solution counter: {}".format(nosolutioncounter))
                    if nosolutioncounter == 5:
                        print("-Add {} new points".format(nrofnewpoints))
                        print(
                            "-Add additional {:1.0f} to budget".format(budgettospend*0.1))
                        print(
                            "-New budget for next design: {}".format(currentcost+budgettospend))
                        print("\n")
                        Xnew = createPD(nrofnewpoints, dim,
                                        "sobol", parameterranges)
                        epsXc = 0.1*np.ones((1, Xnew.shape[0]))
                        meanXc = gp.predictmean(Xnew)

                        gp.adddatapoint(Xnew)
                        gp.adddatapointvalue(meanXc)
                        gp.addaccuracy(epsXc)
                        gp.updateK()
                        #gp.optimizehyperparameter(region, "mean", False)
                        print("\n")

                        NC += nrofnewpoints

                        if nrofnewpoints == 1:
                            currentsol = np.concatenate((np.squeeze(currentsol), np.squeeze(epsXc).reshape(1)))
                        else:
                            currentsol = np.concatenate((np.squeeze(currentsol), np.squeeze(epsXc)))

                        nosolutioncounter = 0

                    else:
                        ' Get maximum error, calcualte comp work to be at x percent of max error '
                        variance = gp.predictvariance(gp.getX)
                        maxpwe = maxpointwiseerror(C1, C2, df, SigmaLL, SigmaP, variance, np.inf)

                        ebeta = maxpwe*beta
                        deltaw = totalcompwork(ebeta)
                        budgettospend += deltaw

                        print("---Adjust budget to spend")
                        print("-Maximum error estiamte: {:1.7f}".format(maxpwe[0]))
                        print("-New \u0394W: {:0.4f}".format(deltaw))
                        print("-New budget to spend: {:0.4f}".format(budgettospend))
                        print("\n")
                else:
                    print("  Budget is not sufficient, stopt adaptive phase.")
                    print("\n")
                    print("Save everything...")
                    gp.savedata(execpath+'/saved_data')
                    return gp, errorlist, costlist

        else:

            ' Set new start value for next design '
            currentsol = sol.x

            ' Set new cummulated cost '
            currentcost = totalcompwork(currentsol)
            costlist.append(currentcost)

            newerrorestimate = globalerrorestimate(
                currentsol, gp, C1, C2, NC, SigmaLL, SigmaP, parameterranges)[0]

            if newerrorestimate < TOL:
                print("\n")
                print("Desired tolerance is still reached, adaptive phase is done.")
                print(" Final error estimate: {:1.6f}".format(newerrorestimate))
                print(" Total used time: {:0.4f} seconds".format(totaltime))
                gp.addaccuracy(currentsol**2, [0, None])
                errorlist.append(newerrorestimate)
                print("Save everything...")
                gp.savedata(execpath+'/saved_data')
                return gp, errorlist, costlist

            print("---No solution found.")
            print("Total used time: {:0.4f} seconds".format(totaltime))

# =============================================================================
#             print(" Add {} new points".format(nrofnewpoints))
#             print("\n")
#
#             Xnew = createPD(nrofnewpoints, dim, "sobol", parameterranges)
#             epsXc = 0.1*np.ones((1, Xnew.shape[0]))
#             meanXc = gp.predictmean(Xnew)
#
#             gp.adddatapoint(Xnew)
#             gp.adddatapointvalue(meanXc)
#             gp.addaccuracy(epsXc)
#             gp.updateK()
#             #gp.optimizehyperparameter(region, "mean", False)
#             NC += nrofnewpoints
#
#             if nrofnewpoints == 1:
#                 currentsol = np.concatenate((np.squeeze(currentsol), np.squeeze(epsXc).reshape(1)))
#             else:
#                 currentsol = np.concatenate((np.squeeze(currentsol), np.squeeze(epsXc)))
# =============================================================================

            ' Get maximum error, calcualte comp work to be at x percent of max error '
            variance = gp.predictvariance(gp.getX)
            maxpwe = maxpointwiseerror(
                C1, C2, df, SigmaLL, SigmaP, variance, np.inf)

            ebeta = beta*maxpwe
            deltaw = totalcompwork(ebeta)
            budgettospend += deltaw
            #nrofdeletedpoints = nrofnewpoints

            print("-Adjust budget to spend")
            print("--Maximum error estiamte: {:1.7f}".format(maxpwe[0]))
            print("--New \u0394W: {:0.4f}".format(deltaw))
            print("--New budget to spend: {:0.4f}".format(budgettospend))
            print("\n")

        counter += 1
        epsilon = currentsol

        if nrofdeletedpoints > 0:
            if epsXc.shape[1] == 1:
                epsilon = np.concatenate(
                    (epsilon, np.sqrt(epsXc).reshape((1))))
            else:
                epsilon = np.concatenate((epsilon, np.sqrt(np.squeeze(epsXc))))
