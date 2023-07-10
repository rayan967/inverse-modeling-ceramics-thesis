import numpy as np

import copy
import scipy
import scipy.optimize as optimize
import time
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from basicfunctions.covariance.cov import *
from basicfunctions.utils.creategrid import *
from gpr.gaussianprocess import *

np.random.seed()


def maxpointwiseerror(C1, C2, df, SigmaLL, SigmaP, variance, norm, scale = 1.0):

    invsigma = np.linalg.inv(SigmaLL)
    invnorm = np.linalg.norm(invsigma, norm)
    N = df.shape[0]
    currentmax = 0
    dim = df.shape[1]
    
    for i in range(N):
        L = np.min(np.linalg.eig( np.outer(df[i,:],invsigma*df[i,:]  ) +
                   np.linalg.inv(SigmaP)+1E-7*np.eye(dim))[0])  # float
        
        if isinstance(L, complex):
            L = L.real
    
        pointwiseerror = (12*C1*invnorm/L + 1/C2) * scale * variance[i,:] 
        if pointwiseerror > currentmax:
            currentmax = pointwiseerror

    return currentmax

def pointwiseerror(C1, C2, df, SigmaLL, SigmaP, variance, norm, scale = 1.0):

    dim = df.shape[0]
    
    invsigma = np.linalg.inv(SigmaLL)
    invnorm = np.linalg.norm(invsigma, norm)
    
    L = np.min(np.linalg.eig( np.outer(df,invsigma*df ) +
                   np.linalg.inv(SigmaP)+1E-3*np.eye(dim))[0])  # float
    
    if isinstance(L, complex):
        L = L.real

    pointwiseerror = (12*C1*invnorm/L + 1/C2) * scale * variance

    return pointwiseerror


def globalerrorestimate(epsilon, gp, C1, C2, NC, SigmaLL, SigmaP, ranges, scale = 1.0):
    """
    Monte Carlo approximation

    Parameters
    ----------
    epsilon : TYPE
        DESCRIPTION.
    gp : TYPE
        DESCRIPTION.
    NC : TYPE
        DESCRIPTION.
    SigmaLL : TYPE
        DESCRIPTION.
    SigmaP : TYPE
        DESCRIPTION.
    ranges : TYPE
        DESCRIPTION.

    Returns
    -------
    globalerrorestimate : TYPE
        DESCRIPTION.

    """

    # Recast
    epsilon = epsilon.reshape((1, -1))

    X = gp.getX #All data , including the candidate points  !!
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
    
    for i in range(N):
        pwee = pointwiseerror(C1, C2, df[i,:], SigmaLL, SigmaP, variance[i,:], np.inf,scale)
        errorsum += pwee**2

    globalerrorestimate = (volofparameterspace/N) * \
        errorsum  # Squared ! Thats why no root

    return globalerrorestimate

def gradientofglobalerrorestimate(epsilon, gp, C1, C2, NC, SigmaLL, SigmaP, ranges, scale = 1.0):

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

    #Calculate KXX once !
    if gp.getXgrad is None:

        matricestest = kernelmatrices(
            X,X,hyperparameter, accuracies)
        KXX =  matricestest[2]
        tmp = KXX - np.diagflat(epsilon**2)
        invK = np.linalg.inv(KXX)

        matsize = N 

    else:
        matrices = kernelmatricesgrad(X, X, gp.getXgrad, hyperparameter,
                                      accuracies, gp.getgradientaccuracy)
        K = matrices[5]
        tmp = K - np.diagflat(np.concatenate((epsilon**2,np.zeros((1,(N-NC)*dim))),axis=1))
        invK = np.linalg.inv(K)

        matsize = N+Ngrad*dim

    # For every point within X
    for i in range(N):
        
        #x = X[i, :].reshape((1, -1))
        # 12*C1*invnorm/L+ 1/C2 * variance
        pwee = pointwiseerror(C1, C2, df[i,:], SigmaLL, SigmaP, variance[i,:], np.inf, scale)

        """ For every point , calculate the derivative of the variance in respect to epsilon \in R^n """
        L = np.min(np.linalg.eig( np.outer(df[i,:],invsigma*df[i,:] ) +
                   np.linalg.inv(SigmaP)+1E-7*np.eye(dim))[0])  # float
        
        if isinstance(L, complex):
            L = L.real
        
        KR = 12/L*invnorm*C1 + (1/C2)

        ' All necessary kernel matrices '
        dvardeps = np.zeros((N, 1))
            
        KxX = tmp[i,:].reshape((1,-1))
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

def totalcompwork(epsilon):
    return np.sum(1/2*epsilon**(-2))
def budgetconstrain(x,budget):
    return budget-np.sum(1/2*x**(-2))
def budgetconstrainjac(x, budget):
    return x**(-3)
def compworkconstrain(x, currentcost, budgettospend):
    return currentcost+budgettospend-np.sum(1/2*x**(-2))
def compworkconstrainjac(x, currentcost, budgettospend):
    return x**(-3)
def fun(x):
    #return np.sin(x[:,0]) + np.cos(3* x[:,1] * x[:,0]) *np.cos(x[:,0])
    #return np.sin(x[:,0]) + np.cos(x[:,1] * x[:,0])
    return x[:,0]**2 + x[:,0]**1
# =============================================================================
# def fun(x):
#     return x[:, 0]**2+x[:, 1]**2+x[:, 2]**2
# =============================================================================

def adapt(gp, epsilon, N, NC, totalbudget, budgettospend, 
          SigmaLL, SigmaP, testranges,
          TOL, TOLe, loweraccuracybound, nrofnewpoints, runpath, execname, 
          beta= 1.2, sqpiter = 800, sqptol = 1E-5):    
    """
    

    Parameters
    ----------
    gp : Gaussian process
        Gaussian process which is getting adapted
    epsilon : np.array()
        Union of all data accuracies. Used as the start alue of the adaption.
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
    testranges : np.array([]) dimxdim
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
    
    print("\n")
    print("------------------------ Start optimization")
    print("Number of initial points:      "+str(N))
    print("Number of candidate points:    "+str(NC))
    print("Total budget:                  "+str(totalbudget))
    print("Desired tolerance:             "+str(TOL))
    print("Lower accuracy bound:          "+str(loweraccuracybound))
    print("Number of adaptive new points: "+str(nrofnewpoints))
    
    
    print("\n")
    errorlist = []
    costlist = []
    dim = gp.getdata[2]
    
    counter = 0
    totaltime = 0
    nosolutioncounter = 0 
    
    currentcost = totalcompwork(epsilon)
    costlist.append(currentcost)
    
    while currentcost < totalbudget:
        
        print("------------------------ Iteration / Design: {}".format(counter))
        print("Current number of points: {} ".format(N+NC))
        print("\n")
        print(" Estimate derivatives...")
        df = gp.predictderivative(gp.getX,True)
        print(" ...done")
        print(" Calculate constants...")
        C1 = np.linalg.norm(df, np.inf)  # float
        C2 = estimateconstants(gp)
        print(" ...done")
        scale = 1.1
        
        """ The bounds have to be adapted to the current data errors at x """
        if totalcompwork(loweraccuracybound*np.ones((1, N+NC))) > totalbudget:
            print("Allowed lower bound exceeds comp. budget.")
            print("\n")
            return gp, errorlist, costlist

        else:
            tmp = np.concatenate(
                (loweraccuracybound*np.ones((1, N+NC)), epsilon.reshape((1, -1)) ),axis = 0).T
            bounds = tuple(map(tuple, tmp))

        globarerror = globalerrorestimate(
            epsilon, gp, C1, C2, NC, SigmaLL, SigmaP, testranges)
        errorlist.append(globarerror[0])
        print(" Current error estimate E^2(D(e)): {:10.6f}".format(
            globarerror[0]))
        print(" Current computational cost:       {:10.4f}".format(currentcost))

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

        ' Minimize E^2(D)'
        print("  Optimization phase: ")
        t0 = time.time()
        sol = scipy.optimize.minimize(globalerrorestimate, epsilon,
                                      args=(gp, C1, C2, NC, SigmaLL,
                                            SigmaP, testranges, scale),
                                      method='SLSQP',
                                      jac=gradientofglobalerrorestimate,
                                      bounds=bounds,
                                      constraints=con,
                                      options={'maxiter': sqpiter, 'ftol': sqptol, 'disp': False})
        t1 = time.time()
        print(t1-t0)
        total_n = t1-t0
        totaltime += total_n
        
        if sol.success == True:

            ' Set new start value for next design '
            epsilon = sol.x

            ' Set new cummulated cost '
            currentcost = totalcompwork(epsilon)
            costlist.append(currentcost)

            if currentcost > totalbudget:
                print("Something went wront. Too much budget spend for this solution.")
                print("Save everything...")
                gp.savedata(runpath+'/saved_data')
                return gp, errorlist, costlist

            print("  Solution found within {} iterations".format(sol.nit))
            newerrorestimate = globalerrorestimate(
                sol.x, gp, C1, C2, NC, SigmaLL, SigmaP, testranges)[0]
            if newerrorestimate < TOL:
                print("\n")
                print("Desired tolerance is reached, adaptive phase is done.")
                print(" Final error estimate: {}".format(newerrorestimate))
                print(" Total used time: {:0.4f} seconds".format(totaltime))
                
                Xcandidate = gp.getX[N:,:]
                #ycandiatenew = np.sum(Xcandidate**3, axis=1)
                #ycandiatenew = np.sin(np.sum(Xcandidate,axis=1))
                ycandiatenew = fun(Xcandidate)
                gp.adddatapointvalue(ycandiatenew.reshape((-1,1)),[N,None])
                
                errorlist.append(newerrorestimate)
                gp.addaccuracy(epsilon**2, [0, N+NC])
                print("Save everything...")
                gp.savedata(runpath+'/saved_data')
                return gp, errorlist, costlist

            print("  Error estimate after optimization: {:10.6f}".format(newerrorestimate))
            print("  Current computational cost: {:10.4f}".format(currentcost))
            print("  New budget for next design: {:10.4f}".format(
                currentcost+budgettospend))
            print("  Used time: {:0.4f} seconds".format(total_n))
            print("\n")
                            
            if np.linalg.norm(newerrorestimate-globarerror) < TOLe:
                print("---No change in error estimate detected.")
                if currentcost < totalbudget:
                    nosolutioncounter += 1
                    print("  No solution counter: {}".format(nosolutioncounter))
                    if nosolutioncounter == 5:
                        print("  Add {} new points".format(nrofnewpoints))
                        print("  Add additional {:10.0f} to budget".format(budgettospend*0.1))
                        print("  New budget for next design: {}".format(currentcost+budgettospend))
                        print("\n")
                        Xnew = createPD(nrofnewpoints, dim, "sobol", testranges)
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
                            epsilon = np.concatenate((np.squeeze(epsilon), np.squeeze(epsXc).reshape(1)))  
                        else:
                            epsilon = np.concatenate((np.squeeze(epsilon), np.squeeze(epsXc)))
                        nosolutioncounter = 0 
                    
                    else:
                        ' Get maximum error, calcualte comp work to be at x percent of max error '
                        variance = gp.predictvariance(gp.getX)
                        maxpwe = maxpointwiseerror(C1, C2, df, SigmaLL, SigmaP, variance, np.inf)
                        
                        ebeta = maxpwe*beta
                        deltaw = totalcompwork(ebeta)
                        budgettospend += deltaw
                        
                        print("  Adjust budget to spend")
                        print("  Maximum error estimate: {}".format(maxpwe[0]))        
                        print("  New \u0394W: {}".format(deltaw))
                        print("  New budget to spend: {}".format(budgettospend))
                        print("\n")
                        #budgettospend += budgettospend*0.25
                        
                                            
                else:
                    print("  Budget is not sufficient, stopt adaptive phase.")
                    print("\n")
                    print("Save everything...")
                    gp.savedata(runpath+'/saved_data')
                    return gp, errorlist, costlist   
        
        else:

            ' Set new start value for next design '
            epsilon = sol.x
            
            ' Set new cummulated cost '
            currentcost = totalcompwork(epsilon)
            costlist.append(currentcost)
            
            newerrorestimate = globalerrorestimate(
                sol.x, gp, C1, C2, NC, SigmaLL, SigmaP, testranges)[0]
            
            if newerrorestimate < TOL:
                print("\n")
                print("Desired tolerance is still reached, adaptive phase is done.")
                print(" Final error estimate: {:10.6f}".format(newerrorestimate))
                print(" Total used time: {:0.4f} seconds".format(totaltime))
                gp.addaccuracy(epsilon**2, [0, N+NC])
                errorlist.append(newerrorestimate)
                print("Save everything...")
                gp.savedata(runpath+'/saved_data')
                return gp, errorlist, costlist
            
            print("---No solution found.")
            print(" Total used time: {:0.4f} seconds".format(totaltime))
            print(" Add {} new points".format(nrofnewpoints))
            print("\n")

            Xnew = createPD(nrofnewpoints, dim, "sobol", testranges)
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
                epsilon = np.concatenate((np.squeeze(epsilon), np.squeeze(epsXc).reshape(1)))  
            else:
                epsilon = np.concatenate((np.squeeze(epsilon), np.squeeze(epsXc)))

            #currentcost += budgettospend
            #budgettospend += budgettospend*0.25
            ' Get maximum error, calcualte comp work to be at x percent of max error '
            variance = gp.predictvariance(gp.getX)
            maxpwe = maxpointwiseerror(C1, C2, df, SigmaLL, SigmaP, variance, np.inf)
            
            ebeta = beta*maxpwe
            deltaw = totalcompwork(ebeta)
            budgettospend += deltaw
            
            print("  Adjust budget to spend")
            print("  Maximum error estiamte: {}".format(maxpwe[0]))        
            print("  New \u0394W: {}".format(deltaw))
            print("  New budget to spend: {}".format(budgettospend))
            print("\n")
            
        ' Recalcualte value for candidate points '
        Xcandidate = gp.getX[N:,:]
        #ycandiatenew = np.sum(Xcandidate**3, axis=1)
        #ycandiatenew = np.sin(np.sum(Xcandidate,axis=1))
        ycandiatenew = fun(Xcandidate)
        gp.adddatapointvalue(ycandiatenew.reshape((-1,1)),[N,None])
        
        counter += 1