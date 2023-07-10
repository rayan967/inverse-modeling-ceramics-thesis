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

from gpr.gaussianprocess import *

def maxpointwiseerror(C1, C2, df, SigmaLL, SigmaP, std, norm, scale=1.0):

# =============================================================================
#     invsigma = np.linalg.inv(SigmaLL)
#     invnorm = np.linalg.norm(invsigma, norm)
#     N = df.shape[0]
#     currentmax = 0
#     dim = df.shape[1]
# 
#     m = df.shape[0]
#     
#     if m == 1:
#         L = np.min(np.linalg.eig(np.outer(df,invsigma* df)+np.linalg.inv(SigmaP))[0])  # float
#     else:
#         L = np.min(np.linalg.eig( df.T@invsigma@df+np.linalg.inv(SigmaP))[0])  # float
#     #L = np.min(np.linalg.eig(np.outer(df,invsigma* df))[0])  # float
#     if isinstance(L, complex):
#         L = L.real
# 
#         pointwiseerror = (12*C1*invnorm/L + 1/C2) * scale * std[i, :]
#         if pointwiseerror > currentmax:
#             currentmax = pointwiseerror
# =============================================================================

    return currentmax

def pointwiseerror(C1, C2, df, SigmaLL, SigmaP,  norm):

    invsigma = np.linalg.inv(SigmaLL)
    invnorm = np.linalg.norm(invsigma, norm)
    
    m = df.shape[0]
    
    if m == 1:
        L = np.min(np.linalg.eig(np.outer(df,invsigma* df)+np.linalg.inv(SigmaP))[0])  
    else:
        L = np.min(np.linalg.eig( df.T@invsigma@df+np.linalg.inv(SigmaP))[0]) 
   
    if isinstance(L, complex):
        L = L.real

    if L <= 1E-4:
        print("Warning: L is too small, set value to: "+str(1))
        L = 1

    KR = (12*C1*invnorm/L + 1/C2)
    return KR

def globalerrorestimate(v, X, yt, hyperparameter, KXX, C1, C2, Nall, SigmaLL, SigmaP, parameterranges, logtransform = False):

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

    'Inverse of KXX-1+V'
    invKV = np.linalg.inv(invKXX+np.diagflat(v))

    'Unit matrix from euclidean vector'
    unitmatrix = np.eye(X.shape[0])

    for i, x in enumerate(X):
        'Local var'
        
        var = unitmatrix[:, i].T@invKV@unitmatrix[:, i]

        'Local derivative estiamte'
        KxX = KXX[i, :].reshape((1, -1))
        df = dGPR(x.reshape((1, -1)), X, KxX, L)@alpha #dim x m, when multible data is given

        'Local error estimation at x \in X'
        norm = np.inf
        KR = pointwiseerror(C1, C2, df.T, SigmaLL, SigmaP,  norm)
        errorsum += KR**2*var

    if logtransform:
        globalerrorestimate = np.log((volofparameterspace/Nall)) + np.log(errorsum)
    else:
        globalerrorestimate = (volofparameterspace/Nall) * errorsum

    return globalerrorestimate  # 14.4

def gradientofglobalerrorestimate(v, X, yt, hyperparameter, KXX, C1, C2, Nall, SigmaLL, SigmaP, parameterranges, logtransform = False):

    v = v.reshape((1, -1))

    sigma = hyperparameter[0]
    Lhyper = hyperparameter[1:]
    volofparameterspace = np.prod(parameterranges[:, 1])

    errorgradsum = 0
    errorsum = 0

    invsigma = np.linalg.inv(SigmaLL)
    norm = np.inf
    invnorm = np.linalg.norm(invsigma, norm)

    'Inverse in eps for df'
    KXXdf = KXX+np.diagflat(v**(-1))
    alpha = np.linalg.solve(KXXdf, yt)

    'Inverse of KXX'
    invKXX = np.linalg.inv(KXX)
    #invKXX[np.abs(invKXX) < 1E-6] = 0.0

    'Inverse of KXX-1+V'
    invKV = np.linalg.inv(invKXX+np.diagflat(v))
    #invKV[np.abs(invKV) < 1E-6] = 0.0

    'Unit matrix from euclidean vector'
    unitmatrix = np.eye(X.shape[0])
    
    'Calculate derivative tensor'
    tensor = np.zeros((Nall,Nall,Nall))
    for kk in range(Nall):
        ei =  unitmatrix[kk,:]
        tensor[:,:,kk] = np.outer(ei.T,ei)

    tmp  = np.einsum("ij,jmk,ml->ilk",invKV,tensor,invKV)

    t0grad = time.perf_counter()
    for i, x in enumerate(X):

        ei =  unitmatrix[i,:]

        'Local derivative estiamte'
        KxX = KXX[i, :].reshape((1, -1))
        df = dGPR(x.reshape((1, -1)), X, KxX, Lhyper)@alpha
        df = df.T
        
        m = df.shape[0]         
        
        if m == 1:
            L = np.min(np.linalg.eig(np.outer(df,invsigma*df)+np.linalg.inv(SigmaP))[0])  # float
        else:
            L = np.min(np.linalg.eig( df.T@invsigma@df+np.linalg.inv(SigmaP))[0])  # float
        if isinstance(L, complex):
            L = L.real
        if L <= 1E-4:
            print("Warning: L is too small, set value to: "+str(1))
            L = 1
        
        KR = (12*C1*invnorm/L + 1/C2)

        gradvar = -np.einsum("l,lmk,m->k",ei.T,tmp,ei)
        errorgradsum += KR * KR * gradvar

    grad = volofparameterspace/Nall * errorgradsum
    
    t1grad = time.perf_counter()
    #print("Time in grad: "+str(t1grad-t0grad))
    return np.squeeze(grad)

def hessianofglobalerrorestimate(v, X, yt, hyperparameter, KXX, C1, C2, Nall, SigmaLL, SigmaP, parameterranges, logtransform = False):

    v = v.reshape((1, -1))

    sigma = hyperparameter[0]
    Lhyper = hyperparameter[1:]
    volofparameterspace = np.prod(parameterranges[:, 1])

    errorhesssum = 0
    errorsum = 0

    invsigma = np.linalg.inv(SigmaLL)
    norm = np.inf
    invnorm = np.linalg.norm(invsigma, norm)

    'Inverse in eps for df'
    KXXdf = KXX+np.diagflat(v**(-1))
    alpha = np.linalg.solve(KXXdf, yt)

    'Inverse of KXX'
    invKXX = np.linalg.inv(KXX)
    #invKXX[np.abs(invKXX) < 1E-6] = 0.0

    'Inverse of KXX-1+V'
    invKV = np.linalg.inv(invKXX+np.diagflat(v))
    #invKV[np.abs(invKV) < 1E-6] = 0.0

    'Unit matrix from euclidean vector'
    unitmatrix = np.eye(X.shape[0])

    'Calculate derivative tensor'
    tensor = np.zeros((Nall,Nall,Nall))
    for kk in range(Nall):
        ei =  unitmatrix[kk,:]
        tensor[:,:,kk] = np.outer(ei.T,ei)

    tmplhs  = np.einsum("ij,jmk->imk",invKV,tensor)
    tmprhs  = np.einsum("jmk,ml->jlk",tensor,invKV)

    t0hess = time.perf_counter()
    for i,x in enumerate(X):

        hessvar = np.zeros((Nall, Nall))
        ei =  unitmatrix[i,:]

        'Local derivative estiamte'
        KxX = KXX[i, :].reshape((1, -1))
        df = dGPR(x.reshape((1, -1)), X, KxX, Lhyper)@alpha
        df = df.T
        
        m = df.shape[0]        
        
        if m == 1:
            L = np.min(np.linalg.eig(np.outer(df,invsigma*df)+np.linalg.inv(SigmaP))[0])  # float
        else:
            L = np.min(np.linalg.eig( df.T@invsigma@df+np.linalg.inv(SigmaP))[0])  # float
        if isinstance(L, complex):
            L = L.real
        if L <= 1E-4:
            print("Warning: L is too small, set value to: "+str(1))
            L = 1
        
        KR = (12*C1*invnorm/L + 1/C2)
        
        
        #for ii in range(Nall):
        for jj in range(0,Nall):
            tmp1 = np.einsum("i,in,nk,kpq,p->q",ei,tmplhs[:,:,jj],invKV,tmprhs[:,:,:jj+1],ei)
            tmp2 = np.einsum("i,ipq,pk,km,m->q",ei,tmplhs[:,:,:jj+1],invKV,tmprhs[:,:,jj],ei)
            hessvar[jj,:jj+1] =  tmp1+tmp2

        diag = np.diagflat(np.diag(hessvar))
        hessvar += hessvar.T 
        hessvar -= diag
    
        errorhesssum += KR**2 * hessvar
        
    hessian = volofparameterspace/Nall * errorhesssum
    
    t1hess = time.perf_counter()
    print("Time in hess: "+str(t1hess-t0hess))
    return hessian

def estimateconstants(gp, df):
    X = gp.getX
    df = np.atleast_2d(df)
    C1 = np.max(np.linalg.norm(df, np.inf, axis=1))  # float
    C2 = 0
    for ii in range(X.shape[0]):
        hess = gp.predicthessian(np.array([X[ii, :]]))
        maxhess = np.linalg.norm(hess, np.inf)
        if maxhess > C2:
            C2 = maxhess
    return C1, C2

""" Inequality functions """
def totalcompwork(v, s=1):
    return np.sum(v**(s))
def totalcompworkeps(epsilon):
    return np.sum(1/2*epsilon**(-2))

""" Data functions """
def fun(x,a):
    return np.sin(x[0])+a*np.cos(x[0]*x[1])
def dfun(x):
    if x.shape[0] > 1:
        return np.array([np.cos(x[:, 0])-x[:, 1]*np.sin(x[:, 0]*x[:, 1]), -x[:, 0]*np.sin(x[:, 0]*x[:, 1])]).T
    else:
        return np.array([np.cos(x[0])-x[1]*np.sin(x[0]*x[1])],[-x[0]*np.sin(x[0]*x[1])])


def adapt(gp, N, NC, totalbudget, budgettospend,
          SigmaLL, SigmaP, parameterranges,
          TOL, TOLe, TOLFEM, TOLFILTER,  nrofnewpoints, execpath, execname,
          beta=1.4, sqpiter=10000, sqptol=1E-1):

    errorlist = []
    realerrorlist = []
    costlist = []
    dim = gp.getdata[2]

    counter = 0
    totaltime = 0
    totalFEM = 0
    nosolutioncounter = 0
    itercounter = 0

    'Epsilon^2 at this points'
    epsilon = np.squeeze(gp.getaccuracy)

    'Solver options'
    xtol = 1E-4
    gtol = 1E-4
    s = 2
    method = "trust-constr"

    currentcost = totalcompworkeps(epsilon)
    costlist.append(currentcost)

    print("\n")
    print("---------------------------------- Start optimization")
    print("Number of initial points:          "+str(N))
    print("Total budget:                      "+str(totalbudget))
    print("Desired tolerance:                 "+str(TOL))
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
        C1, C2 = estimateconstants(gp,df[:,:,0])
        print("  C1: {:10.2f}".format(C1))
        print("  C2: {:10.2f}".format(C2))

        'Calculate KXX once, since this matrix is not changing'
        X, yt, Nall = gp.getX, gp.gety, gp.getdata[0]
        hyperparameter = gp.gethyperparameter
        KXX = kernelmatrix(X, X, hyperparameter)

        'Turn epsilon^2 into v'
        v = epsilon**(-1)

        """ NEW FORMULATION STARTS HERE """
        globalerrorbefore = globalerrorestimate( v, X, yt, hyperparameter, KXX, C1, C2, Nall,
                                                 SigmaLL, SigmaP, parameterranges )
        N = Nall-NC
        print("Global error estimate before optimization:   {:1.8f}".format(globalerrorbefore))
        print("Computational cost before optimization:      {:0.0f}".format(currentcost))
        print("\n")
        print("--- Solve minimization problem")

        'Set start values here'
        if counter == 0:
            v[:Nall] = (budgettospend/Nall)**(1/s)
        else:
            'Adapt initial values by taking the max value of the new solution'
            v[N:] += (budgettospend/NC)**(1/s)

        currentcost = totalcompwork(v, s)

        total_n = 0
        if method == "SLSQP":

            def compworkconstrain(v, currentcost, budgettospend, s=1):
                print("Current solution:    "+str(v))
                print("Current cost of sol. "+str(np.sum(v**s)))
                print("Current Budget:      "+str(currentcost+budgettospend))
                print("KKT:                 "+str(currentcost+budgettospend-np.sum(v**s)))
                print("\n")
                return currentcost+budgettospend-np.sum(v**s)
            def compworkconstrainjac(v, currentcost, budgettospend, s=1):
                return -s*v**(s-1)


            #23.5.2022 !!!!!
            lowerboundvalues = np.concatenate((1E-8*v[:N].reshape(1,-1),1E-8*np.ones((1,NC))),axis = 1)[0].tolist()
            upperboundvalues = (Nall)*[None]
            tmp = zip(lowerboundvalues,upperboundvalues)
            bounds = tuple(map(tuple, tmp))

            arguments = (currentcost, budgettospend,s,)
            con = [{'type': 'eq',
                    'fun': compworkconstrain,
                    'jac': compworkconstrainjac,
                    'args': arguments},]

            """ Add little variation to the solution after the first design"""
            t0 = time.perf_counter()
            sol = scipy.optimize.minimize(globalerrorestimate, v,
                                          args=(X, yt, hyperparameter, KXX, C1, C2,
                                                Nall,  SigmaLL, SigmaP, parameterranges, False),
                                          method=method, tol = sqptol,  bounds = bounds,
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
                s = 2
                return s*(s-1)*v[0]*np.diagflat(x**(s-2))

            nonlinear_constraint = NonlinearConstraint(lambda x: compworkconstrain(x,s),currentcost, currentcost+budgettospend,
                                                       jac=lambda x: compworkconstrainjac(x,s),
                                                       hess=compworkconstrainhess)
            #-np.inf
            t0 = time.perf_counter()
            sol = scipy.optimize.minimize(globalerrorestimate, v,
                                          args=(X, yt, hyperparameter, KXX, C1, C2,
                                                Nall,  SigmaLL, SigmaP, parameterranges),
                                          method=method, tol= 1E-4,
                                          jac = gradientofglobalerrorestimate,
                                          hess = hessianofglobalerrorestimate,
                                          constraints=[nonlinear_constraint],
                                          options={'verbose': 2,'maxiter':10000,'xtol': xtol, 'gtol': gtol})
            t1 = time.perf_counter()
            total_n = t1-t0

        totaltime += total_n

        nrofdeletedpoints = 0
        if sol.success == True:

            print("Solution found within {} iterations".format(sol.nit))
            print("Used time:            {:0.2f} seconds".format(total_n))
            print("Last function value:  {}".format(sol.fun))
            print("\n")

            'Solution for v'
            vsol = sol.x
            print("Found solution: ")
            print("vsol: ")
            print(vsol)
            
            'Adapt solver accuracy when solutions are done within one iteration'
            if sol.nit == 1:
                
                if xtol == 1E-5:
                    print("Highest accuracy within solver is reached.")
                    continue
                
                else:
                    xtol = xtol / 2
                    gtol = gtol / 2              
            
            """ DEBUG SCATTER PRINT """
            execpath ="F:/Uni/Zuse/simlopt/simlopt/data/DEBUG_TRST"
            fig, axs = plt.subplots(1, 1)
            axs.grid(True)
            axs.scatter(X[:N,0],X[:N,1])
            axs.scatter(X[N:,0],X[N:,1],c='red')
            fig.savefig(execpath+'/'+str(counter)+"_"+str(sol.nit)+'.png')


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
                print("Sufficient change in the solutions is detected, optain new simulation values")
                print("for point(s): {}".format(indicesofchangedpoints[1]))

                t0FEM = time.perf_counter()
                print("\n")
                print("--- Start simulation block")
                for jj in range(indicesofchangedpoints[1].shape[0]):
                    currentFEMindex = indicesofchangedpoints[1][jj]

                    ' Get new values for calcualted solution'
                    epsXtnew = currentepsilonsol[currentFEMindex].reshape((1, -1))
                    a = [1.0,0.9,0.8,0.7]
                    ytnew = np.zeros((1,len(a)))
                    for i,a in enumerate(a):
                        ytnew[:,i] = fun(gp.getX[currentFEMindex, :],a).reshape((1, -1))

                    ' Add new value to GP '
                    gp.addaccuracy(epsXtnew**2, currentFEMindex)
                    gp.adddatapointvalue(ytnew, currentFEMindex)

                t1FEM = time.perf_counter()
                totalFEM = t1FEM-t0FEM
                print("Simulation block done within: {:1.4f} s".format(totalFEM))

            ' Filter all points which seemingly have no influence on the global parameter error '
            ' We preemptively filter the list beginning at N since we dont change core points'
            indicesofchangedpoints = np.where(np.abs(np.atleast_2d(np.abs(vsol[N:]-v[N:]) ) < 100))
            print("\n")
            print("--- Point filtering")
            if indicesofchangedpoints[1].size != 0:

                nrofdeletedpoints = indicesofchangedpoints[1].size
                nrofdeletedpoints = nrofnewpoints

                print("Points of no work are detected.")
                print("Delete points with index: {}".format(indicesofchangedpoints[1]+(N)))
                print("  Add {} new candidate points.".format(nrofdeletedpoints))

                ' Problem - the indices do not correspond to the acutal indices within the GP anymore'
                idx = indicesofchangedpoints[1]+(N)
                gp.deletedatapoint(idx)

                ' Delete points from solution vector'
                epsilon = np.delete(epsilon, idx)
                currentepsilonsol = np.delete(currentepsilonsol, idx)
                'Core candiates need to be added as gradient info with high error'
                # Adapt only NC , since initial data is not deleted
                NC -= idx.shape[0]
                print("  Set {} as core points.".format(NC))
                if gp.getXgrad is not None:
                    Xcgrad = gp.getX[N:N+NC]
                    epsXcgrad = 1E10 * np.ones((1, Xcgrad.shape[0]*dim))  # eps**2
                    dfXcgrad = gp.predictderivative(Xcgrad)
                    gp.addgradientdatapoint(Xcgrad)
                    gp.adddgradientdatapointvalue(dfXcgrad)
                    gp.addgradaccuracy(epsXcgrad)

                N += NC
                NC = 0
                
            else:
                print("All points may have sufficient influence on the global parameter error.")
                print("Transfer all candidate points to core points.")
                print("  Set {} as core points.".format(NC))
                nrofdeletedpoints = nrofnewpoints
                print("  Add {} new candidate points.".format(nrofdeletedpoints))
                if gp.getXgrad is not None and NC > 0:
                    Xcgrad = gp.getX[N:N+NC]
                    epsXcgrad = 1E10 * np.ones((1, Xcgrad.shape[0]*dim))  # eps**2
                    dfXcgrad = gp.predictderivative(Xcgrad)
                    gp.addgradientdatapoint(Xcgrad)
                    gp.adddgradientdatapointvalue(dfXcgrad)
                    gp.addgradaccuracy(epsXcgrad)

                N += NC
                NC = 0

            ' Global parameter estimate after optimisation and filtering.'
            vsol = currentepsilonsol**(-2) #Filtered solution
            X, yt, Nall = gp.getX, gp.gety, gp.getdata[0]
            KXX = kernelmatrix(X, X, hyperparameter)

            globalerrorafter = globalerrorestimate(vsol, X, yt, hyperparameter, KXX, C1, C2, Nall,
                               SigmaLL, SigmaP, parameterranges)

            errorlist.append(globalerrorafter)
            currentcost = totalcompwork(vsol,s)
            costlist.append(currentcost)

            'Add new candidate points'
            nrofdeletedpoints = nrofnewpoints
            NC = nrofdeletedpoints
            XC = createPD(NC, dim, "random", parameterranges)
            epsXc = 1E20*np.ones((1, XC.shape[0]))  # eps**2
            meanXc = gp.predictmean(XC)

            gp.adddatapoint(XC)
            gp.adddatapointvalue(meanXc)
            gp.addaccuracy(epsXc)

            if globalerrorafter < TOL:
                print("\n")
                print("--- Convergence ---")
                print("Desired tolerance is reached, adaptive phase is done.")
                print("Final error estimate: {:1.8f}".format(globalerrorafter))
                print("Total used time: {:0.2f} seconds".format(totaltime+totalFEM))
                print("Save everything !")
                gp.savedata(execpath+'/saved_data')
                return gp, errorlist, costlist, realerrorlist

            """ ---------------------------------------------------------- """
            print("\n")
            print("--- New parameters")
            print("Error estimate after optimization and filtering: {:1.8f}".format(globalerrorafter))
            print("Computational cost after optimization:           {:0.0f}".format(currentcost))
            print("New budget for next design:                      {:0.0f}".format(currentcost+budgettospend))
            print("Used time:                                       {:0.2f} seconds".format(total_n))
            print("\n")


        else:

            ' Set new start value for next design '
            vsol = sol.x

            'Calculate KXX once, since this thisd matrix is not changing, ugly but speeds up'
            X, yt, Nall = gp.getX, gp.gety, gp.getdata[0]
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


            budgettospend += 1E7
            N += NC  # All candidates go core
            NC = 0  # There are no furhter candidate points

            print("Adjust budget to spend")
            print("  Maximum error estiamte: {:1.7f}".format(maxpwe[0]))
            print("  New \u0394W: {:0.4f}".format(deltaw))
            print("  New budget to spend: {:0.4f}".format(budgettospend))
            print("\n")

        counter += 1
        
        'If the error descreases too slow we add more budget to spend'
        relchange = np.abs(globalerrorafter-globalerrorbefore)/globalerrorbefore*100
        if relchange < 10:
            print("Relative change: "+str(relchange))
            print("Relative change is below set threshold.")
            
            if currentcost < totalbudget:
                budgettospend *= 5
                print("Adjust budget to spend")
                print("  New budget to spend: {:0.1f}".format(budgettospend))
                print("\n")
            else:
                print("  Budget is not sufficient, stopt adaptive phase.")
                print("\n")
                print("Save everything...")
                gp.savedata(execpath+'/saved_data')
                return gp, errorlist, costlist

        epsilon = vsol**(-1/2) #epsilon

        if nrofdeletedpoints > 0:
            if nrofdeletedpoints == 1:
                epsilon = np.concatenate((epsilon, np.squeeze(epsXc,axis=0)))
            else:
                epsilon = np.concatenate((epsilon, np.squeeze(epsXc)))

        epsilon = epsilon**2
        t1design = time.perf_counter()
        print("Time used for complete design iteration: {:0.2f} seconds".format(t1design-t0design))
        print("\n")