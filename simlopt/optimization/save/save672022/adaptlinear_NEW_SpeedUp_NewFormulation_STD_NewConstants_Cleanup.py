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

from gpr.gaussianprocess import *


def DEBUGGRADIENDTS(X):
    N = X.shape[0]
    dim = X.shape[1]

    a = [1.0,0.9,0.8,0.7]
    dy = np.zeros((N,dim,len(a)))

    for j in range(len(a)):
        for i in range(N):
            x = X[i,:]
            dy[i,:,j]  = np.array([np.cos(x[0])-a[j]*x[1]*np.sin(x[0]*x[1]), -a[j]*x[0]*np.sin(x[0]*x[1])])
    return dy


def estiamteweightfactors(var, X, dy, epsphys):

    dim = X.shape[1]
    delta = 1E-6

    m = dy.shape[2]

    w = np.zeros((X.shape[0]))
    #dy = DEBUGGRADIENDTS(X)

    'Check if epsphys might be a matrix'
    if isinstance(epsphys, (np.floating, float)):
        SigmaLL = epsphys*np.eye(m)

    for i, x in enumerate(X):
        x = x.reshape((1, -1))
        dyatp = dy[i, :, :].T
        varvec = var[i]*np.ones((m))

        'Gauß Newton Step'
        Jprime = dyatp.T @ SigmaLL @ varvec
        w[i] = np.linalg.norm((np.linalg.inv(np.outer(Jprime.T, Jprime)+delta*np.eye((dim)))@Jprime).reshape((1, -1)), 2)
    return w


def MCGlobalEstimate(w,var,Nall,parameterranges):
    var = np.linalg.norm(var*np.ones((1,4)),axis=1)
    volofparameterspace = np.prod(parameterranges[:, 1])
    return np.array([(volofparameterspace/Nall) * np.dot(w,var)])


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

""" Acquisition function """
def acquisitionfunction(gp,XGLEE,epsphys,TOLAcqui,XCdummy=None):

    'Calculate error distribution with new data for '
    dfXC = gp.predictderivative(XGLEE, True)
    varXC = gp.predictvariance(XGLEE)
    w = estiamteweightfactors(varXC, XGLEE, dfXC, epsphys)
    acquisition = varXC* w.reshape((-1,1))

    acquidx = np.where(acquisition >= np.max(acquisition)*TOLAcqui)

    if acquidx[0].size != 0:
        XC = XGLEE[acquidx[0]]

        'Check if XC is alread in X, if so, delete points form XC'
        for i in range(gp.getX.shape[0]):
            currentindex = np.where((XC == gp.getX[i,:].tolist()).all(axis=1))
            if currentindex[0].size != 0:
                print(" Doublicate found.")
                print(" Delete {} from candidate points".format(gp.getX[i,:]))
                XC = np.delete(XC,(currentindex[0][0]),axis=0)
                if XCdummy is not  None:
                    XCdummy = np.vstack((XCdummy,gp.getX[i,:]))
        if XCdummy is not None:
            return XC,XCdummy
        else:
            return XC
    else:
        print(" No changes necessary.")
        return None


""" Computational work evaluation """
def totalcompwork(v, s=1):
    return np.sum(v**(s))
def totalcompworkeps(epsilon):
    return np.sum(1/2*epsilon**(-2))

""" Data functions """
def fun(x, a):
    return np.sin(x[0])+a*np.cos(x[0]*x[1])
def dfun(x):
    if x.shape[0] > 1:
        return np.array([np.cos(x[:, 0])-x[:, 1]*np.sin(x[:, 0]*x[:, 1]), -x[:, 0]*np.sin(x[:, 0]*x[:, 1])]).T
    else:
        return np.array([np.cos(x[0])-x[1]*np.sin(x[0]*x[1])], [-x[0]*np.sin(x[0]*x[1])])


def adapt(gp, totalbudget, incrementalbudget, parameterranges,
          TOL, TOLFEM, TOLFILTER, TOLAcqui,TOLrelchange, epsphys,
          nrofnewpoints,  execpath, execname, adaptgrad):

    dim= gp.getdata[2]

    counter= 0
    totaltime= 0
    totalFEM= 0
    nosolutioncounter= 0
    itercounter= 0

    'Solver options'
    xtol = 1*1E-3
    gtol = 1*1E-3
    itermax = 100000
    s = 2

    """ If withgradient is True, we check if there is already gradient data.
    otherwise we need to at as much gradientdata as trainig data is already available """
    N = gp.getdata[0]

    gradadaptcheck = adaptgrad is True and gp.getXgrad is not None

    if adaptgrad is True and gp.getXgrad is None:
        Xt = gp.getX
        nX = Xt.shape[0]

        epsXgrad = 1E20*np.ones((1,nX*dim))
        dy = gp.predictderivative(Xt)

        gp.addgradientdatapoint(Xt)
        gp.adddgradientdatapointvalue(dy)
        gp.addgradaccuracy(epsXgrad)

        Ngrad = gp.getdata[1] #Is None, when Xgrad is None
    elif gradadaptcheck:
        Ngrad = gp.getdata[1]
        print("Alredy existing gradient data is considered within the minimization.")
    elif gp.getXgrad is not None:
        Ngrad = gp.getdata[1] #Is None, when Xgrad is None

    print("\n")
    print("---------------------------------- Start adaptive phase")
    print("Number of initial points:          "+str(N))
    print("Total budget:                      "+str(totalbudget))
    print("Desired tolerance:                 "+str(TOL))
    print("Number of adaptively added points: "+str(nrofnewpoints))
    print("Workmodel exponent:                "+str(s))
    print("\n")

    "Open logs"
    logpath = os.path.join(execpath+"/", "logs/")
    logpath_general = os.path.join(execpath+"/")
    figurepath = os.path.join(execpath+"/", "iteration_plots/")

    ' Empty array for possible dummy points '
    XCdummy= np.empty((0,dim))

    ' Initial acquisition phase '
    NMC = 25
    XGLEE = createPD(NMC, dim, "grid", parameterranges)
    dfXC = gp.predictderivative(XGLEE, True)
    varXC = gp.predictvariance(XGLEE)
    w = estiamteweightfactors(varXC, XGLEE, dfXC, epsphys)
    NGLEE = XGLEE.shape[0]
    mcglobalerrorafter = MCGlobalEstimate(w,varXC,NGLEE,parameterranges)

    XC = np.array([])
    while XC.size == 0:
        XC,XCdummy = acquisitionfunction(gp,XGLEE,epsphys,TOLAcqui,XCdummy)
        TOLAcqui*=0.99
    NC = XC.shape[0]
    TOLAcqui = 1.0

    print("Found {} potential maxima".format(XC.shape[0]))
    print("Create surrogate with {} candidate point".format((XC.shape[0])))
    print("\n")

    epsXc = 1E20*np.ones((1, XC.shape[0]))
    meanXc = gp.predictmean(XC,True)

    gp.adddatapoint(XC)
    gp.adddatapointvalue(meanXc)
    gp.addaccuracy(epsXc)

    'Epsilon^2 at this points'
    epsilon = np.squeeze(gp.getaccuracy)
    currentcost= totalcompworkeps(epsilon)
    if adaptgrad:

        epsXgrad = 1E20*np.ones((1,XC.shape[0]*dim))
        dyXC = gp.predictderivative(XC)

        gp.addgradientdatapoint(XC)
        gp.adddgradientdatapointvalue(dyXC)
        gp.addgradaccuracy(epsXgrad)

        epsilongrad = np.squeeze(gp.getgradientaccuracy)
        currentcost += totalcompworkeps(epsilongrad)

    try:
        costerrorlog = open(logpath+"costerror.txt","a")
    except IOError:
        print ("Error: File does not appear to exist.")
        return 0

    costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter[0]))
    costerrorlog.write("\n")
    costerrorlog.close()

    while currentcost < totalbudget:

        Nall =  N + NC
        if gradadaptcheck:
            Nallgrad = Ngrad+NC

        X = gp.getX

        try:
            costerrorlog = open(logpath+"costerror.txt","a")
            file = open(logpath_general+"log.txt","a")
        except IOError:
          print ("Error: File does not appear to exist.")
          return 0
        if counter == 0:
            file.write("Error estimate before"+" "+"Used budget"+" "+"Error estimate after")
            file.write("\n")

        'Turn epsilon^2 into v'
        v = epsilon**(-1)
        if gradadaptcheck:
            vgrad = epsilongrad**(-1)

        if counter == 0:
            file.write("Error estimate before"+" "+"Used budget"+" "+"Error estimate after")
            file.write("\n")

        t0design = time.perf_counter()
        print("---------------------------------- Iteration / Design: {}".format(counter))
        print("Current number of points: {} ".format(N))
        print("Current number of candidate points: {} ".format(NC))
        print("\n")

        print("--- A priori error estimate")
        print("Estimate derivatives")
        df = gp.predictderivative(gp.getX, True)
        print("...done")

        """ ------------------------------ MC GLOBAL ERROR ESTIMATION ------------------------------ """
        NMC = 25
        XGLEE = createPD(NMC, dim, "grid", parameterranges)
        NGLEE = XGLEE.shape[0]

        dfGLEE = gp.predictderivative(XGLEE, True)
        varGLEE = gp.predictvariance(XGLEE)
        w = estiamteweightfactors(varGLEE, XGLEE, dfGLEE, epsphys)
        mcglobalerrorbefore = MCGlobalEstimate(w,varGLEE,NGLEE,parameterranges)
        file.write( str(mcglobalerrorbefore[0]) + str(" "))

        ' Plot distribution of error '
        erroratx = np.abs(w.reshape((-1,1))*varGLEE)

        ' Scaling by v'
        maxval = np.max(v[:N])
        minval = np.min(v[:N])

        if maxval == minval:
            sizes = 40*np.ones((v.shape[0]))
        else:
            sizes = v[:N]*((40-5)/(minval-maxval))+(40*maxval-5*minval) / (maxval-minval)

        figcontour, axcontour = plt.subplots()
        axcontour.scatter(X[0:N,0], X[0:N,1],c="black",zorder=2,s = sizes[0:N])
        if XCdummy.size != 0:
            axcontour.scatter(XCdummy[:,0], XCdummy[:,1],marker='^',c="green",zorder=2,s = 30)

        axcontour.scatter(X[N:,0], X[N:,1],marker='x',c="red",zorder=2,s = 15)

        if gp.getXgrad is not None:
            axcontour.scatter(gp.getXgrad[:Ngrad,0], gp.getXgrad[:Ngrad,1],marker='x',c="green",zorder=2,s = 20)

        triang = tri.Triangulation(XGLEE[:,0], XGLEE[:,1])
        refiner = tri.UniformTriRefiner(triang)
        tri_refi, z_test_refi = refiner.refine_field(erroratx[:,0], subdiv=4)
        z_test_refi = np.abs(z_test_refi)
        axcontour.tricontour(tri_refi, z_test_refi, 15, linewidths=0.5, colors='k')
        cntr2 = axcontour.tricontourf(tri_refi, z_test_refi, 15, cmap="RdBu_r")
        #axcontour.tricontour(triang, erroratx[:,0],  linewidths=0.5, colors='k')
        #cntr2 = axcontour.tricontourf(triang, erroratx[:,0], cmap="RdBu_r")

        axcontour.set(xlim=(-0.05, 3.05), ylim=(-0.05, 3.05))
        figcontour.colorbar(cntr2, ax=axcontour)
        axcontour.set_title("MC GLEE: "+str(mcglobalerrorbefore[0]))
        figcontour.savefig(figurepath+"contourplot_"+str(counter)+'.png')
        XCdummy= np.empty((0,dim))

        print("Global error estimate before optimization:   {:1.8f}".format(mcglobalerrorbefore[0]))
        print("Computational cost before optimization:      {:0.0f}".format(currentcost))
        print("\n")

        """ ------------------------------ Solve minimization problem ------------------------------ """

        print("--- Solve minimization problem")
        N = Nall-NC

        'Set start values here'
        if counter == 0:
            #print("Used start value: {}".format( (incrementalbudget/Nall)**(1/s)))
            v[0:]= 10
            print(" Used start value for all points: {}".format(10))
            if gradadaptcheck:
                vgrad[0:] = 10
            print("\n")
        else:
            'Adapt initial values by taking the max value of the new solution'
            if NC == 0:
                v[:Nall] += (incrementalbudget/Nall)**(1/s)
                print(" No candidate points found.")
                print(" Used start value just for candidate points: {}".format( (incrementalbudget/Nall)**(1/s)))
                print("\n")
            else:
                v[N:] += (incrementalbudget/NC)**(1/s)
                print(" Used start value just for candidate points: {}".format( (incrementalbudget/NC)**(1/s)))
                print("\n")


        'Bounds on v'
        lowerbound= v.tolist()
        upperbound= [np.inf]*Nall
        bounds= Bounds(lowerbound, upperbound)
        #bounds.lb[N:] = 0.0

        if gradadaptcheck:
            lowerboundgrad = vgrad.tolist()
            upperboundgrad = [np.inf]*Nallgrad*dim
            boundsgrad = Bounds(lowerboundgrad,upperboundgrad)
            #boundsgrad.lb[Ngrad*dim:] = 0.0

            'Connect bounds'
            lower = np.concatenate((bounds.lb,boundsgrad.lb))
            upper = np.concatenate((bounds.ub,boundsgrad.ub))

            'Build final bound object'
            bounds = Bounds(lower, upper)

            'Combine vs'
            v = np.concatenate((v,vgrad))

        ' Current cost by adding initial values is added to the overall budget '
        currentcost= totalcompwork(v, s)
        total_n= 0
        file.write( str(currentcost+incrementalbudget) + str(" ") )

        'Nonlinear constraints'
        def compworkconstrain(v, s):
            return np.array([np.sum(v**s)])
        def compworkconstrainjac(v, s):
            return np.array([s*v**(s-1)])
        def compworkconstrainhess(x, v):
            s= 2
            return s*(s-1)*v[0]*np.diagflat(x**(s-2))

        nonlinear_constraint= NonlinearConstraint(lambda x: compworkconstrain(x,s), currentcost+incrementalbudget, currentcost+incrementalbudget,
                                                   jac=lambda x: compworkconstrainjac(x,s),
                                                   hess=compworkconstrainhess)
        t0=time.perf_counter()

        ' Check if gradient data is available and set arguments for minimizer '
        if gp.getXgrad is not None:

            hyperparameter = gp.gethyperparameter
            K = kernelmatrixsgrad(X, gp.getXgrad, hyperparameter, gp.getaccuracy*0.0, gp.getgradientaccuracy*0.0)
            var = gp.predictvariance(X)
            wmin = estiamteweightfactors(var, X, df, epsphys)
            Ngrad = gp.getXgrad.shape[0]

            ' Check is gradients should be adapted '
            if adaptgrad:
                tensor = np.zeros(
                    (Nall+Ngrad*dim, Nall+Ngrad*dim, Nall+Ngrad*dim))
                for kk in range(Nall+Ngrad*dim):
                    tensor[kk, kk, kk] = 1
                args = (wmin, X, hyperparameter, K, Nall,
                        tensor, parameterranges, adaptgrad)
                ' ... or not '
            else:
                tensor = np.zeros((Nall+Ngrad*dim, Nall+Ngrad*dim, Nall))
                for kk in range(Nall):
                    tensor[kk, kk, kk] = 1
                args = (wmin, X, hyperparameter, K, Nall,
                        tensor, parameterranges, adaptgrad)
        else:

            hyperparameter = gp.gethyperparameter
            KXX = kernelmatrix(X, X, hyperparameter)
            var = gp.predictvariance(X)
            wmin = estiamteweightfactors(var, X, df, epsphys)

            tensor = np.zeros((Nall, Nall, Nall))
            for kk in range(Nall):
                tensor[kk, kk, kk] = 1
            args = (wmin, X, hyperparameter, KXX, Nall,
                    tensor, parameterranges, adaptgrad)

        sol=scipy.optimize.minimize(targetfunction, v,
                                      args=args,
                                      method='trust-constr',
                                      jac=gradientoftargetfunction,
                                      bounds = bounds,hess=BFGS(),
                                      constraints=[nonlinear_constraint],
                                      options={'verbose': 1, 'maxiter': itermax, 'xtol': xtol, 'gtol': gtol})
        #hess=hessianoftargetfunction,
        #hess=BFGS(),
        #hess=SF1(),
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
            print("Found solution: ")
            if gradadaptcheck:
                print("Point solution: ")
                print(vsol[:Nall])
                print("Gradient solution: ")
                print(vsol[Nall:])
            else:
                print("Point solution: ")
                print(vsol)

            """ DEBUG SCATTER PRINT """
            fig, axs=plt.subplots(1, 1)
            axs.grid(True)
            axs.scatter(X[:N, 0], X[:N, 1])
            axs.scatter(X[N:, 0], X[N:, 1], c='red')
            axs.set_title("Solution "+"Iter: "+str(counter))
            fig.savefig(figurepath+str(counter)+"_"+str(sol.nit)+'.png')

            'Solution for epsilon'
            currentepsilonsol=vsol**(-1/2)
            'Turn eps^2 to eps for comparing'
            epsilon=np.sqrt(epsilon)

            """ ---------- Block for adapting output (y) values ---------- """
            ' Check which point changed in its accuracy. Only if the change is significant a new simulation is done '
            ' since only then the outout value really changed. Otherwise the solution is just set as a new solution.'

            indicesofchangedpoints=np.where(np.abs(np.atleast_2d(epsilon-currentepsilonsol)) > TOLFEM)
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

                    ' Get new values for calcualted solution '
                    epsXtnew=currentepsilonsol[currentFEMindex].reshape((1, -1))
                    a=[1.0, 0.9, 0.8, 0.7]
                    ytnew=np.zeros((1, len(a)))
                    for i, a in enumerate(a):
                        ytnew[:, i]=fun(gp.getX[currentFEMindex, :], a).reshape((1, -1))

                    ' Add new value to GP '
                    gp.addaccuracy(epsXtnew**2, currentFEMindex)
                    gp.adddatapointvalue(ytnew, currentFEMindex)

                t1FEM=time.perf_counter()
                totalFEM=t1FEM-t0FEM
                print("Simulation block done within: {:1.4f} s".format(totalFEM))

            ' Filter all points which seemingly have no influence on the global parameter error '
            ' We preemptively filter the list beginning at N since we dont change core points'
            if counter == 0 :
                indicesofchangedpoints=np.where(np.abs(np.atleast_2d(np.abs(vsol[N:]-v[N:])) < TOLFILTER))
                #indicesofchangedpoints=np.where(np.abs(np.atleast_2d(np.abs(vsol-v)) < 10))
                offsetindex = 0
                idx = indicesofchangedpoints[1]+offsetindex

                NC -= np.where(idx > N)[0].size
                'Check if initial points are within list and subtract them from N'
                N -= np.where(idx <= N)[0].size
            else:
                indicesofchangedpoints=np.where(np.abs(np.atleast_2d(np.abs(vsol[N:]-v[N:])) < TOLFILTER))
                offsetindex = N
                idx=indicesofchangedpoints[1]+(offsetindex)
                NC -= idx.shape[0]
            print("\n")

            print("--- Point filtering")
            if indicesofchangedpoints[1].size != 0:

                #nrofdeletedpoints=indicesofchangedpoints[1].size
                nrofdeletedpoints = nrofnewpoints

                print("Points of no work are detected.")
                print("Delete points with index: {}".format(indicesofchangedpoints[1]+(offsetindex)))
                print("  Add {} new candidate points.".format(nrofdeletedpoints))
                print("\n")
                gp.deletedatapoint(idx)

                ' Delete points from solution vector'
                epsilon=np.delete(epsilon, idx)
                currentepsilonsol=np.delete(currentepsilonsol, idx)
                'Core candiates need to be added as gradient info with high error'

                print("  Set {} as core points.".format(NC))
                N += NC
                NC = 0

            else:
                print("All points may have sufficient influence on the global parameter error.")
                print("Transfer all candidate points to core points.")
                print("  Set {} as core points.".format(NC))
                nrofdeletedpoints=nrofnewpoints
                #print("  Add {} new candidate points.".format(nrofdeletedpoints))
                print("\n")

                N += NC
                NC=0

            """ ------------------------------ MC GLOBAL ERROR ESTIMATION ------------------------------ """
            ' Global parameter estimate after optimisation and filtering.'
            print("--- A posteriori error estimate")
            print("Estimate derivatives for a posteriori estimation")
            dfGLEE = gp.predictderivative(XGLEE, True)
            print("...done")
            varGLEE = gp.predictvariance(XGLEE)

            mcglobalerrorafter = MCGlobalEstimate(w,varGLEE,NGLEE,parameterranges)
            file.write( str(mcglobalerrorafter[0]))
            print("Global error estimate after optimization:   {:1.8f}".format(mcglobalerrorafter[0]))
            print("Computational cost after optimization:      {:0.0f}".format(currentcost))
            print("\n")
            vsol=currentepsilonsol**(-2)  #Filtered solution

            currentcost=totalcompwork(vsol, s)

            costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter[0]))
            costerrorlog.write("\n")

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

            'Add new candidate points'
            print("--- Acquisition phase")
            XC = np.array([])
            while XC.size == 0:
                print(" Adjusting acquisition tolerance")
                XC,Xdummy = acquisitionfunction(gp,XGLEE,epsphys,TOLAcqui,XCdummy)
                TOLAcqui*=0.99
                print(" Current tolerance {}".format(TOLAcqui))
            print(" Number of possible candidate points {}".format(XC.shape[0]))

            NC = XC.shape[0]
            epsXc = 1E20*np.ones((1, XC.shape[0]))  # eps**2
            meanXc = gp.predictmean(XC)
            gp.adddatapoint(XC)
            gp.adddatapointvalue(meanXc)
            gp.addaccuracy(epsXc)

            TOLAcqui = 1.0
            print("Reset tolerance to {} for next design.".format(TOLAcqui))

            print("\n")

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

            """ ---------------------------------------------------------- """
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
            print("  Maximum error estiamte: {:1.7f}".format(maxpwe[0]))
            print("  New \u0394W: {:0.4f}".format(deltaw))
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
        epsilon=vsol**(-1/2)  # epsilon

        if XC.size!=0:
            if XC.shape[0] == 1:
                epsilon=np.concatenate((epsilon, np.squeeze(epsXc, axis=0)))
            elif XC.shape[0] > 1:
                epsilon=np.concatenate((epsilon, np.squeeze(epsXc)))

        epsilon=epsilon**2
        t1design=time.perf_counter()

        print("Time used for complete design iteration: {:0.2f} seconds".format(t1design-t0design))
        print("\n")

        file.write("\n")
        file.close()
        costerrorlog.close()
