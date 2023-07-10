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
        #Jprime = dyatp.T @  (var[i]*np.ones((m,1)))
        #w[i] = np.linalg.norm((np.linalg.inv(np.outer(Jprime.T, Jprime)+delta*np.eye((dim)))@Jprime).reshape((1, -1)), 2) #30.6:13:56

        Jprime = dyatp.T @ SigmaLL @ varvec
        #Jprime = dyatp.T @unitvec
        w[i] = np.linalg.norm((np.linalg.inv(np.outer(Jprime.T, Jprime)+delta*np.eye((dim)))@Jprime).reshape((1, -1)), 2)

    return w


def MCGlobalEstimate(w,var,Nall,parameterranges):
    var = np.linalg.norm(var*np.ones((1,4)),axis=1)
    volofparameterspace = np.prod(parameterranges[:, 1])
    return np.array([(volofparameterspace/Nall) * np.dot(w,var)])


def targetfunction(v, w, X, yt, hyperparameter, KXX, Nall,tensor, parameterranges):

    v = v.reshape((1, -1))

    sigma = hyperparameter[0]
    L = hyperparameter[1:]
    volofparameterspace = np.prod(parameterranges[:, 1])

    errorsum = 0

    'Inverse in eps for df'
    KXXdf = KXX+np.diagflat(v**(-1))
    alpha = np.linalg.solve(KXXdf, yt)

    'Inverse of KXX'
    invKXX = np.linalg.inv(KXX+1E-6*np.eye((X.shape[0])))

    'Inverse of KXX-1+V'
    invKV = np.linalg.inv(invKXX+np.diagflat(v))

    'Unit matrix from euclidean vector'
    unitmatrix = np.eye(X.shape[0])

    globalerrorestimate = (volofparameterspace/Nall) * np.dot(np.diag(invKV),w)
    return globalerrorestimate

def gradientoftargetfunction(v, w, X, yt, hyperparameter, KXX, Nall,tensor, parameterranges):

    v = v.reshape((1, -1))

    sigma = hyperparameter[0]
    Lhyper = hyperparameter[1:]
    volofparameterspace = np.prod(parameterranges[:, 1])

    errorgradsum = 0
    errorsum = 0

    'Inverse in eps for df'
    KXXdf = KXX+np.diagflat(v**(-1))
    alpha = np.linalg.solve(KXXdf, yt)

    'Inverse of KXX'
    invKXX = np.linalg.inv(KXX+1E-6*np.eye((X.shape[0])))
    # invKXX[np.abs(invKXX) < 1E-6] = 0.0

    'Inverse of KXX-1+V'
    invKV = np.linalg.inv(invKXX+np.diagflat(v))
    # invKV[np.abs(invKV) < 1E-6] = 0.0
    t0grad = time.perf_counter()

    for i in range(X.shape[0]):
        gradvar= np.diag(invKV@tensor[:,:,i]@invKV)
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
    acquidx = np.where(acquisition > np.max(acquisition)*TOLAcqui)

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


""" Inequality functions """
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


def adapt(gp, totalbudget, budgettospend, parameterranges,
          TOL, TOLFEM, TOLFILTER, TOLAcqui,TOLrelchange, epsphys,
          nrofnewpoints, execpath, execname):

    errorlist= []
    realerrorlist= []
    costlist= []
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

    N = gp.getdata[0]

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
    acquisition = varXC* w.reshape((-1,1))
    acquidx = np.where(acquisition > np.max(acquisition)*0.85)
    NGLEE = XGLEE.shape[0]
    mcglobalerrorafter = MCGlobalEstimate(w,varXC,NGLEE,parameterranges)
    if acquidx[0].size != 0:
        XC = XGLEE[acquidx[0]]
        NC = XC.shape[0]
    print("Found {} potential maxima".format(acquidx[0].size))
    print("Create surrogate with {} candidate point".format((acquidx[0].size)))
    print("\n")

    ' Initial accuracy of candidate points '
    epsXc = 1E15*np.ones((1, XC.shape[0]))
    meanXc = gp.predictmean(XC,True)

    gp.adddatapoint(XC)
    gp.adddatapointvalue(meanXc)
    gp.addaccuracy(epsXc)

    'Epsilon^2 at this points'
    epsilon= np.squeeze(gp.getaccuracy)

    currentcost= totalcompworkeps(epsilon)
    costlist.append(currentcost)

    try:
        costerrorlog = open(logpath+"costerror.txt","a")
    except IOError:
          print ("Error: File does not appear to exist.")
          return 0

    costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter[0]))
    costerrorlog.write("\n")
    costerrorlog.close()

    while currentcost < totalbudget:

        Nall =  N+NC
        X, yt = gp.getX, gp.gety

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

        if counter == 0:
            file.write("Error estimate before"+" "+"Used budget"+" "+"Error estimate after")
            file.write("\n")

        t0design = time.perf_counter()
        print("---------------------------------- Iteration / Design: {}".format(counter))
        print("Current number of points: {} ".format(N))
        print("Current number of candidate points: {} ".format(NC))
        print("\n")

        print("--- A priori estimate")
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
        triang = tri.Triangulation(XGLEE[:,0], XGLEE[:,1])
        refiner = tri.UniformTriRefiner(triang)
        tri_refi, z_test_refi = refiner.refine_field(erroratx[:,0], subdiv=4)
        z_test_refi = np.abs(z_test_refi)
        axcontour.tricontour(tri_refi, z_test_refi,  linewidths=0.5, colors='k')
        cntr2 = axcontour.tricontourf(tri_refi, z_test_refi, cmap="RdBu_r")
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
            #print("Used start value: {}".format( (budgettospend/Nall)**(1/s)))
            v[:Nall]= 10
            print(" Used start value for all points: {}".format(10))
            print("\n")
        else:
            'Adapt initial values by taking the max value of the new solution'
            if NC == 0:
                v[:Nall] += (budgettospend/Nall)**(1/s)
                print(" No candidate points found.")
                print(" Used start value just for candidate points: {}".format( (budgettospend/Nall)**(1/s)))
                print("\n")
            else:
                v[N:] += (budgettospend/NC)**(1/s)
                print(" Used start value just for candidate points: {}".format( (budgettospend/NC)**(1/s)))
                print("\n")

        currentcost= totalcompwork(v, s)
        total_n= 0

        'Bounds on v'
        upperbound= [np.inf]*Nall
        lowerbound= v.tolist()
        bounds= Bounds(lowerbound, upperbound)
        #if counter > 0:
        bounds.lb[N:] = 1E-6

        'Nonlinear constraints'
        def compworkconstrain(v, s):
            return np.array([np.sum(v**s)])
        def compworkconstrainjac(v, s):
            return np.array([s*v**(s-1)])
        def compworkconstrainhess(x, v):
            s= 2
            return s*(s-1)*v[0]*np.diagflat(x**(s-2))

        'Recalculate w for given X'
        hyperparameter = gp.gethyperparameter
        KXX = kernelmatrix(X, X, hyperparameter)
        var = gp.predictvariance(X)
        wmin = estiamteweightfactors(var, X, df, epsphys)

        file.write( str(currentcost+budgettospend) + str(" ") )
        nonlinear_constraint= NonlinearConstraint(lambda x: compworkconstrain(x, s), currentcost+budgettospend, currentcost+budgettospend,
                                                   jac=lambda x: compworkconstrainjac(x, s),
                                                   hess=compworkconstrainhess)
        t0=time.perf_counter()

        'Calculate derivative tensor'
        unitmatrix = np.eye(X.shape[0])
        tensor = np.zeros((Nall, Nall, Nall))
        for kk in range(Nall):
            ei = unitmatrix[kk, :]
            tensor[:, :, kk] = np.outer(ei.T, ei)

        sol=scipy.optimize.minimize(targetfunction, v,
                                      args=(wmin, X, yt, hyperparameter, KXX,
                                            Nall, tensor, parameterranges),
                                      method='trust-constr',
                                      jac=gradientoftargetfunction,
                                      bounds = bounds,
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
            print("vsol: ")
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

                    ' Get new values for calcualted solution'
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
                if gp.getXgrad is not None:
                    Xcgrad=gp.getX[N:N+NC]
                    epsXcgrad=1E10 * np.ones((1, Xcgrad.shape[0]*dim))  # eps**2
                    dfXcgrad=gp.predictderivative(Xcgrad)
                    gp.addgradientdatapoint(Xcgrad)
                    gp.adddgradientdatapointvalue(dfXcgrad)
                    gp.addgradaccuracy(epsXcgrad)

                N += NC
                NC = 0

            else:
                print("All points may have sufficient influence on the global parameter error.")
                print("Transfer all candidate points to core points.")
                print("  Set {} as core points.".format(NC))
                nrofdeletedpoints=nrofnewpoints
                print("  Add {} new candidate points.".format(nrofdeletedpoints))
                print("\n")
                if gp.getXgrad is not None and NC > 0:
                    Xcgrad=gp.getX[N:N+NC]
                    epsXcgrad=1E10 * np.ones((1, Xcgrad.shape[0]*dim))  # eps**2
                    dfXcgrad=gp.predictderivative(Xcgrad)
                    gp.addgradientdatapoint(Xcgrad)
                    gp.adddgradientdatapointvalue(dfXcgrad)
                    gp.addgradaccuracy(epsXcgrad)

                N += NC
                NC=0

            """ ------------------------------ MC GLOBAL ERROR ESTIMATION ------------------------------ """
            ' Global parameter estimate after optimisation and filtering.'
            print("--- A posteriori estimate")
            print("Estiamte derivatives for a posteriori estimation")
            dfGLEE = gp.predictderivative(XGLEE, True)
            print("...done")
            varGLEE = gp.predictvariance(XGLEE)
            #w = estiamteweightfactors(stdGLEE, XGLEE, dfGLEE, SigmaLL, p0, SigmaP,"wlog")
            mcglobalerrorafter = MCGlobalEstimate(w,varGLEE,NGLEE,parameterranges)
            file.write( str(mcglobalerrorafter[0]))
            print("Global error estimate after optimization:   {:1.8f}".format(mcglobalerrorafter[0]))
            print("Computational cost after optimization:      {:0.0f}".format(currentcost))
            print("\n")
            vsol=currentepsilonsol**(-2)  #Filtered solution

            currentcost=totalcompwork(vsol, s)

            costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter[0]))
            costerrorlog.write("\n")

            errorlist.append(mcglobalerrorafter[0])
            costlist.append(currentcost)

            'If the error descreases too slow we add more budget to spend'
            relchange=np.abs(mcglobalerrorbefore[0]-mcglobalerrorafter[0]) / mcglobalerrorbefore[0]*100
            if relchange < TOLrelchange:
                print("Relative change: "+str(relchange))
                print(" Relative change is below set threshold.")
                print(" Adjusting TOLrelchange to {}".format(0.85))
                TOLAcqui = 0.85
    # =============================================================================
    #             if currentcost < totalbudget:
    #                 budgettospend *= 2
    #                 print("Adjust budget to spend")
    #                 print("  New budget to spend: {:0.1f}".format(budgettospend))
    #                 print("\n")
    #             else:
    #                 print("  Budget is not sufficient, stopt adaptive phase.")
    #                 print("\n")
    #                 print("Save everything...")
    #                 gp.savedata(execpath+'/saved_data')
    #                 return gp, errorlist, costlist
    # =============================================================================
            else:
                print(" No necessary budget adjustments.")
                print("\n")

            'Add new candidate points'
            print("--- Acquisition phase")
            method = "aqcuisistion"

            if method == "random":
                nrofdeletedpoints=nrofnewpoints
                NC=nrofdeletedpoints
                XC=createPD(NC, dim, "random", parameterranges)
                epsXc=1E20*np.ones((1, XC.shape[0]))  # eps**2
                meanXc=gp.predictmean(XC)

                gp.adddatapoint(XC)
                gp.adddatapointvalue(meanXc)
                gp.addaccuracy(epsXc)

            elif method == "aqcuisistion":

# =============================================================================
#                 'Calculate error distribution with new data for '
#                 dfXC = gp.predictderivative(XGLEE, True)
#                 varXC = gp.predictvariance(XGLEE)
#                 w = estiamteweightfactors(varXC, XGLEE, dfXC, epsphys)
#                 acquisition = varXC* w.reshape((-1,1))
#                 acquidx = np.where(acquisition > np.max(acquisition)*TOLAcqui)
#
#
#                 if acquidx[0].size != 0:
#
#                     XC = XGLEE[acquidx[0]]
#                     'Check if XC is alread in X, if so, delete points form XC'
#                     for i in range(gp.getX.shape[0]):
#                         currentindex = np.where((XC == gp.getX[i,:].tolist()).all(axis=1))
#                         if currentindex[0].size != 0:
#                             print(" Doublicate found.")
#                             print(" Delete {} from candidate points".format(gp.getX[i,:]))
#                             XC = np.delete(XC,(currentindex[0][0]),axis=0)
#                             XCdummy = np.vstack((XCdummy,gp.getX[i,:]))
# =============================================================================

                XC = np.array([])

                while XC.size == 0:

                    XC,Xdummy = acquisitionfunction(gp,XGLEE,epsphys,TOLAcqui,XCdummy)
                    TOLAcqui*=0.9

                #if XC.size != 0:
                NC = XC.shape[0]
                epsXc = 1E20*np.ones((1, XC.shape[0]))  # eps**2
                meanXc = gp.predictmean(XC)
                gp.adddatapoint(XC)
                gp.adddatapointvalue(meanXc)
                gp.addaccuracy(epsXc)
# =============================================================================
#                 if XC.size == 0:
#                     print(" All possible candidate points are points already in training data. Distribute CW.")
#                     print("\n")
#                 else:
#                     print(" No changes necessary.")
# =============================================================================
            TOLAcqui = 0.95

            if mcglobalerrorafter[0] < TOL:
                print("\n")
                print("--- Convergence ---")
                print("Desired tolerance is reached, adaptive phase is done.")
                print("Final error estimate: {:1.8f}".format(mcglobalerrorafter[0]))
                print("Total used time: {:0.2f} seconds".format(totaltime+totalFEM))
                print("Save everything !")
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
            print("New budget for next design:                      {:0.0f}".format(currentcost+budgettospend))
            print("Used time:                                       {:0.2f} seconds".format(total_n))
            print("\n")

        else:

            ' Set new start value for next design '
            vsol=sol.x

            ' Set new cummulated cost '
            currentcost=totalcompwork(vsol, s)
            costlist.append(currentcost)

            if globalerrorafter < TOL:
                print("\n")
                print("Desired tolerance is still reached, adaptive phase is done.")
                print(" Final error estimate: {:1.6f}".format(
                    globalerrorafter))
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
            NC=0  # There are no furhter candidate points

            print("Adjust budget to spend")
            print("  Maximum error estiamte: {:1.7f}".format(maxpwe[0]))
            print("  New \u0394W: {:0.4f}".format(deltaw))
            print("  New budget to spend: {:0.4f}".format(budgettospend))
            print("\n")
        counter += 1

        print("--- Adapt solver accuracy and budet")
        if sol.nit == 1:
            print("Solving was done within one iteration.")
            print("Increase accuracy to")
            xtol = xtol*0.5
            gtol = gtol*0.5
            print("xtol: {}".format(xtol))
            print("xtol: {}".format(gtol))
        else:
            print(" No necessary accuracy adjustments.")

# =============================================================================
#         ' Adapt hyperparameter '
#         region = ((5, 20),   (1, 5),   (1, 5))
#         gp.optimizehyperparameter(region, "mean", False)
# =============================================================================


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
