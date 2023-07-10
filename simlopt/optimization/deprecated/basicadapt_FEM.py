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
from mayavi import mlab
from mayavi.mlab import *

def estiamteweightfactors(var, X, dy, epsphys):

    dim = X.shape[1]
    delta = 1E-6

    #m = dy.shape[2]
    m = 1
    w = np.zeros((X.shape[0]))
    #dy = DEBUGGRADIENDTS(X)
    'Check if epsphys might be a matrix'
    if isinstance(epsphys, (np.floating, float)):
        SigmaLL = epsphys*np.eye(m)
        #SigmaLL = np.linalg.inv(SigmaLL)
        SigmaLL = np.linalg.inv(SigmaLL)

    for i, x in enumerate(X):
        #x = x.reshape((1, -1))
        #varvec = var[i]*np.ones((m))
        # Jprime = dyatp.T @ SigmaLL @ varvec  , was standard 8.7.2022 might be wrong since this the derivative of J not r_i !
        # w[i] = np.linalg.norm((np.linalg.inv(np.outer(Jprime.T, Jprime)+delta*np.eye((dim)))@Jprime).reshape((1, -1)), 2)
        #dyatp = dy[i, :, :].T
        #dyatp = dy[i, :, :].T
        dyatp = dy[i, :].T
        Jprime = dyatp.T
        #w[i] = np.linalg.norm((np.linalg.inv(Jprime.T@Jprime+delta*np.eye((m)))@Jprime.T).reshape((1, -1)), np.inf)
        #w[i] = np.linalg.norm((np.linalg.inv(Jprime.T@Jprime+delta*np.eye((m)))@Jprime.T).reshape((1, -1)), 2)
        w[i] = np.linalg.norm(( np.linalg.inv((SigmaLL*np.outer(Jprime.T,Jprime)+delta*np.eye((dim)) )) @ (SigmaLL*Jprime.reshape((-1, 1))))  , 2)
    return w


def MCGlobalEstimate(w,var,Nall,parameterranges):
    #var = np.linalg.norm(var*np.ones((1,4)),axis=1)
    volofparameterspace = np.prod(parameterranges[:, 1])
    #return np.array([(volofparameterspace/Nall) * np.dot(w,var)])
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

""" Acquisition function """
def acquisitionfunction(gp,df,var,w,XGLEE,epsphys,TOLAcqui,XCdummy=None):

    'Calculate error distribution with new data for '
    #w = estiamteweightfactors(var, XGLEE, df, epsphys)
    acquisition = np.sqrt(var)* w.reshape((-1,1))

    acquidx = np.where(acquisition >= np.max(acquisition)*TOLAcqui)

    if acquidx[0].size != 0:
        XC = XGLEE[acquidx[0]]
        'Check if XC is alread in X, if so, delete points form XC'
        for i in range(gp.getX.shape[0]):
            currentindex = np.where((XC == gp.getX[i,:].tolist()).all(axis=1))
            if currentindex[0].size != 0:
                #print(" Doublicate found.")
                #print(" Delete {} from candidate points".format(gp.getX[i,:]))
                XC = np.delete(XC,(currentindex[0][0]),axis=0)
                if XCdummy is not  None:
                    XCdummy = np.vstack((XCdummy,gp.getX[i,:]))
            if XC.size == 0:
                if XCdummy is not None:
                    return XC,XCdummy
                else:
                    return XC
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

def adapt(gp, totalbudget, incrementalbudget, parameterranges,
          TOL, TOLFEM, TOLFILTER, TOLAcqui,TOLrelchange, epsphys,
          nrofnewpoints,  runpath, execpath, execname, adaptgrad):

    'Problem dimension'
    dim= gp.getdata[2]

    'Counter variables'
    counter= 0
    totaltime= 0
    totalFEM= 0
    nosolutioncounter= 0
    itercounter= 0

    'Solver options'
    xtol = 1*1E-4
    gtol = 1*1E-4
    itermax = 100000
    s = 2

    """
    If withgradient is True, we check if there is already gradient data.
    otherwise we need to at as much gradientdata as trainig data is already available """
    N = gp.getdata[0]

    cases = {1:"Case 1: Gradient data is not available.",
             2:"Case 1: Gradient data is available."}

    'Check for which cases are set.'

    if gp.getXgrad is None:
        Ngrad = gp.getdata[1] #Is None, when Xgrad is None
        case = 1
    elif gp.getXgrad is not None:
        Ngrad = gp.getdata[1]
        case = 2

    print("---------------------------------- Start adaptive phase")
    print(cases[case])
    print("Number of initial points:          "+str(N))
    print("Total budget:                      "+str(totalbudget))
    print("Desired tolerance:                 "+str(TOL))
    print("Number of adaptively added points: "+str(nrofnewpoints))
    print("Workmodel exponent:                "+str(s))
    print("\n")

    "Open logs"
    logpath = os.path.join(runpath+"/", "logs/")
    logpath_general = os.path.join(runpath+"/")
    figurepath = os.path.join(runpath+"/", "iteration_plots/")

    ' Empty array for possible dummy points - for printing'
    XCdummy= np.empty((0,dim))

    ' Initial acquisition phase '
    NMC = 15
    XGLEE = createPD(NMC, dim, "grid", parameterranges)
    dfXC = gp.predictderivative(XGLEE, True)
    varXC = gp.predictvariance(XGLEE)
    w = estiamteweightfactors(varXC, XGLEE, dfXC, epsphys)
    NGLEE = XGLEE.shape[0]
    mcglobalinitial = MCGlobalEstimate(w,varXC,NGLEE,parameterranges)

    'Epsilon^2 at this points'
    epsilon = np.squeeze(gp.getaccuracy)
    currentcost= totalcompwork(1/epsilon,s)

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

        N = gp.getX.shape[0]

        ' Logging '
        try:
            costerrorlog = open(logpath+"costerror.txt","a")
            file = open(logpath_general+"log.txt","a")
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
        print("Current number of candidate points: {} ".format(NC))
        print("\n")

        """ ------------------------------ MC GLOBAL ERROR ESTIMATION ------------------------------ """
        dfGLEE  = gp.predictderivative(XGLEE, True)
        varGLEE = gp.predictvariance(XGLEE)
        w = estiamteweightfactors(varGLEE, XGLEE, dfGLEE, epsphys)
        mcglobalerrorbefore = MCGlobalEstimate(w,varGLEE,NGLEE,parameterranges)
        file.write( str(mcglobalerrorbefore[0]) + str(" "))

        XCdummy = np.empty((0,dim))
        print("Global error estimate before optimization:   {:1.8f}".format(mcglobalerrorbefore[0]))
        print("Computational cost before optimization:      {:0.0f}".format(currentcost))
        print("\n")

        """ ------------------------------Acquisition phase ------------------------------ """
        'Add new candidate points'
        print("--- Acquisition phase")
        #NMC = 25
        #XGLEE = createPD(NMC, dim, "grid", parameterranges)

        XC = np.array([])
        while XC.size == 0:
            print(" Adjusting acquisition tolerance")
            XC,Xdummy = acquisitionfunction(gp,dfGLEE,varGLEE,w,XGLEE,epsphys,TOLAcqui,XCdummy)
            TOLAcqui*=0.999999
            if TOLAcqui < 0.5:
                print("No new candidate points were found. Lower overall accuracy.")
                print(" Current tolerance {}".format(TOLAcqui))
                XC = np.array([])
                NC = 0
                break
            print(" Current tolerance {}".format(TOLAcqui))
        print(" Number of possible candidate points {}".format(XC.shape[0]))
        mlab.figure(size=(1000, 1000))
        p = mlab.points3d(XGLEE[:,0], XGLEE[:,1],XGLEE[:,2], np.squeeze(np.sqrt(varGLEE)* w.reshape((-1,1))), colormap = 'gnuplot')
        axes = mlab.axes(color=(0, 0, 0), nb_labels=4)
        axes.title_text_property.color = (0.0, 0.0, 0.0)
        axes.title_text_property.font_family = 'times'
        mlab.orientation_axes()
        mlab.colorbar(orientation='horizontal',nb_labels=5)
        figurepath = os.path.join(runpath+"/", "iteration_plots/")
        mlab.savefig(figurepath+"lee_iter_"+str(counter)+'.png')
        mlab.close()

        TOLAcqui = 1.0
        print("Reset tolerance to {} for next design.".format(TOLAcqui))
        print("\n")

        ' Add found candidate point '
        if XC.size != 0:
            NC = XC.shape[0]
            epsXc = 1E20*np.ones((1, XC.shape[0]))  # eps**2
            meanXc = gp.predictmean(XC)
            gp.adddatapoint(XC)
            gp.adddatapointvalue(meanXc)
            gp.addaccuracy(epsXc)

        mlab.figure( size=(1000, 1000))  # Make background white.
        mlab.points3d(gp.getX[:,0], gp.getX[:,1],gp.getX[:,2],scale_factor = 0.05,colormap = 'gnuplot')
        axes = mlab.axes(color=(0, 0, 0), nb_labels=5)
        axes.title_text_property.color = (0.0, 0.0, 0.0)
        axes.title_text_property.font_family = 'times'
        mlab.orientation_axes()  # Source: <<https://stackoverflow.com/a/26036154/2729627>>.
        figurepath = os.path.join(runpath+"/", "iteration_plots/")
        mlab.savefig(figurepath+"pointcloud_"+str(counter)+'.png')
        mlab.close()


        """ ------------------------------ Solve minimization problem ------------------------------ """

        print("--- Solve minimization problem")

        'Turn epsilon^2 into v'
        epsilon = np.squeeze(gp.getaccuracy)
        v = 1/epsilon

        ' Current cost  '
        total_n= 0

        'Set start values'
        if counter == 0:
            #print("Used start value: {}".format( (incrementalbudget/Nall)**(1/s)))
            v[0:]= 10
            print(" Used start value for all points: {}".format(10))
            print("\n")
        else:
            'Adapt initial values by taking the max value of the new solution'
            if NC == 0:
                v[:Nall] += (incrementalbudget/Nall)**(1/s)
                print(" No candidate points found.")
                print(" Used start value just for candidate points: {}".format((incrementalbudget/Nall)**(1/s)))
                print("\n")
            else:
                if NC > 0:
                    #v[N:] += (incrementalbudget/NC)**(1/s)
                    v[N:] = 0.0
                else:
                    v[:Nall] += (incrementalbudget/Nall)**(1/s)
                print(" Used start value just for candidate points: {}".format((incrementalbudget/NC)**(1/s)))
                print("\n")

        ' Keep track of all points '
        Nall =  N + NC

        'Bounds on v'
        lowerbound= v.tolist()
        upperbound= [np.inf]*Nall
        bounds= Bounds(lowerbound, upperbound)
        bounds.lb[N:] = 0.0

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

        #Case 1 and 2
        X = gp.getX
        hyperparameter = gp.gethyperparameter
        var = gp.predictvariance(X)
        df = gp.predictderivative(gp.getX, True)
        wmin = estiamteweightfactors(var, X, df, epsphys)

        if gp.getXgrad is not None:
            K = kernelmatrixsgrad(X, gp.getXgrad, hyperparameter, gp.getaccuracy*0.0, gp.getgradientaccuracy*0.0)
            Ngrad = gp.getXgrad.shape[0]
            tensor = np.zeros((Nall+Ngrad*dim, Nall+Ngrad*dim, Nall))
            for kk in range(Nall):
                tensor[kk, kk, kk] = 1
        else:
            K = kernelmatrix(X, X, hyperparameter)
            tensor = np.zeros((Nall, Nall, Nall))
            for kk in range(Nall):
                tensor[kk, kk, kk] = 1

        args = (wmin, X, hyperparameter, K, Nall, tensor, parameterranges, adaptgrad)

        sol=scipy.optimize.minimize(targetfunction, v,
                                          args=args,
                                          method='trust-constr',
                                          jac=gradientoftargetfunction,
                                          bounds = bounds,hess=BFGS(),
                                          constraints=[nonlinear_constraint],
                                          options={'verbose': 1, 'maxiter': itermax, 'xtol': xtol, 'gtol': gtol})
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
            print("Found solution with:")
            print(vsol)
            currentcost=totalcompwork(vsol, s)

            'Solution for epsilon'
            currentepsilonsol=1/np.sqrt(vsol)
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

                    ' Get new values for calculated solution '
                    epsXtnew=currentepsilonsol[currentFEMindex].reshape((1, -1))

                    parameter = {"--v0":gp.getX[currentFEMindex, :][0],
                                 "--d1":gp.getX[currentFEMindex, :][1],
                                 "--d2":gp.getX[currentFEMindex, :][2],
                                 "--atol":np.sqrt(epsXtnew[0][0])}

                    runkaskade(execpath, execname, parameter)

                    'Read simulation data and get function value'
                    simulationdata = readtodict(execpath, "dump.log")

                    reached = np.asarray(simulationdata["flag"])

                    if reached is False:
                        print("Simulation did not reached target accuracy.")
                        print("Target accuray: {}".format(epsXtnew[0][0]))
                        print("Reached accuray: {}".format(simulationdata["accuracy"][0]))
                    else:
                        print("Simulation did reached target accuracy.")
                        print("Target accuray: {}".format(epsXtnew[0][0]))
                        print("Reached accuray: {}".format(simulationdata["accuracy"][0]))
                    print("\n")

                    #epsXtnew = np.asarray(simulationdata["accuracy"])
                    epsXtnew = np.atleast_2d(epsXtnew)

                    ytnew = np.asarray(simulationdata["value"])

                    ' Add new value to GP '
                    gp.addaccuracy(epsXtnew**2, currentFEMindex)
                    gp.adddatapointvalue(ytnew, currentFEMindex)

                t1FEM=time.perf_counter()
                totalFEM=t1FEM-t0FEM
                print("Simulation block done within: {:1.4f} s".format(totalFEM))
            #N += NC

            """ ------------------------------ MC GLOBAL ERROR ESTIMATION ------------------------------ """
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

            if mcglobalerrorafter[0] < TOL:
                print("--- Convergence")
                print(" Desired tolerance is reached, adaptive phase is done.")
                print(" Final error estimate: {:1.8f}".format(mcglobalerrorafter[0]))
                print(" Total used time: {:0.2f} seconds".format(totaltime+totalFEM))
                print(" Save everything !")
                gp.savedata(runpath+'/saved_data')
                costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter[0]))
                costerrorlog.write("\n")
                file.close()
                costerrorlog.close()
                return gp

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
                gp.savedata(runpath+'/saved_data')
                return gp

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

# =============================================================================
#         print("--- Adjust hyperparameter")
#         region = ((1,100),   (0.1, 1),   (0.1, 1) ,   (0.1, 1) )
#         gp.optimizehyperparameter(region, "mean", False)
#
# =============================================================================
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

        'Create new solution vector'
        epsilon=1/vsol  # epsilon

        t1design=time.perf_counter()

        print("Time used for complete design iteration: {:0.2f} seconds".format(t1design-t0design))
        print("\n")

        file.write("\n")
        file.close()
        costerrorlog.close()
