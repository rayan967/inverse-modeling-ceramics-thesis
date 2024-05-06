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
from jcm.helper import *
from jcm.martins_problem.scatterproblem import *

def updateDataPointValues(gp,epsilonsolutionbefore,epsilonsolcurrent,NofXt,TOLFEM,adaptgrad,jcmproblem):

    ' Create subvectors '
    currentpointsolutions = epsilonsolcurrent[:NofXt]
    if adaptgrad:
        currentgradsolutions = epsilonsolcurrent[NofXt:]

    epsilon = epsilonsolutionbefore

    dim = gp.getdata[2]

    ' Check if point accuracy was changed enough to recalculate points. '
    indicesofchangedpoints=np.where(np.abs(np.atleast_2d(epsilon[:NofXt]-currentpointsolutions)) > TOLFEM)
    if indicesofchangedpoints[1].size == 0:
        print("\n")
        print("No sufficient change between the solutions.")
        print("Solution is set as new optimal design.")
        gp.addaccuracy(currentpointsolutions,[0,None])
        if adaptgrad:
            gp.addgradaccuracy(currentgradsolutions,[0,None])

    # If there are points, we get the point index , i.e the parameters and recalculate the simulation until the desired accuracy is reached.
    else:
        print("\n")
        print("--- Start simulation block")
        print("Sufficient change in the solutions is detected, optain new simulation values")
        print("for point(s): {}".format(indicesofchangedpoints[1]))

        t0FEM=time.perf_counter()
        print("\n")

        for jj in range(indicesofchangedpoints[1].shape[0]):
            currentFEMindex=indicesofchangedpoints[1][jj]
            epsXtnew=currentpointsolutions[currentFEMindex].reshape((1, -1))
            values = gp.getX[currentFEMindex]
            #theta, phi = [5,7,9,11],[0]
            theta, phi = [5],[0]
            
            model_keys = {
                    "cd"   : float(values[0]),
                    "h"    : 48.3,
                    "swa"  : 87.98,
                    "t"    : float(values[1]),
                    "r_top": float(values[2]),
                    "r_bot": float(values[3])
                    }
                
            res = jcmproblem.model_eval_theta_phi(theta, phi, model_keys=model_keys, der_order=1)
            print("\n")
            
            if not res:
                print("Simulation failed, result is nan")
                print("Resulting")
                print("\n")             
            else:     
                
                ytnew   = np.zeros((gp.m))
                ygradnew = np.zeros((dim,gp.m))

                ' Since we get gradient information anyway, we immediately safe the data '
                for ii in range(len(theta)):
                                      
                    ytnew[ii] = res["Isim"]["P"]["phi_0"][ii]
                                
                    d_Isim_d_cd    = res["d_Isim_d_cd"]["P"]["phi_0"][ii]
                    d_Isim_d_r_bot = res["d_Isim_d_r_bot"]["P"]["phi_0"][ii]
                    d_Isim_d_r_top = res["d_Isim_d_r_top"]["P"]["phi_0"][ii]
                    d_Isim_d_t     = res["d_Isim_d_t"]["P"]["phi_0"][ii]
                    
                    ygradnew[:,ii] = np.array([d_Isim_d_cd,d_Isim_d_r_bot,d_Isim_d_r_top,d_Isim_d_t])
                    
            ' Add new value to GP '
            gp.addaccuracy(epsXtnew, currentFEMindex)
            gp.adddatapointvalue(ytnew, currentFEMindex)
                        
			# We need to check wether we need to adapt the gradients and if so, if the current FEM index is a new point                        
            if adaptgrad and currentFEMindex+1 >= NofXt:
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

                        'Add new value to GP'
                        gp.addgradaccuracy(epsXgradnew,currentgradindex)
                        gp.addgradientdatapointvalue(ygradnew[componentindex],currentgradindex)
                        
                    t1FEM=time.perf_counter()
                    totalFEM=t1FEM-t0FEM
                    #print("Gradient simulation block done within: {:1.4f} s".format(totalFEM))
        t1FEM=time.perf_counter()
        totalFEM=t1FEM-t0FEM
        print("Simulation block done within: {:1.4f} s".format(totalFEM))
        print("\n")


    return gp

def optimizeaccuracies(gp,N,Ngrad,XC,s,epsphys,incrementalbudget,parameterranges,graddataavailable,logger):

    'Initial parameter'
    xtol = 1E-5
    gtol = 1E-5
    itermax = 100000
    dim = gp.getdata[2]
    m = gp.m

    solutionlist = list()
    originallist = list()

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
    v = epsilon**(-1)
    currentcost = totalcompwork(v, s)

    cases = {0: "------ 1: Minimize without graddata at candidate point.",
             1: "------ 2: Minimize with graddata at candidate point."}

    for i in range(2):

        print(cases[i])
        logger.addToFile("------------ " + cases[i])

        """ ------- GRAD DATA NOT PRESENT """
        if graddataavailable == False:

            'Set start value for candidate point'
            v[N:] = 10.0

            'Bounds on v'
            lowerbound= v.tolist()
            upperbound= [np.inf]*(N+NC)
            bounds= Bounds(lowerbound, upperbound)
            bounds.lb[N:] = 1.0 
            
            ' Get GP data '
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

                vgrad[:] = 10

                'Bounds on vgrad'
                lowerboundgrad = vgrad.tolist()
                upperboundgrad = [np.inf]*NC*dim
                boundsgrad = Bounds(lowerboundgrad,upperboundgrad)
                boundsgrad.lb[:] = 1

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

        """ ------- GRAD DATA PRESENT """
        if graddataavailable == True:

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

                K = np.zeros((N+NC+Ngrad*dim,N+NC+Ngrad*dim,m))
                for jj in range(m):
                    K[:,:,jj] =kernelmatrixsgrad(X, Xgrad, hyperparameter[jj,:],
                                                gp.getaccuracy*0.0, gp.getgradientaccuracy*0.0)

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
                Xgrad = gp.getXgrad

                epsilongrad = np.squeeze(gp.getgradientaccuracy)
                vgrad = epsilongrad**(-1)

                'Set new bounds'
                vgrad[Ngrad*dim:] = 10.0
                lowerboundgrad    = vgrad.tolist()
                upperboundgrad    = [np.inf]*((Ngrad+NC)*dim)
                boundsgrad        = Bounds(lowerboundgrad,upperboundgrad)
                boundsgrad.lb[Ngrad*dim:] = 1.0
                
                'Get Kernelmatrices'
                K = np.zeros((N+NC+(Ngrad+NC)*dim, N+NC+(Ngrad+NC)*dim,m))
                for jj in range(m):
                    K[:,:,jj] = kernelmatrixsgrad(X, Xgrad, hyperparameter[jj,:],
                                                  gp.getaccuracy*0.0,
                                                  gp.getgradientaccuracy*0.0)

                tensor = np.zeros((N+NC+(Ngrad+NC)*dim, 
                                   N+NC+(Ngrad+NC)*dim, 
                                   N+NC+(Ngrad+NC)*dim))
                tensor[np.diag_indices(N+NC+(Ngrad+NC)*dim,ndim=3)] = np.ones((N+NC+(Ngrad+NC)*dim))

                optimizegradientdata = True

            'Connect bounds'
            lower = np.concatenate((bounds.lb,boundsgrad.lb))
            upper = np.concatenate((bounds.ub,boundsgrad.ub))

            'Build final bound object'
            bounds = Bounds(lower, upper)

            'Combine vs'
            v = np.concatenate((v,vgrad))

        """ SOLVE OPTIMIZATION """
        nonlinear_constraint= NonlinearConstraint(lambda x: compworkconstrain(x,s),
                                                  currentcost+incrementalbudget,
                                                  currentcost+incrementalbudget,
                                                  jac=lambda x: compworkconstrainjac(x,s))

        args = (wmin, X, K, N+NC, tensor,
                parameterranges, optimizegradientdata)
        method = 'SLSQP'

        sol=scipy.optimize.minimize(targetfunction, v,
                                  args=args,
                                  method=method,
                                  jac=gradientoftargetfunction,
                                  bounds = bounds,
                                  constraints=[nonlinear_constraint],
                                  options={'disp': 1, 'maxiter': itermax, 'ftol': xtol})
        print("\n")
        if sol.success == True and graddataavailable == False:
            vsol = sol.x
            solutionlist, originallist = setsolution(vsol,v,gp,N,NC,dim,i,graddataavailable,solutionlist,originallist,logger)

        elif sol.success == True and graddataavailable == True:
            vsol = sol.x
            solutionlist, originallist = setsolution(vsol,v,gp,N,NC,dim,i,graddataavailable,solutionlist,originallist,logger)

        elif sol.success != True:
            print("Solution not found")
            logger.addToFile("No solution found")
            logger.addToFile("\n")
             
    return solutionlist, originallist
    
def setsolution(vsol,v,gp,N,NC,dim,i,graddataavailable,solutionlist,originallist,logger):

    if graddataavailable == False:

        if i == 0:
            print("Found point solution:")
            prettyprintvector(vsol, dim, False)
            print("\n")

            logger.addToFile("Found solution with:")
            logger.addToFile(str(vsol))
            logger.addToFile("\n")

            costofsolution = totalcompwork(vsol, 4/3)
            print("Cost of solution: {}".format(costofsolution))
            print("\n")
            
            logger.addToFile("Cost of solution:")
            logger.addToFile(str(costofsolution))
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

            costofsolution = totalcompwork(vsol[:N+NC], 4/3)
            costofgradsolution = totalcompwork(vsol[N+NC:], 4/3)
            print("Cost of solution         : {}".format(costofsolution))
            print("Cost of gradient solution: {}".format(costofgradsolution))
            print("\n")
            
            logger.addToFile("Cost of solution:")
            logger.addToFile(str(costofsolution))
            logger.addToFile("\n")
            
            logger.addToFile("Cost of gradient solution:")
            logger.addToFile(str(costofgradsolution))
            logger.addToFile("\n")

            tmpGPR2 = deepcopy(gp)
            epsXt =1/vsol[:N+NC]
            epsXgrad = 1/vsol[N+NC:]
            tmpGPR2.addaccuracy(epsXt,[0,None])
            tmpGPR2.addgradaccuracy(epsXgrad,[0,None])

            solutionlist.append(tmpGPR2)
            originallist.append(v)

        return solutionlist,originallist

    elif graddataavailable == True:

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

            costofsolution     = totalcompwork(vsol[:N+NC], 4/3)
            costofgradsolution = totalcompwork(vsol[N+NC:], 4/3)
            print("Cost of solution         : {}".format(costofsolution))
            print("Cost of gradient solution: {}".format(costofgradsolution))
            print("\n")
            
            logger.addToFile("Cost of solution:")
            logger.addToFile(str(costofsolution))
            logger.addToFile("\n")
            
            logger.addToFile("Cost of gradient solution:")
            logger.addToFile(str(costofgradsolution))
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

            costofsolution     = totalcompwork(vsol[:N+NC], 4/3)
            costofgradsolution = totalcompwork(vsol[N+NC:], 4/3)
            print("Cost of solution         : {}".format(costofsolution))
            print("Cost of gradient solution: {}".format(costofgradsolution))
            print("\n")
            
            logger.addToFile("Cost of solution:")
            logger.addToFile(str(costofsolution))
            logger.addToFile("\n")
            
            logger.addToFile("Cost of gradient solution:")
            logger.addToFile(str(costofgradsolution))
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
            TOL,TOLFEM,TOLAcqui,TOLRELCHANGE,epsphys,
            runpath, execname, adaptgrad , fun):

    'Problem dimension'
    dim= gp.getdata[2]

    'Counter variables'
    counter= 0
    totaltime= 0
    totalFEM= 0
    graddataavailable = False

    'Solver options'
    xtol = 1E-5
    itermax = 10000
    s = np.round(2/3,2)

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

    "Create JCM problem"
    daemon_args =  {"Hostname":"localhost","NThreads":8}
    jcmproblem = MartinsProblem(daemon_args)

    "Open logs"
    logpath = os.path.join(runpath+"/", "logs/")
    figurepath = os.path.join(runpath+"/", "iteration_plots/")

    ' Initial acquisition phase '
    NMC = 8
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
    
    try:
        costerrorlog = open(logpath+"costerror.txt","a")
    except IOError:
      print ("Error: File does not appear to exist.")
      return 0
  
    costerrorlog.write(str(currentcost)+" "+str(mcglobalinitial))
    costerrorlog.write("\n")
    costerrorlog.close()
# =============================================================================
#     if gp.getXgrad is not None:
#         currentcost+= totalcompwork(gp.getgradientaccuracy**(-1),s)
# =============================================================================

    print("Initial point accurcies")
    prettyprintvector(np.squeeze(gp.getaccuracy**(-1)), dim, False)
    print("\n")

    ' If gradient data is available add the costs to the current cost'
    if graddataavailable:
        #epsilongrad = np.squeeze(gp.getgradientaccuracy)
        #currentcost += totalcompwork(epsilongrad**(-1),s)
        print("Initial point accurcies")
        prettyprintvector(np.squeeze(gp.getgradientaccuracy**(-1)), dim, True)

    
    
    while currentcost < totalbudget:
        
        #'Test - sample XGLEE randomly '
        #NMC = 8
        #XGLEE = createPD(NMC, dim, "random", parameterranges)
        
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
        vprior   = 1/epsprior
        currentcostprior = np.round(totalcompwork(vprior, s),0)

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
                
            'Round digits because of JCM'
            XC_rounded = np.round(XC,5)
            print(" Number of possible candidate points: {}".format(XC_rounded.shape[0]))
            print(" Found canditate point(s):            {}".format(XC_rounded[0]))
            print(" Use ith highest value   :            {}".format(index))
            print(" Value at index          :            {}".format(value))
            print("\n")

            logger.addToFile(" Number of possible candidate points: {}".format(XC_rounded.shape[0]))
            logger.addToFile(" Found canditate point(s):            {}".format(XC_rounded[0]))
            logger.addToFile(" Use ith highest value   :            {}".format(index))
            logger.addToFile(" Value at index          :            {}".format(value))
            logger.addToFile("\n")

            NC = 1

        #plotiteration(gp,w,normvar_TEST,N,Ngrad,XGLEE,XC,mcglobalerrorbefore,parameterranges, figurepath,counter)
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
            solutionlist,initialvalues = optimizeaccuracies(gp,N,Ngrad,XC_rounded,s,epsphys,incrementalbudget,parameterranges,graddataavailable,logger)
        else:
            solutionlist,initialvalues = optimizeaccuracies(gp,N,Ngrad,XC_rounded,s,epsphys,incrementalbudget,parameterranges,graddataavailable,logger)

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
            vpost   = 1/epspost
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
                if  relativechange < TOLRELCHANGE:

                    print("Relative change between errors is to small. Neglect optimized gradients.")
                    print("Relative change: {}".format(relativechange))
                    print("\n")
                    logger.addToFile("Relative change between errors is to small. Neglect optimized gradients.")
                    logger.addToFile("Relative change: {}".format(relativechange))
                    logger.addToFile("\n")

                    currentbesterror = mcglobalerrorafter
                    bestcase = 0
                    if graddataavailable:
                        epspost = solutionlist[0].getaccuracies
                    else:
                        epspost = solutionlist[0].getaccuracy
                    bestsolution = 1/epspost
                    optimizegradientdata = False
                    gp = solutionlist[0]
                    epsilon = 1/initialvalues[0]
                    breaked = True

                elif mcglobalerrorafter < currentbesterror and breaked == False:
                    bestcase = i
                    epspost = solutionlist[1].getaccuracies
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

        #currentcost=totalcompwork(bestsolution, s)

        foundcases = {0: "Add point without gradient data.",
                      1: "Add point with gradient data."}

        print("--- Optimization summary")
        print(foundcases[bestcase])
        print("Add point: {}".format(XC[0,:]))
        print("Point accuracies")
        prettyprintvector(bestsolution[0,:N+NC], dim, False)
        if bestcase == 1:
            print("Gradient accuracy")
            prettyprintvector(bestsolution[0,N+NC:], dim, True)
        print("\n")

        logger.addToFile("--- Optimization summary")
        logger.addToFile(str(foundcases[bestcase]))
        logger.addToFile("Add point: {}".format(XC[0,:]))
        logger.addToFile("Point accuracy: {}".format(bestsolution[0,:N+NC]))
        if bestcase == 1:
            logger.addToFile("Gradient accuracy: {}".format(bestsolution[0,N+NC:]))
        #logger.addToFile(str(currentcost)+" "+str(mcglobalerrorafter))
        logger.addToFile("\n")

        """ ------------------------------ Set data ------------------------------ """
        epspost = np.squeeze(epspost)
        gp = updateDataPointValues(gp, epsilon, epspost,
                                   N+NC , TOLFEM,
                                   optimizegradientdata, jcmproblem)
        
        currentcost=totalcompwork(1/gp.getaccuracy, s)
        costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter))
        costerrorlog.write("\n")
        
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

        """ ------------------------------ Adjusting ------------------------------ """
        Nmax = 90
        if N < Nmax:
            print("--- A posteriori hyperparameter adjustment")
            #region = ((0.01, None),   (0.01, None),   (0.01, None),   (0.01, None))
            region = ((0.01, 4),(0.01, 4),(0.01, 4),(0.01, 5))
            region = ((0.01, 6),(0.01, 6),(0.01, 6),(0.01, 5))

            gp.optimizehyperparameter(region, "mean", False)
        else:
            print("--- A posteriori hyperparameter adjustment")
            print("Number of points is higher then "+str(Nmax))
            print("No optimization is performed")
        print("\n")

        print("--- Adapt budget")
        print("Prior incremental budget:   {}".format(incrementalbudget))
        logger.addToFile("Prior incremental budget: {}".format(incrementalbudget))
        incrementalbudget *= 1.1
        incrementalbudget = np.round(incrementalbudget,0)
        print("New incremental budget:     {}".format(incrementalbudget))
        logger.addToFile("New incremental budget:   {}".format(incrementalbudget))
        logger.addToFile("\n")
        
        
        """ ------------------------------ New design ------------------------------ """
        epsilon = epspost
        print("New budget for next design: {:0.0f}".format(currentcost+incrementalbudget))
        print("\n")
        logger.addToFile("New budget for next design: {:0.0f}".format(currentcost+incrementalbudget))
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