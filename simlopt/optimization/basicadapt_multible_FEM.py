import numpy as np

import time
from timeit import default_timer as timer

import scipy
from scipy import optimize

from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.optimize import SR1

import matplotlib.pyplot as plt

from basicfunctions.covariance.cov import *
from basicfunctions.utils.creategrid import *
from basicfunctions.kaskade.kaskadeio import *

from optimization.errormodel_new import *
from optimization.workmodel import *
from optimization.utilities import *

from gpr.gaussianprocess import *

from IOlogging.iotofile import *
from jcm.helper import *
from jcm.martins_problem.scatterproblem import *

""" 
ADAPTED CODE FOR THE USE OF JCM WAVE !!!!
""" 
'JCM imports'
from jcm.helper import *

def adapt(gp, totalbudget, incrementalbudget, parameterranges,
          TOL, TOLFEM, TOLAcqui,TOLrelchange, epsphys,
          runpath, execname, adaptgrad, fun):


    "Create JCM problem"
    daemon_args =  {"Hostname":"localhost","NThreads":8}
    jcmproblem = MartinsProblem(daemon_args)
# =============================================================================
#     from jcm.model_standin.model_standin import gpytorch_multichannel
#     import torch
#     coarse_data_csv = "jcm/model_standin/model_standin/coarse_results.csv"
#     coarse_gp = gpytorch_multichannel.MultChGP(coarse_data_csv, rel_noise_level=1e-2) 
# =============================================================================
    
    'Problem dimension'
    dim = gp.getdata[2]

    'Counter variables'
    counter = 0
    totaltime = 0
    totalFEM = 0

    'Solver options'
    xtol = 1*1E-3
    gtol = 1*1E-3
    itermax = 100000
    s = np.round(2/3,2)

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
        
    print("\n")
    print("---------------------------------- Start adaptive phase")
    print(cases[case])
    print("Number of initial points:          "+str(N))
    print("Total budget:                      "+str(totalbudget))
    print("Desired tolerance:                 "+str(TOL))
    print("Workmodel exponent:                "+str(s))
    print("\n")

    "Open logs"
    logpath = os.path.join(runpath+"/", "logs/")
    logpath_general = os.path.join(runpath+"/")
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

    ' Logging '
    try:
        costerrorlog = open(logpath+"costerror.txt","a")
        
    except IOError:
        print ("Error: File does not appear to exist.")
        return 0

    costerrorlog.write(str(currentcost)+" "+str(mcglobalinitial))
    costerrorlog.write("\n")
    costerrorlog.close()

    NC = 0
    while currentcost < totalbudget:

        N = gp.getX.shape[0]
        ' Logging '
        logger = IOToLog(os.path.join(runpath+"/","iteration_log"),"iteration_"+str(counter))
        try:
            costerrorlog = open(logpath+"costerror.txt","a")
            file = open(logpath_general+"log.txt","a")
            vlog = open(logpath_general+"vlog.txt","a")
        except IOError:
          print ("Error: File does not appear to exist.")
          return 0
        if counter == 0:
            file.write("Error estimate before"+" "+"Used budget"+" "+"Error estimate after")
            file.write("\n")

        t0design = time.perf_counter()
        print("---------------------------------- Iteration / Design: {}".format(counter))
        print("Current number of points: {} ".format(N))
        print("Current number of candidate points: {} ".format(NC))
        print("\n")

        logger.addToFile("---------------------------------- Iteration / Design: {}".format(counter))
        logger.addToFile("Current number of points: {} ".format(N))
        logger.addToFile("Current number of candidate points: {} ".format(NC))
        logger.addToFile("\n")

        'Log current GP hyperparameter'
        logger.addToFile("--- Used hyperparameters")
        for i in range(gp.m):
            hp = gp.gethyperparameter[i,:]
            hpstring = np.array2string(hp, precision=2, separator=',',suppress_small=True)
            logger.addToFile("Hyperparameter experiment "+str(i) +": " +hpstring)
        logger.addToFile("\n")

        """ ------------------------------ MC GLOBAL ERROR ESTIMATION ------------------------------ """
        print("--- A posteriori error estimate")
        logger.addToFile("--- A posteriori error estimate")
        
        t0apriori=time.perf_counter()
        dfGLEE  = gp.predictderivative(XGLEE, True)
        if np.any(np.isnan(dfGLEE)):
            print("STOP")
        varGLEE = gp.predictvariance(XGLEE,True)
        
        normvar = np.linalg.norm(np.sqrt(np.abs(varGLEE)),2,axis=0)**2
        w       = estiamteweightfactors(dfGLEE, epsphys)
        mcglobalerrorbefore = MCGlobalEstimate(w,normvar,NGLEE,parameterranges)
        file.write( str(mcglobalerrorbefore) + str(" "))

        'Calculate actual used work prior'
        epsprior = gp.getaccuracy
        vprior = 1/epsprior
        currentcostprior=totalcompwork(vprior, s)

        print("Global error estimate before optimization:   {:1.5f}".format(mcglobalerrorbefore))
        print("Computational cost before optimization:      {:0.3f}".format(currentcostprior))
        print("\n")
        
        logger.addToFile("Global error estimate before optimization:   {:1.5f}".format(mcglobalerrorbefore))
        logger.addToFile("Computational cost before optimization:      {:0.3f}".format(currentcostprior))
        logger.addToFile("\n")
        
        t1apriori=time.perf_counter()
        tapriori = t1apriori-t0apriori       
        
        """ ------------------------------Acquisition phase ------------------------------ """
        'Add new candidate points'
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
            
            NC = XC.shape[0]
            epsXc = 1E20*np.ones((1, XC.shape[0]))  # eps**2
            meanXc = gp.predictmean(XC,True)
            gp.adddatapoint(XC)
            gp.adddatapointvalue(meanXc)
            gp.addaccuracy(epsXc)

       # plotiteration(gp,w,normvar_TEST,N,Ngrad,XGLEE,XC,mcglobalerrorbefore,parameterranges, figurepath,counter)
        #plotderivativedifference(gp,fun,N,XGLEE,parameterranges,figurepath,counter)
        gp.savedata(runpath+'/saved_data/',str(counter))          
        
        t1acqui = time.perf_counter()
        tacqui = t1acqui-t0acqui            
        
        """ ------------------------------ Solve minimization problem ------------------------------ """
        print("--- Solve minimization problem")
        
        t0minimization = time.perf_counter()
        logger.addToFile("--- Solve minimization problem")

        'Turn epsilon^2 into v'
        epsilon = np.squeeze(gp.getaccuracy)
        v = epsilon**(-1)

        ' Current cost by adding initial values is added to the overall budget '
        currentcost= totalcompwork(v, s)
        file.write(str(currentcost) + str(" "))

        ' Keep track of all points '
        Nall =  N + NC

        if counter == 0:
            #v[0:]= 10
            #print(" Used start value for all points: {}".format(10))
# =============================================================================
#             print(" Use initial values as start values")
#             print("\n")
# =============================================================================
            #v[N:] =( np.min(v[:N])+np.max(v[:N]))/2
            v[N:] = 10
            print("Initial value for new point(s): {}".format(v[N:]))
            print("\n")
            logger.addToFile("Initial value for new point(s): {}".format(v[N:]))
            logger.addToFile("\n")
        else:
            'Adapt initial values by taking the max value of the new solution'
            v[N:] = 10
            #v[N:] =( np.min(v[:N])+np.max(v[:N]))/2
            print("Initial value for new point(s): {}".format(v[N:]))
            print("\n")
            logger.addToFile("Initial value for new point(s): {}".format(v[N:]))
            logger.addToFile("\n")

        'Bounds on v'
        lowerbound = v.tolist()
        upperbound = [np.inf]*Nall
        bounds = Bounds(lowerbound, upperbound)
        bounds.lb[N:] = 1

        #, hess=compworkconstrainhess
        nonlinear_constraint= NonlinearConstraint(lambda x: compworkconstrain(x,s), 
                                                  currentcost+incrementalbudget, 
                                                  currentcost+incrementalbudget,
                                                  jac=lambda x: compworkconstrainjac(x,s))
        #Case 1 and 2
        X = gp.getX
        hyperparameter = gp.gethyperparameter
        df = gp.predictderivative(gp.getX, True)
        wmin = estiamteweightfactors(df, epsphys)

        if gp.getXgrad is not None:
            K = kernelmatrixsgrad(X, gp.getXgrad, hyperparameter[0,:], gp.getaccuracy*0.0, gp.getgradientaccuracy*0.0)
            Ngrad = gp.getXgrad.shape[0]
            tensor = np.zeros((Nall+Ngrad*dim, Nall+Ngrad*dim, Nall))
            for kk in range(Nall):
                tensor[kk, kk, kk] = 1
        else:
            # Tensor for K
            m = gp.m
            K = np.zeros((X.shape[0], X.shape[0],m))
            for i in range(m):
                K[:,:,i] = kernelmatrix(X, X, hyperparameter[i,:])
            
            # Tensor of ones for the gradient
            tensor = np.zeros((Nall, Nall, Nall))
            tensor[np.diag_indices(Nall,ndim=3)] = np.ones((Nall))
                
        #hessianoftargetfunction
        method ='trust-constr'
        method = 'SLSQP'
        args = (wmin, X, K, Nall, tensor, parameterranges, adaptgrad)
        sol=scipy.optimize.minimize(targetfunction, v, args=args,
                                    method = method,
                                    jac=gradientoftargetfunction,
                                    bounds = bounds,
                                    constraints=[nonlinear_constraint],
                                    options={'verbose': 1, 'maxiter': itermax, 'xtol': xtol, 'gtol': gtol})
        t1minimization = time.perf_counter()
        total_n=t1minimization-t0minimization
        totaltime = total_n

        if sol.success == True:

            print("Solution found within {} iterations".format(sol.nit))
            print("Used time:            {:0.2f} seconds".format(total_n))
            print("Last function value:  {}".format(sol.fun))
            print("\n")

            logger.addToFile("Solution found within {} iterations".format(sol.nit))
            logger.addToFile("Used time:            {:0.2f} seconds".format(total_n))
            logger.addToFile("Last function value:  {}".format(sol.fun))
            logger.addToFile("\n")

            'Solution for v'
            vsol=sol.x
            currentcost=totalcompwork(vsol, s)
            print("Found solution with:")
            prettyprintvector(vsol, dim, False)

            logger.addToFile("Found solution with:")
            logger.addToFile(str(vsol))

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
                
                logger.addToFile("\n")
                logger.addToFile("--- Start simulation block")
                logger.addToFile("Sufficient change in the solutions is detected, optain new simulation values")
                logger.addToFile("for point(s): {}".format(indicesofchangedpoints[1]))

                t0FEM=time.perf_counter()
                print("\n")

                for jj in range(indicesofchangedpoints[1].shape[0]):
                    currentFEMindex=indicesofchangedpoints[1][jj]

                    ' Get new values for calculated solution '
                    epsXtnew=currentepsilonsol[currentFEMindex].reshape((1, -1))
        
                    cd = np.round_(gp.getX[currentFEMindex][0],4)
                    t  = np.round_(gp.getX[currentFEMindex][1],4)
                    r_top  = np.round_(gp.getX[currentFEMindex][2],4)
                    r_bot  = np.round_(gp.getX[currentFEMindex][3],4)
                       
                    
                    theta, phi = [5],[0]
                    
                    model_keys = {
                            "cd"   : cd,
                            "h"    : 48.3,
                            "swa"  : 87.98,
                            "t"    : t,
                            "r_top": r_top,
                            "r_bot": r_bot
                            }
                        
                    
# =============================================================================
#                     theta, phi = [5,7],[0]
#                     
#                     model_keys = {
#                             "cd"   : cd,
#                             "h"    : 48.3,
#                             "swa"  : 87.98,
#                             "t"    : t,
#                             "r_top": 10.94,
#                             "r_bot": 8.13
#                             }
# =============================================================================
                    print("Solve simulation for cd: {} and t: {}".format(cd,t))    
                    print("Solve simulation for r_top: {} and r_bot: {}".format(r_top,r_bot))  
                    res = jcmproblem.model_eval_theta_phi(theta, phi, model_keys=model_keys, der_order=1)
                    print("\n")
                    
                    if not res:
                        print("Simulation failed, result is nan")
                        print("Resulting")
                        print("\n")             
                    else:     
                        
                        ytnew   = np.zeros((gp.m))
                        for ii in range(len(theta)):                      
                            ytnew[ii] = res["Isim"]["P"]["phi_0"][ii]


# =============================================================================
#                     m = gp.m
#                     ytnew  = res["Isim"]["P"]["phi_0"]
# 
# =============================================================================
                    ' Add new value to GP '
                    gp.adddatapointvalue(ytnew, currentFEMindex)
                    gp.addaccuracy(epsXtnew**2, currentFEMindex)

                t1FEM=time.perf_counter()
                tFEM=t1FEM-t0FEM
                print("Simulation block done within: {:1.4f} s".format(tFEM))
                print("\n")
                
                logger.addToFile("Simulation block done within: {:1.4f} s".format(tFEM))
                logger.addToFile("\n")

            """ ------------------------------ MC GLOBAL ERROR ESTIMATION ------------------------------ """
            print("--- A posteriori error estimate")
            logger.addToFile("--- A posteriori error estimate")
            
            t0post = time.perf_counter()
            
            'Calculate actual used work'
            epspost = gp.getaccuracy
            vpost = 1/epspost
            currentcostposteriori=totalcompwork(vpost, s)
            
            dfGLEE = gp.predictderivative(XGLEE, True)
            varGLEE = gp.predictvariance(XGLEE,True)
            wpost = estiamteweightfactors(dfGLEE, epsphys)
            normvar = np.linalg.norm(np.sqrt(np.abs(varGLEE)),2,axis=0)**2

            mcglobalerrorafter = MCGlobalEstimate(wpost,normvar,NGLEE,parameterranges)
            file.write( str(mcglobalerrorafter))
            print("Global error estimate after optimization:   {:1.8f}".format(mcglobalerrorafter))
            print("Computational cost after optimization:      {:0.0f}".format(currentcost))
            print("\n")

            logger.addToFile("Global error estimate after optimization:   {:1.8f}".format(mcglobalerrorafter))
            logger.addToFile("Computational cost after optimization:      {:0.0f}".format(currentcost))
            logger.addToFile("\n")

            vsol=currentepsilonsol**(-2)  #Filtered solution
           
            costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter))
            costerrorlog.write("\n")
            
            t1post = time.perf_counter()
            tpost = t1post - t0post
            
            if mcglobalerrorafter <= TOL:
                print("--- Convergence")
                print(" Desired tolerance is reached, adaptive phase is done.")
                print(" Final error estimate: {:1.8f}".format(mcglobalerrorafter))
                print(" Total used time: {:0.2f} seconds".format(totaltime+totalFEM))
                print(" Save everything !")

                logger.addToFile("--- Convergence")
                logger.addToFile(" Desired tolerance is reached, adaptive phase is done.")
                logger.addToFile(" Final error estimate: {:1.8f}".format(mcglobalerrorafter))
                logger.addToFile(" Total used time: {:0.2f} seconds".format(totaltime+totalFEM))
                logger.addToFile(" Save everything !")

                gp.savedata(runpath+'/saved_data')

                costerrorlog.write(str(currentcost)+" "+str(mcglobalerrorafter))
                costerrorlog.write("\n")

                file.close()
                costerrorlog.close()

                return gp
            
            'If the error descreases too slow we add more budget to spend'
            print("--- Adjust budget")
            logger.addToFile("--- Adjust budget")
            relchange = np.abs(mcglobalerrorbefore-mcglobalerrorafter) / mcglobalerrorbefore*100
            if relchange < TOLrelchange:
                print("Relative change: "+str(relchange))
                print(" Relative change is below set threshold.")
                print(" Adjusting TOLrelchange !")
                logger.addToFile("Relative change: "+str(relchange))
                logger.addToFile(" Relative change is below set threshold.")
                logger.addToFile(" Adjusting TOLrelchange !")
                TOLAcqui*=0.9999
            else:
                print("Relative change is sufficient - no necessary budget adjustments.")
                print("\n")
                

                logger.addToFile("Relative change is sufficient - no necessary budget adjustments.")
                logger.addToFile("\n")

            print("--- New parameters")
            print("Error estimate after optimization and filtering: {:1.5f}".format(mcglobalerrorafter))
            print("Computational cost after optimization:           {:0.0f}".format(currentcost))
            print("New budget for next design:                      {:0.0f}".format(currentcost+incrementalbudget))
            print("Used time:                                       {:0.2f} seconds".format(total_n))
            print("\n")

            logger.addToFile("--- New parameters")
            logger.addToFile("Error estimate after optimization and filtering: {:1.5f}".format(mcglobalerrorafter))
            logger.addToFile("Computational cost after optimization:           {:0.0f}".format(currentcost))
            logger.addToFile("New budget for next design:                      {:0.0f}".format(currentcost+incrementalbudget))
            logger.addToFile("Used time:                                       {:0.2f} seconds".format(total_n))
            logger.addToFile("\n")

        else:

            ' Set new start value for next design '
            vsol=sol.x

            ' Set new cummulated cost '
            currentcost=totalcompwork(vsol, s)

            print("\n")
            print("No solution found.")
            print(" " + sol.message)
            print("Total used time: {:0.4f} seconds".format(totaltime))

            incrementalbudget *= 1.1
            N += NC  # All candidates go core
            NC = 0  # There are no furhter candidate points

            print("Adjust budget to spend")
            print("  New budget to spend: {:0.4f}".format(incrementalbudget))
            print("\n")

        counter += 1

        Nmax = 83
        if N < Nmax:
            print("--- A priori hyperparameter adjustment")
            region = ((0.01, 2),(0.01, 2))
            region = ((0.01, 6),(0.01, 6),(0.01, 6),(0.01, 5))
            #region = ((0.1, None),   (0.1, None))
            #region = ((0.1, 3),   (0.1, 3))

            gp.optimizehyperparameter(region, "mean", False)
        else:
            print("--- A priori hyperparameter adjustment")
            print("Number of points is higher then "+str(Nmax))
            print("No optimization is performed")
        print("\n")
        
        print("--- Adapt solver accuracy")
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

        print("--- Adapt buget")
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

        file.write("\n")
        file.close()
        costerrorlog.close()

        logger.addToFile("Times used for a priori error estimate       : {:0.2f} seconds".format(tapriori))
        logger.addToFile("Times used for acquisition phase             : {:0.2f} seconds".format(tacqui))
        logger.addToFile("Times used for solving minimization problem  : {:0.2f} seconds".format(totaltime))
        logger.addToFile("Times used for solving fem simulation        : {:0.2f} seconds".format(tFEM))
        logger.addToFile("Times used for a posteriori error estimate   : {:0.2f} seconds".format(tpost))
        logger.addToFile("Time used for complete design iteration      : {:0.2f} seconds".format(t1design-t0design))
        logger.addToFile("\n")
        logger.closeOutputLog()