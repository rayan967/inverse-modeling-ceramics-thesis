import numpy as np

import copy
import time
from timeit import default_timer as timer

import scipy
import scipy.optimize as optimize
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

from optimization.errormodel_new import *
from optimization.workmodel import *
from optimization.utilities import *

from gpr.gaussianprocess import *
from IOlogging.iotofile import *

from copy import copy
from copy import deepcopy
from jcm.helper import *
from jcm.martins_problem.scatterproblem import *

def adapt(gp, totalbudget, incrementalbudget, parameterranges,
          TOL, TOLFEM, TOLAcqui,TOLrelchange, epsphys,
          runpath, execname, adaptgrad, fun):

    'Problem dimension'
    dim = gp.getdata[2]

    'Counter variables'
    counter = 0
    totaltime = 0
    totalFEM = 0
    nosolutioncounter = 0
    itercounter = 0

    'Solver options'
    xtol = 1*1E-5
    gtol = 1*1E-5
    itermax = 100000
    s = np.round(4/3,2)

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
    print("Workmodel exponent:                "+str(s))
    print("\n")

    "Create JCM problem"
    daemon_args =  {"Hostname":"localhost","NThreads":8}
    jcmproblem = MartinsProblem(daemon_args)


    "Open logs"
    logpath = os.path.join(runpath+"/", "logs/")
    logpath_general = os.path.join(runpath+"/")
    figurepath = os.path.join(runpath+"/", "iteration_plots/")

    ' Empty array for possible dummy points - for printing'
    XCdummy= np.empty((0,dim))

    ' Initial acquisition phase '
    NMC = 9
    XGLEE = createPD(NMC, dim, "grid", parameterranges)
    dfXC = gp.predictderivative(XGLEE, True)
    varXC = gp.predictvariance(XGLEE)
    w = estiamteweightfactors(dfXC, epsphys)
    NGLEE = XGLEE.shape[0]
    mcglobalinitial = MCGlobalEstimate(w,varXC,NGLEE,parameterranges)

    'Epsilon^2 at this points'
    epsilon = np.squeeze(gp.getaccuracy)
    currentcost = totalcompwork(epsilon**(-1),s)

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
        'Check for which cases are set.'
        if gp.getXgrad is None:
            Ngrad = gp.getdata[1] #Is None, when Xgrad is None
        elif gp.getXgrad is not None:
            Ngrad = gp.getdata[1]

        ' Logging '
        logger = IOToLog(os.path.join(runpath+"/","iteration_log"),"iteration_"+str(counter))
        try:
            costerrorlog = open(logpath+"costerror.txt","a")
            file = open(logpath_general+"log.txt","a")
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
        
        costerrorlog.write(str(currentcostprior)+" "+str(mcglobalerrorbefore))
        costerrorlog.write("\n")

        print("Global error estimate before optimization:   {:1.5f}".format(mcglobalerrorbefore))
        print("Computational cost before optimization:      {:0.0f}".format(currentcostprior))
        print("\n")
        
        logger.addToFile("Global error estimate before optimization:   {:1.5f}".format(mcglobalerrorbefore))
        logger.addToFile("Computational cost before optimization:      {:0.0f}".format(currentcostprior))
        logger.addToFile("\n")
        
        t1apriori=time.perf_counter()
        tapriori = t1apriori-t0apriori    

        """ ------------------------------Acquisition phase ------------------------------ """
        'Add new candidate points'
        print("--- Acquisition phase")
        
        t0acqui = time.perf_counter()
        logger.addToFile("--- Acquisition phase")
        normvar_TEST = np.linalg.norm(np.sqrt(np.abs(varGLEE)),2,axis=0)
        XC,index,value = acquisitionfunction(gp,dfGLEE,normvar_TEST,w,XGLEE,epsphys,TOLAcqui)
        
        if XC.size  == 0:
            print("Something went wrong, no candidate point was found.")
            print("\n")
            logger.addToFile("Something went wrong, no candidate point was found.")
            logger.addToFile("Number of possible candidate points: {}".format(XC.shape[0]))
            logger.addToFile("\n")
        TOLAcqui = 1.0


        ' Add found candidate point '
        accuracy = np.sqrt(1E-5)

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
            
            NC        = XC.shape[0]
            epsXc     = accuracy**2*np.ones((1, XC.shape[0]))  # eps**2
            epsXcgrad = accuracy**2*np.ones((1, dim*XC.shape[0]))  # eps**2

            gp.adddatapoint(XC)
            gp.addgradientdatapoint(XC)

            cd     = np.round_(XC[0,0],4)
            t      = np.round_(XC[0,1],4)
            r_top  = np.round_(XC[0,2],4)
            r_bot  = np.round_(XC[0,3],4)
                
            theta, phi = [5],[0]
            
            model_keys = {
                    "cd"   : cd,
                    "h"    : 48.3,
                    "swa"  : 87.98,
                    "t"    : t,
                    "r_top": r_top,
                    "r_bot": r_bot
                    }
            
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
                ygradnew = np.zeros((dim,gp.m))
                for ii in range(len(theta)):                      
                    d_Isim_d_cd    = res["d_Isim_d_cd"]["P"]["phi_0"][ii]
                    d_Isim_d_r_bot = res["d_Isim_d_r_bot"]["P"]["phi_0"][ii]
                    d_Isim_d_r_top = res["d_Isim_d_r_top"]["P"]["phi_0"][ii]
                    d_Isim_d_t     = res["d_Isim_d_t"]["P"]["phi_0"][ii]
                    
                    ytnew[ii] = res["Isim"]["P"]["phi_0"][ii]
                    ygradnew[:,ii] = np.array([d_Isim_d_cd,
                                               d_Isim_d_t,
                                               d_Isim_d_r_top,
                                               d_Isim_d_r_bot])   

            #gp.adddatapointvalue(meanXc)
            gp.adddatapointvalue(np.atleast_2d(ytnew))
            gp.addaccuracy(epsXc)
            
            gp.addgradientdatapointvalue(np.atleast_2d(ygradnew))
            gp.addgradaccuracy(epsXcgrad)
   
        t1acqui = time.perf_counter()
        tacqui = t1acqui-t0acqui


        """ ------------------------------ MC GLOBAL ERROR ESTIMATION ------------------------------ """
        print("--- A posteriori error estimate")
        logger.addToFile("--- A posteriori error estimate")
        
        t0post = time.perf_counter()
        
        'Calculate actual used work'
        epspost = gp.getaccuracy
        vpost = 1/epspost
        currentcost=totalcompwork(vpost, s)
        
        dfGLEE  = gp.predictderivative(XGLEE, True)
        varGLEE = gp.predictvariance(XGLEE,True)
        wpost   = estiamteweightfactors(dfGLEE, epsphys)
        normvar = np.linalg.norm(np.sqrt(np.abs(varGLEE)),2,axis=0)**2

        mcglobalerrorafter = MCGlobalEstimate(wpost,normvar,NGLEE,parameterranges)
        file.write( str(mcglobalerrorafter))
        print("Global error estimate after optimization:   {:1.8f}".format(mcglobalerrorafter))
        print("Computational cost after optimization:      {:0.0f}".format(currentcost))
        print("\n")

        logger.addToFile("Global error estimate after optimization:   {:1.8f}".format(mcglobalerrorafter))
        logger.addToFile("Computational cost after optimization:      {:0.0f}".format(currentcost))
        logger.addToFile("\n")
        
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
        #print("New budget for next design:                      {:0.0f}".format(currentcost+incrementalbudget))
        #print("Used time:                                       {:0.2f} seconds".format(total_n))
        print("\n")

        logger.addToFile("--- New parameters")
        logger.addToFile("Error estimate after optimization and filtering: {:1.5f}".format(mcglobalerrorafter))
        logger.addToFile("Computational cost after optimization:           {:0.0f}".format(currentcost))
        #logger.addToFile("New budget for next design:                      {:0.0f}".format(currentcost+incrementalbudget))
        #logger.addToFile("Used time:                                       {:0.2f} seconds".format(total_n))
        logger.addToFile("\n")
        counter += 1

        Nmax = 10
        if N % 5 == 0:
            print("--- A priori hyperparameter adjustment")
            #region = ((0.01, 10),(0.01, 10),(0.01, 10),(0.01, 10))
            region = ((0.1, 5),(0.1, 4),(0.1, 4),(0.1, 6))
            region = ((0.1, 2),(0.1, 2),(0.1, 2),(0.1, 2))
            gp.optimizehyperparameter(region, "mean", False)
        else:
            print("--- A priori hyperparameter adjustment")
            print("Number of points is higher then "+str(Nmax))
            print("No optimization is performed")
        print("\n")

        gp.savedata(runpath+'/saved_data/',str(counter)) 
     
        t1design=time.perf_counter()
        print("Time used for complete design iteration: {:0.2f} seconds".format(t1design-t0design))
        print("\n")

        file.write("\n")
        file.close()
        costerrorlog.close()
