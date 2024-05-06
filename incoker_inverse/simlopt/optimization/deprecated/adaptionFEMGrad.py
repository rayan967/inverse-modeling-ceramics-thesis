import numpy as np
import matplotlib.pyplot as plt
import copy

from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.utils.creategrid import *
from basicfunctions.utils.creategrid import *
from basicfunctions.utils.arrayhelpers import *
from basicfunctions.kaskade.kaskadeio import *

from hyperparameter.hyperparameteroptimization import *
from hyperparameter.utils.setstartvalues import *
from hyperparameter.utils.crossvalidation import*

from optimization.utils.loss import *
from optimization.confirmation import *
from optimization.utils.helper.optimizegrad import *

from reconstruction.utils.perror import *
from reconstruction.utils.plotGPR import *
from reconstruction.utils.postprocessing import *

from scipy.spatial import distance
from reconstruction.utils.savedata import *

from gpr.gaussianprocess import *

def adaptwithgradients(gp,epsphys,itermax, nsample, region,
                       Wbudget, dptarget, eta, delta, ngp,
                       threshold, verbose, ranges, runpath, execname):

    ' Some initial data '
    counter = 0
    m = 1
    n, dim = gp.getX.shape[0], gp.getX.shape[1]
    cvo = []
    mselist = []
    greedy = False

    runpathsplitted = os.path.split(runpath)
    execpath = runpathsplitted[0]
    runname = runpathsplitted[1]

    def fun(x,dim):
        if dim == 1:
            return x*np.sin(2*x*np.pi)
        elif dim == 2:
            return np.sin(np.sum(x,axis = 1)).reshape(-1,1)
        
        
    ' Lists for evaluation '
    usedworklist = []
    neededworklist = []
    maxdplist = []
    neededworklist.append(np.sum(W(np.sqrt(gp.getaccuracy),dim))) #Accuracy of initial gp
    usedworklist.append(np.sum(W(np.sqrt(gp.getaccuracy),dim))) #Accuracy of initial gp    

    if gp.getXgrad is None:
        ygrad = gp.predictderivative(gp.getX)
        gp.adddgradientdatapointvalue(ygrad)
        gp.addgradientdatapoint(gp.getX)
        gp.addgradaccuracy(1E20*np.ones((1,n*dim)))

    print("\n")
    ' Adaptation '
    while Wbudget > 0:

        'Initialization'
        dp = np.zeros((nsample, dim))
        dphy = np.zeros((nsample, dim))

        print("---------- Iteration: {} ----------".format(counter))

        if dim == 1:
            xs = createPD(nsample, dim, "grid", ranges)
        else:
            xs = createPD(nsample, dim, "random", ranges)

        'Optimize hyperparameter'
        gp.optimizehyperparameter(region, "mean", False)
        if gp.gethyperparameter is not None:
            currenthyperparameter = gp.gethyperparameter
        else:
            gp.sethyperparameter(currenthyperparameter)

        'Predict variance'
        varxs = gp.predictvariance(xs)

        'Predict df at xs'
        df = gp.predictderivative(xs,True)

        ' Calcualte dp = dpv + dpphys for every point in xs '
        print("\n")
        print("Estimate parameter error:")
        for i in range(nsample):
            dp[i, :]   = parametererror(df[i, :], varxs[i], 0.0, m, dim)
            dphy[i, :] = parametererror(df[i, :], 0.0 , epsphys, m, dim)

        ' Stop criterion 1: abort, when at every xs dp are below the targetaccuracy '
        if (np.linalg.norm(dp, 2, axis=1) < dptarget).all():
            print("The estimated parametererror at all sampled points is below the desired threshold - adaptive phase done")


            gp.plot(fun, execpath+'/img/iteration/', "plot_converged", ranges)
            if dim == 1:
                plotparametererrorestimates(gp, epsphys, dptarget, 1000, 1, ranges, None, None, 2, "grid",
                                            runpath+'/estimation_plots/', "iteration_converged",runpath+'/hist_plots/', "histiteration_converged")
            print("Save everything...")
            gp.savedata(runpath+'/saved_data')
            return gp, cvo, mselist

        indexdpphys = np.where(np.logical_and((np.linalg.norm(dphy, 2, axis=1) < dptarget),(np.linalg.norm(df, 2, axis=1) > 0.00001)))
        ' Stop criterion 4: abort, when there is no point at which dpphys < dptarget '
        if indexdpphys[0].size == 0:
            print("No index was found where the parameter error due to the physical measurement was below the target parameter error")
            print(" - Measure more accurate")
            print(" - Increase the target value...")

            gp.plot(fun, execpath+'/img/iteration/', "plot_converged", ranges)
            if dim == 1:
                plotparametererrorestimates(gp, epsphys, dptarget, 1000, 1, ranges, None, None, 2, "grid",
                                            runpath+'/estimation_plots/', "iteration_converged",runpath+'/hist_plots/', "histiteration_converged")

            print("Save everything...")
            gp.savedata(runpath+'/saved_data')
            return gp, cvo, mselist

        ' With the given indices find the values where the ratio is the highest and !! dphys is below the target value '
        ratio = np.linalg.norm(dp[indexdpphys], 2, axis=1) / np.linalg.norm(dphy[indexdpphys], 2, axis=1)
        Iratio = np.argmax(ratio)  # Find index of highest ratio
        I = indexdpphys[0][Iratio] # Find to which point index this ratio belongs
        dpatratio = dp[I]          # Get dp at this point also, np.array(dim,)

        numberofsortedoutpoints = xs.shape[0]-indexdpphys[0].size
        print(f"Number of sorted out points: {numberofsortedoutpoints}")

        if greedy:
            print("Choose greediest strategy")
            if counter == 0:
                dptarget = 0.5*np.linalg.norm(dpatratio, 2)
                print("Set dptarget to: "+str(dptarget))
                print("\n")

        ' Stop criterion 2: abort, when at sample xs dp is below the targetaccuracy '
        if np.linalg.norm(dpatratio, 2) < dptarget:
            #print("dp at point {} of maximum ratio {} is below desisred target accuracy, find point of worst dp in the system".format(xs[I],ratio[Iratio]))
            print("dp at point {} of maximum ratio {} is below desisred target accuracy, return and save".format(
                xs[I], ratio[Iratio]))
            gp.plot(fun, runpath+'/iteration_plots/', "plot_dp_"+str(counter), ranges)
            if dim == 1:
                plotparametererrorestimates(gp, epsphys, dptarget, 1000, 1, ranges,
                                            xs[I], None, 2, "grid", runpath+'/estimation_plots/', "iteration_last_"+str(counter),runpath+'/hist_plots/', "histiteration"+str(counter)+"_dp")
            gp.savedata(runpath+'/saved_data')
            return gp, cvo, mselist

        else:
            np.set_printoptions(precision=4)
            print("At x = {}".format(xs[I]))
            print(" -Ratio: {:g}".format(np.max(ratio)))
            print(" -Norm of estiamted parametererror: {}".format(np.linalg.norm(dpatratio,2)))
            print(" -Predicted derivative: {}".format(df[I]))
            print("\n")
            x = np.array([xs[I]])
            gp.plot(fun,  runpath+'/iteration_plots/', "plot_iteration_"+str(counter), ranges)

            if dim == 1:
                plotparametererrorestimates(gp, epsphys, dptarget, 1000, 1, ranges,
                                            xs[I], None, 2, "grid", runpath+'/estimation_plots/', "iteration_"+str(counter),runpath+'/hist_plots/', "histiteration_"+str(counter))

        '-- Create ghost points around point of maximum parameter error --'
        rangesxghost = []
        for ii in range(dim):
            rangesxghost.append(np.array([x[0, ii]-delta, x[0, ii]+delta]))
        rangesxghost = np.array(rangesxghost)

        for i in range(rangesxghost.shape[0]):
            if rangesxghost[i, 0] < ranges[i, 0]:
                rangesxghost[i, 0] = ranges[i, 0]+0.001
            if rangesxghost[i, 1] > ranges[i, 1]:
                rangesxghost[i, 1] = ranges[i, 1]-0.001
        Xghost = createPD(int(np.ceil((ngp**(1/dim)))), dim, "grid", rangesxghost)

        ' First add x as last point to Xghost '
        Xghost = np.concatenate((Xghost, x), axis=0)

        ' Add them to the existing data with their mean as (best known) data value '
        meanghost = gp.predictmean(Xghost)

        ' Add ghost points to the training data, where for now the errors are infinite '
        Xtextended = np.concatenate((gp.getX, Xghost), axis=0)
        ytextended = np.concatenate((gp.gety, meanghost.reshape(-1, 1)))

        ' Add the error of the ghost points to the existing ones'
        penalty = 1E20
        epsXghost = penalty*np.ones(Xghost.shape[0])
        epsXtextended = np.concatenate((gp.getaccuracy, epsXghost.reshape(1, -1)), axis=1)


        ' When there is gradient data we need to add the ghost point, the values at the ghost point and the accuracies have to be set to penalty'
        Xgradextended = np.concatenate((gp.getXgrad, Xghost), axis=0)
        ygradextended = gp.predictderivative(Xghost)
        ygradextended = np.concatenate((gp.getygrad,ygradextended))

        epsXgradextended = penalty*np.ones((1,Xghost.shape[0]*gp.getdim))
        epsXgradextended = np.concatenate((gp.getgradientaccuracy, epsXgradextended), axis=1)

        ' Find epsilon for every x in Xghost and x itself '
        print("Optimize accuracy:")
        optimalsolution = findepsgrad(x,Xtextended,Xgradextended,ytextended,ygradextended,gp.gethyperparameter.reshape((1, -1)),epsXtextended,epsXgradextended,1,dim,
                                      epsphys,0,n,dptarget,itermax,threshold,Wbudget)

        result,cost,accuracy,pointtype  = optimalsolution[0], optimalsolution[1], optimalsolution[2],optimalsolution[3]

        if type(accuracy) is np.ndarray:
            foundsolution = accuracy[0]
        else:
            foundsolution = accuracy
        #confirmresult(x,copy.copy(gp),epsphys,dptarget,foundsolution**2,runpath+'/confirmation_plots/', "confirmation_"+str(counter)+"_.png")
        confirmresultgrad(x,copy.copy(gp),epsphys,dptarget,foundsolution**2,runpath+'/confirmation_plots/', "confirmation_"+str(counter)+"_.png")
        #confirmresultgrad(x,gp,           epsphys,dptarget,foundsolution,   dptargetfilepath,filename)
        ' Stop criterion 3: abort, when no solution was found '
        if result.size == 0:
            print("No solutions were found within this current point distribution")
            #print("Graphical confirmation: ")
            confirmresult(x, copy.copy(gp), epsphys, dptarget , accuracy**2 ,runpath+'/confirmation_plots/', "confirmation_nosolition.png")
            gp.savedata(runpath+'/saved_data')
            return gp, cvo, mselist

        """ Als Lösung erhält man einen Punkt

        Rückgabewert ist der Lösungspunkt, Index, cost, accuracy, accuracygrad, pointtype

        If pointtype is 0: Nur eps wird angepasst
        If pointtype is 1: eps und epsgrad wird angepasst

        (1) Punkt ist ein neuer Punkt
            (1.1) Am neunen Punkt wird nur eps angepasst
            (1.2) Am neunen Punkt wird eps und epsgrad angepasst
        (2) Punkt ist ein vorhandener Punkt
            (2.1) Am alten Punkt wird nur eps angepasst
            (2.2) Am alten Punkt wird eps und epsgrad angepasst
        """
        currentindex = np.where((gp.getX == result.tolist()).all(axis=1)) #Refactor über index ?

        print("\n")
        print("--------- Simulation phase ")

        if currentindex[0].size==0:
            print("The found point is a new data point...")



            if pointtype == 0: #(1.1)

                simulationresults = createandrun(np.array([result]),accuracy,None,execpath,execname)

                ytnew = simulationresults[0]
                epsXtnew = simulationresults[1]

                ' --- Adapt eps --- '
                gp.adddatapoint(np.array([result]))
                gp.addaccuracy(epsXtnew**2)
                gp.adddatapointvalue(ytnew)

                ' Add point as gradient ghost point '
                ygrad = gp.predictderivative(np.array([result]))

                gp.addgradientdatapoint(np.array([result]))
                gp.addgradaccuracy(1E20*np.ones((1,gp.getdim)))
                gp.adddgradientdatapointvalue(ygrad)

                ' Adapt budget '
                Wbudget = Wbudget - W(epsXtnew, dim)

            elif pointtype == 1: #(1.2)

                simulationresults = createandrun(np.array([result]),accuracy[0],np.squeeze(accuracy[1:],axis=0),execpath,execname)

                ytnew = simulationresults[0]
                epsXtnew = simulationresults[1]
                ygradnew = simulationresults[2]
                epsXgradnew = simulationresults[3]

                ' --- Adapt eps and epsgrad --- '
                gp.adddatapoint(np.array([result]))
                gp.addaccuracy(epsXtnew**2)
                gp.adddatapointvalue(ytnew)

                gp.addgradientdatapoint(np.array([result]))
                gp.addgradaccuracy(epsXgradnew**2)
                gp.adddgradientdatapointvalue(ygradnew)

                ' Adapt budget '
                Wbudget = Wbudget - W(epsXtnew, dim) - W(epsXgradnew, dim)

        else:
            index = currentindex[0][0]

            if pointtype == 0: #(2.1)

                simulationresults = createandrun(np.array([result]),accuracy,None,execpath,execname)

                ytnew = simulationresults[0]
                epsXtnew = simulationresults[1]


                ' Adapt eps of existing point at index '
                gp.addaccuracy(epsXtnew**2,index)
                gp.adddatapointvalue(ytnew,index)

                ' Adapt budget '
                Wbudget = Wbudget - W(epsXtnew, dim)

            elif pointtype == 1: #(2.2)

                simulationresults = createandrun(np.array([result]),accuracy,np.squeeze(accuracy[1:],axis=0),execpath,execname)

                ytnew = simulationresults[0]
                epsXtnew = simulationresults[1]
                ygradnew = simulationresults[2]
                epsXgradnew = simulationresults[3]

                ' Adapt eps and epsgrad of existing point '
                gp.adddatapointvalue(ytnew,index)
                gp.addaccuracy(epsXtnew**2,index)

                gp.adddgradientdatapointvalue(ygrad,index)
                gp.addgradaccuracy(epsXgradnew**2,index)

                ' Adapt budget '
                Wbudget = Wbudget - W(epsXtnew, dim) - W(epsXgradnew, dim)


        print("New budget after adaptive step: {:g}".format(Wbudget[0,0]))
        print("\n")
        
        'Append evaluation data'
        neededworklist.append(neededworklist[counter]+W(accuracy, dim))
        usedworklist.append(usedworklist[counter]+W(epsXtnew, dim)[0,0])

        """ Calculate error measures """
        mse = gp.calculateMSE(fun, 200, ranges, "random")
        #loocv = gp.calculateLOOCV()

        #cvo.append(loocv)
        mselist.append(mse)

        ' Adapt intial data '
        counter = counter + 1
        n = gp.getX.shape[0]
        print("\n")