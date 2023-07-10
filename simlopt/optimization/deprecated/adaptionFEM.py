import numpy as np
import matplotlib.pyplot as plt
import copy

from basicfunctions.utils.creategrid import *
from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.utils.creategrid import *
from basicfunctions.kaskade.kaskadeio import *
from basicfunctions.utils.arrayhelpers import *

from hyperparameter.hyperparameteroptimization import *
from hyperparameter.utils.setstartvalues import *
from hyperparameter.utils.crossvalidation import*

from optimization.utils.loss import *
from optimization.confirmation import *
from optimization.utils.findoptimaleps import *

from reconstruction.utils.perror import *
from reconstruction.utils.plotGPR import *
from reconstruction.utils.postprocessing import *

from scipy.spatial import distance
from reconstruction.utils.savedata import *

from gpr.gaussianprocess import *

def adapt(gp,epsphys,itermax, nsample, region,
          Wbudget, dptarget, eta, delta, ngp,
          threshold, verbose, ranges, runpath, execname):

    ' Some initial data '
    counter = 0
    m = 1
    n, dim = gp.getX.shape[0], gp.getdim
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

        ' Calcualte the mean of dp '
        maxdplist.append(np.max(np.linalg.norm(dp[indexdpphys], 2, axis=1)))
        print("Current highest parmeter error: {}".format(str(maxdplist[counter])))
        print("Current used work: {}".format(str(np.sum(W(np.sqrt(gp.getaccuracy),dim)))))
        print("\n")

        ' Open file to write data '
        filepathaccovercost = os.path.join(runpath,"accovercost.txt")
        filepathneeded = os.path.join(runpath,"neededcost.txt")

        with open(filepathaccovercost,'a+') as f:
            f.write(str(usedworklist[counter])+" "+str(maxdplist[counter])+"\n")
        with open(filepathneeded,'a+') as f:
            f.write(str(neededworklist[counter])+" "+str(maxdplist[counter])+"\n")

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

        ' Find epsilon for every x in Xghost and x itself '
        print("--------- Optimization phase ")
        optimalsolution = findeps(x, Xtextended, gp.getXgrad, ytextended, gp.getygrad, gp.gethyperparameter.reshape((1, -1)), epsXtextended, gp.getgradientaccuracy,
                                         m, dim, epsphys, n, dptarget, itermax, threshold, Wbudget,verbose = False)
        result, index, ccost, accuracy = optimalsolution[0], optimalsolution[1], optimalsolution[2], optimalsolution[3]

        ' Stop criterion 3: abort, when no solution was found '
        if result.size == 0:
            print("No solutions were found within this current point distribution")
            #print("Graphical confirmation: ")
            confirmresult(x, copy.copy(gp), epsphys, dptarget , accuracy**2, None ,runpath+'/confirmation_plots/', "confirmation_nosolition.png")
            gp.savedata(runpath+'/saved_data')
            return gp, cvo, mselist

        print("--------- Simulation phase ")
        print("Adding the point {} with accuracy of {:g}".format(result, accuracy))
        'Check wether the point is new or an old one'
        currentindex = np.where((gp.getX == result.tolist()).all(axis=1))

        if dim == 1:
            parameter = {"--x": result[0],
                         "--eps": accuracy}
        elif dim ==2 :
            parameter = {"--x": result[0],
                         "--y": result[1],
                         "--eps": accuracy}

        elif dim == 3:
# =============================================================================
#             parameter = {"--v0":result[0],
#                          "--d1":result[1],
#                          "--d2":result[2],
#                          "--atol":np.sqrt(accuracy)}
#
# =============================================================================
            parameter = {"--v0":result[0],
                         "--d1":result[1],
                         "--d2":result[2],
                         "--refine":6}

        runkaskade(execpath, execname, parameter)
        print("\n")

        'Read simulation data and get function value'
        simulationdata = readtodict(execpath, "dump.log")

        reached = np.asarray(simulationdata["flag"])
        epsXtnew = np.asarray(simulationdata["accuracy"])
        epsXtnew = epsXtnew.reshape((1, -1))

# =============================================================================
#         'DEBUG'
#         accuracy = np.array([accuracy])
#         epsXtnew = accuracy.reshape((1, -1))
# =============================================================================

        if reached[0] == 1:
            print("Accuracy during simulation reached with: {}".format(epsXtnew[0, 0]))
        elif reached[0] == 0:
            print("Accuracy during simulation was not reached")
            print("Set accuracy to: ".format(epsXtnew[0, 0]))

        'Sanity check'
        confirmresult(result.reshape((1,-1)), copy.copy(gp), epsphys, dptarget , accuracy**2,epsXtnew**2, runpath+'/confirmation_plots/', "confirmation_"+str(counter)+"_.png")
        plosthistogram(gp, 4000, ranges, None,dptarget, 2, runpath+'/hist_plots/',  "hist_"+str(counter)+"_.png")

        'Reshape for concatenate data'
        ytnew = np.asarray(simulationdata["value"])
        ytnew = ytnew.reshape((1, -1))
        print("Simulation value: {}".format(ytnew[0, 0]))

        if currentindex[0].size==0:
            'Then the point at idx is a new point and this is added to the existing data'
            gp.adddatapoint(np.array([result]))
            gp.addaccuracy(epsXtnew**2)
            gp.adddatapointvalue(ytnew)
        else:
            'Dann ist der Punkt schon vorhanden und nur die Genauigkeit wird angepasst und der neu berechnete Wert gesetzt'
            index = currentindex[0][0]
            gp.addaccuracy(epsXtnew**2, index)
            gp.adddatapointvalue(ytnew, index)

        'Adapt budget'
        Wbudget = Wbudget - W(epsXtnew, dim)
        print("New budget after adaptive step: {:g}".format(Wbudget[0,0]))
        print("\n")

        'Append evaluation data'
        neededworklist.append(neededworklist[counter]+W(accuracy, dim))
        usedworklist.append(usedworklist[counter]+W(epsXtnew, dim)[0,0])

        ' Calculate error measures '
        #mse = gp.calculateMSE(fun, 200, ranges, "random")
        loocv = gp.calculateLOOCV()
        cvo.append(loocv)
        #mselist.append(mse)

        ' Adapt intial data '
        counter = counter + 1
        n = gp.getX.shape[0]
        print("\n")
