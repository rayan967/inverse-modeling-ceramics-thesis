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

    ' Container for plotting '
    maxdplist = []
    usedworklist = []
    usedworklist.append(W(gp.getaccuracy, dim)[0,0])

    runpathsplitted = os.path.split(runpath)
    execpath = runpathsplitted[0]
    runname = runpathsplitted[1]

    def fun(x,dim):
        if dim == 1:
            return x*np.sin(2*x*np.pi)
        elif dim == 2:
            return np.sin(np.sum(x,axis = 1)).reshape(-1,1)

    #xequi = np.array([[0.1],[0.2],[0.3],[0.5],[0.6],[0.7],[0.9]])
    xequi = np.genfromtxt('/data/numerik/people/bzfsemle/simlopt/data/3D/dataequi.log')

    print("\n")
    ' Adaptation '
    for ii in range(xequi.shape[0]):

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
            gp.gethyperparameter = currenthyperparameter

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
            return gp, cvo, mselist,maxdplist, usedworklist
        #0.00001
        indexdpphys = np.where(np.logical_and((np.linalg.norm(dphy, 2, axis=1) < dptarget),(np.linalg.norm(df, 2, axis=1) > 0.0001)))
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
            return gp, cvo, mselist,maxdplist, usedworklist


        maxdplist.append(np.mean(np.linalg.norm(dp[indexdpphys], 2, axis=1)))

        numberofsortedoutpoints = xs.shape[0]-indexdpphys[0].size
        print(f"Number of sorted out points: {numberofsortedoutpoints}")

        ' Find epsilon for every x in Xghost and x itself '
        print("--------- Optimization phase ")

        result =  xequi[ii]#For plotting
        accuracy = 0.001

        #confirmresult(result.reshape((1,-1)), copy.copy(gp), epsphys, dptarget , accuracy**2 ,runpath+'/confirmation_plots/', "confirmation_"+str(counter)+"_.png")
        #plosthistogram(gp, 4000, ranges, None,dptarget, np.inf, runpath+'/hist_plots/',  "hist_"+str(counter)+"_.png")

        print("--------- Simulation phase ")
        print("Adding the point {} with accuracy of {:g}".format(result, accuracy))
        'Check wether the point is new or an old one'
        currentindex = np.where((gp.getX == result.tolist()).all(axis=1)) #Refactor über index ?

        if dim == 1:
            parameter = {"--x": result[0],
                         "--eps": accuracy}
        elif dim ==2 :
            parameter = {"--x": result[0],
                         "--y": result[1],
                         "--eps": accuracy}

        elif dim == 3:
            parameter = {"--v0":result[0],
                         "--d1":result[1],
                         "--d2":result[2],
                         "--refine":5}
        runkaskade(execpath, execname, parameter)
        print("\n")

        'Read simulation data and get function value'
        simulationdata = readtodict(execpath, "dump.log")

        reached = np.asarray(simulationdata["flag"])
        epsXtnew = np.asarray(simulationdata["accuracy"])
        epsXtnew = epsXtnew.reshape((1, -1))

        if reached[0] == 0:
            print("Accuracy during simulation reached with: {}".format(epsXtnew[0, 0]))
        elif reached[0] == 1:
            print("Accuracy during simulation was not reached with: {}")
            print("Set accuracy to: ".format(epsXtnew[0, 0]))

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
        if ii > 0:
            usedworklist.append(W(epsXtnew, dim)[0,0]+usedworklist[ii-1])
        print("New budget after adaptive step: {:g}".format(Wbudget[0, 0]))
        print("\n")

        """ Calculate error measures """
        #mse = gp.calculateMSE(fun, 200, ranges, "random")
        loocv = gp.calculateLOOCV()
        cvo.append(loocv)
        #mselist.append(mse)

        ' Adapt intial data '
        counter = counter + 1
        n = gp.getX.shape[0]
        print("\n")
    return gp, cvo, mselist,maxdplist, usedworklist