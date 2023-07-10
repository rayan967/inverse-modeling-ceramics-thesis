import sys
import os
import time
import io
import math

import time
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from mpl_toolkits.mplot3d import Axes3D

import scipy.stats as stats

from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.utils.createfolderstructure import *
from basicfunctions.reconstruction.recon import *
from basicfunctions.utils.plotting import *

from optimization.basicadapt_multible_new import *

from gpr.gaussianprocess import *
from hyperparameter.utils.crossvalidation import*

from optimization.functionprototypes import *
from optimization.testcases import *

plt.close('all')
plt.ioff()

def main():

    ' Path to save adaptive run '
    execpath = '/data/numerik/projects/siMLopt/simlopt/data/'
    execpath = 'D:\Projects\inverse-modeling-ceramics-thesis\simlopt\data'
    execname = None
    
    'Training data '
    testcase        = 1
    Xt,Xgrad        = standardpointdistributions(testcase)
    yt,ygrad,fun    = createdata("himmelblau",Xt)
    epsXt, epsXgrad = createerror(Xt,random=False,graddata=False)
    
    ' Initial problem constants '
    N   = Xt.shape[0]
    dim = Xt.shape[1]
    m   = yt.shape[1]
    
    ' Parameter space boundaries '
    p1lb = np.min(Xt[:,0])
    p1ub = np.max(Xt[:,0])
    p2lb = np.min(Xt[:,1])
    p2ub = np.max(Xt[:,1])
    
    parameterranges = np.array([[p1lb,p1ub], [p2lb,p2ub]])
    print("---------------------------------- Data parameters")
    print("Spatial dimension: {}".format(dim))
    print("Number of features: {}".format(dim))
    print("Number of training points: {}".format(Xt.shape[0]))
    if Xgrad is not None:
        print("Number of ge-training points: {}".format(Xgrad.shape[0]))
    else:
        print("Number of ge-training points: {}".format(0))
    
    ' Parameters for adaptive phase '
    totalbudget         = 1E20          # Total budget to spend
    incrementalbudget   = 1E5           # Incremental budget
    TOL                 = 1E-3          # Overall desired reconstruction tolerance
    TOLFEM              = 0.0           # Reevaluation tolerance
    TOLAcqui            = 1.0           # Acquisition tolerance
    TOLrelchange        = 0             # Tolerance for relative change of global error estimation
    epsphys             = np.array([1E-1,1E-3,1E-2]) # Assumed or known variance of pyhs. measurement!
    adaptgrad           = False         # Toogle if gradient data should be adatped
    
    ' Initial hyperparameter parameters '
    region = ((0.01, 2),   (0.01, 2))
    assert len(region) == dim, "Too much or less hyperparameters for the given problem dimension"
    print("\n")
    
    ' Adaptive phase '
    foldername  = createfoldername("Basic example","2D","1E5")
    runpath     = createfolders(execpath,foldername)
    
    ' Create Gaussian Process Regression'
    if Xgrad is not None:
        gp = GPR(Xt, yt, Xgrad, ygrad, epsXt, epsXgrad)
    else:
        gp = GPR(Xt, yt, None, None, epsXt, None)
    gp.optimizehyperparameter(region, "mean", False)
    print("\n")
    
    print("---------------------------------- Adaptive parameters")
    print("Number of initial data points:       {}".format(N))
    print("Overall stopping tolerance:          {}".format(TOL))
    print("Hyperparameter bounds:               {}".format(region))
    print("\n")
    
    GP_adapted = adapt(gp, totalbudget,incrementalbudget,parameterranges,
                TOL,TOLFEM,TOLAcqui,TOLrelchange,epsphys,
                runpath, execname, adaptgrad , fun)
    
    ' Reconstruction '
    xreal = np.array([[0.5,0.25]])
    a = [1,2,3]
    yreal = np.zeros((xreal.shape[0],len(a)))
    for i,m in enumerate(a):
        yreal[:,i] = fun["function"](xreal,m).reshape((-1,1))

    ' Measurement error (Likelihood) '
    SigmaLL = np.diagflat(epsphys)

    ' Start value, parameter space, solver options'
    p0 = np.array([0.6,0.3])
    pspace = ((0.0, 1.0),(0.0, 1.0),)
    maxiter = 10000000
    toliter = 1E-5
    method = 'Powell'

    starttime       = time.perf_counter()
    recon_result    = reconstruct(GP_adapted, p0, yreal,SigmaLL, pspace, maxiter, toliter, method, xreal )
    recon_parameter = recon_result[0]
    recon_std       = recon_result[1]
    endtime         = time.perf_counter()
    print("Reconstruction done in       {} s".format(endtime - starttime))
    
    
    ' Post processing '
    print("Plot error over cost")
    plotErrorOverCost(runpath,"pdf")
    print("...done")
    print("\n")
    
    print("Plot contours of global error estimate")
    plotErrorContour(GP_adapted,Xt,parameterranges,epsphys,runpath,"pdf")
    print("...done")
    print("\n")
    
    print("Plot histogram")
    plotHistogramOfLocalError(GP_adapted,epsphys,parameterranges,runpath,"pdf",cumulative=False)
    print("...done")
    print("\n")
    
    print("Plot marginal solution")
    plotMarginalSolutions(2,recon_parameter,recon_std,xreal,runpath,"pdf")
    print("...done")
    print("\n")
    

if __name__ == "__main__":
    main()

