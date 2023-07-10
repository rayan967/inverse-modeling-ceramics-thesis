import sys
import os
import time
import io
import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.utils.createfolderstructure import *

from optimization.adaptlinear_NEW import *
from optimization.adaptlinear_NEW_SpeedUp import *


from gpr.gaussianprocess import *

import scipy.stats as stats

from hyperparameter.utils.crossvalidation import*


plt.close('all')
plt.ioff()

""" ----------------------------- Load settings ----------------------------- """
dim = 2
parameterranges = np.array([[0,3], [0,3]])

Xt = np.array([[0.0,0.0],[1.0,0.0],[2.0,0.0],[3.0,0.0],
                [0.0,1.0],[3.0,1.0],[0.0,2.0],[3.0,2.0],
                [0.0,3.0],[1.0,3.0],[2.0,3.0],[3.0,3.0]])
N = Xt.shape[0]


' Create folders '
execpath = 'F:/Uni/Zuse/simlopt/simlopt/data/testdata2D'
execname = './testdata2D'
runpath = createfolders(execpath)

def fun(x):
    return np.sin(x[:,0])+np.cos(x[:,0]*x[:,1])
yt = fun(Xt)
yt = yt.reshape((-1,1))

xreal = np.array([[1,1.5]])
freal = np.ones((1,1))*fun(xreal)

""" -------------------------- Data error ----------------------------- """
epsXt = 1E-1*np.ones((1,N)) #1E-1 for basic setup
Xgrad,ygrad,epsXgrad = None,None,None

print("---------------------------------- Data parameters")
print("Spatial dimension: {}".format(dim))
print("Number of features: {}".format(dim))
print("Number of training points: {}".format(Xt.shape[0]))
if Xgrad is not None:
    print("Number of ge-training points: {}".format(Xgrad.shape[0]))
else:
    print("Number of ge-training points: {}".format(0))

""" -------------------------- Adaption parameters -------------------- """

totalbudget = 1E20
budgettospend = 1E5
TOL = 1E-4#1E-4 for paper
TOLe = 1E-6
TOLFEM = 1E-4
TOLFILTER = 1E-2
loweraccuracybound = 1E-5
nrofnewpoints = 5
epsphys = 1E-5

SigmaLL = np.array([[epsphys]])
SigmaP = np.diagflat((np.array([1E-5, 1E-5])))

""" -------------------------- Candidate poionts -------------------------- """
NC = 60
ranges = np.array([[0.0, 3.0], [0.0, 3.0]])
XC = createPD(NC, 2, "sobol", ranges)

""" -------------------------- HP parameters -------------------------- """
region = ((0.1, 20), (0.1, 20), (0.1, 20))
region = ((1, None), (1, 20), (1, 20))
assert len(region) == dim + 1, "Too much or less hyperparameters for the given problem dimension"

""" -------------------------- Adaption phase ------------------------- """
print("\n")
print("---------------------------------- Adaptive parameters")
print("Number of initial data points:       {}".format(N))
print("Number of initial candidate points:  {}".format(NC))
print("Overall stopping tolerance:          {}".format(TOL))
print("Hyperparameter bounds:               {}".format(region))


' Create Gaussian Process Regression'
if Xgrad is not None:
    gp = GPR(Xt, yt, Xgrad, ygrad, epsXt, epsXgrad)
else:
    gp = GPR(Xt, yt, None, None, epsXt, None)
gp.optimizehyperparameter(region, "mean", False)

' Initial accuracy of candidate points '
epsXc = 1E10*np.ones((1, XC.shape[0]))
meanXc = gp.predictmean(XC)

gp.adddatapoint(XC)
gp.adddatapointvalue(meanXc)
gp.addaccuracy(epsXc)
res = adapt(gp, N, NC, totalbudget,budgettospend,
          SigmaLL, SigmaP,ranges,TOL,TOLe,TOLFEM,TOLFILTER,
          loweraccuracybound,nrofnewpoints,execpath, execname)