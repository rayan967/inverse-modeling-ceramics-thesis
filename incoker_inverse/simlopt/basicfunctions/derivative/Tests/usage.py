from timeit import default_timer as timer

import numpy as np
from basicfunctions.covariance.cov import *
from basicfunctions.utils.creategrid import *
from gpr.gaussianprocess import *

""" 2D Data """
""" Basic 2D data """
testranges = np.array([[0,4],[0,2]])
Xt =  createPD(20 ,2, "random", testranges)
yt = np.sum(Xt**2,axis=1)
ygrad = 2*Xt
epsXt = 1E-4*np.ones((1,Xt.shape[0]))
epsXgrad = 1E-4*np.ones((1,Xt.shape[0]*2))
Xgrad = Xt
verbose = True
gp = GPR(Xt, yt.reshape((-1,1)), Xgrad, ygrad.reshape((-1,1)), epsXt, epsXgrad)
region= ((1, 10), (0.1, 10), (0.1, 10))
gp.optimizehyperparameter(region, "mean", False)
hyperparameter = gp.gethyperparameter.reshape((1,-1))


