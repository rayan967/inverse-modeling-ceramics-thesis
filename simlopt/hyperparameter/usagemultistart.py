import numpy as np
import matplotlib.pyplot as plt

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


""" 2D Training data """

# =============================================================================
# def f2D(X):
#     x = np.array([X[:,0]])
#     y = np.array([X[:,1]])
#     #return np.sin(np.sum(X,axis = 1))
#     return x.T*np.sin(y.T)
# 
# ranges = np.array([[0,5],[0,5]])
# Xt = createPD(5, 2, "random", ranges)
# Xt = np.expand_dims(Xt, axis=2)
# 
# yt = f2D(Xt)
# yt = yt[:,:,0].T
# epsXt = 1E-5*np.ones((1,Xt.shape[0]))
# region = ((1, 1E2), (0.5, 1E2),(0.5, 1E2))
# 
# optimizehyperparametersmultistart(Xt, None, yt, None, epsXt, None, region)
# 
# =============================================================================

""" nD Training data """

dim = 4
def f3D(X):
    return np.sin(np.sum(X,axis = 1))
    #return x.T*np.sin(y.T)

ranges = np.array([[0,5],[0,5],[0,5],[0,5]])
Xt = createPD(5, dim, "random", ranges)
yt = f3D(Xt)

Xt = np.expand_dims(Xt, axis=2)
yt = yt.reshape((-1,1))
epsXt = 1E-5*np.ones((1,Xt.shape[0]))
region = ((1, 1E2), (0.5, 1E2),(0.5, 1E2),(0.5, 1E2),(0.5, 1E2))

optimizehyperparametersmultistart(Xt, None, yt, None, epsXt, None, region)