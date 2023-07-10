import numpy as np
import matplotlib.pyplot as plt


from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from HPOpt.hyperparameteroptimization import *
from basicfunctions.utils.toydata import *
from HPOpt.utils.setstartvalues import *

""" Test for 2D log margonal ll with gradient data """

    
dim = 2
m = 1
n = 2
data = createartificaldata(n, dim, m)
freal = np.ones((1,m))*14.69 # Real parameter set p = 3.8,0.5

""" DEBUG """
Xt = data[:,0:dim].reshape((-1,dim,1))
Xgrad = Xt
yt = data[:,dim,0].reshape((-1,1))
yg = data[:,dim+1:]
ygrad = np.insert(yg[:,:,0],1,[]).reshape(-1,1)
ytilde = np.concatenate((yt,ygrad.reshape(-1,1)))


eps = 1E-4*np.ones((Xt[:,:,0].shape[0]))
epsgrad = 1E-4*np.ones((Xt[:,:,0].shape[0]*2))

H = [12,1.4,1.4]

""" Covariance matrix """
lmllgrad = logmarginallikelihood(H, Xt[:,:,0], Xgrad[:,:,0], yt, ygrad, eps, epsgrad, 1)
grad = logmarginallikelihood_der(H,Xt[:,:,0], Xgrad[:,:,0],yt,ygrad,eps,epsgrad,1)



""" Covariance matrix """
H = [1.0,1.0,1.0]
lmllgrad = logmarginallikelihood(H, Xt[:,:,0], Xgrad[:,:,0], yt, ygrad, eps, epsgrad, 1)
grad = logmarginallikelihood_der(H,Xt[:,:,0], Xgrad[:,:,0],yt,ygrad,eps,epsgrad,1)


'Hyperparameter optmization'
region = ((1E-3, 10), (1E-1, 5), (1E-1, 5))
sigma = np.abs(findsigma(yt))
L = findL(Xt[:,:,0])
H0 = startvalues(sigma, L)
HPm = np.zeros((1, dim + 1))
Hopt = optimizeHP(Xt[:,:,0], Xgrad[:,:,0], yt, ygrad, eps, epsgrad, H0, region, 1)
sigma = Hopt[0]
L = Hopt[1:len(Hopt)]
HPm[0, :] = np.concatenate((np.array([[sigma]]), np.array([L])), axis=1)
print("\n")