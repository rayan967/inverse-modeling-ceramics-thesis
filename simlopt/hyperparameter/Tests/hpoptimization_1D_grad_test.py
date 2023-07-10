import numpy as np
import matplotlib.pyplot as plt


from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from HPOpt.hyperparameteroptimization import *
from basicfunctions.utils.toydata import *
from HPOpt.utils.setstartvalues import *

""" Test for 2D log margonal ll with gradient data """

    
dim = 1
m = 1
n = 3
data = createartificaldata(n, dim, m)
datareal = createartificaldata(1000, dim, m)
freal = np.ones((1,m))*14.69 # Real parameter set p = 3.8,0.5

""" DEBUG """
Xt = data[:,0:dim].reshape((-1,dim,1))
Xgrad = Xt
yt = data[:,dim,0].reshape((-1,1))
yg = data[:,dim+1:]
ygrad = np.insert(yg[:,:,0],1,[]).reshape(-1,1)
ytilde = np.concatenate((yt,ygrad.reshape(-1,1)))


eps = 1E-4*np.ones((Xt[:,:,0].shape[0]))
epsgrad = 1E-4*np.ones((Xt[:,:,0].shape[0]))


""" Covariance matrix """
H = [1.0,1.0]
lmllgrad = logmarginallikelihood(H, Xt[:,:,0], Xgrad[:,:,0], yt, ygrad, eps, epsgrad, 1)
grad = logmarginallikelihood_der(H,Xt[:,:,0], Xgrad[:,:,0],yt,ygrad,eps,epsgrad,1)


'Hyperparameter optmization'
region = ((1E-3, None), (1, 20))
sigma = np.abs(findsigma(yt))
L = findL(Xt[:,:,0])
H0 = startvalues(sigma, L)
HPm = np.zeros((1, dim + 1))
Hopt = optimizeHP(Xt[:,:,0], Xgrad[:,:,0], yt, ygrad, eps, epsgrad, H0, region, 1)
sigma = Hopt[0]
L = Hopt[1:len(Hopt)]
HPm[0, :] = np.concatenate((np.array([[sigma]]), np.array([L])), axis=1)
print("\n")



""" Plot results with optimized hyperparameter """
""" Calculate the variances at xstar using Xtstar for every experiment 1....m"""
x           = createPD(1000, 1, "grid", np.array([[0,5]]))
mat                 = kernelmatricesgrad(x,Xt[:,:,0], Xgrad[:,:,0],HPm[0, :],eps,epsgrad)
variancevector      = np.diag(mat[0] - mat[1]@np.linalg.inv(mat[2])@mat[1].T)
y_pred              = mat[1]@np.linalg.inv(mat[2])@yt
meanegradnhanced = np.concatenate((mat[1],mat[3]),axis = 1) @ (np.linalg.inv(mat[5]) @ ytilde)
variancevectograd = np.diag(mat[0]- np.concatenate((mat[1],mat[3]),axis = 1) @ (np.linalg.inv(mat[5])) @ np.concatenate((mat[1],mat[3]),axis = 1).T)

fig, axs = plt.subplots()
plt.plot(datareal[:,0], datareal[:,1], 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.scatter(Xt, yt+eps.reshape(-1,1),  label='Observations')
#plt.errorbar(Xt[:,0],yt[:,0],epsXt[0], ls='none')
plt.plot(x, meanegradnhanced, 'b-', label='Prediction')
plt.plot(x, y_pred, 'g--', label='Prediction')

plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([meanegradnhanced - np.array([variancevectograd]).reshape(1000,1),
                        (meanegradnhanced + np.array([variancevectograd]).reshape(1000,1))[::-1]]),
         alpha=.5, fc='g', ec='None', label='variance interval')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - np.array([variancevector]).reshape(1000,1),
                        (y_pred + np.array([variancevector]).reshape(1000,1))[::-1]]),
         alpha=.5, fc='b', ec='None', label='variance interval')
plt.grid(True)