import matplotlib.pyplot as plt
import numpy as np
from basicfunctions.utils.creategrid import *
from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.utils.distance import *
from HPOpt.hyperparameteroptimization import *
from Optimization.optimizeaccuracy import *
from basicfunctions.reconstruction.deltap import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from HPOpt.utils.logmarginal import *
from HPOpt.utils.logmarginalgrad import *


plt.close('all')


print("---- Loading data ----")
try:
    data = np.load('D:/Uni/Zuse/simlopt/simlopt/Reconstruction/Tests/8.npy')
    print("Data done...")
    freal = np.load('D:/Uni/Zuse/simlopt/simlopt/Reconstruction/Tests/100.npy')
except:
    print("Could not open the files...")

""" Extract data """
Xt = data[:,0].reshape((-1,1))
yt = data[:,1]

N = Xt.shape[0]

""" Extract gradient data """
Xgrad = Xt
ygrad = data[:,2]

ranges      = np.array([[0.3,0.5]])
x           = createPD(100, 1, "grid", ranges)
""" -------------------------- Prescale data -------------------------- """
scalefactor = 1E9
yt          = yt    * scalefactor
ygrad       = ygrad * scalefactor

Xreal       = freal[:,0].reshape((-1,1))
yreal       = freal[:,1] * scalefactor
ygradreal  =  freal[:,2] * scalefactor

""" -------------------------- Data error ----------------------------- """
epsXt      = 1E-5*np.ones((1,Xt.shape[0]))
epsXtgrad  = 1E-5*np.ones((1,Xt.shape[0]))

""" -------------------------- HP OPT ----------------------------- """
region      = ((1e-3, None), (1E-3, None))
H0          = []
sigma       = 10
H0.append(sigma)
L           = [1]
H0          = H0+L
gflag       = 0

Hopt            = optimizeHP(Xt,None,yt,None,epsXt,None,H0,region, gflag)
sigma           = Hopt[0]
L               = Hopt[1:len(Hopt)]
HPm             = np.concatenate((np.array([[sigma]]),np.array([L])),axis = 1)


""" -------------------------- Calculate the derivatives -------------------------- """
matdf               = kernelmatrices(x,Xt,HPm.T, epsXt)
derivatives         = (dGPR(x,Xt,matdf[1],L)@(np.linalg.inv(matdf[2])@yt)).T
df                  = ygradreal

""" Calculate the mean and the variance """
mat                 = kernelmatrices(x,Xt,HPm.T,epsXt)
variancevector      = np.diag(mat[0] - mat[1]@np.linalg.inv(mat[2])@mat[1].T)

y_pred  = mat[1]@np.linalg.inv(mat[2])@yt
y_pred  = y_pred.reshape((-1,1))
y_real  = yreal


""" ------------------------------------------ WITH GRADIENT DATA ------------------------------------------ """
H0          = []
sigma       = 1.
H0.append(sigma)
L               = [1.]
H0              = H0+L
gflag           = 0
Hopt            = optimizeHP(Xt,Xgrad,yt,ygrad,epsXt,epsXtgrad,H0,region, 1)
sigma           = Hopt[0]
L               = Hopt[1:len(Hopt)]
HPgrad          = np.concatenate((np.array([[sigma]]),np.array([L])),axis = 1)

ytilde              = np.concatenate((yt,ygrad))
mat                 = kernelmatricesgrad(x,Xt,Xgrad,HPgrad.T,epsXt,epsXtgrad)
variancevectorgrad  = np.diag(mat[0] -np.concatenate((mat[1],mat[3]),axis = 1)@np.linalg.inv(mat[5])@(np.concatenate((mat[1],mat[3]),axis = 1)).T)
y_predgrad          = np.concatenate((mat[1],mat[3]),axis = 1) @ (np.linalg.inv(mat[5]) @ ytilde)
derivativesgrad     = (dGPRgrad(x,Xt,Xgrad,sigma,L) @ (np.linalg.inv(mat[5]) @ ytilde)).T

""" Plot """

fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')

axs[0].plot(Xreal, yreal, 'r:', label=r'$f(x) = x\,\sin(x)$')
axs[0].scatter(Xt, yt,  label='Observations')
axs[0].scatter(Xgrad, yt,c = 'g', marker = "*", label='Observations')

axs[0].plot(x, y_pred, 'b-', label='Prediction')
axs[0].plot(x, y_predgrad, 'g--', label='Prediction')

axs[0].fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - np.array([variancevector]).reshape(100,1),
                        (y_pred + np.array([variancevector]).reshape(100,1))[::-1]]),
         alpha=.25, fc='b', ec='None', label='variance interval')
axs[0].fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_predgrad - np.array([variancevectorgrad]).reshape(100,1),
                        (y_predgrad + np.array([variancevectorgrad]).reshape(100,1))[::-1]]),
         alpha=.25, fc='g', ec='None', label='variance interval')
axs[0].grid(True)
axs[0].set_xlabel('$x$')
axs[0].set_ylabel('$f(x)$')
axs[1].plot(x, derivatives.T,'b-')
axs[1].plot(x, derivativesgrad.T,'g--')
axs[1].plot(x,df, 'r:')
axs[1].grid(True)
axs[1].set_xlabel('$x$')
axs[1].set_ylabel('$1/df$')




