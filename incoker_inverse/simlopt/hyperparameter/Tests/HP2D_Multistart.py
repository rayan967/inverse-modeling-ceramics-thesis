import numpy as np
from HPOpt.hyperparameteroptimization import *
from basicfunctions.reconstruction.datahandling import *

import matplotlib.pyplot as plt



""" ----------------------------- Load settings ----------------------------- """


data = np.loadtxt("/data/numerik/people/bzfsemle/simlopt/data/2D/trainingdata2D_asym.log", dtype=float)
data = np.loadtxt("/data/numerik/people/bzfsemle/simlopt/data/1D/trainingdata1D.log", dtype=float)
data = np.expand_dims(data, axis=2)

#TODO : DATA HANDLE FOR GRADIENT DATA

dim = 1
m = 1
standardizedata = False

""" DEBUG """
ranges      = np.array([[0,1]])
x           = createPD(100, 1, "grid", ranges)
Xt = data[:,0:dim].reshape((-1,dim,1))
yt = data[:,dim].reshape((-1,1))
freal = np.ones((1,m))*0.160890 # sin(x+y) , y = 0.523 - x

if standardizedata == True:
    print("Standardize data with {} spatial features".format(dim))
    datascaled = handledata(data, freal, dim, standardizedata)
    Xt = datascaled[0]
    yt = datascaled[1]
    Xgrad = datascaled[2]
    ygrad = datascaled[3]
    scaler = datascaled[4]
    m = datascaled[5]
    features = datascaled[6]

else:
    datascaled = handledata(data, freal, dim, standardizedata)
    Xt = datascaled[0]
    yt = datascaled[1]
    Xgrad = datascaled[2]
    ygrad = datascaled[3]
    scaler = datascaled[4]
    m = datascaled[5]
    features = datascaled[6]

n = Xt.shape[0]

""" -------------------------- Data error ----------------------------- """
epsXt = (data[:,-1,0].reshape((m,Xt.shape[0])))**2
#ygrad = np.array([])
#Xgrad = np.array([]) # HAS TO BE CHANGED FOR  for FEM example
epsXgrad = 1E-5 * np.ones((m, Xgrad.shape[0] * dim))

print("Data parameters ")
print(" Spatial dimension: {}".format(dim))
print(" Number of experiments: {}".format(m))
print(" Number of features: {}".format(features))
print(" Number of training points: {}".format(Xt.shape[0]))
print(" Number of ge-training points: {}".format(Xgrad.shape[0]))

""" -------------------------- HP parameters -------------------------- """
#bounds = ((1E-2,1E5),(1,1E5),(1,1E5))
bounds = ((-1E1, 1E3), (-1E5, 1E5))
""" ------------------------------------------ NON GRADIENT ------------------------------------------"""


#HPm = optimizehyperparametersmultistart( Xt,None, yt, None, epsXt, None, bounds)
HPm = optimizehyperparameters(Xt, None, yt, None, epsXt, None, bounds, "mean")
#HPmgrad = optimizehyperparametersmultistart( Xt,Xgrad, yt, ygrad, epsXt, epsXgrad, bounds)


""" ------------------------------------------ WITH GRADIENT DATA ------------------------------------------ """

mat                 = kernelmatrices(x,Xt[:,:,0],HPm[0,:],epsXt)
variancevector      = np.diag(mat[0] -mat[1]@np.linalg.inv(mat[2])@mat[1].T)
y_pred              = mat[1] @ (np.linalg.inv(mat[2]) @ yt)

# =============================================================================
#
# ytilde              = np.concatenate((yt,ygrad))
# mat                 = kernelmatricesgrad(x,Xt[:,:,0],Xgrad[:,:,0],HPmgrad,epsXt,epsXgrad)
# variancevectorgrad  = np.diag(mat[0] -np.concatenate((mat[1],mat[3]),axis = 1)@np.linalg.inv(mat[5])@(np.concatenate((mat[1],mat[3]),axis = 1)).T)
# y_predgrad          = np.concatenate((mat[1],mat[3]),axis = 1) @ (np.linalg.inv(mat[5]) @ ytilde)
# =============================================================================

""" REAL DATA """
#df = dfreal(x)


""" Plotting """
fig, axs = plt.subplots(1)
fig.suptitle('Vertically stacked subplots')

axs.plot(x,x*np.sin((2*np.pi)*x) , 'r:', label=r'$f(x) = x\,\sin(x)$')

axs.scatter(Xt, yt,  label='Observations')
# =============================================================================
# axs.scatter(Xgrad, yt,c = 'g', marker = "*", label='Observations')
#
# =============================================================================
axs.plot(x, y_pred, 'b-', label='Prediction')
# =============================================================================
# axs.plot(x, y_predgrad, 'g--', label='Prediction')
#
# =============================================================================
axs.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - np.array([variancevector]).reshape(100,1),
                        (y_pred + np.array([variancevector]).reshape(100,1))[::-1]]),
         alpha=.25, fc='b', ec='None', label='variance interval')
# =============================================================================
# axs.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([y_predgrad - np.array([variancevectorgrad]).reshape(100,1),
#                         (y_predgrad + np.array([variancevectorgrad]).reshape(100,1))[::-1]]),
#          alpha=.25, fc='g', ec='None', label='variance interval')
# =============================================================================

axs.grid(True)

plt.show()



