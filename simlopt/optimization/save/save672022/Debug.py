import numpy as np
import matplotlib.pyplot as plt

from basicfunctions.utils.creategrid import *
from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from HPOpt.hyperparameteroptimization import *
from basicfunctions.reconstruction.deltap import *
from HPOpt.utils.setstartvalues import *
from basicfunctions.utils.creategrid import *
from sklearn.preprocessing import StandardScaler
from scipy import optimize
from confirmation import *

from scipy.optimize import Bounds

plt.close('all')


def data(X):
    return X*np.sin(X)
    #return np.sin(X)
    
def ddata(X):
    return X*np.cos(X)+np.sin(X)
    #return np.cos(X)

' Lossfunction and its derivate '
def loss(w0, idx, x, Xtextended, ytextended, epsXtextended, HPm):
    """ Calcualte eps(W) """
    current_eps = epsofw(w0, d=1)

    """ Set new eps-training data """
    epsXtextended[idx] = current_eps**2

    """ Calculate dp with new eps at x """
    matrices = kernelmatrices(x, Xtextended, HPm[0, :], epsXtextended)
    metaerror = matrices[0] - (matrices[1]@np.linalg.inv(matrices[2])@matrices[1].T)
        
    L = HPm[0, 1:]
    alpha = np.linalg.inv(matrices[2]) @ ytextended
    df = dGPR(x, Xtextended, matrices[1], L)@alpha

    SigmaInv = np.diagflat(1/metaerror)
    A = np.linalg.inv(df.T@SigmaInv@df)
    B = df.T@SigmaInv@metaerror.T
    dp = -A@B

    return (dp-1E-4)**2

def dloss(w0, idx, x, Xtextended, ytextended, epsXtextended, HPm):
    """ Calcualte eps(W) """
    current_eps = epsofw(w0, d=1)

    """ Set new eps-training data """
    epsXtextended[idx] = current_eps**2

    """ Calcualte dp with new eps at x """
    matrices = kernelmatrices(x, Xtextended, HPm[0, :], epsXtextended)
    metaerror = matrices[0] - \
        (matrices[1]@np.linalg.inv(matrices[2])@matrices[1].T)

    L = HPm[0, 1:]
    alpha = np.linalg.inv(matrices[2]) @ ytextended
    df = dGPR(x, Xtextended, matrices[1], L)@alpha

    SigmaInv = np.diagflat(1/metaerror)
    A = np.linalg.inv(df.T@SigmaInv@df)
    B = df.T@SigmaInv@metaerror.T
    dp = -A@B
    
    """ Calculate dpdw with new eps at x """

    dfdepsj = np.zeros((m, dim))
    dEdepsi = np.zeros((Xtextended.shape[0], Xtextended.shape[0]))

    eps = epsXtextended
    epsj = epsXtextended[idx]
    dEdepsi[idx, idx] = 2*np.sqrt(epsj)

    matricesdf = kernelmatrices(x, Xtextended, HPm[0, :], eps)
    L = HPm[0, 1:]

    invK = np.linalg.inv(matricesdf[2])

    dfdepsj[0, :] = -dGPR(x, Xtextended, matricesdf[1],L) @ (invK @ dEdepsi @ invK) @ ytextended
    #dfdepsj[i,:]  mit Mathematica gecheckt

    """ Ableitung var nach deps """
    dvardepsj = np.zeros((m, 1))
    dvardepsj[0, :] = matricesdf[1] @ invK @ dEdepsi @ invK @ matricesdf[1].T
    #ddvardepsj[i,:]  mit Mathematica gecheckt

    """ Ableitung SigmaInv nach deps """
    dSigmaInvdepsi = np.zeros((m, m))
    dSigmaInvdepsi[0, 0] = (matricesdf[1]@invK @ dEdepsi @
                            invK@matricesdf[1].T)[0, 0]
    dAdepsj = -A@(dfdepsj.T@SigmaInv@df - df.T@SigmaInv @
                  dSigmaInvdepsi@SigmaInv @ df + df.T @ SigmaInv@dfdepsj)@A
    dBdepsj = dfdepsj.T@SigmaInv@metaerror.T - \
        df.T@SigmaInv@dSigmaInvdepsi@SigmaInv@metaerror.T + df.T@SigmaInv@dvardepsj
    ddpdepsj = -((dAdepsj@B) + (A@dBdepsj))
    dndpdeps = 2*np.dot((dp-1E-4).T, ddpdepsj)

    ' deps / dw '
    w = W(epsj)
    depsdW = deps(w)
    
    ' Compose everything'
    dndpdW = dndpdeps * depsdW 
    ' return '
    return dndpdW[0,0]

def hess(w0, idx, x, Xtextended, ytextended, epsXtextended, HPm):
    
    
    hess = np.zeros([2])
    h = np.array([0,1E-6])
    
    for i in range(2):
        """ Calcualte eps(W) """
        current_eps = epsofw(w0+h[i], d=1)
    
        """ Set new eps-training data """
        epsXtextended[idx] = current_eps**2
    
        """ Calcualte dp with new eps at x """
        matrices = kernelmatrices(x, Xtextended, HPm[0, :], epsXtextended)
        metaerror = matrices[0] - \
            (matrices[1]@np.linalg.inv(matrices[2])@matrices[1].T)
    
        L = HPm[0, 1:]
        alpha = np.linalg.inv(matrices[2]) @ ytextended
        df = dGPR(x, Xtextended, matrices[1], L)@alpha
    
        SigmaInv = np.diagflat(1/metaerror)
        A = np.linalg.inv(df.T@SigmaInv@df)
        B = df.T@SigmaInv@metaerror.T
        dp = -A@B
        
        """ Calculate dpdw with new eps at x """
    
        dfdepsj = np.zeros((m, dim))
        dEdepsi = np.zeros((Xtextended.shape[0], Xtextended.shape[0]))
    
        eps = epsXtextended
        epsj = epsXtextended[idx]
        dEdepsi[idx, idx] = 2*np.sqrt(epsj)
    
        matricesdf = kernelmatrices(x, Xtextended, HPm[0, :], eps)
        L = HPm[0, 1:]
    
        invK = np.linalg.inv(matricesdf[2])
    
        dfdepsj[0, :] = -dGPR(x, Xtextended, matricesdf[1],L) @ (invK @ dEdepsi @ invK) @ ytextended
        #dfdepsj[i,:]  mit Mathematica gecheckt
    
        """ Ableitung var nach deps """
        dvardepsj = np.zeros((m, 1))
        dvardepsj[0, :] = matricesdf[1] @ invK @ dEdepsi @ invK @ matricesdf[1].T
        #ddvardepsj[i,:]  mit Mathematica gecheckt
    
        """ Ableitung SigmaInv nach deps """
        dSigmaInvdepsi = np.zeros((m, m))
        dSigmaInvdepsi[0, 0] = (matricesdf[1]@invK @ dEdepsi @
                                invK@matricesdf[1].T)[0, 0]
        dAdepsj = -A@(dfdepsj.T@SigmaInv@df - df.T@SigmaInv @
                      dSigmaInvdepsi@SigmaInv @ df + df.T @ SigmaInv@dfdepsj)@A
        dBdepsj = dfdepsj.T@SigmaInv@metaerror.T - \
            df.T@SigmaInv@dSigmaInvdepsi@SigmaInv@metaerror.T + df.T@SigmaInv@dvardepsj
        ddpdepsj = -((dAdepsj@B) + (A@dBdepsj))
        dndpdeps = 2*np.dot((dp-1E-3).T, ddpdepsj)
    
        ' deps / dw '
        w = W(epsj)
        depsdW = deps(w)
        
        ' Compose everything'
        dndpdW = dndpdeps * depsdW 
        hess[i] = dndpdW
    return (hess[1]-hess[0])/h[1]

' Initial data '
dim = 1
np.random.seed(42)

err =  np.array([1E-3])
lowerlimit = -2
upperlimit = 10
numberofinitialpoints = 8
# = createPD(numberofinitialpoints, 1, "grid", np.array([[lowerlimit, upperlimit]]))

X= np.array([[-2],[0],[6],[7],[8]])

epsXt = np.random.choice(err, X.shape[0])
' Add a randomly chosen error to the data points '
yt = data(X)+np.sqrt(epsXt.reshape((-1, 1)))
n = X.shape[0]
m = 1

'Hyperparameter optmization'
region = ((1E-3, None), (1, 20))
sigma = np.abs(findsigma(yt))
L = findL(X)
H0 = startvalues(sigma, L)
HPm = np.zeros((1, 1 + 1))
Hopt = optimizeHP(X, None, yt, None, epsXt, None, H0,region, 0)
sigma = Hopt[0]
L = Hopt[1:len(Hopt)]
HPm[0, :] = np.concatenate((np.array([[sigma]]), np.array([L])), axis=1)
print("\n")


xplot = createPD(1000, 1, "grid", np.array([[lowerlimit, upperlimit]]))
mat = kernelmatrices(xplot, X, HPm[0, :], epsXt)
variancevector = np.diag(mat[0] - mat[1]@np.linalg.inv(mat[2])@mat[1].T)
y_pred = mat[1]@np.linalg.inv(mat[2])@yt[:, 0]

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(xplot, y_pred, 'b--', label='Prediction')
ax1.plot(xplot, data(xplot), 'r--', label='$f(x)=x^2$')
ax1.fill(np.concatenate([xplot, xplot[::-1]]),
         np.concatenate([y_pred.reshape(1000, 1) - np.array([variancevector]).reshape(1000, 1),
                        (y_pred.reshape(1000, 1) + np.array([variancevector]).reshape(1000, 1))[::-1]]),
         alpha=.5, fc='b', ec='None', label='variance interval')
ax1.scatter(X[0:n], yt[0:n], color ="blue", label='Observations')
ax1.scatter(X[n:], yt[n:], color='red',  label='Observations')
ax1.errorbar(
    X.squeeze(),
    yt.squeeze(),
    yerr=(np.sqrt(epsXt)).squeeze(),
    lw=0,
    elinewidth=1.0,
    color="C1",
)
ax1.grid(True)

thresHold = 1E-3
delta = 0.25
nGP = 5

x = np.array([[3]])
initsize = X.shape[0]
'-- Create ghost points around point of maximum parameter error --'
Xghost = np.linspace(x-delta, x+delta, num=nGP).reshape((-1, 1)) 
if (Xghost < lowerlimit).any():
    Xghost = np.linspace(lowerlimit, x+delta, num=nGP).reshape((-1, 1))
elif (Xghost > upperlimit).any():
    Xghost = np.linspace(x-delta, upperlimit, num=nGP).reshape((-1, 1))
    
' Add them to the existing data '
matxghost = kernelmatrices(Xghost, X, HPm[0, :], epsXt)
alpha = np.linalg.inv(matxghost[2]) @ yt
meanghost = matxghost[1]@alpha.reshape(-1, 1)

' Add ghost points to the training data, where for now the errors are infinite '
Xtextended = np.concatenate((X, Xghost), axis=0)
ytextended = np.concatenate((yt, data(Xghost)))

' Add the error of the ghost points to the existign ones'
epsXghost = 1E10*np.ones(Xghost.shape[0])
epsXtextended = np.concatenate((epsXt, epsXghost), axis=0)

epsinitial  = np.concatenate((epsXt, epsXghost), axis=0)

' Inline work model '
def W(eps, d=1):
    return 1/d * eps**(-d)

def epsofw(W, d=1):
    return d*W**(-1/d)

def deps(W, d=1):
    return -(d*W)**(-((d+1)/d))
Wbudget = 1E5
threshold = 1E-6

""" Inequalites
	1. w0 < W < Wbudget
    2. threshold < epsilon_i< eps0
	"""
def cons_f(x):
    d = 1
    return [(d*x[0])**(-1/d)]
def cons_J(x):
    d = 1
    return [-(x[0])**(-((d+1)/d))]  


idx = 7
if idx >= initsize:
    w0 = W(1E-1)
    bounds = Bounds([w0],[Wbudget])
    nonlinear_constraint = NonlinearConstraint(cons_f, threshold, epsofw(w0, d=1), jac=cons_J, hess=BFGS())
    
    print("Index: {}, point x: {}(e), initial work value {}".format(idx, Xtextended[idx],w0))
    dw = 0  
    print("Initial eps: {}".format(epsofw(w0)))
else:
    w0 = W(np.sqrt(epsXtextended[idx]))
    bounds = Bounds([w0],[Wbudget])
    nonlinear_constraint = NonlinearConstraint(cons_f, threshold, epsofw(w0, d=1), jac=cons_J, hess=BFGS())
    print("Index: {}, point x: {}(i), initial work value {}".format(idx, Xtextended[idx],w0))
    print("Initial accuracy: {}".format(np.sqrt(epsXtextended[idx])))  
    dw = 0
#hessian= hess(w0, idx, x, Xtextended, ytextended, epsXtextended, HPm)
res  = minimize(loss, w0, args = (idx, x, Xtextended, ytextended, epsXtextended, HPm),
                method= 'trust-constr', jac= dloss,hess=BFGS(),
                constraints=[nonlinear_constraint], bounds = bounds,
                options={'maxiter':2000,'disp': True, 'gtol':1E-20,'xtol':1E-9 ,'barrier_tol': 1e-10})
                #options={'maxiter': 750,'ftol': 1E-20,'gtol': 1e-20,'iprint': 1,'disp': True })
print(res.x)
print(res.fun)
print(np.sqrt(res.fun))
print(epsofw(res.x))

'Calcualte dp = dpv + dpphys for every point in xs '
eps = np.array([[1E-1,1E-2,1E-3,1E-4,1E-5,1E-6,1E-7,1E-8,1E-9]])
n = eps.shape[1]
dp = np.zeros((1, n))
epsplot = np.zeros((1, n)) 

for i in range(Xtextended.shape[0]): 
    for w in range(n):
        
        #x = np.array([Xtextended[i,:]])            
        
        'Calculate variance at every point in xs'
        epsXtextended[i] = eps[0,w]
        matxs = kernelmatrices(x, Xtextended, HPm[0, :], epsXtextended)
        varxs = matxs[0] - matxs[1]@np.linalg.inv(matxs[2])@matxs[1].T
                
        'Calculate df at xs'
        L = HPm[0,1:]
        df = np.zeros((1, 1))
        alpha = np.linalg.inv(matxs[2])@ytextended
        df = dGPR(x, Xtextended, matxs[1], L)@alpha
        
        dp[0,w] = -(varxs)/ df
        #print("dp {} at eps {:g} costs {}".format( np.abs(dp[idx,w]),epsXt[idx],W(epsXt[idx], d=1)))
    
        epsplot[0,w] = np.sqrt(eps[0,w])
    
    #print(i)
    if i >= initsize:
        epsXtextended[i] = 1E10
    else:
        epsXtextended[i] = 1E-3

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) 
    
    ax.plot(epsplot[0][:],np.abs(dp[0][:]), label="x: {}".format(Xtextended[i]))     
    ax.hlines(thresHold, np.sqrt(1E-9), np.sqrt(1E-1),color = 'red', linestyles='dashed')
    ax.set_xscale('log')
    ax.set_yscale('log')         
    ax.grid(True)    
    ax.set_xlabel(r'$\log{\varepsilon}$')
    ax.set_ylabel(r'$\log{\left \| \delta p \right \|_2}$')
    ax.legend()
    dp = np.zeros((1, n))




