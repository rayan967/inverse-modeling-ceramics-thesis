import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *

from os import path

def confirmresult(x,gp,epsphys,dptarget,foundsolution,seteps,filepath,filename):

    mean = gp.predictmeanDEBUG(x)
    dim = gp.getdim

    'Add x to the given set'
    gp.adddatapoint(x)
    gp.adddatapointvalue(mean)
    gp.addaccuracy(np.array([1E-1]).reshape(1,-1))

    'Calcualte dp = dpv + dpphys for every point in xs '
    eps = (10.**(-np.arange(1,9))).reshape((1,-1))

    #eps = np.array([[1E-1,1E-2,1E-3,1E-4,1E-5,1E-6,1E-7,1E-8,1E-9]])
    epssecondaxis = eps.tolist()
    n = eps.shape[1]
    dp = np.zeros((1, n))
    dpwithoutphys = np.zeros((1, n))
    dpphys = np.zeros((1, n))
    epsplot = np.zeros((1, n))

    for w in range(n):

        idx = gp.getX.shape[0]-1
        gp.addaccuracy(eps[0,w],idx)

        epsplot[0,w] = eps[0,w]

        'Calculate variance at every point in xs'
        varxs = gp.predictvarianceDEBUG(x)

        'Calculate df at xs'
        df = gp.predictderivativeDEBUG(x)

        if dim == 1 :
            dp[0,w] = -(varxs + epsphys )/ df
            dpwithoutphys[0,w] = -(varxs)/ df
            dpphys[0,w]= -( epsphys )/ df
        else:
            dp[0,w] =  np.linalg.norm(-(varxs+epsphys)/(np.dot(df.T,df))*df,2)
            dpwithoutphys[0,w] =  np.linalg.norm(-(varxs)/(np.dot(df.T,df))*df,2)
            dpphys[0,w]= np.linalg.norm(-(epsphys)/(np.dot(df.T,df))*df,2)


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epsplot[0][:],np.abs(dp[0][:]), label="dp at x")
    ax.plot(epsplot[0][:],np.abs(dpphys[0][:]),'g--',label="dp from phys.")
    ax.plot(epsplot[0][:],np.abs(dpwithoutphys[0][:]),'c-',label="dp without phys.")

    if foundsolution is not None:
        ax.axvline(foundsolution,color = 'orange', linestyle='--',label="found solution")
    if seteps is not None:
        ax.axvline(seteps,color = 'green', linestyle='--',label="set solution")

    ax.hlines(dptarget, 1E-8, 1E-1,color = 'red', linestyles='dashed',label="target accuracy")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks((10.**(-np.arange(1,9))).tolist())
    ax.grid(True)
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(r'${\left \| \delta p \right \|_2}$')
    ax.legend()
    ax.figure.savefig(os.path.join(filepath,filename))



def confirmresultgrad(x,gp,epsphys,dptarget,foundsolution,filepath,filename):
    eps = (10.**(-np.arange(1,9))).reshape((1,-1))
    n = eps.shape[1]

    'dp withouth gradient information'
    dp = np.zeros((1, n))
    dpwithoutphys = np.zeros((1, n))
    dpphys = np.zeros((1, n))
    varlist= []
    dflist = []

    'dp with gradient information '
    dpgrad = np.zeros((1, n))
    dpwithoutphysgrad = np.zeros((1, n))
    dpphysgrad = np.zeros((1, n))
    varlistgrad = []
    dflistgrad = []

    'dp with constant gradient information '
    dpgradconst = np.zeros((1, n))
    dpwithoutphysgrad = np.zeros((1, n))
    dpphysgradconst = np.zeros((1, n))
    varlistgradconst = []
    dflistgradconst = []

    'dp with constant information '
    dpepsconst = np.zeros((1, n))
    dpwithoutphysgrad = np.zeros((1, n))


    mean = gp.predictmean(x)
    df = gp.predictderivative(x)
    dim = gp.getdim

    for i in range (4):

        """
        i = 0: Nur epsilon wird variiert
        i = 1: Gradienteninformationen werden hinzugefügt, aber konstant gelassen, epsilon wird variiert
        i = 2: Gradienteninformationen werden hinzugefügt, epsgrad und epsilon werden variiert
        """


        epsplot = np.zeros((1, n))

        if i == 0:
            'Add x to the given set as data point'
            gp.adddatapoint(x)
            gp.adddatapointvalue(mean)
            gp.addaccuracy(np.array([1E-1]).reshape(1,-1))

            for w in range(n):

                idx = gp.getX.shape[0]-1
                gp.addaccuracy(eps[0,w],idx)

                epsplot[0,w] = eps[0,w]

                'Calculate variance and df'
                varxs = gp.predictvariance(x)
                df = gp.predictderivative(x)

                varlist.append(varxs[0,0])
                dflist.append(df[0,0])

                dp[0,w] = -(varxs + epsphys)/ df
                dpwithoutphys[0,w] = -(varxs)/ df
                dpphys[0,w]= -( epsphys )/ df

        if i == 1:
            'Add x to the given set as data point'
            gp.adddatapoint(x)
            gp.adddatapointvalue(mean)
            gp.addaccuracy(np.array([1E-1]).reshape(1,-1))

            gp.addgradientdatapoint(x)
            gp.adddgradientdatapointvalue(df)
            gp.addgradaccuracy(np.array([1E-1]).reshape(1,-1))

            for w in range(n):

                idx = gp.getX.shape[0]-1
                gp.addaccuracy(eps[0,w],idx)

                epsplot[0,w] = eps[0,w]

                'Calculate variance and df'
                varxs = gp.predictvariance(x)
                df = gp.predictderivative(x)

                varlistgradconst.append(varxs[0,0])
                dflist.append(df[0,0])

                dpgradconst[0,w] = -(varxs + epsphys)/ df
                dpwithoutphys[0,w] = -(varxs)/ df
                dpphysgradconst[0,w]= -( epsphys )/ df

        if i == 2:
            'Add x to the given set as data point'
            gp.adddatapoint(x)
            gp.adddatapointvalue(mean)
            gp.addaccuracy(np.array([1E-1]).reshape(1,-1))

            gp.addgradientdatapoint(x)
            gp.adddgradientdatapointvalue(df)
            gp.addgradaccuracy(np.array([1E-1]).reshape(1,-1))

            for w in range(n):

                idx = gp.getX.shape[0]-1
                gp.addaccuracy(eps[0,w],idx)
                gp.addgradaccuracy(eps[0,w],idx)

                epsplot[0,w] = eps[0,w]

                'Calculate variance and df'
                varxs = gp.predictvariance(x)
                df = gp.predictderivative(x)

                varlistgrad.append(varxs[0,0])
                dflist.append(df[0,0])

                dpgrad[0,w] = -(varxs + epsphys)/ df
                dpwithoutphys[0,w] = -(varxs)/ df
                dpphysgrad[0,w]= -( epsphys )/ df

        if i == 3:
            'Add x to the given set as data point'
            gp.adddatapoint(x)
            gp.adddatapointvalue(mean)
            gp.addaccuracy(np.array([1E-1]).reshape(1,-1))

            gp.addgradientdatapoint(x)
            gp.adddgradientdatapointvalue(df)
            gp.addgradaccuracy(np.array([1E-1]).reshape(1,-1))

            for w in range(n):

                idx = gp.getX.shape[0]-1
                gp.addgradaccuracy(eps[0,w],idx)

                epsplot[0,w] = eps[0,w]

                'Calculate variance and df'
                varxs = gp.predictvariance(x)
                df = gp.predictderivative(x)

                varlistgrad.append(varxs[0,0])
                dflist.append(df[0,0])

                dpepsconst[0,w] = -(varxs + epsphys)/ df

        print("\n")

        'Delete data points for new case'
        if i == 0:
                gp.deletedatapoint()
        if i == 1:
                gp.deletedatapoint()
                gp.deletegradientdatapoint()
        if i == 2:
                gp.deletedatapoint()
                gp.deletegradientdatapoint()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(epsplot[0][:],np.abs(dp[0][:]), label="dp at x")
    #ax.plot(epsplot[0][:],np.abs(dpphys[0][:]),'g--',label="dp from phys.")
    #ax.plot(epsplot[0][:],np.abs(dpwithoutphys[0][:]),'c-',label="dp without phys.")

    ax.plot(epsplot[0][:],np.abs(dpgrad[0][:]), label="dp_grad at x")
    #ax.plot(epsplot[0][:],np.abs(dpphysgrad[0][:]),'g-.',label="dp_grad from phys.")
    #ax.plot(epsplot[0][:],np.abs(dpphysgrad[0][:]),'c-',label="dp_grad without phys.")

    ax.plot(epsplot[0][:],np.abs(dpgradconst[0][:]), label="dp_grad_const at x")
    #ax.plot(epsplot[0][:],np.abs(dpphysgradconst[0][:]),'g-.',label="dp_grad_const from phys.")

    #ax.plot(epsplot[0][:],np.abs(dpepsconst[0][:]), label="dp_eps_const")
    #ax.plot(epsplot[0][:],np.abs(dpphysgradconst[0][:]),'g-.',label="dp_grad_const from phys.")

    if foundsolution is not None:
        ax.axvline(foundsolution,color = 'orange', linestyle='--',label="found solution")

    ax.hlines(dptarget, 1E-8, 1E-1,color = 'red', linestyles='dashed',label="target accuracy")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks((10.**(-np.arange(1,9))).tolist())
    ax.grid(True)
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(r'${\left \| \delta p \right \|_2}$')
    ax.legend()
    ax.figure.savefig(os.path.join(filepath,filename))