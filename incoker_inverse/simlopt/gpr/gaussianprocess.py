import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

from os import path

from incoker_inverse.simlopt.basicfunctions.covariance.cov import *
from incoker_inverse.simlopt.basicfunctions.derivative.dGPR import *
from incoker_inverse.simlopt.basicfunctions.utils.creategrid import *

from incoker_inverse.simlopt.hyperparameter.hyperparameteroptimization import *
from incoker_inverse.simlopt.hyperparameter.utils.setstartvalues import *
from incoker_inverse.simlopt.hyperparameter.utils.crossvalidation import*

from incoker_inverse.simlopt.basicfunctions.utils.creategrid import *

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from incoker_inverse.simlopt.gpr.variationalgaussianprocess import *

class ShapeError(Exception):

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return 'ShapeError, {0} '.format(self.message)
        return 'ShapeError has been raised'


class GPR:

    def __init__(self, Xt, yt, Xgrad, ygrad, epsxt, epsxgrad, hyperparameter=None):
        """

        Constructor for the Gaussian Process Regression.

        Parameters
        ----------
        Xt : np.array(nxd)
            Training data
        yt : np.array(nx1)
            Values at training data
        Xgrad : np.array(nxd)
            Gradient training data
        ygrad : np.array(1xd*n)
            Gradient values at gradient training data
        epsxt : np.array(1xn)
            Accuracies at training data
        epsxgrad : np.array(1xd*n)
            Accuracies at gradient training data
        hyperparameter : np.array(1,), optional
            Hyperparameter. Can be none. The default is None.

        Returns
        -------
        None.

        """
        self.X = Xt
        self.yt = yt
        self.Xgrad = Xgrad
        self.ygrad = ygrad
        self.epsxt = epsxt
        self.epsxgrad = epsxgrad
        self.hyperparameter = hyperparameter
        self.dim = self.X.shape[1]
        self.initialamount = self.X.shape[0]

        self.m = self.yt.shape[1]

        if self.hyperparameter is None:
            print("GPR - WARNING: ")
            print("No hyperparamter are set - setting unit parameters")
            self.hyperparameter = np.ones((self.m,self.dim))
        else:
            #Reshape hyperparameter to a (,1) array
            if self.hyperparameter.ndim > 1:
                pass
                #print("Reshaping hyperparamter to needed shape (1,)")
                #self.hyperparameter = np.squeeze(self.hyperparameter, axis=0)

        if Xgrad is None:
            self.Kxx = np.array([])
            self.KxX = np.array([])
            self.KXX = np.array([])
            self.KxXgrad = None
            self.KXXgrad = None
            self.KXgradXgrad = None
            self.K = None
            self.ytilde = None

        else:
            self.Kxx = np.array([])
            self.KxX = np.array([])
            self.KXX = np.array([])
            self.KxXgrad = np.array([])
            self.KXXgrad = np.array([])
            self.KXgradXgrad = np.array([])
            self.K = np.array([])
            self.ytilde = np.concatenate((self.yt, self.ygrad))

    @property
    def getdata(self):

        if self.Xgrad is not None:
            return self.X.shape[0], self.Xgrad.shape[0], self.dim
        return self.X.shape[0], None, self.dim

    @property
    def getX(self):
        return self.X

    @property
    def getXgrad(self):
        return self.Xgrad

    @property
    def gety(self):
        return self.yt

    @property
    def getygrad(self):
        return self.ygrad

    @property
    def getytilde(self):
        return self.ytilde

    @property
    def getaccuracy(self):
        return self.epsxt

    @property
    def getgradientaccuracy(self):
        return self.epsxgrad

    @property
    def getaccuracies(self):
        return np.concatenate((self.epsxt,self.epsxgrad),axis=1)
    @property
    def gethyperparameter(self):
        return self.hyperparameter

    @property
    def getdim(self):
        return self.dim

    def calculatecovariance(self, x=None):
        """


        Parameters
        ----------
        x : np.array(1,d), optional
            Value at where the covariance between all data points is calculated.

        Returns
        -------
        None.

        """

        if x is None:
            x = self.X
        else:

            try:
                assert(x.shape != (1,))
            except AssertionError:
                error = "Shape of x is: " + \
                    str(x.shape) + ". Shape should be a (1,dim)-matrix"
                raise(AssertionError(error))

        if self.Xgrad is None:
            self.Kxx, self.KxX, self.KXX = kernelmatrices(
                x, self.X, self.hyperparameter, self.epsxt)
        else:
            self.Kxx,self.KxX,self.KXX,self.KxXgrad,self.KXXgrad,self.K = kernelmatricesgrad(
                x, self.X, self.Xgrad, self.hyperparameter, self.epsxt, self.epsxgrad)

    def predictmean(self, x, mgauss = False):
        """


        Parameters
        ----------
        x : np.array(m,d), optional
            m values at where the mean is predicited.

        Returns
        -------
        mean : np.array([[m,1]])
            Predicted mean at x.

        """

        if x is None:
            print("No point(s) are set, returning...")
            return None

        try:
            assert(x.shape != (1,))
        except AssertionError:

            error = "Shape of x is: " + \
                str(x.shape) + ". Shape should be a (1,dim)-matrix"
            raise(AssertionError(error))

        if self.Xgrad is None:

            if mgauss == False:
                _,self.KxX, self.KXX = kernelmatrices(
                    x, self.X, self.hyperparameter[0,:], self.epsxt)
                alpha = np.linalg.solve(self.KXX, self.yt)
                mean = self.KxX@alpha
            else:
                m = self.yt.shape[1]
                n = x.shape[0]
                mean = np.zeros((n,m))


                for i in range(m):
                    _,KxX, KXX = kernelmatrices(x, self.X, self.hyperparameter[i,:], self.epsxt)
                    alpha = np.linalg.solve(KXX, self.yt[:,i])
                    mean[:,i] = KxX@alpha
        else:

            m = self.yt.shape[1]
            n = x.shape[0]
            mean = np.zeros((n,m))


            for i in range(m):

                _,KxX,_, KxXgrad,_, K = kernelmatricesgrad(x, self.X, self.Xgrad, self.hyperparameter[i,:], self.epsxt, self.epsxgrad)
                
                KXXtXXg = np.concatenate((KxX,KxXgrad), axis=1)
                alpha = np.linalg.solve(K,self.ytilde[:,i])
                mean[:,i] = KXXtXXg@alpha

        return mean

    def predictvariance(self, x, multi = False):
        """


        Parameters
        ----------
        x : np.array(m,d), optional
            m values at where the mean is predicited.

        Returns
        -------
        np.array([m,1])
            Predicted variance at x

        """

        if x is None:
            print("No point(s) are set, returning...")
            return None

        try:
            assert(x.shape != (1,))
        except AssertionError:
            error = "Shape of x is: " + \
                str(x.shape) + ". Shape should be a (1,dim)-matrix"
            raise(AssertionError(error))

        if self.Xgrad is None:
            if multi:
                if x.shape[0] > 1:
                    var = np.zeros((self.m,x.shape[0]))
                    for i in range(self.m):
                        Kxx, KxX, KXX = kernelmatrices(x, self.X, self.hyperparameter[i,:], self.epsxt)
                        var[i,:] = np.diagonal(Kxx- (KxX)@np.linalg.inv(KXX)@(KxX).T)
                    return var
                else:
                    var = np.zeros((1,self.m))
                    for i in range(self.m):
                        Kxx, KxX, KXX = kernelmatrices(x, self.X, self.hyperparameter[i,:], self.epsxt)
                        var[0,i] = Kxx- (KxX)@np.linalg.inv(KXX)@(KxX).T
                return np.abs(var)

            else:
                Kxx, KxX, KXX = kernelmatrices(x, self.X, self.hyperparameter[0,:], self.epsxt)
                var = Kxx-(KxX)@np.linalg.inv(KXX)@(KxX).T

        else:
            if multi:
                if x.shape[0] > 1:
                    var = np.zeros((self.m,x.shape[0]))
                    for i in range(self.m):
                        Kxx,KxX,_, KxXgrad,_, K = kernelmatricesgrad(x, self.X, self.Xgrad, self.hyperparameter[i,:], self.epsxt, self.epsxgrad)
                        KXXtXXg = np.concatenate((KxX, KxXgrad), axis=1)
                        var[i,:] = np.diagonal(Kxx-KXXtXXg@np.linalg.inv(K)@KXXtXXg.T)
                else:
                    var = np.zeros((1,self.m))
                    for i in range(self.m):
                        Kxx,KxX,_, KxXgrad,_, K = kernelmatricesgrad(x, self.X, self.Xgrad, self.hyperparameter[i,:], self.epsxt, self.epsxgrad)
                        KXXtXXg = np.concatenate((KxX, KxXgrad), axis=1)
                        var[0,i] = Kxx-KXXtXXg@np.linalg.inv(K)@KXXtXXg.T
               
                return np.abs(var)
            else:
                Kxx,KxX,_, KxXgrad,_, K = kernelmatricesgrad(x, self.X, self.Xgrad, self.hyperparameter[0,:], self.epsxt, self.epsxgrad)
                KXXtXXg = np.concatenate((KxX, KxXgrad), axis=1)
                var = Kxx-KXXtXXg@np.linalg.inv(K)@KXXtXXg.T
                
        return np.diagonal(var).reshape((-1, 1))

    def updateK(self, x=None):

        if x is None:
            x = self.X

        if self.Xgrad is None:
            covmatrices = kernelmatrices(
                x, self.X, self.hyperparameter, self.epsxt)
            self.KXX = covmatrices[2]

        else:
            covmatrices = kernelmatricesgrad(x, self.X, self.Xgrad, self.hyperparameter, self.epsxt, self.epsxgrad)
            self.K =  covmatrices[5]


    def predictderivative(self, x, asmatrix=False):
        """

        Parameters
        ----------
        x : np.array(1,dim)
            Point at which the derivative gets evaluated.

        asmatrix : bool

        Returns
        -------
        df : TYPE
            Gradient value at point x.
        df = [ f11,...,f1d | f21,...,f2d | .... | fn1,...,fnd ]^T

        if m>1 and asmatrix == False

        f11_1 | f11_2 | f11_m
        .     | .     | .
        .     | .     | .
        f1d_1 | f1d_2 | f1d_m
        .     | .     | .
        .     | .     | .
        .     | .     | .
        fn1_1 | fn1_2 | fn1_m
        .     | .     | .
        .     | .     | .
        fnd_1 | fnd_2 | fnd_m


        if asmatrix == True and m = 1

        df = f11 , ... , f1d
             f21 , ... , f2d
             .
             .
             fn1 , ... , fnd


        if asmatrix == True and m > 1

        return as tensor

        """
        L = self.hyperparameter[0:]
        sigma = 1

        m = self.yt.shape[1]
        n = x.shape[0]
        dim = x.shape[1]

        if self.Xgrad is None:
            if asmatrix:
                dftensor = np.zeros((n,dim,m))
                for ii in range(self.m):
                    L = self.hyperparameter[ii,:]
                    _,KxX, KXX = kernelmatrices(x, self.X, self.hyperparameter[ii,:], self.epsxt)
                    alpha = np.linalg.solve(KXX, self.yt[:,ii])
                    df = dGPR(x, self.X, KxX, L)@alpha
                    tmp = df.reshape((-1,dim)) #Changed 30.08.2022
                    dftensor[:,:,ii] = tmp
            else:
                dftensor = np.zeros((dim*n,m))
                for ii in range(m):
                    L = self.hyperparameter[ii,:]
                    _,KxX,KXX = kernelmatrices(x, self.X, self.hyperparameter[ii,:], self.epsxt)
                    alpha = np.linalg.solve(KXX, self.yt[:,ii])
                    #tmp =  dGPR(x, self.X, KxX, L)@alpha
                    dftensor[:,ii] = dGPR(x, self.X, KxX, L)@alpha
        else:
            if asmatrix:
                dftensor = np.zeros((n,dim,m))
                
                for ii in range(self.m):
                    L = self.hyperparameter[ii,:]
                    K = kernelmatrixsgrad(self.X,self.Xgrad,self.hyperparameter[ii,:], self.epsxt, self.epsxgrad)
                    alpha = np.linalg.solve(K, self.ytilde[:,ii])
                    df = dGPRgrad(x, self.X, self.Xgrad, sigma, L)@alpha
                    tmp =  df.reshape((-1,dim)) #Changed 30.08.2022
                    dftensor[:,:,ii] = tmp
            else:
                dftensor = np.zeros((n*dim,m))
                sigma = 1.0
                for ii in range(self.m):
                    L = self.hyperparameter[ii,:]
                    K = kernelmatrixsgrad(self.X,self.Xgrad,self.hyperparameter[ii,:], self.epsxt, self.epsxgrad)
                    alpha = np.linalg.solve(K, self.ytilde[:,ii])
                    dftensor[:,ii] = dGPRgrad(x, self.X, self.Xgrad, sigma, L)@alpha
        return dftensor

    def predicthessian(self, x):
        """

        Parameters
        ----------
        x : np.array(1,dim)
            Point at which the hessian gets evaluated.

        asmatrix : bool

        Returns
        -------
        df : TYPE
            Hessian value at point x.


        """

        covmatrices = kernelmatrices(
            x, self.X, self.hyperparameter, self.epsxt)
        self.KxX, self.KXX = covmatrices[1], covmatrices[2]

        L = self.hyperparameter[1:]
        sigma = self.hyperparameter[0]

        Xt = self.X

        N1 = x.shape[0]
        N3 = Xt.shape[0]

        # Prescale data
        Xscaled  = x/L
        Xtscaled = Xt/L

        # Build kernel matrices
        n1sq = np.sum(x**2,axis=1);
        n3sq = np.sum(Xtscaled**2,axis=1);

        DXgradXgrad = np.transpose(np.outer(np.ones(N3),n1sq)) + np.outer(np.ones(N1),n3sq)-2*(np.dot(Xscaled,np.transpose(Xtscaled)))
        KXgXg = sigma**2 * np.exp(-DXgradXgrad / 2.0)

        tmprow = np.array([])
        Kfdy = np.array([])
        for i in range(0,N1):
            xi = x[i,:]
            for j in range(0,N3):
                xj = Xt[j,:]
                diff = np.outer(((xi-xj)/(L**2)),((xi-xj)/(L**2)))
                tmp = KXgXg[i,j]*( diff - np.diag(1/L**2))
                if j == 0:
                    tmprow = tmp
                else:
                    tmprow = np.concatenate((tmprow,tmp),axis=1);
            if i == 0:
                Kfdy = tmprow
            else:
                Kfdy = np.concatenate((Kfdy,tmprow),axis=0);

        hessian = np.zeros((self.dim,self.dim))
        for ii in range(self.dim):
            subvec = np.zeros((self.dim,self.X.shape[0]))
            for jj in range(self.X.shape[0]):
                idx = self.dim*jj+ii
                sub = Kfdy[:,idx]
                subvec[:,jj]  = sub
            ddm = subvec@np.linalg.inv(self.KXX)@self.yt
            hessian[:,ii] = ddm[:,0]

        return hessian

    def predictderivativevariance(self, x):
        """


        Parameters
        ----------
        x : np.array(m,d), optional
            m values at where the mean is predicited.

        Returns
        -------
        np.array([m,1])
            Predicted variance at x

        """

        if x is None:
            print("No point(s) are set, returning...")
            return None

        try:
            assert(x.shape != (1,))
        except AssertionError:
            error = "Shape of x is: " + \
                str(x.shape) + ". Shape should be a (1,dim)-matrix"
            raise(AssertionError(error))


        if self.Xgrad is None:
            covmatrices = kernelmatrices(
                x, self.X, self.hyperparameter, self.epsxt)
            self.Kxx, self.KxX, self.KXX = covmatrices[0], covmatrices[1], covmatrices[2]
            var = self.Kxx - (self.KxX)@np.linalg.inv(self.KXX)@(self.KxX).T

        else:
            covmatrices = kernelmatricesgrad(
                x, self.X, self.Xgrad, self.hyperparameter, self.epsxt, self.epsxgrad)
            self.Kxx, self.KxX, self.KxXgrad, self.K = covmatrices[0], covmatrices[1], covmatrices[3], covmatrices[5]
            KXXtXXg = np.concatenate((self.KxX, self.KxXgrad), axis=1)
            var = self.Kxx - KXXtXXg@np.linalg.inv(self.K)@KXXtXXg.T

        return np.diagonal(var).reshape((-1, 1))


    def optimizehyperparameter(self, region, startvaluescheme, gridsearch=False):

        if gridsearch:
            self.hyperparameter = optimizehyperparametersmultistart(
                self.X, self.Xgrad, self.yt, self.ygrad, self.epsxt, self.epsxgrad, region)
        else:

            if self.Xgrad is None:
                if self.yt.shape[1]>=1:
                    for i in range( self.yt.shape[1]):
                        print("Optimize Hyperparameter for dataset {}".format(i))
                        self.hyperparameter[i,:] = optimizehyperparameters(self.X, None, self.yt[:,i], None, self.epsxt, None, region, startvaluescheme)
                        print("\n")
                else:
                    self.hyperparameter = np.atleast_2d(optimizehyperparameters(
                    self.X, self.Xgrad, self.yt, self.ygrad, self.epsxt, self.epsxgrad, region, startvaluescheme))

            else:
                if self.yt.shape[1]>=1:
                    for i in range( self.yt.shape[1]):
                        print("Optimize Hyperparameter for dataset {}".format(i))
                        self.hyperparameter[i,:] = optimizehyperparameters(self.X, self.Xgrad, self.yt[:,i], self.ygrad[:,i], self.epsxt, self.epsxgrad, region, startvaluescheme)
                        print("\n")
                else:
                    self.hyperparameter = np.atleast_2d(optimizehyperparameters(
                    self.X, self.Xgrad, self.yt, self.ygrad, self.epsxt, self.epsxgrad, region, startvaluescheme))

    def adddatapoint(self, x):
        """


        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.X = np.concatenate((self.X, x), axis=0)


    def adddatapointvalue(self, y, idx=None):
        """


        Parameters
        ----------
        y : np.array(m,dim)
            Values at x which are added to the current training data


        Returns
        -------
        None.

        """


        if idx is not None:
            try:
                if type(idx) is list:
                    if idx[1] is None:
                        self.yt[idx[0]:, :] = y
                    else:
                        self.yt[idx[0]:idx[1], :] = y
                else:
                    self.yt[idx, :] = y

            except IndexError as e:
                print(e)
        else:
            self.yt = np.concatenate((self.yt, y), axis=0)

        if self.Xgrad is not None:
            if self.ygrad is None:
                self.ytilde = y
            else:
                self.ytilde = np.concatenate((self.yt,self.ygrad ))

    def addgradientdatapoint(self, xgrad):
        """


        Parameters
        ----------
        xgrad : np.array([m,dim])

        Returns
        -------
        None.

        """

        if self.Xgrad is not None:
            self.Xgrad = np.concatenate((self.Xgrad, xgrad), axis=0)
        else:
            self.Xgrad = xgrad

    def addgradientdatapointvalue(self, ygrad, idx=None):
        """


        Parameters
        ----------
        ygrad : np.array(m,dim)
                Gradient values at x which are added to the current training data
                ygrad = [y_11,...,y1d, .... , ym1,...,ymd]^T

        Returns
        -------
        None.

        """
        if idx is not None:
            try:
                if type(idx) is list:
                    if idx[1] is None:
                        self.ygrad[idx[0]:, :] = ygrad
                    else:
                        self.ygrad[idx[0]:idx[1], :] = ygrad
                else:
                    self.ygrad[idx, :] = ygrad

            except IndexError as e:
                print(e)
        else:
            if self.ygrad is not None:
                self.ygrad = np.concatenate((self.ygrad, ygrad), axis=0)
            else:
                self.ygrad = ygrad

        self.ytilde = np.concatenate((self.yt, self.ygrad))

    def addaccuracy(self, eps, idx=None):
        if idx is not None:
            try:
                if type(idx) is list:
                    if idx[1] is None:
                        self.epsxt[0, idx[0]:] = eps
                    else:
                        self.epsxt[0, idx[0]:idx[1]] = eps
                else:
                    self.epsxt[0, idx] = eps
            except IndexError as e:
                print(e)
        else:
            self.epsxt = np.concatenate((self.epsxt, eps), axis=1)

    def addgradaccuracy(self, epsgrad, idx=None):
        if idx is not None:
            if type(idx) is list:
                if idx[1] is None:
                    self.epsxgrad[0, idx[0]:] = epsgrad
                else:
                    self.epsxgrad[0, idx[0]:idx[1]] = epsgrad
            else:
                try:
                    self.epsxgrad[0, idx] = epsgrad
                except IndexError as e:
                    print(e)
        else:
            if self.epsxgrad is None:
                self.epsxgrad = epsgrad
            else:
                self.epsxgrad = np.concatenate((self.epsxgrad, epsgrad), axis=1)

    def extenddata(self, x, y, ygrad):
        self.X = np.concatenate((self.X, x), axis=0)
        self.Xgrad = np.concatenate((self.Xgrad, x), axis=0)
        self.yt = np.concatenate((self.yt, y), axis=0)
        self.ygrad = np.concatenate((self.ygrad, ygrad), axis=0)

    def deletedatapoint(self,idx=None):

        if idx is None:

            if self.dim > 1:

                self.X = self.X[:-1]
                self.yt = np.delete(self.yt,-1).reshape((-1,1))
                self.epsxt = np.delete(self.epsxt,-1).reshape((1,-1))

            else:
                self.X = np.delete(self.X,-1).reshape((-1,1))
                self.yt = np.delete(self.yt,-1).reshape((-1,1))
                self.epsxt = np.delete(self.epsxt,-1).reshape((1,-1))

            if self.Xgrad is not None:
                self.ytilde = np.concatenate((self.yt,self.ygrad ))

        else:
            self.X = np.delete(self.X,idx,axis=0)
            self.yt = np.delete(self.yt,idx,axis=0)
            self.epsxt = np.delete(self.epsxt,idx,axis=1)
            if self.Xgrad is not None:
                self.ytilde = np.concatenate((self.yt,self.ygrad ))


    def deletegradientdatapoint(self,idx=None):

        if self.Xgrad is not None:

            if idx is None:

                if self.Xgrad.shape[0] == 1:
                    'Is there only one point given set to None'
                    self.Xgrad = None
                    self.ygrad = None
                    self.epsxgrad = None

                else:

                    if self.dim > 1.:
                        'Delete just the last element'
                        self.Xgrad = self.Xgrad[:-1]
                        self.ygrad =  self.ygrad[:-2]
                        self.epsxgrad = self.epsxgrad[0,:-2].reshape((1,-1))

                    else:
                        self.Xgrad = np.delete(self.Xgrad,-1).reshape((-1,1))
                        self.ygrad = np.delete(self.ygrad,-1).reshape((-1,1))
                        self.epsxgrad = np.delete(self.epsxgrad,-1).reshape((1,-1))

                    self.ytilde = np.concatenate((self.yt,self.ygrad ))

            else:

                if self.dim > 1:
                    self.Xgrad = np.delete(self.Xgrad,idx,axis=0)
                    self.ygrad = np.delete(self.ygrad,[idx*self.dim,idx*self.dim+self.dim-1]).reshape((-1,1))
                    self.epsxgrad = np.delete(self.epsxgrad,[idx*self.dim,idx*self.dim+self.dim-1]).reshape((1,-1))
                    self.ytilde = np.concatenate((self.yt,self.ygrad))
                else:
                    'When data points are deleted for dim>1 we need to delete index->index*dim'


       

    def calculateLOOCV(self, k, verbose=False):
        """
        Parameters
        ----------
        verbose : bool, optional
            Bool for verbosity. The default is False.

        Returns
        -------
        mse : float
            LOOCV error for the given (enhanced) training data.

        We assume that when gradient data is present there are at the same postion as
        the training data itself, such that Xt = Xgrad

        """
        #Check if folding is possible
        if self.X.shape[0]%k != 0:
            print("Data couldn't be devided in equal parts. Choose other set size")
            return
        if k == 1:
            print("Data cant't be devided in one set. Choose other set size")
            return

        n = self.X.shape[0]
        dim = self.dim
        if self.Xgrad is not None:
            ngrad = self.Xgrad.shape[0]
        else:
            ngrad = 0

        setrange=int(self.X.shape[0]/k)
        error = np.zeros((k))

        start = timer()
        if verbose:
            print("Data is devided into {} sets".format(int(k)))

        if self.Xgrad is None:
            for i in range(k):

                """ Walking bounds """
                lower = i*setrange
                upper = (i+1)*setrange

                """ Validation set """
                vset = self.X[lower:upper,:].reshape((-1,dim))
                vval = self.yt[lower:upper].reshape((-1,1))

                """ Training set """
                tset = np.delete(self.X, np.s_[lower:upper], axis=0)
                tval = np.delete(self.yt, np.s_[lower:upper]).reshape((-1,1))
                eps = np.delete(self.epsxt[0,:],  np.s_[lower:upper])

                """ Calcualte mean """
                mat  = kernelmatrices(vset,tset,self.hyperparameter,eps)
                mean = mat[1]@(np.linalg.solve(mat[2],tval))


                """ Calculate error measure """
                error[i] = np.sqrt((1 / setrange)*np.sum( (vval-mean)**2))

        else:
            for i in range(k):

                """ Walking bounds """
                lower = i*setrange
                upper = (i+1)*setrange

                """ Validation set """
                vset = self.X[lower:upper,:].reshape((-1,dim))
                vval = self.yt[lower:upper].reshape((-1,1))

                """ Training set """
                tset = np.delete(self.X, np.s_[lower:upper], axis=0)
                tsetgrad = tset
                tval = np.delete(self.yt, np.s_[lower:upper]).reshape((-1,1))

                tvalgrad = np.delete(self.ygrad, np.s_[lower*dim:upper*dim]).reshape((-1,1))

                ttilde = np.concatenate((tval,tvalgrad))

                eps = np.delete(self.epsxt[0,:],  np.s_[lower:upper])
                epsgrad = np.delete(self.epsxgrad[0,:],  np.s_[lower*dim:upper*dim])


                mat  = kernelmatricesgrad(vset,tset,tsetgrad,self.hyperparameter,eps,epsgrad)
                mean = np.concatenate((mat[1], mat[3]), axis=1)@(np.linalg.solve(mat[5],ttilde))

                """ Calculate error measure """
                error[i] = np.sqrt((1 / setrange)*np.sum( (vval-mean)**2))

        """ Calculate mean error """
        loocv = np.sqrt((1 / k)*np.sum(error**2))
        end = timer()

        if verbose:
            print("Number of data points: {}".format(self.X.shape[0]))
            if self.Xgrad is None:
                print("Number of gradient data points: {}".format(0))
            else:
                print("Number of gradient data points: {}".format(self.Xgrad.shape[0]))
            print("Elapsed time: "+str((end - start))+" s")
            print("Mean cross validation error: {:g}".format(loocv))

        return loocv

    def calculateMSE(self, fun, numberofsamplepoints, ranges, gridtype, verbose=False):

        if fun is None:
            return None

        start = timer()

        xs = createPD(numberofsamplepoints, self.dim, gridtype, ranges)

        mean = self.predictmean(xs)
        error = (fun(xs,self.dim)-mean)**2
        mse = np.sqrt(1 / xs.shape[0]*np.sum(error))

        end = timer()

        if verbose is True:
            print("--- MSE ---")
            print("Number of data points: {}".format(self.X.shape[0]))
            if self.Xgrad is None:
                print("Number of gradient data points: {}".format(0))
            else:
                print("Number of gradient data points: {}".format(self.Xgrad.shape[0]))
            print("Mean squared error: {:g}".format(mse))
            print("Elapsed time: "+str((end - start))+" s")
            print("\n")

        return mse

    def plot(self, fun, filepath, filename, region):

        if self.dim >= 3:
            print("Plotting is only possible for 1 or 2 dimensional problems")
            return None

        if self.dim == 1:
            numofpts = 200
            Xplot1d = createPD(numofpts, 1, "grid", region)
            meanplot = self.predictmean(Xplot1d)
            variancevector = self.predictvariance(Xplot1d)

            fig, axs = plt.subplots(1)
            if fun is not None:
                axs.plot(Xplot1d, fun(Xplot1d,self.dim), 'r:', label=r'$f(x) = x\,\sin(x)$')

            axs.scatter(self.X, self.yt,  label='Observations')
            axs.plot(Xplot1d, meanplot, 'b-', label='Prediction')
            axs.fill(np.concatenate([Xplot1d, Xplot1d[::-1]]),
                     np.concatenate([meanplot.reshape((-1, 1)) - 2*np.array([variancevector]).reshape(numofpts, 1),
                                    (meanplot.reshape((-1, 1)) + 2*np.array([variancevector]).reshape(numofpts, 1))[::-1]]),
                     alpha=.25, fc='b', ec='None', label=r'$2\sigma$')

            axs.errorbar(self.X, np.squeeze(self.yt) ,yerr =  np.squeeze(np.sqrt(self.epsxt), axis=0), fmt='.k')
            #axs.grid(True)
            axs.set_xlabel("x")
            axs.set_ylabel("f(x)")
            axs.legend()

        elif self.dim ==2 :
            fig, axs = plt.subplots(1, 1)
            axs.scatter(self.getX[:,0], self.getX[:,1],zorder=2)  # Initial points

            idx = np.where((np.sqrt(self.epsxt)[0, 0:self.initialamount] != 0.001))
            alteredinitialpoints = self.getX[idx[0], :]

            if self.getX.shape[0]>self.initialamount:
                axs.scatter(self.getX[self.initialamount:,0], self.getX[self.initialamount:,1],
                        color='orange', marker='x', zorder=2)  # Added points

            axs.scatter(alteredinitialpoints[:, 0], alteredinitialpoints[:, 1],
                        color='orange', marker='x', zorder=2)  # Altered initial points

            ' Plot small frame around Omega '
            framex = [region[0][0], region[0][0],
                      region[0][1], region[0][1], region[0][0]]
            framey = [region[1][0], region[1][1],
                      region[1][1], region[1][0], region[1][0]]
            axs.plot(framex, framey, 'k--', alpha=0.5, zorder=1)

            axs.grid(True)
            axs.set_ylabel("p2")
            axs.set_xlabel("p1")

        if filepath is None:
            fp = os.path.join(pathlib.Path(__file__).parent.resolve(), filename+'.pdf')
        elif path.isdir(filepath):
            fp = os.path.join(filepath, filename+'.pdf')
        fig.savefig(fp,format="pdf")

    def savedata(self, filepath, filename = None):

        if not path.isdir(filepath):
            filepath = pathlib.Path(__file__).parent.resolve()
        if filename is None:
            np.save(filepath+"/X.npy", self.X)
            np.save(filepath+'/yt.npy', self.yt)
            np.save(filepath+'/Xgrad.npy', self.Xgrad)
            np.save(filepath+'/ygrad.npy', self.ygrad)
            np.save(filepath+'/HPm.npy', self.hyperparameter)
            np.save(filepath+'/epsXt.npy', self.epsxt)
            np.save(filepath+'/epsXgrad.npy', self.epsxgrad)
        else:
            np.save(filepath+'X_'+filename+".npy", self.X)
            np.save(filepath+'yt_'+filename+".npy", self.yt)
            np.save(filepath+'Xgrad_'+filename+".npy", self.Xgrad)
            np.save(filepath+'ygrad_'+filename+".npy", self.ygrad)
            np.save(filepath+'HPm_'+filename+".npy", self.hyperparameter)
            np.save(filepath+'epsXt_'+filename+".npy", self.epsxt)
            np.save(filepath+'epsXgrad_'+filename+".npy", self.epsxgrad)