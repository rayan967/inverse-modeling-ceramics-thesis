""" Script for testing two  different implementations of the covariance matrix """
import numpy as np

n = 5

X1 = np.linspace(0,5,n).reshape(n,-1)
X2 = np.linspace(0,5,n).reshape(n,-1)

""" Implementation 1 """

N1 = X1.shape[0]
D1 = X1.shape[1]

N2 = X2.shape[0]
D2 = X2.shape[1]

assert D1 == D2, "Dimensions must be equal"

# Preallocation of covariance matrices
KXX = np.zeros((N1, N1))
KXY = np.zeros((N1, N2))
KYY = np.zeros((N2, N2))

sigma = 1
l_mat = 2

# Prescale data
X1 = X1/l_mat
X2 = X2/l_mat

n1sq = np.sum(X1**2,axis=1);
n2sq = np.sum(X2**2,axis=1);

DXX = np.transpose(np.outer(np.ones(N1),n1sq)) + np.outer(np.ones(N1),n1sq)-2* (np.dot(X1,np.transpose(X1)))
KXX = sigma**2 * np.exp(-DXX / 2.0)

DXY = np.transpose(np.outer(np.ones(N2),n1sq)) + np.outer(np.ones(N1),n2sq)-2* (np.dot(X1,np.transpose(X2)))
KXY = sigma**2 * np.exp(-DXY / 2.0)

DYY = np.transpose(np.outer(np.ones(N2),n2sq)) + np.outer(np.ones(N2),n2sq)-2* (np.dot(X2,np.transpose(X2)))
KYY = sigma**2 * np.exp(-DYY / 2.0)


""" Implementation 2 """
X1 = np.linspace(0,5,n).reshape(n,-1)
X2 = np.linspace(0,5,n).reshape(n,-1)

def rbf(x,y,sigma,l):
    return sigma**2 * np.exp(-0.5*(((x-y)/l)**2))


KYYstar = np.zeros((N2,N2))
for ii  in range(0 , N2):
    for jj  in range(0 , N2):
        KYYstar[ii,jj] = rbf(X1[ii],X2[jj],sigma,l_mat)
