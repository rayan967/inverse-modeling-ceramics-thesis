import numpy as np


def totalcompwork(v, s=1):
    return np.sum(v**(s))

def totalcompworkeps(epsilon):
    return np.sum(1/2*epsilon**(-2))

'Nonlinear constraints'
def compworkconstrain(v, s):
    return np.array([np.sum(v**s)])
def compworkconstrainjac(v, s):
    return np.array([s*v**(s-1)])
def compworkconstrainhess(x, v):
    #s = np.round(4/3,2)
    s = 2
    if s == 1:
        return np.zeros((x.shape[0],x.shape[0]))
    return s*(s-1)*v[0]*np.diagflat(x**(s-2))