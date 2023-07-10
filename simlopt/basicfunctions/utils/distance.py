# -*- coding: utf-8 -*-

import numpy as np


def distance(x,X,eps):
    """


    Parameters
    ----------
    x : np.array 1 x d
        Point which is getting checked

    X :  np.array n x d
         All other points in the hyperspace

    eps : TYPE
         Allowed minimal distance between points

    Returns
    -------
    np.array 1xd
        Point which is closest to x
        or
        None, if no point was found

    """

    # Get dimensions
    dimx  = np.shape(x)[1]
    dimX  = np.shape(X)[1]

    assert dimx == dimX

    dist = np.sqrt(np.sum((x-X)**2, axis=1))

    I = np.argmin(dist)
    value = np.amin(dist)

    if value < eps:
        return X[I,:]
    else:
        return np.array([])

