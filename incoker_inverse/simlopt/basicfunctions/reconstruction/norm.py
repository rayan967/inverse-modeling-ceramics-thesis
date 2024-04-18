import numpy as np

def sigmanorm(x,y,Sigma):
    """

    Parameters
    ----------
    x,y : np.array 1 x m
        DESCRIPTION.

    Sigma : np.array m x m
        DESCRIPTION.

    Returns
    -------
    float: (x-y).T @ Sigma @ (x-y) = ||(x-y)||^2_Sigma

    """
    
    if Sigma.shape[0] >1:

        """ Check if x is a row vector, then change it to a coloumn vector """
        if x.shape[1] > x.shape[0]:
            x = x.T
        """ Check if y is a row vector, then change it to a coloumn vector """
        if y.shape[1] > y.shape[0]:
            y = y.T
    
        return (x-y).T @ Sigma @ (x-y)

    else:
        """ Check if x is a row vector, then change it to a coloumn vector """
        if x.shape[1] > x.shape[0]:
            x = x.T
        """ Check if y is a row vector, then change it to a coloumn vector """
        if y.shape[1] > y.shape[0]:
            y = y.T
    
        return np.dot((x-y).T ,(Sigma @ (x-y).T).T)