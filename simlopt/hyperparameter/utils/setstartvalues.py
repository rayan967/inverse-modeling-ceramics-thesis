import numpy as np


def startvalues(sigma, L):

    H0 = [sigma]
    H0 += L
    return H0


# =============================================================================
# def findsigma(yt):
# 
#     sort = np.sort(yt)
#     diff = np.zeros((len(sort)-1))
#     for i in range(len(sort)-1):
#         #diff[i] = sort[0,i+1]-sort[0,i]
#         diff[i] = sort[i+1]-sort[i]
# 
#     if np.sum(diff == 0) == 0:
#         return 1/len(sort) * np.sum(diff, axis=0)
#     else:
#         return 1/(len(sort) - np.sum(diff == 0)) * np.sum(diff, axis=0)
# =============================================================================

# =============================================================================
# def findL(Xt):
#
#     dim = Xt.shape[1]
#     L = []
#
#     if dim == 1:
#         for i in range(dim):
#             sort = np.sort(Xt[:])
#             diff = np.zeros((len(sort)-1))
#
#             for i in range(len(sort)-1):
#                 diff[i] = sort[i+1]-sort[i]
#
#             if np.sum(diff == 0) == 0:
#                 Lstart = 1/len(sort) * np.sum(diff,axis = 0)
#                 L.append(Lstart)
#             else:
#                 Lstart = 1/(len(sort) -np.sum(diff == 0) )* np.sum(diff,axis = 0)
#                 L.append(Lstart)
#         return L
#
#     for i in range(dim):
#         sort = np.sort(Xt[dim,:])
#         diff = np.zeros((len(sort)-1))
#
#         for i in range(len(sort)-1):
#             diff[i] = sort[i+1]-sort[i]
#
#         if np.sum(diff == 0) == 0:
#             Lstart = 1/len(sort) * np.sum(diff,axis = 0)
#             L.append(Lstart)
#         else:
#             Lstart = 1/(len(sort) -np.sum(diff == 0) )* np.sum(diff,axis = 0)
#             L.append(Lstart)
#     return L
# =============================================================================

def findsigma(yt):
    """
    
    Find start values using the idea of:
        Nalika Ulapane: Hyper-Parameter Initialization for Squared Exponential Kernel- based Gaussian Process Regression

    Parameters
    ----------
    yt : np.array([n])
        Training data values

    Returns
    -------
    sigma : float
        Initial value for data variance

    """
    sort = np.sort(yt[:])
    diff = 0

    for i in range(len(sort)-1):
        counter = 0
        for i in range(len(sort)-1):
            diff += abs(sort[i+1]-sort[i])
            if sort[i+1] != sort[i]:
                counter+=1
        sigma = diff/counter
        if sigma == 0 :
            sigma = 1
            
    return sigma

def findL(Xt):
    """
    
    Find start values using the idea of:
        Nalika Ulapane: Hyper-Parameter Initialization for Squared Exponential Kernel- based Gaussian Process Regression

    Parameters
    ----------
    Xt : np.array([n,d])
        Training data

    Returns
    -------
    L : list of size dim
        Initial values for lenght scales

    """
    L = []
    dim = Xt.shape[1]

    for i in range(dim):
        sort = Xt[:, i]
        diff = 0
        counter = 0
        for i in range(len(sort)-1):
            diff += abs(sort[i+1]-sort[i])
            if sort[i+1] != sort[i]:
                counter += 1
        sol = diff/counter
        if sol == 0:
            sol = 1
        L.append(sol)
    return L
