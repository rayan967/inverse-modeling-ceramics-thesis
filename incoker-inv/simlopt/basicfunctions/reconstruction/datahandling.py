import numpy as np
from sklearn.preprocessing import StandardScaler


def handledata(data, freal, spatialfeatures, standardize):
    """


    Parameters
    ----------
    data : n x nboffeatures x m

    freal : 1xm
        experimental data

    spatialfeatures : int
        number of spatial features, dimension of parameter space

    Returns
    -------
    Xt : n x dim x m
        training data
    yt : n x m
        fuction values
    Xgrad : n x dim x m
        gradient training data
    ygrad : n*dim x m
        gradient values
    scaler : 3 x features x m
        data structure of scaling parameters
    m : int
        number of experiments
    features : int
        number of overall features

    """
    m = data.shape[2]
    features = data.shape[1]-1 #-1 because the last value is epsxt

    standardscaler = StandardScaler()
    scaler = np.zeros((3, features, m))

    """ scaler data structure
    feature mean 1->nf
    feature sqrt(var) 1->nf
    feature var 1->nf
    """
    if standardize == True:
        
        for i in range(m):
            data[:, :, i] = standardscaler.fit_transform(data[:, :, i])
            scaler[0, :, i] = standardscaler.mean_
            scaler[1, :, i] = standardscaler.scale_
            scaler[2, :, i] = standardscaler.var_
            freal[0, i] = (freal[0, i] - scaler[0, spatialfeatures, i]) / scaler[1, spatialfeatures, i]
    
        """ Extract data  """
        Xt = data[:, 0:spatialfeatures, :]
        yt = data[:, spatialfeatures, :]
    
        """ Extract gradient data """
        Xgrad = data[:, 0:spatialfeatures, :]
        yg = data[:, spatialfeatures + 1:, :]
    
        ygrad = np.zeros((yg.shape[0] * spatialfeatures, m))
        for i in range(0, m):
            ygrad[:, i] = np.insert(yg[:, :, i], 1, [])
    
    
        return Xt, yt, Xgrad, ygrad, scaler, m, features
    
    else:  
        
        for i in range(m):
            scaler[0, :, i] = np.zeros((1,features))
            scaler[1, :, i] = np.ones((1,features))
            scaler[2, :, i] = np.ones((1,features))
            freal[0, i] = (freal[0, i] - scaler[0, spatialfeatures, i]) / scaler[1, spatialfeatures, i]
        
        """ Extract data  """
        Xt = data[:, 0:spatialfeatures, :]
        yt = data[:, spatialfeatures, :]
    
        """ Extract gradient data """
        Xgrad = data[:, 0:spatialfeatures, :]
        yg = data[:, spatialfeatures + 1:features, :]
    
        ygrad = np.zeros((yg.shape[0] * spatialfeatures, m))
        for i in range(0, m):
            ygrad[:, i] = np.insert(yg[:, :, i], 1, [])
    
    
        return Xt, yt, Xgrad, ygrad, scaler, m, features