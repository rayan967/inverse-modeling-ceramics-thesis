import numpy as np

def vectortomatrix(vector,dim):

    matrix = np.zeros((int(vector.shape[0]/dim),dim))
    for i in range(dim):
        matrix[:,i] =  np.squeeze(vector[i::dim])

    return matrix