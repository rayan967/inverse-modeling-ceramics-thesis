import numpy as np
from optimization.functionprototypes import *


def standardpointdistributions(case,graddata=False):
    
    if case == 0:
        Xt = np.array([[0.0,0.0],[1.0,0.0],[2.0,0.0],[3.0,0.0],
                       [0.0,1.0],[3.0,1.0],[0.0,2.0],[3.0,2.0],
                       [0.0,3.0],[1.0,3.0],[2.0,3.0],[3.0,3.0]])
    elif case == 1:
        Xt = np.array([[0.0,0.0],[1.0,0.0],
                       [0.0,1.0],[1.0,1.0]])
    elif case == 2:
        Xt = np.array([[0.0,0.0],[0.5,0.0],[1.0,0.0],
                       [0.0,0.5],[1.0,0.5],
                       [0.0,1.0],[0.5,1.0],[1.0,1.0]])
    elif case == 3:
        Xt = np.array([[0.0,0.0],[3.0,0.0],
                        [0.0,3.0],[3.0,3.0]])
        
    elif case == 4:
            Xt = np.array([[0.0,0.0],[2.0,0.0],
                           [0.0,2.0],[2.0,2.0],
                           [1.0,1.0]])
    Xgrad = None
    if graddata : 
        Xgrad = Xt
    
    return Xt,Xgrad
        

def createdata(case,Xt,graddata=False):
    
    graddata = None
    ygrad = None
    
    if case == "exponential":
        fun = exponentialwithrotationparameter()
        yt1 = fun["function"](Xt,0).reshape((-1,1))
        yt2 = fun["function"](Xt,2).reshape((-1,1))
        yt3 = fun["function"](Xt,4).reshape((-1,1))
        #yt4 = fun["function"](Xt,1.5).reshape((-1,1))
        #yt5 = fun["function"](Xt,2).reshape((-1,1))
        #yt6 = fun["function"](Xt,2.5).reshape((-1,1))
        #yt7 = fun["function"](Xt,3).reshape((-1,1))
        yt = np.concatenate((yt1,yt2,yt3),axis=1)
                      
        if graddata:
            yg1 = fun["gradient"](Xt,0).reshape((-1,1))
            yg2 = fun["gradient"](Xt,2).reshape((-1,1))
            yg3 = fun["gradient"](Xt,4).reshape((-1,1))
            ygrad = np.concatenate((yg1,yg2,yg3),axis=1)
        
    elif case == "himmelblau":
        fun = himmelblauwithparameter()
        yt1 = fun["function"](Xt,1).reshape((-1,1))
        yt2 = fun["function"](Xt,2).reshape((-1,1))
        yt3 = fun["function"](Xt,3).reshape((-1,1))      
        yt = np.concatenate((yt1,yt2,yt3),axis=1)
        
        if graddata:
            yg1 = fun["gradient"](Xt,0).reshape((-1,1))
            yg2 = fun["gradient"](Xt,2).reshape((-1,1))
            yg3 = fun["gradient"](Xt,4).reshape((-1,1))
            ygrad = np.concatenate((yg1,yg2,yg3),axis=1)
            
    elif case == "sphere":
        fun = spherewithparameter()
        yt1 = fun["function"](Xt,1).reshape((-1,1))
        yt2 = fun["function"](Xt,2).reshape((-1,1))
        yt3 = fun["function"](Xt,3).reshape((-1,1))      
        yt = np.concatenate((yt1,yt2,yt3),axis=1)
        
        if graddata:
            yg1 = fun["gradient"](Xt,0).reshape((-1,1))
            yg2 = fun["gradient"](Xt,2).reshape((-1,1))
            yg3 = fun["gradient"](Xt,4).reshape((-1,1))
            ygrad = np.concatenate((yg1,yg2,yg3),axis=1)

 
    return yt,ygrad,fun

def createerror(Xt,random=False,graddata=False):
    
    N   = Xt.shape[0]
    dim = Xt.shape[1]
    
    epsXgrad = None
    if random:
        epsXt = np.random.rand(N) * (0.1-0.025) + 0.025
        epsXt = epsXt.reshape((1,N))
    
        if graddata:
            epsXgrad = np.random.rand(N*dim) * (0.1-0.025) + 0.025
                
    else:
        vardata = 1E-4  #i.e eps = 1E-2
        vardata = 1E-10 #i.e eps = 1E-5
        vardata = 1E-6 #i.e eps = 1E-3
        vardata = 1E-8 #i.e eps = 1E-4
        vardata = 1E-12 #i.e eps = 1E-6
        vardata = 1E-1
        epsXt = vardata*np.ones((1,N)) #1E-1 for basic setup
    
        if graddata:
            vargraddata = 1E-1
            epsXgrad = vargraddata*np.ones((1,N*dim))
    
    return epsXt, epsXgrad