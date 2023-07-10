import numpy as np

def sumofsines():
    
    
    def fun(x):
        return np.sin(x[:,0])+np.cos(x[:,0]*x[:,1])  
    
    def grad(x):
        return np.array([np.cos(x[:,0])-x[:,1]*np.sin(x[:,0]*x[:,1]),-x[:,0]*np.sin(x[:,0]*x[:,1])]).T
   
    return {"function" : fun,"gradient" : grad}

def sumofsineswithparameter(): 
    
    def fun(x,a):
        return np.sin(x[:,0])+a*np.cos(x[:,0]*x[:,1])  
    
    def grad(x,a):
        return np.array([np.cos(x[:,0])-a*x[:,1]*np.sin(x[:,0]*x[:,1]),-a*x[:,0]*np.sin(x[:,0]*x[:,1])]).T

    return  {"function" : fun,"gradient" : grad}


def himmelblauwithparameter(): 
    
    def fun(x,a):
        return (x[:,0]**2+x[:,1]-11)**2+a*(x[:,0]+x[:,1]**2-7)**2
    
    def grad(x,a):
        return  np.array([2*(x[:,0]**2+x[:,1]-11)*x[:,0]*2+a*2*(x[:,0]+x[:,1]**2-7),
                2*(x[:,0]**2+x[:,1]-11)+2*a*(x[:,0]+x[:,1]**2-7)*x[:,1]*2]).T

    return  {"function" : fun,"gradient" : grad}

def spherewithparameter(): 
    
    def fun(x,a):
        return a*(x[:,0]**2+x[:,1]**2)
    
    def grad(x,a):
        return np.array([2*a*x[:,0],2*a*x[:,1]]).T

    return  {"function" : fun,"gradient" : grad}

def grad(x):
    return np.array([np.cos(x[:,0])-np.sin(x[:,0]*x[:,1])*x[:,1],-np.sin(x[:,0]*x[:,1])*x[:,0]]).T

def rosenbrock(x):
    return (1-x[:,0])**2+(x[:,1]-x[:,0]**2)**2

def parabola(x):
    return (x[:,0])**2+(x[:,1])**2


# =============================================================================
# def exponentialwithparameter(): 
#     
#     def fun(x,a):
#         return a*np.exp(5*x[:,0]*x[:,1])
#     
#     def grad(x,a):
#         return  np.array([ 5*a*x[:,1]*np.exp(5*x[:,0]*x[:,1]) , 5*a*x[:,0]*np.exp(5*x[:,0]*x[:,1]) ]).T
#     return  {"function" : fun,"gradient" : grad}
# =============================================================================


def exponentialwithparameter(): 
    
    def fun(x,a):
        return x[:,0]**a+x[:,1]**a
    
    def grad(x,a):
        return  np.array([ (x[:,0]**(a-1))*a , (x[:,1]**(a-1))*a ]).T
    return  {"function" : fun,"gradient" : grad}


def exponentialwithrotationparameter(): 
    
    def fun(x,a):
        return  ((np.cos(a)*x[:,0]+np.sin(a)*x[:,1]) +(-np.sin(a)*x[:,0]+np.cos(a)*x[:,1]))**2
    
    def grad(x,a):
        dx = 2* (np.cos(a) - np.sin(a))* (x[:,0]*np.cos(a) + x[:,1]*np.cos(a) - x[:,0]*np.sin(a) + x[:,1]*np.sin(a))
        dy = 2* (np.cos(a) + np.sin(a))* (x[:,0]*np.cos(a) + x[:,1]*np.cos(a) - x[:,0]*np.sin(a) + x[:,1]*np.sin(a))
        return  np.array([ dx , dy ]).T
    
    return  {"function" : fun,"gradient" : grad}