import numpy as np
import matplotlib.pyplot as plt
from basicfunctions.utils.creategrid import *

def rastrigin(x,a):
    n = x.shape[0]
    d = x.shape[1]
    return a * d+ np.sum(x**2-10*np.cos(2*np.pi*x),axis=1)

def sphere(x):
    return np.sum(x**2,axis=1)

def rosenbrock(x):
    pass

def ackley(x):
    """ The function is usually evaluated on the hypercube xi ∈ [-32.768, 32.768], for all i = 1, …, d,
    although it may also be restricted to a smaller domain."""
    return -20*np.exp(-0.2*np.sqrt(0.5*(np.sum(x**2,axis=1))))-np.exp(.5*np.sum(np.cos(2*np.pi*x),axis=1))+np.e+20

def beale(x):
    if x.shape[1] > 2:
        print("Function is a R^2->R function")
        return -1
    else:
        return (1.5-x[:,0]+x[:,0]*x[:,1])**2+(2.25-x[:,0]+x[:,0]*x[:,1]**2)**2+(2.625-x[:,0]+x[:,0]*x[:,1]**3)**2

def goldensteinprice(x):
    if x.shape[1] > 2:
        print("Function is a R^2->R function")
        return -1
    else:
        x,y = x[:,0],x[:,1]
        return (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*(x**2)+48*y-36*x*y+27*(y**2)))

def booth(x):
    if x.shape[1] > 2:
        print("Function is a R^2->R function")
        return -1
    else:
        x,y = x[:,0],x[:,1]
        return (x+2*y-7)**2+(2*x+y-5)**2

def bukinno6(x):
    """ The function is usually evaluated on the rectangle x1 ∈ [-15, -5], x2 ∈ [-3, 3]."""
    if x.shape[1] > 2:
        print("Function is a R^2->R function")
        return -1
    else:
        x,y = x[:,0],x[:,1]
        return 100*np.sqrt(np.abs(y-0.01*x**2))+0.01*np.abs(x+10)

def levino13(x):
    if x.shape[1] > 2:
        print("Function is a R^2->R function")
        return -1
    else:
        x,y = x[:,0],x[:,1]
        return np.sin(3*np.pi*x)**2+(x-1)**2*(1+np.sin(3*np.pi*x)**2)+(y-1)**2*(1+np.sin(3*np.pi*y)**2)

def himmelblau(x):
    if x.shape[1] > 2:
        print("Function is a R^2->R function")
        return -1
    else:
        x,y = x[:,0],x[:,1]
        return (x**2+y-11)**2+(x+y**2-7)**2

def threehumpcamel(x):
    if x.shape[1] > 2:
        print("Function is a R^2->R function")
        return -1
    else:
        x,y = x[:,0],x[:,1]
        return 2*x**2-1.05*x**4+x**6/6+x*y+y**2

def earsom(x):
    if x.shape[1] > 2:
        print("Function is a R^2->R function")
        return -1
    else:
        x,y = x[:,0],x[:,1]
        return -np.cos(x)*np.cos(y)*np.exp(-(((x-np.pi)**2+(y-np.pi)**2)))

def eggholder(x):
    """ The function is usually evaluated on the square xi ∈ [-512, 512], for all i = 1, 2."""
    if x.shape[1] > 2:
        print("Function is a R^2->R function")
        return -1
    else:
        x,y = x[:,0],x[:,1]
        return -(y+47)*np.sin(np.sqrt(np.abs(x/2+(y+47))))-x*np.sin(np.sqrt(np.abs(x-(y+47))))

def hoeldertable(x):
    """ The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2. """
    if x.shape[1] > 2:
        print("Function is a R^2->R function")
        return -1
    else:
        x,y = x[:,0],x[:,1]
        return -np.abs(np.sin(x)*np.cos(y)*np.exp(np.abs(1-np.sqrt(x**2+y**2)/np.pi)))

def styblinksitang(x):
    return np.sum(x**4-16*x**2+5*x,axis=1)/2

def dropwave(x):
    """ The function is usually evaluated on the square xi ∈ [-5.12, 5.12], for all i = 1, 2."""
    if x.shape[1] > 2:
        print("Function is a R^2->R function")
        return -1
    else:
        x,y = x[:,0],x[:,1]
        return -(1+np.cos(12*np.sqrt(x**2+y**2)))/(0.5*(x**2+y**2)+2)

def schwefel(x):
    """ The function is usually evaluated on the hypercube xi ∈ [-500, 500], for all i = 1, …, d."""
    d = x.shape[1]
    return 418.9829*d-np.sum(x*np.sin(np.sqrt(np.abs(x))),axis=1)


testranges = np.array([[-2,2], [-2,2]])
testdata = createPD(40, 2, "grid", testranges)
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
yvalues = sphere(testdata)
surf = ax.plot_trisurf(
    testdata[:, 0], testdata[:, 1], yvalues, linewidth=0, antialiased=True, cmap=plt.cm.Spectral)

ax.set_xlabel(r"$x_0$")
ax.set_ylabel(r"$x_1$")
ax.set_zlabel(r"$x_2$")