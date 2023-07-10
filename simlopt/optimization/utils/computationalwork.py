import numpy as np

' Inline work model '
def W(eps, d):
    return (1/d) * eps**(-d)

def epsofw(W, d):
    return (d*W+1E-6)**(-1/d)

def deps(W, d):
    return -(d*W+1E-6)**(-((d+1)/d))

def depsgrad(Wgrad, d):
    return -(d*Wgrad+1E-6)**(-((d+1)/d))

def epsofwgrad(W, d):
    return (d*W+1E-6)**(-1/d)

def Wgrad(epsgrad,d):
    return (1/d) * epsgrad**(-d)