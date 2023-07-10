from pyDOE import lhs
from scipy.stats import qmc

import sys
import numpy as np
import warnings


def createPD(nbofpoints, dim, gtype, ranges):

    """
      Parameters
      ----------
      nbofpoints : TYPE
          DESCRIPTION.
      dim : TYPE
          DESCRIPTION.
      type : TYPE
          DESCRIPTION.
      ranges : np.array
             | lower bound p1 , upper bound p1 |
             | lower bound p2 , upper bound p2 |
             | ....                            |
             | lower bound pd , upper bound pd |
      Returns
      -------
      None.
    """
    #assert dim == ranges.shape[0]

    if gtype == "latin":
        grid = ranges[:,0] +  (ranges[:,1]-ranges[:,0])*lhs(dim, samples = nbofpoints )
        return grid

    elif gtype == "random":
        grid = ranges[:,0] +  (ranges[:,1]-ranges[:,0])*np.random.rand(nbofpoints,dim )
        return grid

    elif gtype == "grid":
        allG = []
        for i in range(0,ranges.shape[0]):
            allG.append([(ranges[i,0] + (ranges[i,1]-ranges[i,0])*np.linspace(0,1,nbofpoints))])# Create linspaces
        out = np.meshgrid(*allG)
        outtmp = []
        for j in range(0,ranges.shape[0]):
            outtmp.append( out[j].reshape(-1))
        grid = np.vstack([*outtmp]).T
        return grid
    
    elif gtype == "sobol":
        #print("Warning: if you call this method more than once you should come up with a way of reusing the sobol_sampler, or else you will use different designs on each call.")
        sobol_sampler = qmc.Sobol(d=dim)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid = qmc.scale(
                sobol_sampler.random(nbofpoints),
                ranges[:,0],
                ranges[:,1],
            )
        return grid
    
    elif gtype == "halton":
        halot_sampler = qmc.Halton(d=dim)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid = qmc.scale(
                halot_sampler.random(nbofpoints),
                ranges[:,0],
                ranges[:,1],
            )
        return grid
    
    else:
        return 1
"""
nbOfPoints = 1000
ranges = np.array([[15,35],[40,60],[84,90],[2.5,6.5],[3.5,18],[0,10]])
gridneu = createPD(4, 6, "random", ranges)
"""