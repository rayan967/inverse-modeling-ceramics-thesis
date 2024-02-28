from pyDOE import lhs
from scipy.stats import qmc

import sys
import numpy as np
import warnings


def is_within_exclusion_zone(point, exclusion_zones, weights):
    for center, radius in exclusion_zones:
        if weighted_distance(point, center, weights) < radius:
            return True
    return False

def weighted_distance(point_a, point_b, weights):
    """
    Calculate the weighted Euclidean distance between two points.

    :param point_a: First point (array-like).
    :param point_b: Second point (array-like).
    :param weights: Weights for each dimension (array-like).
    :return: Weighted distance.
    """
    diff = np.array(point_a) - np.array(point_b)
    weighted_diff = diff * weights
    return np.sqrt(np.sum(weighted_diff ** 2))

def createPD(nbofpoints, dim, gtype, ranges, exclusion_zones=None):
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
    weights = calculate_weights(ranges)

    if gtype == "latin":
        grid = ranges[:,0] +  (ranges[:,1]-ranges[:,0])*lhs(dim, samples = nbofpoints )
        if exclusion_zones:
            filtered_grid = []
            for point in grid:
                if not is_within_exclusion_zone(point, exclusion_zones, weights):
                    filtered_grid.append(point)

            grid = np.array(filtered_grid)

        return grid


    elif gtype == "random":
        grid = ranges[:,0] +  (ranges[:,1]-ranges[:,0])*np.random.rand(nbofpoints,dim )
        if exclusion_zones:
            filtered_grid = []
            for point in grid:
                if not is_within_exclusion_zone(point, exclusion_zones, weights):
                    filtered_grid.append(point)

            grid = np.array(filtered_grid)

        return grid


    elif gtype == "grid":
        # Calculate the number of points along each dimension
        points_per_dim = int(np.ceil(nbofpoints ** (1 / dim)))

        # Generate grid
        grid = np.meshgrid(*[np.linspace(ranges[i, 0], ranges[i, 1], points_per_dim) for i in range(dim)])
        grid = np.vstack(map(np.ravel, grid)).T


        if exclusion_zones:
            filtered_grid = []
            for point in grid:
                if not is_within_exclusion_zone(point, exclusion_zones, weights):
                    filtered_grid.append(point)

            grid = np.array(filtered_grid)

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
        if exclusion_zones:
            filtered_grid = []
            for point in grid:
                if not is_within_exclusion_zone(point, exclusion_zones, weights):
                    filtered_grid.append(point)

            grid = np.array(filtered_grid)

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
        if exclusion_zones:
            filtered_grid = []
            for point in grid:
                if not is_within_exclusion_zone(point, exclusion_zones, weights):
                    filtered_grid.append(point)

            grid = np.array(filtered_grid)

        return grid
    else:
        return 1

def calculate_weights(parameterranges):
    """
    Calculate weights for each parameter inversely proportional to their range.

    :param parameterranges: Array of parameter ranges.
    :return: Array of weights.
    """
    ranges = parameterranges[:, 1] - parameterranges[:, 0]
    weights = 1 / ranges
    return weights
"""
nbOfPoints = 1000
ranges = np.array([[15,35],[40,60],[84,90],[2.5,6.5],[3.5,18],[0,10]])
gridneu = createPD(4, 6, "random", ranges)
"""