"""Plots for prediocted subspace for 2d model"""

import joblib
import numpy as np
from matplotlib import pyplot as plt


considered_properties = [
    'thermal_conductivity',
    'thermal_expansion',
    'young_modulus',
   'poisson_ratio',
]

for property_name in considered_properties:

    # Change next line to switch between different models in models folder
    models = joblib.load("models/2d_model.joblib")["models"]

    X = models[property_name]['X_train']
    X_test = models[property_name]['X_test']
    Y = models[property_name]['y_train']
    y_test = models[property_name]['y_test']


    prop = 21
    # Dense grid
    v2_values = np.linspace(min(X[:, 0]), max(X[:, 0]), num=100)
    rho_values = np.linspace(min(X[:, 1]), max(X[:, 1]), num=100)
    v2_grid, rho_grid = np.meshgrid(v2_values, rho_values)

    feature_grid = np.vstack([v2_grid.ravel(), rho_grid.ravel()]).T

    predictions, uncertainty = (models[property_name]['pipe'].predict(feature_grid, return_std=True))

    predictions_grid = predictions.reshape(v2_grid.shape)
    uncertainty_grid = uncertainty.reshape(v2_grid.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(v2_grid, rho_grid, uncertainty_grid)

    ax.set_xlabel('Volume Fraction (zirconia)')
    ax.set_ylabel('Particle Size Ratio')
    ax.set_zlabel(str(property_name))

    ax.set_title('Uncertainty in '+str(property_name) +' predictions')

    plt.show()