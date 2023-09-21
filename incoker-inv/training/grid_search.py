import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, DotProduct, WhiteKernel
from sklearn.model_selection import GridSearchCV
import pathlib
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from skopt.learning.gaussian_process.kernels import RationalQuadratic

# Features and properties
considered_features = [
    'volume_fraction_4', 'volume_fraction_10', 'volume_fraction_1',
    'chord_length_mean_4', 'chord_length_mean_10', 'chord_length_mean_1',
    'chord_length_variance_4', 'chord_length_variance_10', 'chord_length_variance_1'
]
considered_properties = ['young_modulus']


# Load data
training_data = pathlib.Path("../data/training_data_rve_database.npy")
data = np.load(training_data)
data['thermal_expansion'] *= 1e6

X = np.vstack(tuple(data[f] for f in considered_features)).T
Y = np.vstack(tuple(data[p] for p in considered_properties)).T
clean_indices = np.argwhere(~np.isnan(Y))
y_clean = Y[clean_indices.flatten()]
X_clean = X[clean_indices.flatten()]
scaler = StandardScaler()
Xt = scaler.fit_transform(X_clean)
# Define kernels
kernels = [
    Matern(),
    RBF(),
    DotProduct(),
    RationalQuadratic(),
    Matern() + WhiteKernel(),
    RBF() + WhiteKernel(),
    DotProduct() + WhiteKernel(),
    RationalQuadratic() + WhiteKernel(),
    Matern() * RBF() + WhiteKernel(),
]

# Define parameter grid
param_grid = {
    "kernel": kernels,
    "alpha": [1e-10, 1e-8, 1e-5, 1e-3, 1e-1, 1e0, 1e1]
}

# Run grid search
gpr = GaussianProcessRegressor(normalize_y=True, random_state=0)
grid_search = GridSearchCV(gpr, param_grid, scoring='r2', cv=5)
grid_search.fit(X_clean, y_clean)

# Print best parameters
print("Best parameters found: ", grid_search.best_params_)
print("Custom accuracy with best parameters: ", grid_search.best_score_)

