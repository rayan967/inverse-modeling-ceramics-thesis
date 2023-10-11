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
considered_properties = [
    'poisson_ratio',
]



def extract_XY_2(data):
    """Use for 2 features."""

    filtered_indices = np.where(data['volume_fraction_1'] == 0.0)

    chord_length_ratio = data['chord_length_mean_4'][filtered_indices] / data['chord_length_mean_10'][filtered_indices]

    volume_fraction_4 = data['volume_fraction_4'][filtered_indices]

    X = np.vstack((volume_fraction_4, chord_length_ratio)).T

    Y = np.vstack(tuple(data[p][filtered_indices] for p in considered_properties)).T

    global considered_features

    considered_features = [
    'volume_fraction_4',
    'chord_length_ratio'
]

    return X, Y

# Load data
training_data = pathlib.Path("data/training_data_rve_database.npy")
data = np.load(training_data)
data['thermal_expansion'] *= 1e6


Y, X = extract_XY_2(data)

# pick a single property
x = X[:, 0]
x_float = np.array([float(val) if val != b'nan' else np.nan for val in x])

# ignore NaNs in the data
clean_indices = np.argwhere(~np.isnan(x_float))
x_clean = x_float[clean_indices.flatten()].reshape(-1, 1)
Y_clean = Y[clean_indices.flatten()]




scaler = StandardScaler()
Xt = scaler.fit_transform(x_clean)
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
grid_search.fit(x_clean, Y_clean)

# Print best parameters
print("Best parameters found: ", grid_search.best_params_)
print("Custom accuracy with best parameters: ", grid_search.best_score_)



