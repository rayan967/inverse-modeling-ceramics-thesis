import numpy as np
from scikeras.wrappers import KerasRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, DotProduct, WhiteKernel
from sklearn.model_selection import GridSearchCV
import pathlib
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from skopt.learning.gaussian_process.kernels import RationalQuadratic
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import pathlib
from sklearn.preprocessing import StandardScaler

# Features and properties
considered_features = [
    'volume_fraction_4', 'volume_fraction_10', 'volume_fraction_1',
    'chord_length_mean_4', 'chord_length_mean_10', 'chord_length_mean_1',
    'chord_length_variance_4', 'chord_length_variance_10', 'chord_length_variance_1'
]
considered_properties = [
    'thermal_expansion',
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

scaler2 = StandardScaler()
yt = scaler2.fit_transform(Y_clean)


mlp = MLPRegressor(max_iter=500)  # setting max_iter to a higher value to ensure convergence

# Define the grid search parameters
param_grid = {
    'hidden_layer_sizes': [(x,) for x in [10, 50, 100]] + [(x, x) for x in [10, 50, 100]] + [(10, 10, 10), (50, 50, 50), (100, 100, 100)],
    'activation': ['relu', 'tanh', 'logistic'],  # logistic is sigmoid
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'batch_size': ['auto', 10, 50],
    'learning_rate': ['constant', 'adaptive'],
}

grid_search = GridSearchCV(mlp, param_grid, scoring='r2', cv=5)
grid_search.fit(x_clean, Y_clean)

# Print best parameters
print("Best parameters found: ", grid_search.best_params_)
print("R-squared with best parameters: ", grid_search.best_score_)
