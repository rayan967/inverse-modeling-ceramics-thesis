import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, DotProduct, WhiteKernel, RationalQuadratic
from sklearn.model_selection import GridSearchCV
import pathlib
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# Features and properties
considered_features = [
    'volume_fraction_4', 'volume_fraction_1',
    'chord_length_mean_4', 'chord_length_mean_10', 'chord_length_mean_1',
    'chord_length_variance_4', 'chord_length_variance_10', 'chord_length_variance_1'
]
considered_properties = [
    'thermal_conductivity',
    'thermal_expansion',
    'young_modulus',
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


def extract_XY_3(data):
    """Use for 3 features."""

    chord_length_ratio = data['chord_length_mean_4'] / data['chord_length_mean_10']
    X = np.vstack((data['volume_fraction_4'], data['volume_fraction_1'], chord_length_ratio)).T
    Y = np.vstack(tuple(data[p] for p in considered_properties)).T

    global considered_features

    considered_features = [
    'volume_fraction_4', 'volume_fraction_1',
    'chord_length_ratio'
]
    return X, Y


def extract_XY(data):
    """Use for 8 features."""

    X = np.vstack(tuple(data[f] for f in considered_features)).T
    Y = np.vstack(tuple(data[p] for p in considered_properties)).T

    return X, Y


# Load data
training_data = pathlib.Path("./data/training_data_rve_database.npy")
data = np.load(training_data)
data['thermal_expansion'] *= 1e6

X = np.vstack(tuple(data[f] for f in considered_features)).T
Y = np.vstack(tuple(data[p] for p in considered_properties)).T

X, Y = extract_XY(data)
fig, axs = plt.subplots(2, 2, figsize=(20, 16))  # Adjust the size as needed
axs = axs.flatten()  # Flatten the array to easily index subplots
for i, property_name in enumerate(considered_properties):

    y = Y[:, i]
    y_float = np.array([float(val) if val != b'nan' else np.nan for val in y])
    clean_indices = np.argwhere(~np.isnan(y_float))
    y_clean = y_float[clean_indices.flatten()]
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
        RBF() * DotProduct() + WhiteKernel(),
    ]

    # Define parameter grid
    param_grid = {
        "kernel": kernels,
        "alpha": [1e-10, 1e-8, 1e-5, 1e-3, 1e-1, 1e0, 1e1],
    }

    # Run grid search
    gpr = GaussianProcessRegressor(random_state=0, normalize_y=True)
    grid_search = GridSearchCV(gpr, param_grid, scoring='r2', cv=5)
    grid_search.fit(X_clean, y_clean)

    # Print best parameters
    print("Best parameters found: ", grid_search.best_params_)
    print("Custom accuracy with best parameters: ", grid_search.best_score_)

    results = pd.DataFrame(grid_search.cv_results_)

    # Simplify the DataFrame and convert 'param_kernel' to string
    results_simplified = results.copy()


    def kernel_to_string(kernel):
        """
        Convert a kernel object to a string representation.

        Args:
        kernel (sklearn.gaussian_process.kernels.Kernel): The kernel object.

        Returns:
        str: The string representation of the kernel.
        """
        if isinstance(kernel, sklearn.gaussian_process.kernels.Sum):
            return f"{kernel_to_string(kernel.k1)} + {kernel_to_string(kernel.k2)}"
        elif isinstance(kernel, sklearn.gaussian_process.kernels.Product):
            return f"{kernel_to_string(kernel.k1)} * {kernel_to_string(kernel.k2)}"
        else:
            # Map kernel classes to abbreviations
            kernel_name = kernel.__class__.__name__
            if kernel_name == 'Matern':
                return 'M'
            elif kernel_name == 'RBF':
                return 'RBF'
            elif kernel_name == 'RationalQuadratic':
                return 'RQ'
            elif kernel_name == 'WhiteKernel':
                return 'W'
            elif kernel_name == 'DotProduct':
                return 'DP'
            else:
                return kernel_name  # Fallback for any other kernel types


    # Use this function instead of str
    results_simplified = results.copy()
    results_simplified['param_kernel'] = results_simplified['param_kernel'].apply(kernel_to_string)
    heatmap_data = results_simplified.pivot(index='param_kernel', columns='param_alpha', values='mean_test_score')

    # Plot the heatmap in the i-th subplot
    sns.heatmap(heatmap_data, annot=True, annot_kws={"size": 14}, cmap='RdBu', fmt=".3e", vmin=-0.1, vmax=1, ax=axs[i])
    axs[i].set_title(f'Hyperparameter Tuning for {property_name}')
    axs[i].set_ylabel('Kernel')
    axs[i].set_xlabel('Noise level')


plt.tight_layout()
plt.show()