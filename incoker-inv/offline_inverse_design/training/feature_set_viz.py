import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    DotProduct,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

considered_features = [
    "volume_fraction_4",
    "volume_fraction_1",
    "chord_length_mean_4",
    "chord_length_mean_10",
    "chord_length_mean_1",
    "chord_length_variance_4",
    "chord_length_variance_10",
    "chord_length_variance_1",
]

considered_properties = [
    "thermal_conductivity",
    "thermal_expansion",
    "young_modulus",
    "poisson_ratio",
]


# Define feature extraction functions
def extract_XY_2(data):
    filtered_indices = np.where(data["volume_fraction_1"] == 0.0)
    chord_length_ratio = data["chord_length_mean_4"][filtered_indices] / data["chord_length_mean_10"][filtered_indices]
    volume_fraction_4 = data["volume_fraction_4"][filtered_indices]
    X = np.vstack((volume_fraction_4, chord_length_ratio)).T
    Y = np.vstack(tuple(data[p][filtered_indices] for p in considered_properties)).T
    return X, Y


def extract_XY_3(data):
    chord_length_ratio = data["chord_length_mean_4"] / data["chord_length_mean_10"]
    X = np.vstack((data["volume_fraction_4"], data["volume_fraction_1"], chord_length_ratio)).T
    Y = np.vstack(tuple(data[p] for p in considered_properties)).T
    return X, Y


def extract_XY(data):
    X = np.vstack(tuple(data[f] for f in considered_features)).T
    Y = np.vstack(tuple(data[p] for p in considered_properties)).T
    return X, Y


# Load data
training_data = pathlib.Path("./data/training_data_rve_database.npy")
data = np.load(training_data)
data["thermal_expansion"] *= 1e6

# Define kernels with WhiteKernel
kernels_with_white = [
    RBF() + WhiteKernel(),
    Matern() + WhiteKernel(),
    RBF() * DotProduct() + WhiteKernel(),
    RationalQuadratic() + WhiteKernel(),
    DotProduct + WhiteKernel(),
]
feature_extraction_functions = {
    "8-Feature set": extract_XY,
    "3-Feature set": extract_XY_3,
    "2-Feature set": extract_XY_2,
}


# Function to convert kernel to string
def kernel_to_string(kernel):
    if isinstance(kernel, sklearn.gaussian_process.kernels.Sum):
        return f"{kernel_to_string(kernel.k1)} + {kernel_to_string(kernel.k2)}"
    elif isinstance(kernel, sklearn.gaussian_process.kernels.Product):
        return f"{kernel_to_string(kernel.k1)} * {kernel_to_string(kernel.k2)}"
    else:
        kernel_name = kernel.__class__.__name__
        if kernel_name == "Matern":
            return "M"
        elif kernel_name == "RBF":
            return "RBF"
        elif kernel_name == "RationalQuadratic":
            return "RQ"
        elif kernel_name == "WhiteKernel":
            return "W"
        elif kernel_name == "DotProduct":
            return "DP"
        else:
            return kernel_name


kernels_with_white = [
    DotProduct() + WhiteKernel(),
    RBF() + WhiteKernel(),
    Matern() + WhiteKernel(),
    RationalQuadratic() + WhiteKernel(),
    RBF() * DotProduct() + WhiteKernel(),
]

# Define the parameter grid focusing only on the kernels
param_grid = {
    "kernel": kernels_with_white,
}


# Define a function to perform grid search and return results
def perform_grid_search(X, Y, property_index, feature_set):
    y = Y[:, property_index]
    y_float = np.array([float(val) if val != b"nan" else np.nan for val in y])
    clean_indices = np.argwhere(~np.isnan(y_float))
    y_clean = y_float[clean_indices.flatten()]
    X_clean = X[clean_indices.flatten()]
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_clean)

    gpr = GaussianProcessRegressor(random_state=0, normalize_y=True)
    grid_search = GridSearchCV(gpr, param_grid, scoring="r2", cv=5)
    grid_search.fit(X_clean, y_clean)

    results = pd.DataFrame(grid_search.cv_results_)
    results["param_kernel"] = results["param_kernel"].apply(kernel_to_string)
    results["feature_set"] = feature_set
    return results


# Create subplots for each property
fig, axs = plt.subplots(2, 2, figsize=(20, 16))
axs = axs.flatten()

for i, property_name in enumerate(considered_properties):
    final_results = pd.DataFrame()  # Initialize an empty DataFrame for each property
    # Prepare data for each feature set
    for feature_set, extract_func in zip(
        ["8-Feature set", "3-Feature set", "2-Feature set"], [extract_XY, extract_XY_3, extract_XY_2]
    ):
        X, Y = extract_func(data)
        results = perform_grid_search(X, Y, i, feature_set)

        # Concatenate the results
        final_results = pd.concat([final_results, results])

    # After collecting results for a property, pivot and plot on the corresponding subplot
    final_pivot = final_results.pivot_table(index="param_kernel", columns="feature_set", values="mean_test_score")
    sns.heatmap(final_pivot, annot=True, annot_kws={"size": 14}, cmap="RdBu", fmt=".3e", vmin=-0.1, vmax=1, ax=axs[i])
    axs[i].set_title(f"R2 Scores for {property_name}")

    axs[i].set_ylabel("")
    axs[i].set_xlabel("")

plt.tight_layout()
plt.show()
