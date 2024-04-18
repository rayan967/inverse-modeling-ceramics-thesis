"""
Convert an adaptive Gaussian Process (GP) model from simlopt's implementation to the Scikit-learn GP framework.

This script takes a trained adaptive GP model file as input, transforms it for compatibility with Scikit-learn,
and evaluates its performance using standard metrics. It also provides functionality for visualizing model predictions
and exporting the converted model for future use. The script is designed to handle command-line arguments to facilitate
easy interaction and automation.
"""

import argparse
import pathlib

import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF, WhiteKernel


def main(adapt_model, export_model_file, property_name, plots=False):
    """
    Convert and evaluate an adaptive GP model using Scikit-learn's GaussianProcessRegressor.

    Load a model trained with a specialized GP implementation, preprocess the data, fit a Scikit-learn Gaussian Process
    model, and evaluate its performance using various metrics. Optionally, generate plots to visualize the model's
    predictions. If specified, export the converted model for future use.

    Args:
        adapt_model (pathlib.Path): Path to the adaptive GP model file.
        export_model_file (pathlib.Path): Optional path to save the converted GP model.
        property_name (str): Material property to model, used to label outputs.
        plots (bool): Flag to indicate whether to produce visualization plots of the model performance.

    Returns:
        dict: A dictionary containing details and performance metrics of the trained Scikit-learn GP model.
    """
    models = {}

    considered_features = ["volume_fraction_4", "chord_length_ratio"]

    print("Features: ", str(considered_features))

    agp = joblib.load(adapt_model)
    X, y = agp.X, agp.yt

    y_float = np.array([float(val) if val != b"nan" else np.nan for val in y])

    # ignore NaNs in the data
    clean_indices = np.argwhere(~np.isnan(y_float))
    y_clean = y_float[clean_indices.flatten()]
    X_clean = X[clean_indices.flatten()]

    # create a pipeline object for training using the best parameters
    pipe = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), normalize_y=True))

    # split in test and train data
    X_train, X_test, y_train, y_test = X_clean, X_clean, y_clean, y_clean
    pipe.fit(X_train, y_train)

    models[property_name] = {"pipe": pipe, "features": considered_features}
    models[property_name]["X_train"] = X_train
    models[property_name]["X_test"] = X_test
    models[property_name]["y_train"] = y_train
    models[property_name]["y_test"] = y_test

    # Evaluate the model using the score method
    score = pipe.score(X_test, y_test)
    print("Model R-squared score on test set:", score)

    # Evaluate the model using the score method
    score = pipe.score(X_train, y_train)
    print("Model R-squared score on training set:", score)

    y_pred = pipe.predict(X_test)
    min_pred_value = np.min(y_pred)
    max_pred_value = np.max(y_pred)
    print(min_pred_value, max_pred_value)

    print(f"Property {property_name}")
    print("   score on train set = ", pipe.score(X_train, y_train))
    print("   score on test  set = ", pipe.score(X_test, y_test))

    print("Accuracy: ", metrics.r2_score(y_test, y_pred))

    # The mean squared error
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    models[property_name]["rmse"] = rmse
    print("   Root mean squared error: %.2f" % rmse)
    # The coefficient of determination: 1 is perfect prediction
    print("   Coefficient of determination: %.5f" % r2_score(y_test, y_pred))

    custom_scorer = make_scorer(metrics.r2_score, greater_is_better=True)

    # cross validation
    cv = cross_validate(pipe, X_clean, y_clean, cv=5, scoring=custom_scorer)
    cv_score_mean = cv["test_score"].mean()
    cv_score_std = cv["test_score"].std()
    models[property_name]["cv_score_mean"] = cv_score_mean
    models[property_name]["cv_score_std"] = cv_score_std
    print("   %0.5f accuracy with a standard deviation of %0.5f" % (cv_score_mean, cv_score_std))

    if plots:
        plt.figure()
        plt.scatter(X_test[:, 0], y_test, label="Actual", color="blue", marker="o")
        plt.scatter(X_test[:, 0], y_pred, label="Predicted", color="red", marker="x")

        plt.xlabel("Volume Fraction Zirconia")
        plt.ylabel(property_name)
        plt.legend()
        plt.show()

        v2_values = np.linspace(min(X_train[:, 0]), max(X_train[:, 0]), num=100)
        rho_values = np.linspace(min(X_train[:, 1]), max(X_train[:, 1]), num=100)
        v2_grid, rho_grid = np.meshgrid(v2_values, rho_values)

        # Flatten the grid to pass it to the model for prediction
        feature_grid = np.vstack([v2_grid.ravel(), rho_grid.ravel()]).T

        # Assume 'model' is your trained 2-feature model
        # Make predictions
        predictions = models[property_name]["pipe"].predict(feature_grid)

        # Reshape the predictions to match the shape of the grid
        predictions_grid = predictions.reshape(v2_grid.shape)

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(v2_grid, rho_grid, predictions_grid)

        # Add labels and title
        ax.set_xlabel("Volume Fraction (zirconia)")
        ax.set_ylabel("Particle Size Ratio")
        ax.set_zlabel(property_name)
        ax.set_title("3D Surface of Predicted Material Property")

        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Scatter plot for actual points
        ax.scatter(X_test[:, 0], X_test[:, 1], y_test, label="Actual", color="blue", marker="o")

        # Scatter plot for optimized points
        ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, label="Predictions with ML model", color="red", marker="x")

        # Set labels
        ax.set_xlabel("Volume Fraction Zirconia")
        ax.set_ylabel("Particle Size Ratio (rho)")
        ax.set_zlabel(property_name)

        # Show legend
        ax.legend()

        plt.show()

    for prop in ["young_modulus", "poisson_ratio", "thermal_conductivity", "thermal_expansion"]:
        if prop in models:
            print(
                f"{prop}: {models[prop]['cv_score_mean']:.3f}, {models[prop]['cv_score_std']:.1e}, "
                f"{models[prop]['rmse']:.1e}"
            )

    # export model for use in other projects
    if export_model_file is not None:
        import sys
        from datetime import date

        import pkg_resources

        installed_packages = pkg_resources.working_set
        installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])

        # store python code in current directory for reproducibility
        local_python_files = list(pathlib.Path().glob("*.py"))
        local_python_code = [f.read_text() for f in local_python_files]

        exported_model = {
            "models": models,
            "version_info": {
                "date": date.today().isoformat(),
                "python": sys.version,
                "packages": installed_packages_list,
            },
            "python_files": local_python_code,
        }
        joblib.dump(exported_model, export_model_file)
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models using structure-property data of " "RVEs.")
    parser.add_argument("adapt_model", type=pathlib.Path, help="Path to the adaptive GP model to convert")
    parser.add_argument(
        "--export_model_file",
        type=pathlib.Path,
        required=False,
        help="Path to a file where the trained models will be exported to.",
    )
    parser.add_argument(
        "--property_name",
        type=str,
        choices=["thermal_conductivity", "thermal_expansion", "young_modulus", "poisson_ratio"],
        required=True,
        help="Name of the property",
    )
    parser.add_argument("--plots", action="store_true", help="Specify if plots should be shown.")

    args = parser.parse_args()

    main(args.adapt_model, args.export_model_file, args.property_name, plots=args.plots)
