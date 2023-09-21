import argparse
import pathlib
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import WhiteKernel, RBF, Matern
from sklearn.metrics import mean_squared_error, r2_score

BEST_PARAMETERS = {
    'thermal_conductivity_composite': {'alpha': 1e-05,
                                       'kernel': Matern(length_scale=1, nu=1.5) + WhiteKernel(noise_level=1)}
}


def main(train_data_file, export_model_file):
    training_data = pathlib.Path(train_data_file)
    if not training_data.exists():
        print(f"Error: training data path {training_data} does not exist.")

    df = pd.read_csv(training_data)
    models = {}
    # Calculate chord_length_mean_ratio
    df['chord_length_mean_ratio'] = df['chord_length_mean_zro2'] / df['chord_length_mean_al2o3']
    df.dropna(inplace=True)

    features = ['volume_fraction_zro2', 'chord_length_mean_ratio']

    X = df[features].values
    Y = df['thermal_conductivity_composite'].values
    property_name = 'thermal_conductivity_composite'
    # GPR parameters
    best_params = BEST_PARAMETERS['thermal_conductivity_composite']
    kernel = best_params['kernel']
    alpha = best_params['alpha']

    pipe = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, alpha=alpha)
    )

    # split test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    pipe.fit(X_train, y_train)

    score = pipe.score(X_test, y_test)
    print("Model R-squared score on test set:", score)

    y_pred = pipe.predict(X_test)

    v2_values = np.linspace(min(X_train[:, 0]), max(X_train[:, 0]), num=100)
    rho_values = np.linspace(min(X_train[:, 1]), max(X_train[:, 1]), num=100)
    v2_grid, rho_grid = np.meshgrid(v2_values, rho_values)

    # Flatten the grid to pass it to the model for prediction
    feature_grid = np.vstack([v2_grid.ravel(), rho_grid.ravel()]).T

    predictions = pipe.predict(feature_grid)

    # Reshape the predictions to match the shape of the grid
    predictions_grid = predictions.reshape(v2_grid.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(v2_grid, rho_grid, predictions_grid)

    ax.set_xlabel('Volume Fraction (zirconia)')
    ax.set_ylabel('Particle Size Ratio')
    ax.set_zlabel('thermal_conductivity')
    ax.set_title('3D Surface of Predicted Material Property')

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for actual points
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, label="Actual", color='blue', marker='o')

    # Scatter plot for optimized points
    ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, label="Predictions with ML model", color='red',
               marker='x')

    # Set labels
    ax.set_xlabel('Volume Fraction Zirconia')
    ax.set_ylabel('Particle Size Ratio (rho)')
    ax.set_zlabel('thermal_conductivity')
    ax.legend()

    plt.show()

    models[property_name] = {'pipe': pipe, 'features': features}
    models[property_name]['X_train'] = X_train
    models[property_name]['X_test'] = X_test
    models[property_name]['y_train'] = y_train
    models[property_name]['y_test'] = y_test

    # Exporting model if the export_model_file is specified
    if export_model_file is not None:
        from datetime import date
        import pkg_resources
        import sys
        installed_packages = pkg_resources.working_set
        installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
                                          for i in installed_packages])

        # store python code in current directory for reproducibility
        local_python_files = list(pathlib.Path().glob('*.py'))
        local_python_code = [f.read_text() for f in local_python_files]

        exported_model = {
            'models': models,
            'version_info':
                {
                    'date': date.today().isoformat(),
                    'python': sys.version,
                    'packages': installed_packages_list
                },
            'python_files': local_python_code
        }
        joblib.dump(exported_model, export_model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ML models using structure-property data.')
    parser.add_argument('train_data_file', type=pathlib.Path, help='Path to the CSV file with training data.')
    parser.add_argument('--export_model_file', type=pathlib.Path, required=False,
                        help='Path to a file where the trained model will be exported to.')
    args = parser.parse_args()

    main(args.train_data_file, args.export_model_file)
