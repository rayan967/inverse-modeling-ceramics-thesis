import argparse
import pathlib
import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import WhiteKernel, RBF, Matern
from sklearn.metrics import make_scorer


considered_features = [
    'volume_fraction_4', 'volume_fraction_1',
    'chord_length_mean_4', 'chord_length_mean_10', 'chord_length_mean_1',
    'chord_length_variance_4', 'chord_length_variance_10', 'chord_length_variance_1'
]

# material properties to consider in training
considered_properties = [
    'thermal_conductivity',
    'thermal_expansion',
    'young_modulus',
    'poisson_ratio',
]

property_ax_dict = {
    'thermal_conductivity':'CTC [W/(m*K)]',
    'thermal_expansion':'CTE [ppm/K]',
    'young_modulus':'Young\'s Modulus[GPa]',
    'poisson_ratio':'Poisson Ratio',
}


def main(train_data_file, export_model_file, number_of_features, plots=False):

    training_data = pathlib.Path(train_data_file)
    if not training_data.exists():
        print(f"Error: training data path {training_data} does not exist.")

    if training_data.suffix == '.npy':
        data = np.load(training_data)
    else:
        print("Invalid data")

    print(f"loaded {data.shape[0]} training data pairs")

    data['thermal_expansion'] *= 1e6


    if number_of_features == 3:
        Y, X = extract_XY_3(data)
    elif number_of_features == 2:
        Y, X = extract_XY_2(data)
    else:
        Y, X = extract_XY(data)




    model_types = ["NN", "GPR", "RF"]
    best_models = {}

    for i, property_name in enumerate(considered_properties):

        # pick a single property
        x = X[:, i]
        x_float = np.array([float(val) if val != b'nan' else np.nan for val in x])

        # ignore NaNs in the data
        clean_indices = np.argwhere(~np.isnan(x_float))
        x_clean = x_float[clean_indices.flatten()].reshape(-1,1)
        Y_clean = Y[clean_indices.flatten()]

        # Change next line for different feature sets from models folder
        gbr_models = joblib.load("models/2_GBR.joblib")["models"][property_name]

        X_train, X_test, y_train, y_test = train_test_split(x_clean, Y_clean, random_state=0)

        best_score = -float("inf")
        best_model_name = None
        best_model = None
        nn_predictions = {}
        gpr_predictions = {}
        rf_predictions = {}

        for model_name in model_types:
            model = train_model_with(model_name, X_train, y_train)
            print("Model:", model_name)
            print("Property:", property_name)

            y_pred = model.predict(X_test)
            xr_pred = gbr_models['pipe'].predict(y_pred)

            rscore = accuracy_test(xr_pred,X_test)
            rm = rmse(xr_pred,X_test)
            print("\nScores based on test data: \n")
            print("RR.Score:",str(rscore))
            print("RRMSE:",str(rm))


            x_min, x_max = np.min(X_train[:, 0]), np.max(X_train[:, 0])

            property_range = np.linspace(x_min, x_max, 100)



            # Predict microstructures for the range of property values
            predictions = model.predict(property_range.reshape(-1, 1))

            xr_pred = gbr_models['pipe'].predict(predictions)

            # Store predictions based on model type
            if model_name == "NN":
                nn_predictions[property_name] = predictions
            elif model_name == "GPR":
                gpr_predictions[property_name] = predictions
            elif model_name == "RF":
                rf_predictions[property_name] = predictions

            rscore = accuracy_test(xr_pred,property_range)
            rm = rmse(xr_pred,property_range)

            print("\nScores based on property range: \n")
            print("RR.Score:",str(rscore))
            print("RMSE:",str(rm))
            print("##########")

            plot_property_vs_volume_fraction(
                predictions,
                property_range,
                property_name,
                "Volume Fraction",
                "Particle Size Ratio",
                property_ax_dict[property_name],
                gbr_models,
                train_points=( y_train[:, 0], y_train[:, 1], X_train,)
            )
            if rscore > best_score:
                best_score = rscore
                best_model_name = model_name
                best_model = model

        plot_predictions_for_property(nn_predictions[property_name],
                                         gpr_predictions[property_name],
                                         rf_predictions[property_name],
                                         property_range,
                                         "Volume Fraction Zirconia",
                                         "Particle Size Ratio",
                                         property_ax_dict[property_name],
                                         gbr_models,
                                         train_points=(y_train[:, 0], y_train[:, 1], X_train))
        best_models[property_name] = {
            'model': best_model,
            'model_type': best_model_name,
            'score': best_score
        }



def plot_predictions_for_property(nn_predictions, gpr_predictions, rf_predictions, property_range, xlabel, ylabel, zlabel, models, train_points=None):
    """
    Plot property vs. volume fraction for different models.

    Parameters:
    - nn_predictions, gpr_predictions, rf_predictions: arrays of predictions from different models
    - property_name, xlabel, ylabel, zlabel: strings, titles and labels for the plot
    - models: dictionary containing model details (for surface plot)
    - train_points: tuple of three arrays (X_train, y_train, Z_train) used to plot the training points. Optional.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    Xt = models['X_train']
    v2_values = np.linspace(min(Xt[:, 0]), max(Xt[:, 0]), num=50)
    rho_values = np.linspace(min(Xt[:, 1]), max(Xt[:, 1]), num=50)
    v2_grid, rho_grid = np.meshgrid(v2_values, rho_values)

    feature_grid = np.vstack([v2_grid.ravel(), rho_grid.ravel()]).T

    predictions = (models['pipe'].predict(feature_grid))

    predictions_grid = predictions.reshape(v2_grid.shape)
    ax.plot_surface(v2_grid, rho_grid, predictions_grid, rstride=1, cstride=1,
                    color='b', alpha=0.1, )  # Set color and transparency)

    # Plot NN predictions
    ax.scatter3D(nn_predictions[:, 0], nn_predictions[:, 1], property_range, c='b', marker='o', alpha=0.6, label='NN Predictions', linewidth = 0.1)

    # Plot GPR predictions
    ax.scatter3D(gpr_predictions[:, 0], gpr_predictions[:, 1], property_range, c='g', marker='^', alpha=0.6, label='GPR Predictions')

    # Plot RF predictions
    ax.scatter3D(rf_predictions[:, 0], rf_predictions[:, 1], property_range, c='r', marker='s', alpha=0.6, label='RF Predictions')

    if train_points:
        X_train, y_train, Z_train = train_points
        ax.scatter3D(X_train, y_train, Z_train, c='m', marker='x', alpha=0.3, label='Ground Truth')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend()
    plt.show()

def plot_property_vs_volume_fraction(prediction, property_range, property_name, xlabel, ylabel, zlabel, models, train_points=None):
    """
    Plot property vs. volume fraction.

    Parameters:
    - x_values, y_values, z_values: arrays of the same length, coordinates of the points to plot
    - title, xlabel, ylabel, zlabel: strings, titles and labels for the plot
    - train_points: tuple of three arrays (X_train, y_train, Z_train) used to plot the training points. Optional.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Xt = models['X_train']
    v2_values = np.linspace(min(Xt[:, 0]), max(Xt[:, 0]), num=50)
    rho_values = np.linspace(min(Xt[:, 1]), max(Xt[:, 1]), num=50)
    v2_grid, rho_grid = np.meshgrid(v2_values, rho_values)

    feature_grid = np.vstack([v2_grid.ravel(), rho_grid.ravel()]).T

    predictions = (models['pipe'].predict(feature_grid))

    predictions_grid = predictions.reshape(v2_grid.shape)
    ax.plot_surface(v2_grid, rho_grid, predictions_grid, rstride=1, cstride=1,
                    color='b', alpha=0.1, )  # Set color and transparency)

    # Plot predictions
    ax.scatter3D(prediction[:, 0], prediction[:, 1], property_range, c='b', marker='o', alpha=0.6, label='Predictions')

    if train_points:
        X_train, y_train, Z_train = train_points
        ax.scatter3D(X_train, y_train, Z_train, c='r', marker='x', alpha=0.6, label='Actual points')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend()
    plt.show()

def accuracy_test(y_pred, y_test):
    """
    Parameters
    ----------
    model : GPR model
    X_test : np.array
        Test data (features).
    y_test : np.array
        Test data (true values).
    tolerance : float
        Tolerance for the accuracy score.

    Returns
    -------
    score : float
        Accuracy score between 0 and 100.
    """


    # Calculate whether each prediction is within the tolerance of the true value
    score = metrics.r2_score(y_true=y_test, y_pred=y_pred)

    return score

def mse(y_pred, y_test):
    """
    Parameters
    ----------
    model : GPR model
    X_test : np.array
        Test data (features).
    y_test : np.array
        Test data (true values).
    tolerance : float
        Tolerance for the accuracy score.

    Returns
    -------
    score : float
        Accuracy score between 0 and 100.
    """


    # Calculate whether each prediction is within the tolerance of the true value
    score = metrics.mean_squared_error(y_true=y_test, y_pred=y_pred)

    return score

def rmse(y_pred, y_test):
    """
    Parameters
    ----------
    model : GPR model
    X_test : np.array
        Test data (features).
    y_test : np.array
        Test data (true values).
    tolerance : float
        Tolerance for the accuracy score.

    Returns
    -------
    score : float
        Accuracy score between 0 and 100.
    """


    # Calculate whether each prediction is within the tolerance of the true value
    score = metrics.mean_absolute_error(y_true=y_test, y_pred=y_pred)

    return score


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

def train_model_with(model_name, X_train, y_train):
    """Helper function to train with different models."""
    if model_name == "NN":
        from sklearn.neural_network import MLPRegressor
        model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100,100))
        )
    elif model_name == "GPR":
        kernel = RBF() + WhiteKernel()
        model = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel))
    elif model_name == "RF":
        from sklearn.ensemble import RandomForestRegressor
        model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100))
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model.fit(X_train, y_train)
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ML models using structure-property data of '
                                                 'RVEs.')
    parser.add_argument('train_data_file', type=pathlib.Path,
                        help='Path to the database of RVE structures and simulation results or '
                             'numpy file with training data already loaded')
    parser.add_argument('--export_model_file', type=pathlib.Path, required=False,
                        help='Path to a file where the trained models will be exported to.')
    parser.add_argument('--number_of_features', type=int, required=True,
                        help='Number of features, supports 8 or 3 or 2.')
    args = parser.parse_args()

    main(args.train_data_file, args.export_model_file, args.number_of_features)