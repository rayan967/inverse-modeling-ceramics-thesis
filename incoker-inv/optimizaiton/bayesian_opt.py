import argparse
import pathlib

import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from standard_training import extract_XY, extract_XY_3
from skopt import Optimizer, gp_minimize
from scipy.optimize import minimize, LinearConstraint

considered_features = [
    'volume_fraction_4', 'volume_fraction_10', 'volume_fraction_1',
    'chord_length_mean_4', 'chord_length_mean_10', 'chord_length_mean_1',
    'chord_length_variance_4', 'chord_length_variance_10', 'chord_length_variance_1'
]

# material properties to consider in training
considered_properties = [
    'thermal_conductivity',
    'young_modulus',
    'poisson_ratio',
    'thermal_expansion',
]


def main(prop1, property_name):
    print("starting opt")

    training_data = pathlib.Path("../data/training_data_rve_database.npy")
    if not training_data.exists():
        print(f"Error: training data path {training_data} does not exist.")

    if training_data.suffix == '.npy':
        data = np.load(training_data)
    else:
        print("Invalid data")

    print(f"loaded {data.shape[0]} training data pairs")

    data['thermal_expansion'] *= 1e6

    models = joblib.load("../models/3d_model.joblib")["models"]
    features = models[property_name]['features']


    X = models[property_name]['X_train']
    X_test = models[property_name]['X_test']
    Y = models[property_name]['y_train']
    y_test = models[property_name]['y_test']

    # Compute the minimum and maximum values for each feature in the training data
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)

    # Define the search space dimensions based on the minimum and maximum values
    dimensions = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]
    print(dimensions)
    """
    res = gp_minimize(lambda x: objective_function(x, prop, models, property_name),
                     dimensions,
                     n_calls=100,  # Number of iterations)
                     )

    optimal_x = res.x

    optimal_microstructure = convert_x_to_microstructure(optimal_x, features)

    # Evaluate the desired property using the optimal microstructure
    optimal_property_value = predict_property(property_name, optimal_microstructure, models)

    print(optimal_microstructure)
    print("Error in optimisation: "+str(np.abs(prop - optimal_property_value)))

    """
    # Running subspace optimizations with Bayesian optimisation
    features = models[property_name]['features']
    print("Features: ", str(features))

    prop_bounds = (min(Y), max(Y))

    # Grid of 100 points across property bounds for plot
    num_points = 100
    prop_values = np.linspace(prop_bounds[0], prop_bounds[1], num_points)

    optimal_microstructures = []
    optimal_volume_fractions_4 = []
    optimal_properties = []

    actual_volume_fractions_4 = X[:, 0]
    actual_properties = Y
    count = 0
    error_bars_min = []
    error_bars_max = []

    for prop in prop_values:
        count += 1
        res = gp_minimize(lambda x: objective_function(x, prop, models, property_name),
                          dimensions,
                          n_calls=100,
                          )

        optimal_x = res.x
        optimal_microstructure = convert_x_to_microstructure(optimal_x, features)

        optimal_microstructures.append(optimal_microstructure)
        # Store the optimal volume fraction and thermal expansion value
        optimal_volume_fractions_4.append(optimal_microstructure['volume_fraction_4'])
        optimal_properties.append(prop)

    predicted_properties = []

    for optimal_microstructure in optimal_microstructures:
        predicted_value = predict_property(property_name, optimal_microstructure, models)
        predicted_properties.append(predicted_value)

    predicted_properties = np.array(predicted_properties)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(prop_values, predicted_properties))
    print(f"Root Mean Square Error (RMSE) between predicted and desired values: {rmse}")

    # Calculate R2 score for optimized and actual feature sets
    r2 = r2_score(prop_values, predicted_properties)
    print(f"R2 score between predicted and desired values: {r2*100}")


    plt.figure()
    plt.scatter(actual_volume_fractions_4, actual_properties, label="Actual",  color='blue', marker='o')
    plt.scatter(optimal_volume_fractions_4, optimal_properties, label="Optimized", color='red', marker='x')
    plt.xlabel("Volume Fraction Zirconia")
    plt.ylabel(property_name)
    plt.legend()
    plt.show()



def convert_x_to_microstructure(x, features):
    if len(features) == 9:
        volume_fraction_4 = x[0]
        volume_fraction_10 = x[1]
        volume_fraction_1 = x[2]
        chord_length_mean_4 = x[3]
        chord_length_mean_10 = x[4]
        chord_length_mean_1 = x[5]
        chord_length_variance_4 = x[6]
        chord_length_variance_10 = x[7]
        chord_length_variance_1 = x[8]

        # Construct the microstructure representation
        microstructure = {
            'volume_fraction_4': volume_fraction_4,
            'volume_fraction_10': volume_fraction_10,
            'volume_fraction_1': volume_fraction_1,
            'chord_length_mean_4': chord_length_mean_4,
            'chord_length_mean_10': chord_length_mean_10,
            'chord_length_mean_1': chord_length_mean_1,
            'chord_length_variance_4': chord_length_variance_4,
            'chord_length_variance_10': chord_length_variance_10,
            'chord_length_variance_1': chord_length_variance_1,
        }
    elif len(features) == 8:
        volume_fraction_4 = x[0]
        volume_fraction_1 = x[1]
        chord_length_mean_4 = x[2]
        chord_length_mean_10 = x[3]
        chord_length_mean_1 = x[4]
        chord_length_variance_4 = x[5]
        chord_length_variance_10 = x[6]
        chord_length_variance_1 = x[7]

        # Construct the microstructure representation
        microstructure = {
            'volume_fraction_4': volume_fraction_4,
            'volume_fraction_1': volume_fraction_1,
            'chord_length_mean_4': chord_length_mean_4,
            'chord_length_mean_10': chord_length_mean_10,
            'chord_length_mean_1': chord_length_mean_1,
            'chord_length_variance_4': chord_length_variance_4,
            'chord_length_variance_10': chord_length_variance_10,
            'chord_length_variance_1': chord_length_variance_1,
        }
    elif len(features) == 3:
        volume_fraction_4 = x[0]
        volume_fraction_1 = x[1]
        chord_length_ratio = x[2]

        # Construct the microstructure representation
        microstructure = {
            'volume_fraction_4': volume_fraction_4,
            'volume_fraction_1': volume_fraction_1,
            'chord_length_ratio': chord_length_ratio,
        }
    elif len(features) == 2:
        volume_fraction_4 = x[0]
        chord_length_ratio = x[1]

        # Construct the microstructure representation
        microstructure = {
            'volume_fraction_4': volume_fraction_4,
            'chord_length_ratio': chord_length_ratio,
        }

    return microstructure


def objective_function(x, desired_property, models, property_name):
    features = models[property_name]['features']
    microstructure = convert_x_to_microstructure(x, features)
    #print("Current microstructure:", str(microstructure))
    predicted_property = predict_property(property_name, microstructure, models)
    discrepancy = predicted_property - desired_property
    #print("Objective value: ", str(discrepancy ** 2))
    return (discrepancy ** 2)  # Return the squared discrepancy for minimization


def predict_property(property_name, microstructure, models):
    model = models[property_name]['pipe']
    features = models[property_name]['features']
    # Extract the features required by the specific property's GPR model
    microstructure_features = [microstructure[feature] for feature in features]
    # Prepare the input for prediction
    X = np.array(microstructure_features).reshape(1, -1)
    # Use the GPR model to predict the property value
    predicted_value = model.predict(X)
    return predicted_value[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimise microstructure for given property.')
    parser.add_argument('prop', type=float,
                        help='Expected property')
    parser.add_argument('prop_name', type=str,
                        help='Expected property')
    args = parser.parse_args()

    main(args.prop, args.prop_name)
