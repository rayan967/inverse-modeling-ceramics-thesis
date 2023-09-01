import argparse
import pathlib

import joblib
import numpy as np
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


def main(prop, property_name):
    print("starting opt")

    training_data = pathlib.Path("training_data_rve_database.npy")
    if not training_data.exists():
        print(f"Error: training data path {training_data} does not exist.")

    if training_data.suffix == '.npy':
        data = np.load(training_data)
    else:
        print("Invalid data")

    print(f"loaded {data.shape[0]} training data pairs")

    data['thermal_expansion'] *= 1e6

    models = joblib.load("models/8d-features.joblib")["models"]
    features = models[property_name]['features']

    X, Y = extract_XY(data)

    # Compute the minimum and maximum values for each feature in the training data
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)

    # Define the search space dimensions based on the minimum and maximum values
    dimensions = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]
    print(dimensions)

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

    return microstructure


def objective_function(x, desired_property, models, property_name):
    features = models[property_name]['features']

    # Convert the input vector x to a microstructure representation
    microstructure = convert_x_to_microstructure(x, features)

    # Predict the desired property using the trained GPR models
    predicted_property = predict_property(property_name, microstructure, models)

    # Calculate the difference between the predicted property and the desired property
    discrepancy = predicted_property - desired_property

    return abs(discrepancy)  # Return the absolute value of the discrepancy (for minimization)


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
