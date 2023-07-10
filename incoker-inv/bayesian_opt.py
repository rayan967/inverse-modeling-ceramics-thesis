import argparse
import pathlib

import joblib
import numpy as np
from training import extract_XY
from skopt import Optimizer
from scipy.optimize import minimize

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


def main(prop):
    print("starting opt")

    training_data = pathlib.Path('training_data_rve_database.npy')
    if not training_data.exists():
        print(f"Error: training data path {training_data} does not exist.")

    if training_data.suffix == '.npy':
        data = np.load(training_data)
    else:
        print("Invalid data")

    print(f"loaded {data.shape[0]} training data pairs")

    data['thermal_expansion'] *= 1e6
    X, Y = extract_XY(data)

    models = joblib.load("trained_models.joblib")["models"]
    # Compute the minimum and maximum values for each feature in the training data
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)

    # Define the search space dimensions based on the minimum and maximum values
    dimensions = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]

    # Initialize the Bayesian optimization algorithm with the objective function and search space
    optimizer = Optimizer(dimensions, base_estimator="GP", acq_func="EI")

    num_iterations = 100  # Define the desired number of iterations

    # Run the optimization loop
    for _ in range(num_iterations):
        # Select the next candidate point for evaluation
        x_next = optimizer.ask()

        # Evaluate the objective function at the candidate point
        y_next = objective_function(x_next, prop, models)

        # Provide the observed point to the optimizer for updating the surrogate model
        optimizer.tell(x_next, y_next)

    # Obtain the optimal microstructure from the best solution found
    res = minimize(lambda x: objective_function(x, prop, models), x0=optimizer.ask(), bounds=dimensions)
    optimal_x = res.x
    optimal_microstructure = convert_x_to_microstructure(optimal_x)

    # Evaluate the desired property using the optimal microstructure
    optimal_property_value = predict_property('thermal_expansion', optimal_microstructure, models)

    print(optimal_microstructure)


def convert_x_to_microstructure(x):
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

    return microstructure


def objective_function(x, desired_property, models):
    # Convert the input vector x to a microstructure representation
    microstructure = convert_x_to_microstructure(x)

    # Predict the desired property using the trained GPR models
    predicted_property = predict_property('thermal_expansion', microstructure, models)

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
    args = parser.parse_args()

    main(args.prop)
