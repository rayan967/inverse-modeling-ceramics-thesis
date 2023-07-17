import argparse
import pathlib
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


considered_features = [
    'volume_fraction_4', 'volume_fraction_10', 'volume_fraction_1',
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


def main(train_data_file, export_model_file):

    training_data = pathlib.Path(train_data_file)
    if not training_data.exists():
        print(f"Error: training data path {training_data} does not exist.")

    if training_data.suffix == '.npy':
        data = np.load(training_data)
    else:
        print("Invalid data")

    print(f"loaded {data.shape[0]} training data pairs")

    data['thermal_expansion'] *= 1e6

    models = {}

    for property_name in considered_properties:
        y = np.vstack(tuple(data[f] for f in considered_features)).T
        X = data[property_name]

        # Remove NaN values from the data
        valid_indices = ~np.isnan(X)
        X_clean = X[valid_indices].reshape(-1, 1)
        y_clean = y[valid_indices]

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam')

        # Train the model
        model.fit(X_train_scaled, y_train)

        # Evaluate the model on the test set
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        score = model.score(X_test_scaled, y_test)

        print("--------------------------")
        print(f"Property: {property_name}")
        print("Test MSE:", mse)
        print("Test MAE:", mae)
        print("Score:", score)
        print("--------------------------")
        models[property_name] = model

    # Export the trained models
    if export_model_file is not None:
        joblib.dump(models, export_model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ML models using structure-property data of RVEs.')
    parser.add_argument('train_data_file', type=pathlib.Path,
                        help='Path to the database of RVE structures and simulation results or '
                             'numpy file with training data already loaded')
    parser.add_argument('--export_model_file', type=pathlib.Path, required=False,
                        help='Path to a file where the trained models will be exported to.')
    args = parser.parse_args()

    main(args.train_data_file, args.export_model_file)
