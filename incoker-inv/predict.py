import argparse

import joblib
import numpy as np
import json


def main(microstructure):
    models = joblib.load("models/8d-features.joblib")["models"]
    model = models['thermal_conductivity']['pipe']
    features = models['thermal_conductivity']['features']

    # Extract the features required by the specific property's GPR model
    microstructure_features = [microstructure[feature] for feature in features]
    # Prepare the input for prediction
    X = np.array(microstructure_features).reshape(1, -1)
    # Use the GPR model to predict the property value
    predicted_value = model.predict(X)
    print(predicted_value)
    return predicted_value[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict property')
    parser.add_argument('prop', type=json.loads,
                        help='Expected property in JSON format')
    args = parser.parse_args()

    main(args.prop)