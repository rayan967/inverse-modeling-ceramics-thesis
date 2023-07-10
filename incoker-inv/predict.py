import argparse

import joblib
import numpy as np


def main(microstructure):
    models = joblib.load("trained_models.joblib")["models"]
    model = models['thermal_expansion']['pipe']
    features = models['thermal_expansion']['features']

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
    parser.add_argument('prop', type=str,
                        help='Expected property')
    args = parser.parse_args()

    import json

    dict = json.loads(args.prop)
    args = parser.parse_args()

    main(dict)