from pathlib import Path
import main
import pickle


def predict(X):
    # Load the model and scaler from the saved file
    with open(str(Path(__file__).parents[1] / 'code/model.pickle'), 'rb') as f:
        model, label_encoders, scaler = pickle.load(f)

    main.cleaning_steps(X)  # Perform Cleaning
    main.perform_feature_engineering(X)  # Perform Feature Engineering
    # Label Encoding
    for column, label_encoder in label_encoders.items():
        X[column] = label_encoder.transform(X[column])
    X = scaler.transform(X)  # Standardize
    pred = model.predict(X)
    return pred
