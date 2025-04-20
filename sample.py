def predict_disease_probabilities(models, clf, scaler, user_input):
    # Ensure the user input is a batch of data (even if it's just one sample)
    user_input = user_input.reshape(1, -1)

    # Scale the input data
    user_input_scaled = scaler.transform(user_input)

    # Convert to tensor for model processing
    user_input_tensor = torch.tensor(user_input_scaled, dtype=torch.float32)

    # Get the encoded features from the models
    encoded_features = []
    for model in models:
        model.eval()  # Set to evaluation mode
        with torch.no_grad():
            encoded, _ = model(user_input_tensor)
        encoded_features.append(encoded.numpy())

    # Stack all encoded features
    encoded_features = np.hstack(encoded_features)

    # Predict the disease probabilities
    probabilities = clf.predict_proba(encoded_features)[0]

    # Verify we have probabilities for all classes
    if len(probabilities) != len(disease_mapping):
        print(f"Warning: Got {len(probabilities)} probabilities for {len(disease_mapping)} diseases")

    # Create a full set of probabilities for all diseases
    full_probabilities = {disease_mapping[i]: 0.0 for i in range(len(disease_mapping))}
    for class_idx, prob in zip(clf.classes_, probabilities):
        full_probabilities[disease_mapping[class_idx]] = prob

    return full_probabilities

def load_models(input_dim, hidden_dim, base_path="saved_models"):

    models = []

    # Load encoder-decoder models
    i = 0
    while True:
        model_path = os.path.join(base_path, f'encoder_decoder_{i}.pth')
        if not os.path.exists(model_path):
            break

        model = EncoderDecoder(input_dim, hidden_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models.append(model)
        i += 1

    # Load classifier
    clf_path = os.path.join(base_path, 'classifier.joblib')
    clf = joblib.load(clf_path)

    # Load scaler
    scaler_path = os.path.join(base_path, 'scaler.joblib')
    scaler = joblib.load(scaler_path)

    return models, clf, scaler

import numpy as np
import torch
import joblib
from pathlib import Path

# Constants
hidden_dim = 128  # Same as used during training
input_dim = 5     # Example dimension, adjust based on your actual input dimension

def load_and_predict(input_values):
    """
    Load saved models and make predictions with provided input values.

    Parameters:
    input_values (list): List of float values for prediction
    """
    try:
        # Load saved models
        models, clf, scaler = load_models(input_dim, hidden_dim, "saved_models")
        print("Models loaded successfully!")

        # Convert input to numpy array and reshape
        user_input = np.array(input_values, dtype=np.float32)

        # Ensure correct dimensionality
        if len(user_input) < input_dim:
            user_input = np.pad(user_input, (0, input_dim - len(user_input)), mode='constant')
        elif len(user_input) > input_dim:
            user_input = user_input[:input_dim]

        # Get predictions
        disease_probabilities = predict_disease_probabilities(models, clf, scaler, user_input)

        # Sort probabilities by value in descending order
        sorted_probs = dict(sorted(disease_probabilities.items(),
                                 key=lambda x: x[1],
                                 reverse=True))

        # Print results
        print("\nDisease Probability Rankings:")
        print("-" * 50)
        for disease, prob in sorted_probs.items():
            probability_percentage = prob * 100
            print(f"{disease:<25}: {probability_percentage:>6.2f}%")

    except FileNotFoundError:
        print("Error: Model files not found. Please ensure models are saved in the 'saved_models' directory.")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

# Example usage with sample input
if __name__ == "__main__":
    # Sample input values (example values, adjust based on your actual input requirements)
    sample_input = [2.13216596653557,	1.15657,	2.85210280364193	]

    print("Making predictions with sample input:", sample_input)
    print("-" * 50)
    load_and_predict(sample_input)


