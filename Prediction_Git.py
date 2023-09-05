import numpy as np
import argparse
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('-sessiondir', dest='sessiondir', help='Session Dir', nargs='?', default='.')
parser.add_argument('-modelName', dest='modelName', help='Name of the model, see models.py', nargs='?',
                    default='model_lstm')
args, unknown = parser.parse_known_args()

# This decodes the np array of features generated for training the model

def Anne_decode(numpy_array):
    int_to_nuc = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    decoded_sequences = []
    for one_hot_seq in numpy_array:
        # Get the index of the max value along the last axis (i.e. for each nucleotide)
        max_indices = np.argmax(one_hot_seq, axis=-1)
        # Map the integers back to nucleotides using the dictionary
        decoded_seq = ''.join([int_to_nuc[i] for i in max_indices])
        # Add the decoded sequence to the list
        decoded_sequences.append(decoded_seq)
    return decoded_sequences


def predict_and_format(prediction_features, threshold):
    # Assuming model.predict gives probabilities for both classes
    # And we are interested in the second class (index 1)
    prediction_probs = model.predict(prediction_features)[:, 1]

    # Decode the sequences
    decoded_sequences = Anne_decode(prediction_features)

    # Initialize result list
    formatted_results = []

    for seq, prob in zip(decoded_sequences, prediction_probs):
        # Apply the threshold to make a binary prediction
        pred_class = 1 if prob > threshold else 0

        # Format the output string
        formatted_output = f"{seq}: {pred_class}"

        # Add to the result list
        formatted_results.append(formatted_output)

    return formatted_results


checkpoint_filepath = args.sessiondir + '/' + args.modelName + '.h5'

model = load_model(checkpoint_filepath)

# Here you can load any sequences, properly one-hot-encoded, and run a prediction on them

prediction_features = np.load('prediction_features.npy')

# Set any threshold you want

threshold = 0.62

formatted_results = predict_and_format(prediction_features, threshold)

for res in formatted_results:
    print(res)