import os
import pickle
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc

# Global variables
dataset = []
training_features = []
training_labels = []
test_features = []
test_labels = []

# Generate dataset if not exists
def generate_dataset():
    directory = "Data/genres_original"
    output_file = "my1.dat"
    if not os.path.exists(output_file):
        with open(output_file, "wb") as fS:
            genre_label = 0
            for genre in os.listdir(directory):
                genre_label += 1
                if genre_label > 10:
                    break
                for file in os.listdir(os.path.join(directory, genre)):
                    try:
                        rate, sig = wav.read(os.path.join(directory, genre, file))
                        mfcc_feat = mfcc(sig, rate, winlen=0.020, nfft=1024, appendEnergy=False)
                        mean_matrix = mfcc_feat.mean(0)
                        feature_data = (mean_matrix, genre_label)
                        pickle.dump(feature_data, fS)
                    except Exception as e:
                        print(f"Error processing {file} in {genre}: {e}")

# Load dataset and split into training and testing sets
def load_dataset(filename, split=0.7):
    with open(filename, 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                break
    for features, _, label in dataset:  # Unpacking correctly
        if np.random.rand() < split:
            training_features.append(features)
            training_labels.append(label)
        else:
            test_features.append(features)
            test_labels.append(label)

