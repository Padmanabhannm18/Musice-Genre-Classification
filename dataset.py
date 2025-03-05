import os
import pickle
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc

# Global dataset variables
dataset = []
training_features = []
training_labels = []
test_features = []
test_labels = []

def generate_dataset():
    """Generates a dataset of music genres if not already created."""
    directory = "Data/genres_original"
    output_file = "my1.dat"

    if not os.path.exists(output_file):
        with open(output_file, "wb") as f:
            genre_label = 0
            for folder in os.listdir(directory):
                genre_label += 1
                if genre_label == 11:
                    break
                for file in os.listdir(f"{directory}/{folder}"):
                    try:
                        rate, sig = wav.read(f"{directory}/{folder}/{file}")
                        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
                        covariance = np.cov(np.transpose(mfcc_feat))
                        mean_matrix = mfcc_feat.mean(0)
                        feature_data = (mean_matrix, covariance, genre_label)
                        pickle.dump(feature_data, f)
                    except Exception as e:
                        print(f"Error processing {file}: {e}")

def load_dataset(filename="my1.dat", split=0.7):
    """Loads dataset and splits into training and test sets."""
    global dataset, training_features, training_labels, test_features, test_labels
    
    with open(filename, 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                break

    for data in dataset:
        features, _, label = data
        if np.random.rand() < split:
            training_features.append(features)
            training_labels.append(label)
        else:
            test_features.append(features)
            test_labels.append(label)
