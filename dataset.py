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
    directory123 = "Data/genres_original"
    output_file = "my1.dat"
    if not os.path.exists(output_file):
        with open(output_file, "wb") as fS:
            iS = 0
            for folderS in os.listdir(directory123):
                iS += 1
                if iS == 11:
                    break
                for fileS in os.listdir(directory123 + "/" + folderS):
                    try:
                        (rate, sig) = wav.read(directory123 + "/" + folderS + "/" + fileS)
                        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
                        covariance = np.cov(np.matrix.transpose(mfcc_feat))
                        mean_matrix = mfcc_feat.mean(0)
                        featureS = (mean_matrix, covariance, iS)
                        pickle.dump(featureS, fS)
                    except Exception as e:
                        print("Got an exception:", e, "in folder:", folderS, "filename:", fileS)

# Load dataset and split into training and testing sets
def load_dataset(filename, split=0.7):
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

