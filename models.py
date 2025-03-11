from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc

# Train KNN model
def train_knn(training_features, training_labels):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(training_features, training_labels)
    return knn

# Predict genre using KNN
def predict_genre(audio_path, knn_model):
    rate, sig = wav.read(audio_path)
    mfcc_feat = mfcc(sig, rate, winlen=0.020, nfft=1024, appendEnergy=False)
    feature = mfcc_feat.mean(0)
    prediction = knn_model.predict([feature])[0]
    return prediction
