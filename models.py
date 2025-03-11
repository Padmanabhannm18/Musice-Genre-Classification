from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc

# Dictionary to store model performance
model_performance = {}

# Train selected models
def train_models(training_features, training_labels, test_features, test_labels):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300),
        "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(300,), max_iter=500)
    }
    
    for name, model in models.items():
        model.fit(training_features, training_labels)
        predictions = model.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        model_performance[name] = (model, accuracy * 100)
        # print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    
    return max(model_performance.items(), key=lambda x: x[1][1])[1][0]  # Return the best model

# Predict genre using the best model
def predict_genre(audio_path, best_model):
    rate, sig = wav.read(audio_path)
    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    feature = mfcc_feat.mean(0)
    prediction = best_model.predict([feature])[0]
    return prediction
