from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc

# Dictionary to store model performance
model_performance = {}

# Train multiple models
def train_models(training_features, training_labels, test_features, test_labels):
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=300),
        "SVM": SVC(kernel='linear', probability=True),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300),
        "Naive Bayes": GaussianNB(),
        "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(300,), max_iter=500)
    }
    for name, model in models.items():
        model.fit(training_features, training_labels)
        predictions = model.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        model_performance[name] = (model, accuracy * 100)

# Predict genre using the best model
def predict_genre(audio_path, best_model):
    rate, sig = wav.read(audio_path)
    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    feature = mfcc_feat.mean(0)
    prediction = best_model.predict([feature])[0]
    return prediction
