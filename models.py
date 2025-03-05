from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

# Model performance storage
model_performance = {}

def train_models(training_features, training_labels, test_features, test_labels):
    """Trains multiple models and stores their performance."""
    global model_performance
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
        accuracy = accuracy_score(test_labels, predictions) * 100
        model_performance[name] = (model, accuracy)

def predict_genre(audio_path, best_model):
    """Predicts the genre of an uploaded song using the best model."""
    from utils import extract_features
    feature = extract_features(audio_path)
    return best_model.predict([feature])[0]

def get_best_model():
    """Returns the best performing model."""
    best_model_name = max(model_performance, key=lambda name: model_performance[name][1])
    return best_model_name, model_performance[best_model_name]
