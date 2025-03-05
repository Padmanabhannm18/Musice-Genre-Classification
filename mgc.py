import streamlit as st
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
# Global variables
dataset = []
training_features = []
training_labels = []
test_features = []
test_labels = []
model_performance = {}

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
def loadDataset(filename, split=0.7):
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

# Train multiple models
def train_models():
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
        accuracy = accuracy*100
        model_performance[name] = (model, accuracy)

# Predict genre using the best model
def predict_genre(audio_path, best_model):
    rate, sig = wav.read(audio_path)
    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    feature = mfcc_feat.mean(0)
    prediction = best_model.predict([feature])[0]
    return prediction

# Streamlit UI
st.title("🎵 Music Genre Classification with Multiple Models")

# Generate dataset
generate_dataset()
loadDataset("my1.dat")
train_models()

# Display model performances

# Upload and predict
uploaded_file = st.file_uploader("Upload a song", type=["wav"])
if uploaded_file:
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    st.audio(temp_audio_path, format="audio/wav")
    
    if st.button("Predict Genre"):
        try:
            st.subheader("Model Performance")
            performance_data = {
                "Model": [name for name in model_performance.keys()],
                "Accuracy (%)": [accuracy for _, accuracy in model_performance.values()]
            }
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df)  # Display the DataFrame in Streamlit

            # Find the best model
            best_model_name = max(model_performance, key=lambda name: model_performance[name][1])
            best_model, best_accuracy = model_performance[best_model_name]

            st.success(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.2f}%")

            # Predict the genre using the best model
            genre = predict_genre(temp_audio_path, best_model)
            genres = {1: "blues", 2: "classical", 3: "country", 4: "disco", 5: "hiphop",
                    6: "jazz", 7: "metal", 8: "pop", 9: "reggae", 10: "rock"}
            genre_name = genres.get(genre, "Unknown")
            st.success(f"Predicted Genre: {genre_name}")
        except Exception as e:
            st.error(f"Error predicting genre: {e}")

