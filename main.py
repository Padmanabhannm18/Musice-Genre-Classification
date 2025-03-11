import streamlit as st
import pandas as pd
from dataset import generate_dataset, load_dataset, training_features, training_labels, test_features, test_labels
from models import train_knn, predict_genre
from utils import save_uploaded_file, GENRE_LABELS

# Streamlit UI
st.title("🎵 Music Genre Classification")

# Generate dataset
generate_dataset()
load_dataset("my1.dat")

# Train KNN model
knn_model = train_knn(training_features, training_labels)

# File uploader
uploaded_file = st.file_uploader("Upload a song", type=["wav"])
if uploaded_file:
    temp_audio_path = save_uploaded_file(uploaded_file)
    st.audio(temp_audio_path, format="audio/wav")

    if st.button("Predict Genre"):
        try:
            # Predict the genre using the trained KNN model
            genre = predict_genre(temp_audio_path, knn_model)
            genre_name = GENRE_LABELS.get(genre, "Unknown")
            st.success(f"Predicted Genre: {genre_name}")
        except Exception as e:
            st.error(f"Error predicting genre: {e}")
