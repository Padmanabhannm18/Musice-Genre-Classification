import streamlit as st
import pandas as pd
from dataset import generate_dataset, load_dataset, training_features, training_labels, test_features, test_labels
from models import train_models, predict_genre, model_performance
from utils import get_best_model, save_uploaded_file, GENRE_LABELS

# Streamlit UI
st.title("🎵 Music Genre Classification with Multiple Models")

# Generate dataset
generate_dataset()
load_dataset("my1.dat")

# Train models
train_models(training_features, training_labels, test_features, test_labels)

# File uploader
uploaded_file = st.file_uploader("Upload a song", type=["wav"])
if uploaded_file:
    temp_audio_path = save_uploaded_file(uploaded_file)
    st.audio(temp_audio_path, format="audio/wav")
    
    if st.button("Predict Genre"):
        try:
            st.subheader("Model Performance")
            performance_data = {
                "Model": [name for name in model_performance.keys()],
                "Accuracy (%)": [accuracy for _, accuracy in model_performance.values()]
            }
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df)

            # Find the best model
            best_model_name, best_model, best_accuracy = get_best_model(model_performance)

            st.success(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.2f}%")

            # Predict the genre using the best model
            genre = predict_genre(temp_audio_path, best_model)
            genre_name = GENRE_LABELS.get(genre, "Unknown")
            st.success(f"Predicted Genre: {genre_name}")
        except Exception as e:
            st.error(f"Error predicting genre: {e}")
