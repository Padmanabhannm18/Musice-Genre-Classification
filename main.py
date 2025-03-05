import streamlit as st
import pandas as pd
import os
from dataset import generate_dataset, load_dataset, training_features, training_labels, test_features, test_labels
from models import train_models, predict_genre, get_best_model, model_performance

# Define genre mapping
genres = {
    1: "blues", 2: "classical", 3: "country", 4: "disco", 5: "hiphop",
    6: "jazz", 7: "metal", 8: "pop", 9: "reggae", 10: "rock"
}

if __name__ == "__main__":
    st.title("🎵 Music Genre Classification with Multiple Models")

    # Generate and load dataset
    if not os.path.exists("my1.dat"):
        generate_dataset()
    load_dataset()

    # Train models
    train_models(training_features, training_labels, test_features, test_labels)

    # File uploader for music
    uploaded_file = st.file_uploader("Upload a song", type=["wav"])
    
    if uploaded_file:
        temp_audio_path = "temp_audio.wav"
        with open(temp_audio_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        st.audio(temp_audio_path, format="audio/wav")

        # Predict genre when the button is clicked
        if st.button("Predict Genre"):
            try:
                # Display model performance
                st.subheader("Model Performance")
                performance_df = pd.DataFrame({
                    "Model": [name for name in model_performance.keys()],
                    "Accuracy (%)": [accuracy for _, accuracy in model_performance.values()]
                })
                st.dataframe(performance_df)

                # Get best model
                best_model_name, (best_model, best_accuracy) = get_best_model()
                st.success(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.2f}%")

                # Predict genre
                genre = predict_genre(temp_audio_path, best_model)
                genre_name = genres.get(genre, "Unknown")
                st.success(f"Predicted Genre: {genre_name}")
            except Exception as e:
                st.error(f"Error predicting genre: {e}")
