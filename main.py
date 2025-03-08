import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
from utils import extract_mel_spectrogram

# Load trained model
model = tf.keras.models.load_model("genre_classification_model.h5")

genres = {
    0: "blues", 1: "classical", 2: "country", 3: "disco", 4: "hiphop",
    5: "jazz", 6: "metal", 7: "pop", 8: "reggae", 9: "rock"
}

st.title("🎵 Deep Learning Music Genre Classification")

uploaded_file = st.file_uploader("Upload a song", type=["wav"])

if uploaded_file:
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    st.audio(temp_audio_path, format="audio/wav")

    if st.button("Predict Genre"):
        try:
            mel_spec = extract_mel_spectrogram(temp_audio_path)
            mel_spec = np.expand_dims(mel_spec, axis=(0, -1))  # Add batch & channel dimension

            prediction = model.predict(mel_spec)
            predicted_genre = genres[np.argmax(prediction)]

            st.success(f"Predicted Genre: {predicted_genre}")
        except Exception as e:
            st.error(f"Error predicting genre: {e}")
