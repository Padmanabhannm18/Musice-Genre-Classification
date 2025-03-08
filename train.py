import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from models import train_model
from utils import extract_mel_spectrogram

genres = {
    "blues": 0, "classical": 1, "country": 2, "disco": 3, "hiphop": 4,
    "jazz": 5, "metal": 6, "pop": 7, "reggae": 8, "rock": 9
}

DATASET_DIR = "Data/genres_original"

# Load dataset
features, labels = [], []
for genre, label in genres.items():
    genre_dir = os.path.join(DATASET_DIR, genre)
    for file in os.listdir(genre_dir):
        try:
            file_path = os.path.join(genre_dir, file)
            mel_spec = extract_mel_spectrogram(file_path)
            mel_spec = np.expand_dims(mel_spec, axis=-1)  # Add channel dimension
            features.append(mel_spec)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Convert to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Split dataset
x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train model
model, history = train_model(x_train, y_train, x_val, y_val)
