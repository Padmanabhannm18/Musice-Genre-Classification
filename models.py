import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def build_cnn_model(input_shape=(128, 128, 1), num_classes=10):
    """Builds a Convolutional Neural Network model."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model(x_train, y_train, x_val, y_val, epochs=20, batch_size=32):
    """Trains the CNN model."""
    model = build_cnn_model()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size)
    model.save("genre_classification_model.h5")
    return model, history
