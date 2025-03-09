import os

# Function to get best performing model
def get_best_model(model_performance):
    best_model_name = max(model_performance, key=lambda name: model_performance[name][1])
    best_model, best_accuracy = model_performance[best_model_name]
    return best_model_name, best_model, best_accuracy

# Function to save uploaded file
def save_uploaded_file(uploaded_file, temp_audio_path="temp_audio.wav"):
    with open(temp_audio_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    return temp_audio_path

# Mapping of genre labels
GENRE_LABELS = {
    1: "blues", 2: "classical", 3: "country", 4: "disco", 5: "hiphop",
    6: "jazz", 7: "metal", 8: "pop", 9: "reggae", 10: "rock"
}
