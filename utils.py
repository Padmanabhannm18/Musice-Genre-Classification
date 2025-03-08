import librosa
import numpy as np

def extract_mel_spectrogram(audio_path, n_mels=128, n_fft=2048, hop_length=512):
    """Extracts Mel Spectrogram features from an audio file."""
    y, sr = librosa.load(audio_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db
