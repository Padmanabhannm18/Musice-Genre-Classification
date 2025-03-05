import scipy.io.wavfile as wav
from python_speech_features import mfcc
import numpy as np

def extract_features(audio_path):
    """Extracts MFCC features from an audio file."""
    rate, sig = wav.read(audio_path)
    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    return mfcc_feat.mean(0)
