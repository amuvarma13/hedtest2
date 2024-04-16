import wave
import numpy as np
import librosa

def load_wav_as_numpy(filename):
    audio, sr = librosa.load(filename, sr=16000)
    return audio
