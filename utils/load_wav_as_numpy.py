import wave
import numpy as np

def load_wav_as_numpy(filename):
    # Open the WAV file
    with wave.open(filename, 'r') as wav_file:
        # Extract Audio Parameters
        n_channels = wav_file.getnchannels()
        n_frames = wav_file.getnframes()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()

        # Read audio frames as bytes
        audio_frames = wav_file.readframes(n_frames)

        # Convert bytes to numpy array based on sample width
        if sample_width == 1:
            # 8-bit audio
            dtype = np.uint8
        elif sample_width == 2:
            # 16-bit audio
            dtype = np.int16
        elif sample_width == 4:
            # 32-bit audio
            dtype = np.int32
        else:
            raise ValueError("Unsupported sample width")

        # Create a numpy array from bytes, using the correct data type
        data = np.frombuffer(audio_frames, dtype=dtype)

        # If stereo, convert shape to (n_frames, n_channels)
        if n_channels > 1:
            data = np.reshape(data, (n_frames, n_channels))

    return data


