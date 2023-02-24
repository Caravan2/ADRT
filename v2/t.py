import os
import librosa
import numpy as np



data_dir = '../dataset'

# List of digit labels
labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

max_length = 0

# Loop over all digit recordings
for label in labels:
    digit_dir = os.path.join(data_dir, label)
    for file in os.listdir(digit_dir)[:10]:
        if file.endswith('.wav'):
            # Load the audio file
            audio_path = os.path.join(digit_dir, file)
            audio, sr = librosa.load(audio_path, sr=44100, mono=True)

            # Extract MFCC features and flatten the resulting feature vector
            mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
            mfccs_flat = mfccs.flatten()

            # Update the maximum length if necessary
            if len(mfccs_flat) > max_length:
                max_length = len(mfccs_flat)

# Print the maximum length
print(f"Maximum length of flattened MFCC feature vectors: {max_length}")