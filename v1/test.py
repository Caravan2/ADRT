import os
import random
import tensorflow as tf
import numpy as np

import librosa

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

digit_labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
labels = digit_labels

# Load the audio file and convert to mel-spectrogram
folder = random.choice(digit_labels)
file = random.choice(os.listdir(f'dataset/{folder}'))
file_path = f'dataset/{folder}/{file}'
# file_path  = "1.m4a"

y, sr = librosa.load(file_path)
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128, fmax=8000)
mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

# Split the mel-spectrogram into 1 second chunks
mel_chunks = librosa.util.frame(mel_spec, frame_length=32, hop_length=64, axis=1)
X = []
for chunk in mel_chunks.T:
    chunk = np.expand_dims(chunk, axis=-1) # Add a third dimenstion to chunk
    chunk_resized = tf.image.resize(chunk, (32, 32)).numpy() # Resize the mel-spectrogram chunk to 32x32

    # Add the mel-spectrogram chunk to the input data
    X.append(chunk_resized)

# Convert the input data to a numpy array and add a batch dimension
X = np.array(X) / 255.0
X = X.reshape(-1, 32, 32, 1)

# LOAD THE MODEL
model = tf.keras.models.load_model("Digit_Recognition_0_0_1.h5")

# Use the trained model to predict the letter
y_pred = model.predict(X)
letter_index = np.argmax(y_pred.mean(axis=0))
print(letter_index)
letter = labels[letter_index]

print('Predicted letter:', letter, "Real letter:", folder)