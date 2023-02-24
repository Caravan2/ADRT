import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import librosa
import numpy as np

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

MIN_DURATION = 1.0

# Load the Speech Commands Dataset
X_train = []
y_train = []
X_test = []
y_test = []
digit_labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
letter_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
labels = digit_labels

# Loop through the audio files
for label in labels:
    for file in os.listdir(f"dataset/{label}")[:1000]:
        # Load the audio file
        file_path = os.path.join(f"dataset/{label}", file)
        audio, sr = librosa.load(file_path)

        # Skip any audio files that are too short
        if len(audio) < MIN_DURATION * sr:
            continue

        # Convert the audio to a mel spectrogram
        spectrogram = librosa.feature.melspectrogram(audio, sr=sr, n_fft=1024, hop_length=512, n_mels=128, fmax=8000)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        # Split the spectrogram into 1-second chunks 
        # int(MIN_DURATION * sr / 512)
        chunks = librosa.util.frame(spectrogram, frame_length=32, hop_length=int(MIN_DURATION * sr / 512), axis=1)

        # Add each chunk to the training or testing data and label
        for chunk in chunks.T:
            chunk = np.expand_dims(chunk, axis=-1)  # Add a third dimension to chunk
            chunk_resized = tf.image.resize(chunk, (32, 32)).numpy()  # Resize the mel-spectrogram chunk to 32x32
            if np.random.rand() < 0.8:
                X_train.append(chunk_resized)
                y_train.append(labels.index(label))
            else:
                X_test.append(chunk_resized)
                y_test.append(labels.index(label))

# Convert the data to numpy arrays and normalize
X_train = np.array(X_train) / 255.0
y_train = tf.keras.utils.to_categorical(np.array(y_train))
X_test = np.array(X_test) / 255.0
y_test = tf.keras.utils.to_categorical(np.array(y_test))

# Define the model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(26, activation='softmax'))


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_data=(X_test, y_test), verbose=2)


# Save the Model
model.save("Digit_Recognition_0_0_1.h5")