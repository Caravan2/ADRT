import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import sounddevice as sd

# Path to directory containing digit recordings
data_dir = '../dataset'

# List of digit labels
labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# Initialize empty lists to store features and labels
features = []
target = []

# Loop over all digit recordings
for label in labels:
    digit_dir = os.path.join(data_dir, label)
    for file in os.listdir(digit_dir):
        if file.endswith('.wav'):
            file_path = os.path.join(digit_dir, file)
            # Load audio file and extract MFCC features
            signal, sr = librosa.load(file_path, sr=16000)
            mfccs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=13)
            # Pad MFCC feature arrays with zeros to have the same shape
            max_length = 32
            mfccs_padded = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
            # Add MFCC features and label to features and target lists
            features.append(mfccs_padded)
            target.append(label)

# Calculate maximum length of all MFCC feature arrays
max_length = max(len(mfccs_flat) for mfccs_flat in features)
# Pad MFCC feature arrays with zeros to have the same shape
features = [np.pad(mfccs_flat, (0, max_length - len(mfccs_flat)), mode='constant') for mfccs_flat in features]
# Convert features and target lists to numpy arrays
features = np.stack(features)
n_samples, n_steps, n_features = features.shape
features = features.reshape(n_samples, n_steps * n_features)
target = np.array(target)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train SVM classifier
clf = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
clf.fit(X_train, y_train)

# Test the classifier on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy of the classifier
accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)

# Save trained model to file
with open('digit_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

# # Load the saved model from file
# with open('digit_classifier.pkl', 'rb') as f:
#     clf = pickle.load(f)

# max_length = 32
# def detect_digit(num_seconds=3):
#     # Record audio for num_seconds seconds
#     recording = sd.rec(int(num_seconds * 44100), samplerate=44100, channels=1)
#     sd.wait()
    
#     # Extract MFCC features from recorded audio
#     mfccs = librosa.feature.mfcc(recording.flatten(), sr=44100, n_mfcc=13)
#     # Truncate mfccs if necessary
#     if mfccs.shape[1] > max_length:
#         mfccs = mfccs[:, :max_length]
    
#     # Flatten and pad the feature vector
#     mfccs_flat = np.pad(mfccs.flatten(), (0, max_length * 13 - len(mfccs.flatten())), mode='constant')
#     # Reshape the feature vector to have the same shape as the feature vectors used to train the SVC model
#     mfccs_flat = mfccs_flat.reshape(1, -1)
    
#     # Use the trained model to predict the label of the recorded audio
#     label = clf.predict(mfccs_flat)[0]
    
#     # Calculate the probability of the predicted label
#     proba = np.max(clf.predict_proba(mfccs_flat))
    
#     # If the probability is greater than a threshold, return the label
#     print(proba, '\n\n')
#     if proba > 0.9:
#         return label
#     else:
#         return None

# # Test the detect_digit function
# while True:
#     label = detect_digit()
#     if label:
#         print('Detected digit:', label)
#         break