import numpy as np
import librosa
import pickle
import sounddevice as sd
from scipy.io.wavfile import write

# Load the saved model from file
with open('digit_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

max_length = 32
sr = 16000
def detect_digit(num_seconds=1):
    # SET A DEVICE IN OUR CASE USB CAMERA DEVICE NUMBER 20, to query use sd.query_devices()
    sd.default.device = 20
    # Record audio for num_seconds seconds
    recording = sd.rec(int(num_seconds * sr), samplerate=sr, channels=1)
    sd.wait()
    write('output.wav', 44100, recording)
    
    # Extract MFCC features from recorded audio
    mfccs = librosa.feature.mfcc(recording.flatten(), sr=sr, n_mfcc=13)
    # Truncate mfccs if necessary
    if mfccs.shape[1] > max_length:
        mfccs = mfccs[:, :max_length]
    
    # Flatten and pad the feature vector
    mfccs_flat = np.pad(mfccs.flatten(), (0, max_length * 13 - len(mfccs.flatten())), mode='constant')
    # Reshape the feature vector to have the same shape as the feature vectors used to train the SVC model
    mfccs_flat = mfccs_flat.reshape(1, -1)
    
    # Use the trained model to predict the label of the recorded audio
    label = clf.predict(mfccs_flat)[0]
    
    # Calculate the probability of the predicted label
    proba = np.max(clf.predict_proba(mfccs_flat))
    
    # If the probability is greater than a threshold, return the label
    print(proba, label, '\n\n')
    # if proba > 0.9:
    #     return label
    # else:
    #     return None
    if proba > 0.7:
        return label
    else:
        return None

# Test the detect_digit function
while True:
    label = detect_digit()
    if label:
        print('Detected digit:', label)
        break