import os
import random
import numpy as np
import librosa
import pickle


model_pickle_name = "digit_classifier.pkl"
max_length = 32

def load_the_model():
    with open(model_pickle_name, 'rb') as f:
        clf = pickle.load(f)
    return clf


def open_file_and_get_mfccs_flat(filename):
    signal, sr = librosa.load(filename, sr=16000)
    mfccs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=13)

    mfccs_flat = np.pad(mfccs.flatten(), (0, max_length * 13 - len(mfccs.flatten())), mode='constant')
    mfccs_flat = mfccs_flat.reshape(1, -1)
    return mfccs_flat


def detect_digit(clf, filename: str):
    mfccs_flat = open_file_and_get_mfccs_flat(filename)
    
    # Use the trained model to predict the label of the recorded audio
    label = clf.predict(mfccs_flat)[0]
    # Calculate the probability of the predicted label
    proba = np.max(clf.predict_proba(mfccs_flat))

    return label, proba 


filename = "output.wav"
clf = load_the_model()
predicted_digit, probability = detect_digit(clf, filename)
print(f"Predicted digit is {predicted_digit} with probability {probability}")
