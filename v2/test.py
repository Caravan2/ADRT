import os
import random
import numpy as np
import librosa
import pickle


# Path to directory containing digit recordings
data_dir = '../dataset'

# List of digit labels
labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

model_pickle_name = "digit_classifier.pkl"
max_length = 32


def load_the_model():
    with open(model_pickle_name, 'rb') as f:
        clf = pickle.load(f)
    return clf


def open_random_file_and_get_mfccs_flat(digit):
    digit_dir = os.path.join(data_dir, digit)
    file_dir = random.choice(os.listdir(digit_dir))
    file_path = os.path.join(digit_dir, file_dir)

    signal, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=13)

    mfccs_flat = np.pad(mfccs.flatten(), (0, max_length * 13 - len(mfccs.flatten())), mode='constant')
    mfccs_flat = mfccs_flat.reshape(1, -1)
    return mfccs_flat


def detect_digit(clf, digit: str):
    mfccs_flat = open_random_file_and_get_mfccs_flat(digit)
    
    # Use the trained model to predict the label of the recorded audio
    label = clf.predict(mfccs_flat)[0]
    
    # Calculate the probability of the predicted label
    proba = np.max(clf.predict_proba(mfccs_flat))

    return label, proba 


total = 10000
correct_guess = 0
wrong_guess = 0

clf = load_the_model()

for i in range(total):
    actual_digit = random.choice(labels)
    predicted_digit, probability = detect_digit(clf, actual_digit)
    print(f"Actual digit is {actual_digit}, Predicted digit is {predicted_digit} with probability {probability}")

    if actual_digit == predicted_digit:
        correct_guess += 1
    else:
        wrong_guess += 1

print('\n\n')
print(f"Correct Guesses {correct_guess}/{total}. Accuracy {correct_guess/total}")
print(f"Wrong Guesses {wrong_guess}/{total}. Loss {wrong_guess/total}")