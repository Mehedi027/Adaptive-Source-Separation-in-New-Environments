import os
import librosa
import numpy as np
import json
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("audio_classification_rnn.h5")
print("Model loaded successfully.")

# Load the category names
with open("categories.json", "r") as f:
    category_map = json.load(f)
print("Category map loaded successfully.")


# Function to extract MFCC features for inference
def extract_mfcc(file_path, sr=16000, n_mfcc=13):
    try:
        y, _ = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.mean(mfcc, axis=1)  # Average over time
        return mfcc
    except Exception as e:

        print(f"Error processing file {file_path}: {e}")
        return None


# Path to the new audio file (change this to the actual file you want to test)
new_audio_path = "C:/Users/mehed/OneDrive/Desktop/RME/Sem1/ADSP/Project/sample_classroom_audio.wav"  # Replace with your audio file path

# Extract MFCC features from the new audio file
mfcc_features = extract_mfcc(new_audio_path)
if mfcc_features is not None:
    mfcc_features = np.expand_dims(mfcc_features, axis=(0, -1))  # Prepare for RNN
    prediction = model.predict(mfcc_features)
    predicted_class_index = np.argmax(prediction, axis=-1)[0]
    predicted_category = category_map[predicted_class_index]

    print(f"Predicted category: {predicted_category}")
else:
    print("Failed to extract MFCC features.")
