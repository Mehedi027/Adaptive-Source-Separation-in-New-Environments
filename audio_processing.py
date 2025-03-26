import os
import pandas as pd
import librosa
import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Paths to the ESC-50 metadata and audio files
esc50_metadata_path = "C:/Users/mehed/OneDrive/Desktop/RME/Sem1/ADSP/Project/esc50.csv"
esc50_audio_path = "C:/Users/mehed/OneDrive/Desktop/RME/Sem1/ADSP/Project/audio"

# Load the metadata
metadata = pd.read_csv(esc50_metadata_path)

# Display the first few rows of the metadata to ensure it's loaded correctly
print(metadata.head())

# Function to extract MFCCs from audio files
def extract_mfcc(file_path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc, axis=1)  # Average over time
    return mfcc

# Extract features and labels
features = []
labels = []

print("Extracting MFCC features from audio files...")
for index, row in metadata.iterrows():
    file_path = os.path.join(esc50_audio_path, row['filename'])
    try:
        mfcc = extract_mfcc(file_path)
        features.append(mfcc)
        labels.append(row['category'])
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Convert features and labels to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

# Save category names (you can store these in a JSON file)
categories = label_encoder.classes_  # This will give you the list of categories
with open("categories.json", "w") as f:
    json.dump(categories.tolist(), f)
print("Category map saved as 'categories.json'.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, categorical_labels, test_size=0.2, random_state=42)

#RNN model
model = Sequential([
    SimpleRNN(128, input_shape=(X_train.shape[1], 1), activation='relu'),
    Dense(64, activation='relu'),
    Dense(categorical_labels.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape the input data to fit RNN's expected input shape
X_train_reshaped = X_train[..., np.newaxis]
X_test_reshaped = X_test[..., np.newaxis]

# Train the model
print("Training the model...")
history = model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, validation_data=(X_test_reshaped, y_test))

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test_reshaped, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Save the model
model.save("audio_classification_rnn.h5")
print("Model saved as 'audio_classification_rnn.h5'")
