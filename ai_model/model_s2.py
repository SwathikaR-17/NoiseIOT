import os
import numpy as np
import librosa

# Paths
DATASET_PATH = "ESC-50-master/ESC-50-master/audio/"
META_PATH = "ESC-50-master/ESC-50-master/meta/esc50.csv"

# Load metadata file
import pandas as pd
metadata = pd.read_csv(META_PATH)

features = []
labels = []

print("➡ Starting MFCC feature extraction...")

for index, row in metadata.iterrows():
    file_name = row['filename']
    label = row['category']

    file_path = os.path.join(DATASET_PATH, file_name)

    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)

        # Extract MFCCs (most common 40)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Take mean over time so it becomes (40,)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        features.append(mfcc_mean)
        labels.append(label)

    except Exception as e:
        print("Error processing:", file_name, "->", e)

print("✔ Feature extraction completed!")
print("Total files processed:", len(features))

# Convert to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Save features
np.save("features.npy", X)
np.save("labels.npy", y)

print("✔ Saved: features.npy and labels.npy")
