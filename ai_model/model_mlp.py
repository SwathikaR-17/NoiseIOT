import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
ESC50_CSV = r"ESC-50-master/meta/esc50.csv"
AUDIO_DIR = r"ESC-50-master/audio"

# -----------------------------
# LOAD METADATA
# -----------------------------
print("Loading metadata...")
df = pd.read_csv(ESC50_CSV)

labels = df['category'].values
files = df['filename'].values

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

print("Extracting features...")

X = []
y_clean = []

for filename, label_idx in zip(files, y):
    full_path = os.path.join(AUDIO_DIR, filename)

    feat = extract_features(full_path)
    if feat is not None:
        X.append(feat)
        y_clean.append(label_idx)

X = np.array(X)
y_clean = np.array(y_clean)

print(f"Features extracted: {X.shape}")

# -----------------------------
# TRAIN/TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_clean, test_size=0.2, random_state=42, stratify=y_clean
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# BUILD SIMPLE MLP
# -----------------------------
model = Sequential([
    Dense(256, activation='relu', input_shape=(40,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y_clean)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# -----------------------------
# TRAIN
# -----------------------------
print("Training model...")
model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# -----------------------------
# EVALUATE
# -----------------------------
preds = model.predict(X_test)
pred_labels = np.argmax(preds, axis=1)

acc = accuracy_score(y_test, pred_labels)
print(f"\nFinal Accuracy: {acc * 100:.2f}%")

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("noise_mlp_model.h5")
print("Model saved as noise_mlp_model.h5")

