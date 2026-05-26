import numpy as np
import joblib
import os

# Load extracted features
print("Loading extracted features...")
base_path = os.path.dirname(__file__)  # ai_model folder

features = np.load(os.path.join(base_path, "features", "features.npy"))
labels = np.load(os.path.join(base_path, "features", "labels.npy"))

print("Training model...")
from sklearn.svm import SVC
model = SVC(kernel="rbf", C=10)

model.fit(features, labels)

model_path = os.path.join(base_path, "models", "svm_model.pkl")
joblib.dump(model, model_path)

print("Model saved to:", model_path)
