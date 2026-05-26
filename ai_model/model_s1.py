# model_s1.py
# Stage 1: Load ESC-50, convert each audio to mel-spec (no librosa), save dataset arrays.

import os
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from scipy import signal
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ------------------------
# CONFIG
# ------------------------
ESC_PATH = r"C:\AINoiseDetector\ESC-50-master\ESC-50-master"   # <-- confirmed by you
META_PATH = os.path.join(ESC_PATH, "meta", "esc50.csv")
AUDIO_PATH = os.path.join(ESC_PATH, "audio")

TARGET_SR = 44100           # target sample rate (ESC-50 files are typically 44100)
DURATION = 5.0              # seconds each sample (ESC-50 is 5s)
SAMPLES = int(TARGET_SR * DURATION)

N_MEL = 64
FRAME_LENGTH = 1024
FRAME_STEP = 512
FFT_LENGTH = 1024
OUT_SIZE = (128, 128)       # resize mel-spec to this (height, width)
OUTPUT_NPY = os.path.join("..", "processed_data.npz")  # saved relative to ai_model folder

# ------------------------
# Sanity checks
# ------------------------
if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"Metadata not found at {META_PATH}")
if not os.path.isdir(AUDIO_PATH):
    raise FileNotFoundError(f"Audio folder not found at {AUDIO_PATH}")

# ------------------------
# Helpers
# ------------------------
def load_and_fix_audio(path, target_sr=TARGET_SR, samples=SAMPLES):
    """Load file with soundfile, convert stereo->mono, resample if needed, pad/truncate to fixed length."""
    audio, sr = sf.read(path, always_2d=False)  # audio: (n,) or (n,2)
    # to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    # resample if needed
    if sr != target_sr:
        num_samples = int(len(audio) * target_sr / sr)
        audio = signal.resample(audio, num_samples)
        sr = target_sr
    # pad or truncate
    if len(audio) < samples:
        pad_len = samples - len(audio)
        audio = np.pad(audio, (0, pad_len))
    else:
        audio = audio[:samples]
    return audio.astype(np.float32), sr

def audio_to_melspec_np(audio, sr=TARGET_SR,
                        frame_length=FRAME_LENGTH, frame_step=FRAME_STEP,
                        fft_length=FFT_LENGTH, n_mel=N_MEL, out_size=OUT_SIZE):
    """Convert 1D numpy audio to log-mel spectrogram numpy array shape (out_size[0], out_size[1], 1)."""
    # Convert to tensorflow tensor
    audio_tf = tf.convert_to_tensor(audio, dtype=tf.float32)

    # STFT -> shape [frames, fft_bins]
    stft_tf = tf.signal.stft(audio_tf,
                             frame_length=frame_length,
                             frame_step=frame_step,
                             fft_length=fft_length,
                             window_fn=tf.signal.hann_window,
                             pad_end=False)
    spectrogram = tf.abs(stft_tf)  # magnitude

    # Mel filter
    num_spectrogram_bins = spectrogram.shape[-1]
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mel,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sr,
        lower_edge_hertz=0.0,
        upper_edge_hertz=float(sr / 2.0)
    )  # shape [spectrogram_bins, n_mel]

    # spectrogram: [frames, bins], mel = spectrogram @ mel_weights -> [frames, n_mel]
    mel_spec = tf.matmul(spectrogram, mel_weight_matrix)

    # log + small const for numerical stability
    log_mel = tf.math.log(mel_spec + 1e-6)

    # transpose so mel bins are rows (like image) -> shape [n_mel, frames]
    log_mel_T = tf.transpose(log_mel)

    # expand channel and resize to out_size
    log_mel_T_exp = tf.expand_dims(log_mel_T, axis=-1)  # [n_mel, frames, 1]
    log_mel_resized = tf.image.resize(log_mel_T_exp, out_size)

    return log_mel_resized.numpy().astype(np.float32)  # shape (out_size[0], out_size[1], 1)

# ------------------------
# Load metadata
# ------------------------
df = pd.read_csv(META_PATH)
print("Metadata loaded. Total rows:", len(df))

# ------------------------
# Build arrays (this may take some time)
# ------------------------
X_list = []
y_list = []
failed = 0

print("Converting audio → mel-spectrograms (this can take some minutes)...")
for i, row in tqdm(df.iterrows(), total=len(df)):
    fname = row['filename']
    label = int(row['target'])
    fpath = os.path.join(AUDIO_PATH, fname)
    try:
        audio, sr = load_and_fix_audio(fpath)
        mel = audio_to_melspec_np(audio, sr=sr)
        X_list.append(mel)
        y_list.append(label)
    except Exception as e:
        failed += 1
        print(f"Failed {fname}: {e}")

X = np.stack(X_list, axis=0)
y = np.array(y_list, dtype=np.int32)

print(f"Converted: X.shape={X.shape}, y.shape={y.shape}, failed={failed}")

# ------------------------
# Save processed arrays for fast reuse
# ------------------------
np.savez_compressed(OUTPUT_NPY, X=X, y=y)
print("Saved processed data to:", OUTPUT_NPY)

# ------------------------
# Quick train/test split (optional)
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
print("Train/test split ->", X_train.shape, X_test.shape)
