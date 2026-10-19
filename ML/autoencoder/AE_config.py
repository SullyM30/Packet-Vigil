# AE_config.py — Central configuration for the Autoencoder model.
# All hyperparameters, file paths, and device settings live here so
# that AE_train.py, Autoencoder.py, and the online pipeline can share
# a single source of truth.

import os
import torch

# ── Project paths ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATA_DIR = os.path.join(PROJECT_ROOT, "Train and Test files")
MODEL_DIR = os.path.join(PROJECT_ROOT, "scripts", "autoencoder")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

TRAIN_FILE = os.path.join(DATA_DIR, "AE_TRAINFILE.csv")
TEST_FILE = os.path.join(DATA_DIR, "X_test.csv")

# ── Model architecture ───────────────────────────────────────────────
INPUT_DIM = 115              # Number of AfterImage features per packet
LATENT_DIM = 8               # Bottleneck size
HIDDEN_DIMS = [64, 36]       # Encoder hidden-layer widths (decoder mirrors in reverse)

# ── Training hyperparameters ─────────────────────────────────────────
BATCH_SIZE = 1024
LEARNING_RATE = 1e-04
EPOCHS = 8
VALIDATION_SPLIT = 0.2
SEED = 1124                  # Reproducibility seed (best found for full dataset)
SPARSITY_TARGET = 0.05       # Desired average activation per latent neuron
SPARSITY_WEIGHT = 1e-3       # KL-divergence penalty weight for sparsity
DROPOUT_P = 0.1              # Dropout probability in encoder layers
NOISE_STD = 0.1              # Std-dev of Gaussian noise added during denoising training

# ── Device setup ─────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.xpu.is_available():
    DEVICE = torch.device("xpu")
else:
    DEVICE = torch.device("cpu")

print(f"[INFO] Using device: {DEVICE}")

# ── Output paths ─────────────────────────────────────────────────────
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "autoencoder.pt")
SCALER_SAVE_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
