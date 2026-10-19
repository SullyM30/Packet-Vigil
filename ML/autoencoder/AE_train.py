# AE_train.py — Training script for the denoising autoencoder.
# Loads benign-only traffic from AE_TRAINFILE.csv, fits a MinMaxScaler,
# then trains the autoencoder with Gaussian noise injection and a
# KL-divergence sparsity penalty on the latent layer.

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import joblib
from tqdm import tqdm

from scripts.autoencoder.Autoencoder import Autoencoder
import scripts.autoencoder.AE_config as cfg


def kl_divergence(p, p_hat):
    """Compute KL divergence between target sparsity `p` and observed
    average activation `p_hat`.  Used to encourage sparse latent codes."""
    return torch.sum(
        p * torch.log(p / (p_hat + 1e-10)) +
        (1 - p) * torch.log((1 - p) / (1 - p_hat + 1e-10))
    )


def load_data(file_path):
    """Read a benign-only CSV, fit a MinMaxScaler, and return a
    TensorDataset where input == target (reconstruction objective)."""
    df = pd.read_csv(file_path)
    print(df.head(1))
    print(df.dtypes)

    # Scale features to [0, 1] for stable AE training
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df.values)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Persist the scaler so the online pipeline can reuse it at inference
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, cfg.SCALER_SAVE_PATH)
    print(f"Scaler saved to {cfg.SCALER_SAVE_PATH}")

    # For an autoencoder the target is the same as the input
    dataset = TensorDataset(X_tensor, X_tensor)
    return dataset


def train():
    """Run the full training loop and save the model weights."""
    torch.manual_seed(cfg.SEED)

    dataset = load_data(cfg.TRAIN_FILE)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    model = Autoencoder(cfg.INPUT_DIM, cfg.LATENT_DIM, cfg.HIDDEN_DIMS).to(cfg.DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    model.train()
    for epoch in range(cfg.EPOCHS):
        epoch_loss = 0
        for x_batch, y_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}"):
            x_batch = x_batch.to(cfg.DEVICE)

            # Inject Gaussian noise for denoising training
            noisy_input = x_batch + torch.randn_like(x_batch) * cfg.NOISE_STD

            optimizer.zero_grad()
            reconstructed = model(noisy_input)
            recon_loss = criterion(reconstructed, x_batch)  # Reconstruct the clean input

            # Sparsity regularization via KL divergence on latent activations
            p_hat = torch.mean(model.latent, dim=0)
            sparsity_loss = kl_divergence(torch.full_like(p_hat, cfg.SPARSITY_TARGET), p_hat)

            total_loss = recon_loss + cfg.SPARSITY_WEIGHT * sparsity_loss
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        avg_activation = torch.mean(model.latent).item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{cfg.EPOCHS}] | Loss: {avg_loss:.6f} | Avg latent activation: {avg_activation:.4f}")

    # Save trained weights
    torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
    print(f"[INFO] Model saved to {cfg.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
