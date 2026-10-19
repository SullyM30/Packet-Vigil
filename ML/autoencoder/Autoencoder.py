# Autoencoder.py — Feed-forward denoising autoencoder for reconstructing
# benign network traffic.  Trained on normal packets so that anomalous
# (attack) packets produce high reconstruction error.

import torch
import torch.nn as nn
import scripts.autoencoder.AE_config as cfg


class Autoencoder(nn.Module):
    """Symmetric feed-forward autoencoder.

    Encoder compresses input_dim -> hidden_dims -> latent_dim.
    Decoder mirrors the encoder in reverse to reconstruct the original input.
    A sigmoid is applied to the latent representation so that a KL-divergence
    sparsity penalty can be used during training (see AE_train.py).
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int]):
        super(Autoencoder, self).__init__()

        # ── Build encoder: input -> hidden layers -> latent bottleneck ──
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(cfg.DROPOUT_P))
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ── Build decoder: latent -> hidden layers (reversed) -> input dim ──
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        self.latent = torch.sigmoid(latent)  # Stored for sparsity penalty in training
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
