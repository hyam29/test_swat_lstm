# src/models/lstm.py
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMAutoEncoder(nn.Module):
    """
    LSTM AutoEncoder for multivariate time-series reconstruction.

    Input:  (B, T, F)
    Output: (B, T, F)
    """
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder: sequence -> last hidden state
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder: repeat latent across time -> reconstruct sequence
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to be 3D (B,T,F), got shape={tuple(x.shape)}")

        B, T, F = x.shape
        if F != self.n_features:
            raise ValueError(f"Expected feature dim F={self.n_features}, got F={F}")

        # Encode
        _, (h_n, _) = self.encoder(x)      # h_n: (L, B, H)
        latent = h_n[-1]                   # (B, H)

        # Decode
        dec_in = latent.unsqueeze(1).repeat(1, T, 1)   # (B, T, H)
        dec_out, _ = self.decoder(dec_in)              # (B, T, H)
        x_hat = self.output_layer(dec_out)             # (B, T, F)
        return x_hat
