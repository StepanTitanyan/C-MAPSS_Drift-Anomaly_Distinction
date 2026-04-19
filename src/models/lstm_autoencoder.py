"""
src/models/lstm_autoencoder.py
==============================
LSTM Autoencoder for reconstruction-based anomaly detection.

Architecture:
    Encoder: LSTM processes input sequence → compress to latent vector
    Decoder: LSTM reconstructs the original sequence from the latent vector

Anomaly detection logic:
    Train on normal sequences → learns to reconstruct normal patterns well.
    At test time, abnormal sequences have higher reconstruction error.

Why include this alongside forecasting models?
    Some anomaly types (e.g., subtle coordinated multi-sensor shifts) may
    not produce large forecasting errors but may produce large reconstruction
    errors. Having both detection paradigms strengthens the evaluation.
"""

import torch
import torch.nn as nn
from typing import Tuple


class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder.

    Encoder compresses the input window into a fixed-size latent vector.
    Decoder reconstructs the input window from that vector.

    Parameters
    ----------
    input_size : int
        Number of sensors.
    hidden_size : int
        LSTM hidden dimension.
    latent_dim : int
        Size of the bottleneck representation.
    num_layers : int
        LSTM layers in encoder and decoder.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Compress hidden state to latent
        self.to_latent = nn.Linear(hidden_size, latent_dim)

        # Expand latent back to hidden for decoder initialization
        self.from_latent = nn.Linear(latent_dim, hidden_size)

        # Decoder
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection
        self.output_layer = nn.Linear(hidden_size, input_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to latent vector.

        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        latent : (batch, latent_dim)
        """
        _, (h_n, _) = self.encoder(x)
        # Use last layer's hidden state
        latent = self.to_latent(h_n[-1])
        return latent

    def decode(
        self, latent: torch.Tensor, seq_len: int, x_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode latent vector back to a sequence.

        Uses teacher forcing: feeds the actual input (shifted) to the decoder.

        Parameters
        ----------
        latent : (batch, latent_dim)
        seq_len : int
        x_input : (batch, seq_len, input_size) — original input for teacher forcing

        Returns
        -------
        reconstruction : (batch, seq_len, input_size)
        """
        batch_size = latent.size(0)

        # Initialize decoder hidden state from latent
        h_0 = self.from_latent(latent).unsqueeze(0)
        # Repeat for all layers
        h_0 = h_0.repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)

        # Feed reversed input to decoder (common autoencoder trick)
        reversed_input = torch.flip(x_input, dims=[1])

        decoder_out, _ = self.decoder(reversed_input, (h_0, c_0))
        reconstruction = self.output_layer(decoder_out)

        # Reverse back to original order
        reconstruction = torch.flip(reconstruction, dims=[1])

        return reconstruction

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → decode.

        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        reconstruction : (batch, seq_len, input_size)
        latent : (batch, latent_dim)
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent, x.size(1), x)
        return reconstruction, latent

    def compute_reconstruction_error(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-sample reconstruction MSE (used as anomaly score).

        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        error : (batch,) — mean reconstruction error per sample.
        """
        reconstruction, _ = self.forward(x)
        error = ((x - reconstruction) ** 2).mean(dim=(1, 2))
        return error
