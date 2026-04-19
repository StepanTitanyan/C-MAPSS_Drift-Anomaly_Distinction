"""
src/models/gaussian_lstm.py
===========================
Probabilistic LSTM Forecaster — the core model of this project.

Architecture:
    Input: (batch, window_size, num_sensors)
      → LSTM layers (stacked, with dropout between layers)
      → Last hidden state h[-1] of shape (batch, hidden_size)
      → Two parallel heads:
          → Linear → mu    (predicted mean, shape: batch × num_sensors)
          → Linear → softplus + floor → sigma (predicted std, shape: batch × num_sensors)

Why two heads?
    A standard LSTM forecaster only outputs μ (point prediction).
    Our model also outputs σ — a per-sensor uncertainty estimate.

    This means the model learns not just WHAT to predict, but HOW CONFIDENT
    it is in that prediction. This is critical because:
    1. The anomaly score becomes the NLL: a deviation in a high-confidence
       region is more alarming than the same deviation in a low-confidence region.
    2. The predicted σ itself becomes a feature: if σ increases during drift
       (the model recognizes unfamiliar patterns), that's a signal the
       second-stage classifier can use.

Training:
    Use GaussianNLLLoss from src/training/losses.py.
    The loss penalizes both inaccurate means AND poorly calibrated uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GaussianLSTM(nn.Module):
    """
    Probabilistic LSTM for multivariate time-series forecasting.

    Predicts the next-step sensor values as a diagonal Gaussian distribution:
        x_{t+1} ~ N(μ_{t+1}, diag(σ²_{t+1}))

    Parameters
    ----------
    input_size : int
        Number of input features (= number of sensors).
    hidden_size : int
        LSTM hidden state dimension.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout between LSTM layers (applied only if num_layers > 1).
    sigma_min : float
        Minimum floor for predicted σ to prevent NLL collapse.
        Without this, the model could drive σ → 0 on well-predicted points,
        making NLL → -∞, which destabilizes training.
    output_size : int
        Number of output features (= number of sensors, typically same as input_size).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        sigma_min: float = 1e-4,
        output_size: int = None,
    ):
        super().__init__()

        if output_size is None:
            output_size = input_size

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.sigma_min = sigma_min

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Mean head: predicts expected next-step values
        self.mu_head = nn.Linear(hidden_size, output_size)

        # Sigma head: predicts uncertainty (std) for each sensor
        # Raw output → softplus → + sigma_min
        self.sigma_head = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, window_size, input_size)
            Input sensor window.

        Returns
        -------
        mu : torch.Tensor of shape (batch, output_size)
            Predicted mean for next step.
        sigma : torch.Tensor of shape (batch, output_size)
            Predicted standard deviation for next step (always > sigma_min).
        """
        # Run through LSTM
        # lstm_out: (batch, window_size, hidden_size) — all hidden states
        # h_n: (num_layers, batch, hidden_size) — final hidden state per layer
        # c_n: (num_layers, batch, hidden_size) — final cell state per layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last layer's final hidden state as the sequence representation
        # h_n[-1] has shape (batch, hidden_size)
        last_hidden = h_n[-1]

        # Predict mean
        mu = self.mu_head(last_hidden)

        # Predict sigma with softplus + floor
        # softplus(x) = log(1 + exp(x)), which is smooth and always > 0
        sigma_raw = self.sigma_head(last_hidden)
        sigma = F.softplus(sigma_raw) + self.sigma_min

        return mu, sigma

    def predict(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference mode prediction (no gradients).

        Parameters
        ----------
        x : torch.Tensor of shape (batch, window_size, input_size)

        Returns
        -------
        mu, sigma as numpy arrays.
        """
        self.eval()
        with torch.no_grad():
            mu, sigma = self.forward(x)
        return mu.cpu().numpy(), sigma.cpu().numpy()


class DeterministicLSTM(nn.Module):
    """
    Standard (non-probabilistic) LSTM forecaster for baseline comparison.

    Same architecture as GaussianLSTM but only outputs μ.
    Sigma is returned as a dummy tensor of ones (for interface compatibility).

    This lets us use the same training loop, loss interface, and evaluation
    code for both deterministic and probabilistic models.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = None,
    ):
        super().__init__()

        if output_size is None:
            output_size = input_size

        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns (mu, sigma) where sigma is a dummy tensor of ones.
        This keeps the interface consistent with GaussianLSTM.
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        mu = self.fc(last_hidden)

        # Dummy sigma (ones) so the MSE loss wrapper can ignore it
        sigma = torch.ones_like(mu)

        return mu, sigma
