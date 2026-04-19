"""
src/models/gaussian_gru.py
==========================
Probabilistic GRU Forecaster — alternative backbone to LSTM.

GRU (Gated Recurrent Unit) is structurally simpler than LSTM:
- No separate cell state (c_t), only hidden state (h_t).
- Fewer parameters (~25% fewer than LSTM with same hidden_size).
- Often trains faster and sometimes matches LSTM performance.

We include both to empirically compare in the ablation study.
The architecture mirrors GaussianLSTM: same dual-head (μ, σ) design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GaussianGRU(nn.Module):
    """
    Probabilistic GRU for multivariate time-series forecasting.

    Parameters are identical to GaussianLSTM for fair comparison.
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

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.mu_head = nn.Linear(hidden_size, output_size)
        self.sigma_head = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, window_size, input_size)

        Returns
        -------
        mu : (batch, output_size) — predicted mean
        sigma : (batch, output_size) — predicted std (> sigma_min)
        """
        # GRU returns: output (batch, seq_len, hidden), h_n (num_layers, batch, hidden)
        gru_out, h_n = self.gru(x)
        last_hidden = h_n[-1]

        mu = self.mu_head(last_hidden)
        sigma = F.softplus(self.sigma_head(last_hidden)) + self.sigma_min

        return mu, sigma


class DeterministicGRU(nn.Module):
    """
    Standard (non-probabilistic) GRU forecaster for baseline comparison.
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

        self.gru = nn.GRU(
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
        gru_out, h_n = self.gru(x)
        last_hidden = h_n[-1]
        mu = self.fc(last_hidden)
        sigma = torch.ones_like(mu)
        return mu, sigma
