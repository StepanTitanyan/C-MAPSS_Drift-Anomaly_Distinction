"""
src/training/losses.py
======================
Loss functions for training temporal models.

The key innovation is the Gaussian NLL loss:
- Instead of just predicting the next sensor values (MSE loss),
  the model predicts BOTH a mean and a standard deviation per sensor.
- The loss penalizes the model for being wrong AND for being overconfident.
- A large error with large predicted σ is less penalized than a large error
  with tiny predicted σ.

This is fundamental to the probabilistic approach: the model learns to output
calibrated uncertainty estimates, which become features for anomaly scoring
and drift/anomaly distinction.

Mathematical form:
    NLL = (1/d) Σ_j [ log(σ_j) + (x_j - μ_j)² / (2·σ_j²) ]

where:
    x_j = true sensor value
    μ_j = predicted mean
    σ_j = predicted standard deviation
    d   = number of sensors
"""

import torch
import torch.nn as nn
import math


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood loss for probabilistic forecasting.

    The model outputs (mu, sigma) and we compute the NLL of the true
    observation under the predicted Gaussian distribution.

    This loss has two important properties:
    1. It rewards accurate predictions (small |x - μ|).
    2. It rewards honest uncertainty (σ should be large when the model
       is likely to be wrong, and small when it's likely to be right).

    Parameters
    ----------
    reduction : str
        "mean" (average over batch and sensors) or "none" (per-sample loss).
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL.

        Parameters
        ----------
        mu : torch.Tensor of shape (batch, d)
            Predicted means.
        sigma : torch.Tensor of shape (batch, d)
            Predicted standard deviations (must be > 0).
        target : torch.Tensor of shape (batch, d)
            True sensor values.

        Returns
        -------
        torch.Tensor
            Scalar loss if reduction="mean", or (batch,) if reduction="none".
        """
        # NLL per sensor: log(σ) + (x - μ)² / (2σ²)
        # We also add 0.5 * log(2π) for mathematical correctness,
        # though it's a constant and doesn't affect optimization.
        variance = sigma ** 2
        nll = 0.5 * (
            torch.log(variance)
            + (target - mu) ** 2 / variance
            + math.log(2 * math.pi)
        )

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "none":
            # Average over sensors, keep batch dimension
            return nll.mean(dim=-1)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class MSELossWrapper(nn.Module):
    """
    Standard MSE loss, wrapped to have a consistent interface.

    Used for deterministic baseline models.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,   # ignored for MSE
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MSE (ignores sigma).

        Parameters match GaussianNLLLoss for compatibility.
        """
        return self.mse(mu, target)


def get_loss_function(loss_type: str, reduction: str = "mean") -> nn.Module:
    """
    Factory function to get the appropriate loss.

    Parameters
    ----------
    loss_type : str
        "nll" for Gaussian NLL, "mse" for standard MSE.
    reduction : str
        "mean" or "none".

    Returns
    -------
    nn.Module
    """
    if loss_type == "nll":
        return GaussianNLLLoss(reduction=reduction)
    elif loss_type == "mse":
        return MSELossWrapper(reduction=reduction)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'nll' or 'mse'.")
