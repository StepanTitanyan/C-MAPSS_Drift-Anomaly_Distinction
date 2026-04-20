"""
src/models/tranad.py
====================
Practical TranAD-style transformer baseline adapted to this project.

Adaptation choices for this repo:
- same engine-level FD001 split
- same 7 selected sensors
- same window length
- same healthy-only training assumption
- same synthetic anomaly evaluation pipeline

This implementation keeps the parts of TranAD that fit naturally here:
- transformer sequence encoder
- two-phase self-conditioning
- second-phase prediction used for anomaly scoring

Because the rest of this project is framed as next-step health modelling,
we adapt TranAD from window reconstruction to *window -> next-step prediction*.
That keeps the data protocol and labels aligned with the URD pipeline.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TranAD(nn.Module):
    """
    TranAD-style transformer baseline adapted for next-step prediction.

    Forward pass returns two predictions:
      - pred_phase1: coarse prediction
      - pred_phase2: self-conditioned refined prediction
    """

    def __init__(
        self,
        input_size: int,
        window_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.window_size = window_size
        self.d_model = d_model

        self.input_proj = nn.Linear(input_size, d_model)
        self.posenc = PositionalEncoding(d_model, max_len=window_size + 2)

        enc_layer1 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        enc_layer2 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder1 = nn.TransformerEncoder(enc_layer1, num_layers=num_layers)
        self.encoder2 = nn.TransformerEncoder(enc_layer2, num_layers=num_layers)

        self.focus_proj = nn.Linear(input_size, d_model)

        head_in = 2 * d_model
        self.head1 = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, input_size),
        )
        self.head2 = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, input_size),
        )

    def _context(self, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([h[:, -1, :], h.mean(dim=1)], dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.posenc(self.input_proj(x))
        h1 = self.encoder1(z)
        pred1 = self.head1(self._context(h1))

        # Practical self-conditioning: emphasise sensors where phase-1 prediction
        # disagrees with the most recent observed step. This keeps inference causal.
        focus = torch.sigmoid((pred1 - x[:, -1, :]) ** 2)
        z2 = self.posenc(self.input_proj(x) + self.focus_proj(focus).unsqueeze(1))
        h2 = self.encoder2(z2)
        pred2 = self.head2(self._context(h2))
        return pred1, pred2

    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        _, pred2 = self.forward(x)
        return pred2


class TranADTrainer:
    """Dedicated trainer because TranAD uses a two-phase loss."""

    def __init__(
        self,
        model: TranAD,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
        patience: int = 15,
        lr_patience: int = 7,
        lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        gradient_clip_norm: float = 1.0,
        phase2_weight: float = 1.5,
        device: str = "cpu",
        checkpoint_dir: Optional[str] = None,
        model_name: str = "tranad",
    ):
        self.model = model.to(device)
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.gradient_clip_norm = gradient_clip_norm
        self.phase2_weight = phase2_weight
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
        )
        self.history = {"train_loss": [], "val_loss": [], "lr": []}
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_no_improve = 0
        self.mse = nn.MSELoss()

    def _loss(self, pred1: torch.Tensor, pred2: torch.Tensor, target: torch.Tensor, epoch_idx: int) -> torch.Tensor:
        progress = min(epoch_idx / max(self.max_epochs, 1), 1.0)
        phase1_weight = max(0.25, 1.0 - progress)
        loss1 = self.mse(pred1, target)
        loss2 = self.mse(pred2, target)
        return phase1_weight * loss1 + self.phase2_weight * loss2

    def _train_one_epoch(self, loader: DataLoader, epoch_idx: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            pred1, pred2 = self.model(X_batch)
            loss = self._loss(pred1, pred2, y_batch, epoch_idx)
            loss.backward()
            if self.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
            bs = X_batch.size(0)
            total_loss += loss.item() * bs
            n_samples += bs
        return total_loss / max(n_samples, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, epoch_idx: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            pred1, pred2 = self.model(X_batch)
            loss = self._loss(pred1, pred2, y_batch, epoch_idx)
            bs = X_batch.size(0)
            total_loss += loss.item() * bs
            n_samples += bs
        return total_loss / max(n_samples, 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, verbose: bool = True) -> Dict[str, list]:
        start_time = time.time()
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_one_epoch(train_loader, epoch)
            val_loss = self._evaluate(val_loader, epoch)
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
                if self.checkpoint_dir:
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    torch.save(self.best_model_state, os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt"))
            else:
                self.epochs_no_improve += 1

            if verbose:
                elapsed = time.time() - start_time
                print(
                    f"Epoch [{epoch:3d}/{self.max_epochs}] "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                    f"LR: {current_lr:.2e} | Best: {self.best_val_loss:.6f} | Time: {elapsed:.0f}s"
                )

            if self.epochs_no_improve >= self.patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        return self.history


class TranADScorer:
    """Healthy-validation normalisation for TranAD next-step squared error scores."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        scores = np.mean((y_true - y_pred) ** 2, axis=1)
        self.mean_ = float(np.mean(scores))
        self.std_ = max(float(np.std(scores)), 1e-8)

    def score(self, y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> np.ndarray:
        scores = np.mean((y_true - y_pred) ** 2, axis=1)
        if normalize:
            scores = (scores - self.mean_) / self.std_
        return scores

    def compute_thresholds(self, y_true: np.ndarray, y_pred: np.ndarray, percentiles=None) -> Dict[float, float]:
        if percentiles is None:
            percentiles = [95.0, 97.5, 99.0]
        scores = self.score(y_true, y_pred, normalize=True)
        return {p: float(np.percentile(scores, p)) for p in percentiles}
