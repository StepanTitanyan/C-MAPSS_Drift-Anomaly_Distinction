"""
src/training/trainer.py
=======================
Universal training loop for all neural models in the project.

Features:
- Works with both probabilistic (NLL) and deterministic (MSE) models
- Early stopping on validation loss
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping to prevent explosions
- Best-model checkpointing
- Training history logging (for loss curves)

The trainer doesn't know or care whether the model is LSTM, GRU, or
autoencoder — it only requires that model.forward(x) returns (mu, sigma)
and uses the loss function from src/training/losses.py.
"""

import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple

from src.training.losses import get_loss_function


class Trainer:
    """
    Trains a neural model with early stopping and LR scheduling.

    Parameters
    ----------
    model : nn.Module
        Must have forward(x) → (mu, sigma).
    loss_type : str
        "nll" for Gaussian NLL, "mse" for standard MSE.
    learning_rate : float
    weight_decay : float
        L2 regularization strength.
    max_epochs : int
    patience : int
        Early stopping patience (epochs without improvement).
    lr_patience : int
        LR scheduler patience.
    lr_factor : float
        LR reduction factor.
    min_lr : float
        Minimum learning rate.
    gradient_clip_norm : float
        Max gradient norm for clipping.
    device : str
        "cuda" or "cpu".
    checkpoint_dir : str or None
        Directory to save best model. None = don't save.
    model_name : str
        Name for the checkpoint file.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_type: str = "nll",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
        patience: int = 15,
        lr_patience: int = 7,
        lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        gradient_clip_norm: float = 1.0,
        device: str = "cpu",
        checkpoint_dir: Optional[str] = None,
        model_name: str = "model",
    ):
        self.model = model.to(device)
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        # Loss function
        self.criterion = get_loss_function(loss_type)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
        )

        # Tracking
        self.history = {"train_loss": [], "val_loss": [], "lr": []}
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_no_improve = 0

    def _train_one_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            mu, sigma = self.model(X_batch)
            loss = self.criterion(mu, sigma, y_batch)

            loss.backward()

            # Gradient clipping
            if self.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )

            self.optimizer.step()

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        return total_loss / n_samples

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mu, sigma = self.model(X_batch)
            loss = self.criterion(mu, sigma, y_batch)

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        return total_loss / n_samples

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the full training loop.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        verbose : bool
            Print progress every epoch.

        Returns
        -------
        dict: Training history with "train_loss", "val_loss", "lr" lists.
        """
        start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            # Train
            train_loss = self._train_one_epoch(train_loader)

            # Validate
            val_loss = self._evaluate(val_loader)

            # LR scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0

                # Save checkpoint
                if self.checkpoint_dir:
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    path = os.path.join(
                        self.checkpoint_dir, f"{self.model_name}_best.pt"
                    )
                    torch.save(self.best_model_state, path)
            else:
                self.epochs_no_improve += 1

            # Print progress
            if verbose:
                elapsed = time.time() - start_time
                print(
                    f"Epoch [{epoch:3d}/{self.max_epochs}] "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Best: {self.best_val_loss:.6f} | "
                    f"Time: {elapsed:.0f}s"
                )

            # Early stopping
            if self.epochs_no_improve >= self.patience:
                if verbose:
                    print(
                        f"\nEarly stopping at epoch {epoch} "
                        f"(no improvement for {self.patience} epochs)"
                    )
                break

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        total_time = time.time() - start_time
        if verbose:
            print(f"\nTraining complete in {total_time:.1f}s")
            print(f"Best validation loss: {self.best_val_loss:.6f}")

        return self.history

    def load_best_model(self, path: Optional[str] = None):
        """Load a saved best model checkpoint."""
        if path is None:
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
            else:
                raise RuntimeError("No best model state available.")
        else:
            state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state)
