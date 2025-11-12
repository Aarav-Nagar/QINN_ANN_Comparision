import time
import torch
import torch.nn as nn
import numpy as np


class EnsembleTrainer:
    """Trainer for EnsembleModel with OneCycleLR, early stopping, checkpointing, gradient clipping."""

    def __init__(self, model, config, logger, device: str = "cpu"):
        self.model = model.to(device)
        self.config = config
        self.logger = logger
        self.device = device

        self.optimizer = None
        self.scheduler = None
        self.loss_fn = nn.SmoothL1Loss()

        self.best_state = None
        self.best_score = float("inf")

    def setup_optimizer(self, lr: float = None, weight_decay: float = None):
        lr = lr if lr is not None else getattr(self.config, "LEARNING_RATE", 1e-3)
        weight_decay = weight_decay if weight_decay is not None else getattr(self.config, "ANN_L2_REGULARIZATION", 1e-5)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def fit(self, train_loader, val_loader, epochs: int = 20, use_onecycle: bool = True, max_lr: float = None, early_stopping_patience: int = 10, chkpt_name: str = None):
        if self.optimizer is None:
            self.setup_optimizer()

        steps_per_epoch = max(1, len(train_loader))
        max_lr = max_lr if max_lr is not None else self.optimizer.param_groups[0]["lr"]

        if use_onecycle:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr, total_steps=epochs * steps_per_epoch)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5)

        patience = 0
        for epoch in range(epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)

            try:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            except Exception:
                pass

            val_rmse = val_metrics.get("rmse", float("inf"))
            if val_rmse < self.best_score:
                self.best_score = val_rmse
                self.best_state = {"epoch": epoch, "model_state": self.model.state_dict()}
                patience = 0
                if chkpt_name:
                    try:
                        torch.save(self.best_state, chkpt_name)
                    except Exception:
                        pass
            else:
                patience += 1

            self.logger.info(f"Epoch {epoch}/{epochs}: Train Loss={train_metrics['loss']:.6f}, Val RMSE={val_rmse:.6f}, LR={self.optimizer.param_groups[0]['lr']:.2e}")

            if patience >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        if self.best_state is not None:
            try:
                self.model.load_state_dict(self.best_state["model_state"])
            except Exception:
                pass

        return self.best_score

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        nb = 0
        for batch in loader:
            x = batch[0].to(self.device).float()
            y = batch[1].to(self.device).float()
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.loss_fn(out["regression"], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            running_loss += loss.item()
            nb += 1
        return {"loss": running_loss / max(1, nb)}

    def validate_epoch(self, loader):
        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device).float()
                y = batch[1].to(self.device).float()
                out = self.model(x)
                preds.append(out["regression"].cpu().numpy())
                trues.append(y.cpu().numpy())

        preds = np.concatenate(preds) if preds else np.array([])
        trues = np.concatenate(trues) if trues else np.array([])
        if preds.size:
            rmse = float(((preds - trues) ** 2).mean() ** 0.5)
            loss = float(((preds - trues) ** 2).mean())
        else:
            rmse = float("inf"); loss = float("inf")
        return {"loss": loss, "rmse": rmse}

    def predict(self, loader):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device).float()
                out = self.model(x)
                preds.append(out["regression"].cpu().numpy())
        return np.concatenate(preds) if preds else np.array([])
