"""
Ensemble subpackage.

Exports EnsembleModel and EnsembleTrainer for cleaner imports:
from models.Ensemble import EnsembleModel, EnsembleTrainer
"""
from .ensemble_model import EnsembleModel
from .ensemble_trainer import EnsembleTrainer

__all__ = ["EnsembleModel", "EnsembleTrainer"]
