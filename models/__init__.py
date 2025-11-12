"""models package

Expose commonly used model classes at package level for convenience.

Exports:
- AdvancedANN, AdvancedANNTrainer  (from ann_model.py)
- QINN, QINNTrainer                (from qinn_model.py)
- EnsembleModel, EnsembleTrainer   (from Ensemble subpackage)
"""

import os

from .ann_model import AdvancedANN, AdvancedANNTrainer
from .qinn_model import QINN, QINNTrainer

# Try to import Ensemble classes from the Ensemble subpackage. If import
# fails (e.g. during partial package initialization), set placeholders and
# write the traceback to disk for debugging.
try:
    from .Ensemble.ensemble_model import EnsembleModel
    from .Ensemble.ensemble_trainer import EnsembleTrainer
except Exception:
    EnsembleModel = None
    EnsembleTrainer = None
    try:
        import traceback

        err_path = os.path.join(os.path.dirname(__file__), 'models_init_error.txt')
        with open(err_path, 'w') as _f:
            _f.write(traceback.format_exc())
    except Exception:
        # Best-effort only; don't raise during import
        pass

# Make names available at package level
globals()['EnsembleModel'] = EnsembleModel
globals()['EnsembleTrainer'] = EnsembleTrainer

__all__ = [
    'AdvancedANN',
    'AdvancedANNTrainer',
    'QINN',
    'QINNTrainer',
    'EnsembleModel',
    'EnsembleTrainer',
]