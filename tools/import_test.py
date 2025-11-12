import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    import models
    from models import AdvancedANN, QINN
    # EnsembleModel may be exposed at package level or only in subpackage
    try:
        from models import EnsembleModel
    except Exception:
        from models.Ensemble import EnsembleModel
    import models.Ensemble as E
    print('Imports OK:', AdvancedANN.__name__, QINN.__name__, EnsembleModel.__name__)
except Exception as e:
    print('Import error:', e)
    raise
