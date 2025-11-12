import sys, traceback
ROOT = r"c:\Users\aarav\OneDrive\IT\Projects\ScienceFair"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print('sys.path[0]=', sys.path[0])
try:
    import models
    print('Imported models OK')
    print('models module:', models)
    print('models.__file__:', getattr(models, '__file__', None))
    names = [n for n in dir(models) if not n.startswith('_')]
    print('Available names in models:', names)
    print("Has EnsembleModel attr?", hasattr(models, 'EnsembleModel'))
    print('models.__dict__ keys sample:', list(models.__dict__.keys())[:40])
except Exception as e:
    print('Failed to import models:')
    traceback.print_exc()

print('\nAttempt import models.Ensemble')
try:
    import models.Ensemble as E
    print('Imported models.Ensemble OK')
    print('Ensemble names:', [n for n in dir(E) if not n.startswith('_')])
except Exception as e:
    print('Failed to import models.Ensemble:')
    traceback.print_exc()

print('\nAttempt from models import EnsembleModel')
try:
    from models import EnsembleModel
    print('from models import EnsembleModel OK ->', EnsembleModel)
except Exception as e:
    print('Failed from models import EnsembleModel:')
    traceback.print_exc()
