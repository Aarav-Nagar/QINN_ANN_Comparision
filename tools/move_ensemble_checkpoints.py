"""Move ensemble checkpoint .pth files into models/Ensemble/checkpoints."""
import shutil, glob, os

root = os.path.dirname(os.path.dirname(__file__))
models_dir = os.path.join(root, 'models')
dest = os.path.join(models_dir, 'Ensemble', 'checkpoints')
os.makedirs(dest, exist_ok=True)
patterns = [
    os.path.join(models_dir, 'ensemble_*_TSLA_*_chkpt.pth'),
    os.path.join(models_dir, 'ensemble_safe_*_TSLA_*_chkpt.pth'),
    os.path.join(models_dir, 'ensemble_TSLA_*_chkpt.pth'),
]
moved = 0
for pattern in patterns:
    for f in glob.glob(pattern):
        try:
            shutil.move(f, dest)
            print('Moved', os.path.basename(f))
            moved += 1
        except Exception as e:
            print('Failed', f, e)
if moved == 0:
    print('No files matched.')
else:
    print(f'Moved {moved} files.')
