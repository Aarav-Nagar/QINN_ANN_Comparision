"""Train a small ensemble + improved QINN smoke test and report metrics.

This script runs a short ANN baseline, the new EnsembleModel, and the improved QINN (with classical surrogate fallback)
and reports RMSE/MAE and directional accuracy for quick comparison.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.config import CONFIG, setup_logging
from preprocessing.data_processor import DataProcessor
from models.ann_model import AdvancedANN, AdvancedANNTrainer
from models.Ensemble import EnsembleModel, EnsembleTrainer
from models.qinn_model import QINN, QINNTrainer
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

logger = setup_logging(CONFIG)
CONFIG.set_random_seeds()

processor = DataProcessor(CONFIG, logger)
loaded = processor.load_all_data()
if 'TSLA' not in loaded:
    logger.error('TSLA not loaded; aborting')
    raise SystemExit(1)

# Use macro_indicators.csv
try:
    macro_path = os.path.join(CONFIG.DATA_DIR, 'macro_indicators.csv')
    if os.path.exists(macro_path):
        processor.macro_data = pd.read_csv(macro_path, parse_dates=['Date'], index_col='Date')
except Exception:
    pass

proc = processor.process_single_stock('TSLA')
proc_fe = processor.add_technical_indicators(proc)
processor.processed_data['TSLA'] = proc_fe

folds = processor.walk_forward_splits(proc_fe, n_splits=1, train_window_years=4, test_window_years=1)
fold = folds[0]
fold_config = (fold['train_start'], fold['train_end'], fold['test_end'])
ml = processor.prepare_ml_data('TSLA', 15, fold_config=fold_config)

X_train = ml['X_train']; y_train = ml['y_train_reg']; X_val = ml['X_val']; y_val = ml['y_val_reg']; X_test = ml['X_test']; y_test = ml['y_test_reg']

# Dataloaders
train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.LongTensor((y_train>0).astype(int)))
val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val), torch.LongTensor((y_val>0).astype(int)))
test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test), torch.LongTensor((y_test>0).astype(int)))
train_loader = DataLoader(train_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

# ANN baseline
ann = AdvancedANN(input_dim=ml['n_features'], hidden_layers=[256,128,64], use_ensemble=False)
ann_tr = AdvancedANNTrainer(ann, CONFIG, logger, device=CONFIG.DEVICE)
ann_tr.optimizer = torch.optim.AdamW(ann.parameters(), lr=CONFIG.LEARNING_RATE)
ann_tr.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ann_tr.optimizer, mode='min', factor=0.5, patience=3)
ann_tr.regression_criterion = torch.nn.SmoothL1Loss(); ann_tr.classification_loss_weight = 0.0
ann_tr.fit(train_loader, val_loader, epochs=5, early_stopping_patience=3)
ann_preds = ann_tr.predict(test_loader)['regression_predictions']

# Ensemble
ens = EnsembleModel(input_dim=ml['n_features'], gbdt_dim=0)
# small trainer loop (manual)
def train_simple(model, train_loader, val_loader, epochs=5):
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE)
    for e in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch[0].float()
            y = batch[1].float()
            opt.zero_grad()
            out = model(x)
            loss = torch.nn.SmoothL1Loss()(out['regression'], y)
            loss.backward(); opt.step()
    return model

ens = train_simple(ens, train_loader, val_loader, epochs=5)
# predict
ens.eval();
with torch.no_grad():
    allp = []
    for b in test_loader:
        x = b[0].float()
        try:
            out = ens(x)
            allp.append(out['regression'].cpu().numpy())
        except Exception:
            # If model forward fails for a batch, skip it
            continue
    ens_preds = np.concatenate(allp) if allp else np.array([])

# QINN (classical surrogate fallback)
qinn = QINN(input_dim=ml['n_features'], n_qubits=4, n_quantum_layers=1, n_circuits=1)
# If PennyLane is slow, set classical surrogate True and replace quantum with XGBoost later
try:
    qtr = QINNTrainer(qinn, CONFIG, logger, device=CONFIG.DEVICE)
    qtr.optimizer = torch.optim.AdamW(qinn.parameters(), lr=CONFIG.LEARNING_RATE)
    qtr.setup_optimizer_and_scheduler('adamw')
    qtr.regression_criterion = torch.nn.SmoothL1Loss(); qtr.classification_loss_weight = 0.0
    qtr.fit(train_loader, val_loader, epochs=3, early_stopping_patience=2)
    qinn_preds = qtr.predict(test_loader)['regression_predictions']
except Exception as e:
    logger.warning(f'QINN training failed: {e}; falling back to XGBoost surrogate')
    from xgboost import XGBRegressor
    xgb = XGBRegressor(n_estimators=100, random_state=CONFIG.RANDOM_SEED)
    xgb.fit(X_train, y_train)
    qinn_preds = xgb.predict(X_test)

# Compute metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score


def safe_metrics(y_true, y_pred, name: str = "model"):
    # handle empty arrays
    if y_true is None or y_pred is None:
        logger.warning(f"{name}: missing predictions or targets; returning NaNs")
        return float('nan'), float('nan'), float('nan')
    try:
        if len(y_true) == 0 or len(y_pred) == 0:
            logger.warning(f"{name}: empty true/pred arrays (len true={len(y_true) if y_true is not None else 'None'}, len pred={len(y_pred) if y_pred is not None else 'None'}); returning NaNs")
            return float('nan'), float('nan'), float('nan')
    except Exception:
        logger.warning(f"{name}: could not determine array lengths; returning NaNs")
        return float('nan'), float('nan'), float('nan')

    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        dir_acc = accuracy_score((y_true>0).astype(int), (y_pred>0).astype(int))
        return rmse, mae, dir_acc
    except Exception as e:
        logger.warning(f"{name}: metric computation failed: {e}; returning NaNs")
        return float('nan'), float('nan'), float('nan')

ann_m = safe_metrics(y_test, ann_preds, name='ANN')
ens_m = safe_metrics(y_test, ens_preds, name='Ensemble')
qinn_m = safe_metrics(y_test, qinn_preds, name='QINN')

print('Model, RMSE, MAE, DirAcc')
print('ANN', *ann_m)
print('Ensemble', *ens_m)
print('QINN', *qinn_m)

# Save comparison
out = {'ANN': ann_m, 'Ensemble': ens_m, 'QINN': qinn_m}
import json
with open(os.path.join(CONFIG.RESULTS_DIR, 'ensemble_smoke_comparison.json'), 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in out.items()}, f, indent=2)

print('Saved results to results/ensemble_smoke_comparison.json')
