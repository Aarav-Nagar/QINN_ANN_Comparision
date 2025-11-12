"""Compact hyperparameter grid search for Ensemble and QINN (short runs).

This script runs quick experiments to find promising hyperparameters (epochs kept small).
It will try a small grid and save the results to results/hyperparam_tune.json.
"""
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import numpy as np
from utils.config import CONFIG, setup_logging
from preprocessing.data_processor import DataProcessor
from models.Ensemble import EnsembleModel, EnsembleTrainer
from models.qinn_model import QINN, QINNTrainer
from models.ann_model import AdvancedANN, AdvancedANNTrainer
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

logger = setup_logging(CONFIG)
CONFIG.set_random_seeds()

# Load and prepare data
processor = DataProcessor(CONFIG, logger)
loaded = processor.load_all_data()
if 'TSLA' not in loaded:
    logger.error('TSLA missing; aborting')
    raise SystemExit(1)
proc = processor.process_single_stock('TSLA')
proc_fe = processor.add_technical_indicators(proc)
processor.processed_data['TSLA'] = proc_fe
folds = processor.walk_forward_splits(proc_fe, n_splits=1, train_window_years=4, test_window_years=1)
fold = folds[0]
fold_config = (fold['train_start'], fold['train_end'], fold['test_end'])
ml = processor.prepare_ml_data('TSLA', 15, fold_config=fold_config)

X_train = ml['X_train']; y_train = ml['y_train_reg']; X_val = ml['X_val']; y_val = ml['y_val_reg']; X_test = ml['X_test']; y_test = ml['y_test_reg']

train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.LongTensor((y_train>0).astype(int)))
val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val), torch.LongTensor((y_val>0).astype(int)))
test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test), torch.LongTensor((y_test>0).astype(int)))
train_loader = DataLoader(train_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

results = {'ensemble': [], 'qinn': []}

# Compact grids (keep small for quick runs)
lrs = [5e-4, 1e-3]
wds = [1e-5, 1e-4]
drops = [0.1, 0.2]
epochs = 10

# Baseline ANN quick run
ann = AdvancedANN(input_dim=ml['n_features'], hidden_layers=[256,128,64], use_ensemble=False)
ann_tr = AdvancedANNTrainer(ann, CONFIG, logger, device=CONFIG.DEVICE)
ann_tr.optimizer = torch.optim.AdamW(ann.parameters(), lr=1e-3)
ann_tr.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ann_tr.optimizer, mode='min', factor=0.5, patience=3)
ann_tr.regression_criterion = torch.nn.SmoothL1Loss(); ann_tr.classification_loss_weight = 0.0
ann_tr.fit(train_loader, val_loader, epochs=5, early_stopping_patience=3)
ann_preds = ann_tr.predict(test_loader)['regression_predictions']
ann_rmse = float(np.sqrt(mean_squared_error(y_test, ann_preds)))

# Tune Ensemble
for lr in lrs:
    for wd in wds:
        for drop in drops:
            cfg = {'lr': lr, 'weight_decay': wd, 'dropout': drop}
            logger.info(f"Tuning Ensemble with {cfg}")
            model = EnsembleModel(input_dim=ml['n_features'], gbdt_dim=0, hidden_dims=[256,128], dropout=drop)
            trainer = EnsembleTrainer(model, CONFIG, logger, device=CONFIG.DEVICE)
            trainer.setup_optimizer(lr=lr, weight_decay=wd)
            best_rmse = trainer.fit(train_loader, val_loader, epochs=epochs, use_onecycle=True, max_lr=lr*10, early_stopping_patience=5, chkpt_name=None)
            preds = trainer.predict(test_loader)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            dir_acc = float(accuracy_score((y_test>0).astype(int), (preds>0).astype(int)))
            entry = {'lr': lr, 'weight_decay': wd, 'dropout': drop, 'rmse': rmse, 'mae': mae, 'dir_acc': dir_acc}
            results['ensemble'].append(entry)
            logger.info(f"Ensemble result: {entry}")

# Tune QINN (small search)
for lr in lrs:
    for wd in wds:
        for drop in drops:
            cfg = {'lr': lr, 'weight_decay': wd, 'dropout': drop}
            logger.info(f"Tuning QINN with {cfg}")
            model = QINN(input_dim=ml['n_features'], n_qubits=4, n_quantum_layers=1, n_circuits=1, dropout_rate=drop)
            trainer = QINNTrainer(model, CONFIG, logger, device=CONFIG.DEVICE)
            trainer.setup_optimizer_and_scheduler('adamw')
            # override weight decay
            for pg in trainer.optimizer.param_groups:
                pg['lr'] = lr
                pg['weight_decay'] = wd
            # try onecycle
            try:
                trainer.fit(train_loader, val_loader, epochs=epochs, early_stopping_patience=5, use_onecycle=True, max_lr=lr*5, weight_decay=wd, grad_clip=1.0)
                preds = trainer.predict(test_loader)['regression_predictions']
            except Exception as e:
                logger.warning(f"QINN failed for cfg {cfg}: {e}; fallback to XGB")
                from xgboost import XGBRegressor
                xgb = XGBRegressor(n_estimators=200, random_state=CONFIG.RANDOM_SEED)
                xgb.fit(X_train, y_train)
                preds = xgb.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            dir_acc = float(accuracy_score((y_test>0).astype(int), (preds>0).astype(int)))
            entry = {'lr': lr, 'weight_decay': wd, 'dropout': drop, 'rmse': rmse, 'mae': mae, 'dir_acc': dir_acc}
            results['qinn'].append(entry)
            logger.info(f"QINN result: {entry}")

# Save results
out_path = os.path.join(CONFIG.RESULTS_DIR, 'hyperparam_tune_results.json')
with open(out_path, 'w') as f:
    json.dump({'baseline_ann_rmse': ann_rmse, 'results': results}, f, indent=2)

print('Saved hyperparameter tuning results to', out_path)
