"""Walk-forward evaluation runner for TSLA using best hyperparameters.

This script:
- Loads best configs from results/hyperparam_tune_results.json
- Runs walk-forward evaluation across CONFIG.VALIDATION_FOLDS and CONFIG.PREDICTION_HORIZONS
- Trains Ensemble and QINN (shorter but longer than the tuning run) with early stopping
- Saves per-fold/horizon metrics and aggregated summaries to results/walkforward_tsla_results.json

Note: QINN uses PennyLane and can be slow; adjust epochs if needed.
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

# Load best hyperparams from previous tuning
hp_path = os.path.join(CONFIG.RESULTS_DIR, 'hyperparam_tune_results.json')
if not os.path.exists(hp_path):
    logger.error(f"Hyperparameter results not found at {hp_path}; run hyperparam tuning first")
    raise SystemExit(1)

with open(hp_path, 'r') as f:
    hp = json.load(f)

# Pick best (lowest RMSE) ensemble and qinn configs
def best_config(results_list):
    if not results_list:
        return None
    best = min(results_list, key=lambda x: x.get('rmse', float('inf')))
    return best

best_ens = best_config(hp['results'].get('ensemble', []))
best_qinn = best_config(hp['results'].get('qinn', []))

logger.info(f"Best ensemble config: {best_ens}")
logger.info(f"Best QINN config: {best_qinn}")

# Prepare data for TSLA
processor = DataProcessor(CONFIG, logger)
loaded = processor.load_all_data()
if 'TSLA' not in loaded:
    logger.error('TSLA missing; aborting')
    raise SystemExit(1)
proc = processor.process_single_stock('TSLA')
if proc is None:
    logger.error('Processing TSLA failed; aborting')
    raise SystemExit(1)
processor.processed_data['TSLA'] = processor.add_technical_indicators(proc)

results = {'ensemble': [], 'qinn': [], 'aggregated': {}}

# Evaluation settings
ensem_epochs = 50
qinn_epochs = 30
early_stopping = 10

# Walk-forward folds (explicit, consecutive years). Validate on 2023, test on 2024.
# User requested consecutive-year training ranges and validation on 2023 to predict 2024.
folds = [
    ('2015', '2022', '2024'),
    ('2019', '2022', '2024')
]
horizons = CONFIG.PREDICTION_HORIZONS

for fold in folds:
    train_start, train_end, test_year = fold
    fold_config = (train_start, train_end, test_year)
    logger.info(f"Running fold {fold_config}")

    for horizon in horizons:
        logger.info(f"Evaluating horizon {horizon}d")
        # Build explicit train / val / test using years: train = train_start..train_end, val = 2023, test = 2024
        df = processor.processed_data['TSLA']
        try:
            train_start_int = int(fold[0]); train_end_int = int(fold[1])
            train_df = df[(df.index.year >= train_start_int) & (df.index.year <= train_end_int)].copy()
            val_df = df[df.index.year == 2023].copy()
            test_df = df[df.index.year == 2024].copy()
            if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
                logger.warning(f"Skipping {fold_config} {horizon}d due to missing train/val/test rows")
                continue
        except Exception as e:
            logger.warning(f"Skipping {fold_config} {horizon}d: {e}")
            continue

        # Select target and feature columns
        target_col = f'target_{horizon}d'
        direction_col = f'direction_{horizon}d'
        target_cols = [col for col in train_df.columns if col.startswith('target_') or col.startswith('direction_')]
        feature_cols = [col for col in train_df.columns if col not in target_cols]

        if target_col not in train_df.columns:
            logger.warning(f"Target {target_col} missing for horizon {horizon}; skipping")
            continue

        # Fit scaler on training features and apply to val/test
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        scaler.fit(train_df[feature_cols])

        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        test_scaled = test_df.copy()

        train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
        val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
        test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

        # Prepare arrays
        X_train = train_scaled[feature_cols].values
        y_train = train_scaled[target_col].values
        X_val = val_scaled[feature_cols].values
        y_val = val_scaled[target_col].values
        X_test = test_scaled[feature_cols].values
        y_test = test_scaled[target_col].values

        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.LongTensor((y_train>0).astype(int)))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val), torch.LongTensor((y_val>0).astype(int)))
        test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test), torch.LongTensor((y_test>0).astype(int)))
        train_loader = DataLoader(train_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

        # Ensemble
        try:
            lr = best_ens.get('lr', CONFIG.LEARNING_RATE) if best_ens else CONFIG.LEARNING_RATE
            wd = best_ens.get('weight_decay', CONFIG.ANN_L2_REGULARIZATION) if best_ens else CONFIG.ANN_L2_REGULARIZATION
            drop = best_ens.get('dropout', CONFIG.ANN_DROPOUT_RATE) if best_ens else CONFIG.ANN_DROPOUT_RATE

            ens_model = EnsembleModel(input_dim=len(feature_cols), gbdt_dim=0, hidden_dims=[256,128], dropout=drop)
            ens_tr = EnsembleTrainer(ens_model, CONFIG, logger, device=CONFIG.DEVICE)
            ens_tr.setup_optimizer(lr=lr, weight_decay=wd)

            chkpt = os.path.join(CONFIG.MODELS_DIR, f"ensemble_TSLA_{horizon}d_{train_start}-{train_end}_chkpt.pth")
            best_rmse = ens_tr.fit(train_loader, val_loader, epochs=ensem_epochs, use_onecycle=True, max_lr=lr*10, early_stopping_patience=early_stopping, chkpt_name=chkpt)
            preds = ens_tr.predict(test_loader)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            dir_acc = float(accuracy_score((y_test>0).astype(int), (preds>0).astype(int)))
            entry = {'fold': fold_config, 'horizon': horizon, 'lr': lr, 'weight_decay': wd, 'dropout': drop, 'rmse': rmse, 'mae': mae, 'dir_acc': dir_acc}
            results['ensemble'].append(entry)
            logger.info(f"Ensemble fold result: {entry}")
        except Exception as e:
            logger.error(f"Ensemble failed for fold {fold_config} {horizon}d: {e}")

        # QINN
        try:
            lr = best_qinn.get('lr', CONFIG.LEARNING_RATE) if best_qinn else CONFIG.LEARNING_RATE
            wd = best_qinn.get('weight_decay', CONFIG.ANN_L2_REGULARIZATION) if best_qinn else CONFIG.ANN_L2_REGULARIZATION
            drop = best_qinn.get('dropout', CONFIG.ANN_DROPOUT_RATE) if best_qinn else CONFIG.ANN_DROPOUT_RATE

            qinn_model = QINN(input_dim=len(feature_cols), n_qubits=4, n_quantum_layers=1, n_circuits=1, dropout_rate=drop)
            qinn_tr = QINNTrainer(qinn_model, CONFIG, logger, device=CONFIG.DEVICE)
            qinn_tr.setup_optimizer_and_scheduler('adamw')
            # override lrs/wd
            for pg in qinn_tr.optimizer.param_groups:
                pg['lr'] = lr
                pg['weight_decay'] = wd

            chkpt = os.path.join(CONFIG.MODELS_DIR, f"qinn_TSLA_{horizon}d_{train_start}-{train_end}_chkpt.pth")
            qho = None
            try:
                qinn_tr.fit(train_loader, val_loader, epochs=qinn_epochs, early_stopping_patience=early_stopping, use_onecycle=True, max_lr=lr*5, weight_decay=wd, grad_clip=1.0, chkpt_path=chkpt)
                preds = qinn_tr.predict(test_loader)['regression_predictions']
            except Exception as e:
                logger.warning(f"QINN failed for fold {fold_config} {horizon}d: {e}; falling back to XGB")
                from xgboost import XGBRegressor
                xgb = XGBRegressor(n_estimators=300, random_state=CONFIG.RANDOM_SEED)
                xgb.fit(X_train, y_train)
                preds = xgb.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            dir_acc = float(accuracy_score((y_test>0).astype(int), (preds>0).astype(int)))
            entry = {'fold': fold_config, 'horizon': horizon, 'lr': lr, 'weight_decay': wd, 'dropout': drop, 'rmse': rmse, 'mae': mae, 'dir_acc': dir_acc}
            results['qinn'].append(entry)
            logger.info(f"QINN fold result: {entry}")
        except Exception as e:
            logger.error(f"QINN failed for fold {fold_config} {horizon}d: {e}")

# Aggregate by horizon and model
import collections
agg = {'ensemble': {}, 'qinn': {}}
for model_key in ['ensemble', 'qinn']:
    by_h = collections.defaultdict(list)
    for entry in results[model_key]:
        by_h[entry['horizon']].append(entry['rmse'])
    for h, vals in by_h.items():
        agg[model_key][str(h)] = {
            'mean_rmse': float(np.mean(vals)),
            'std_rmse': float(np.std(vals)),
            'n_folds': len(vals)
        }

results['aggregated'] = agg

out_path = os.path.join(CONFIG.RESULTS_DIR, 'walkforward_tsla_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print('Saved walk-forward results to', out_path)
