"""Walk-forward smoke test: Stage 1+2 verification

Loads TSLA, creates advanced features/targets, performs a single walk-forward split,
trains a small ANN for a few epochs and reports RMSE/MAE.
"""
import os
import sys
import json
import time

# Ensure project root is on sys.path so local packages import correctly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.config import CONFIG, setup_logging
from preprocessing.data_processor import DataProcessor
from models.ann_model import AdvancedANN, AdvancedANNTrainer
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

logger = setup_logging(CONFIG)
CONFIG.set_random_seeds()

processor = DataProcessor(CONFIG, logger)

# Ensure TSLA exists in config stocks for this smoke test
if 'TSLA' not in CONFIG.STOCKS:
    CONFIG.STOCKS.insert(0, 'TSLA')

# Load data
loaded = processor.load_all_data()
if 'TSLA' not in loaded:
    logger.error('TSLA not loaded; aborting smoke test')
    raise SystemExit(1)

# Use macro_indicators.csv if present
try:
    macro_path = os.path.join(CONFIG.DATA_DIR, 'macro_indicators.csv')
    if os.path.exists(macro_path):
        processor.macro_data = pd.read_csv(macro_path, parse_dates=['Date'], index_col='Date')
        logger.info('Loaded macro_indicators.csv for market features')
except Exception as e:
    logger.warning(f'Could not load macro_indicators.csv: {e}')

# Process TSLA
processor.stock_data['TSLA'] = loaded['TSLA']
proc = processor.process_single_stock('TSLA')
if proc is None:
    logger.error('Processing TSLA failed; aborting')
    raise SystemExit(1)

# Add technical indicators
proc_fe = processor.add_technical_indicators(proc)
processor.processed_data['TSLA'] = proc_fe

# Walk-forward folds
folds = processor.walk_forward_splits(proc_fe, n_splits=1, train_window_years=4, test_window_years=1)
if not folds:
    logger.error('No walk-forward folds generated; aborting')
    raise SystemExit(1)

fold = folds[0]
fold_config = (fold['train_start'], fold['train_end'], fold['test_end'])

# Prepare ML data for 15d horizon
ml = processor.prepare_ml_data('TSLA', 15, fold_config=fold_config)

# Small ANN for smoke training
X_train = ml['X_train']
y_train = ml['y_train_reg']
X_val = ml['X_val']
y_val = ml['y_val_reg']
X_test = ml['X_test']
y_test = ml['y_test_reg']

# Convert to torch loaders
train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.LongTensor((y_train>0).astype(int)))
val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val), torch.LongTensor((y_val>0).astype(int)))
test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test), torch.LongTensor((y_test>0).astype(int)))

train_loader = DataLoader(train_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

model = AdvancedANN(input_dim=ml['n_features'], hidden_layers=[256,128,64], use_ensemble=False)
trainer = AdvancedANNTrainer(model, CONFIG, logger, device=CONFIG.DEVICE)
trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE)
trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode='min', factor=0.5, patience=3)
trainer.regression_criterion = torch.nn.SmoothL1Loss()
trainer.classification_loss_weight = 0.0  # focus on numeric

# Short training
history = trainer.fit(train_loader, val_loader, epochs=3, early_stopping_patience=2)

# Predict test
preds = trainer.predict(test_loader)['regression_predictions']

rmse = np.sqrt(((preds - y_test) ** 2).mean())
mae = np.mean(np.abs(preds - y_test))

out = {
    'rmse': float(rmse),
    'mae': float(mae),
    'n_test': len(y_test)
}

out_path = os.path.join(CONFIG.RESULTS_DIR, 'wf_smoke_test_tsla_15d.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)

logger.info(f"Smoke test finished: RMSE={rmse:.6f}, MAE={mae:.6f}, saved: {out_path}")
print(json.dumps(out, indent=2))
