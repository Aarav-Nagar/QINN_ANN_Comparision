"""
Complete Main Experiment - ANN vs QINN Stock Prediction
Runs comprehensive experiments across multiple stocks, horizons, and test configurations
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from preprocessing.data_processor import DataProcessor
from models.ann_model import AdvancedANN, AdvancedANNTrainer
from models.qinn_model import QINN, QINNTrainer
from utils.config import CONFIG, setup_logging

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Experiment parameters
STOCKS = [
    'NVDA', 'MS', 'TSLA', 'TEAM',       # Tech
    'C', 'BRK-B', 'BAC',                # Finance
    'PFE', 'UNH', 'NKTR',               # Healthcare
    'CVX', 'XOM',                       # Energy
    'TRIP', 'LULU',                     # Travel/Retail
    'AVAV'                              # Aerospace
]

TEST_CONFIGS = {
    'long_historical': {
        'name': 'Long Historical (2015-2025)',
        'fold_config': ('2015', '2023', '2024'),
        'description': '8 years training (2015-2022) → val 2023 → test 2024'
    },
    'recent_data': {
        'name': 'Recent Data (2020-2025)',
        'fold_config': ('2020', '2023', '2024'),
        'description': '3 years training (2020-2022) → val 2023 → test 2024'
    }
}

HORIZONS = {
    '1d': 1,
    '15d': 15,
    '1mo': 21,
    '3mo': 63
}

MODELS = ['ann', 'qinn']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(title, width=80, char='='):
    """Print formatted header"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")

def format_time(seconds):
    """Format seconds to readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100
    
    # Directional accuracy
    if len(y_true) > 1:
        true_dir = np.diff(y_true) > 0
        pred_dir = np.diff(y_pred) > 0
        dir_acc = np.mean(true_dir == pred_dir) * 100
    else:
        dir_acc = 0.0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'directional_accuracy': float(dir_acc)
    }

def prepare_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """Prepare PyTorch DataLoaders"""
    # Create classification targets (up/down)
    y_train_cls = (y_train > 0).astype(int)
    y_val_cls = (y_val > 0).astype(int)
    y_test_cls = (y_test > 0).astype(int)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
        torch.LongTensor(y_train_cls)
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
        torch.LongTensor(y_val_cls)
    )
    
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test),
        torch.LongTensor(y_test_cls)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(stock, horizon_name, horizon_days, test_name, fold_config,
                         model_type, processor, processed_stocks, config, logger, device):
    """Run a single experiment configuration"""
    
    experiment_id = f"{stock}_{horizon_name}_{test_name}_{model_type}"
    logger.info(f"Starting: {experiment_id}")
    
    try:
        # Prepare ML data
        ml_data = processor.prepare_ml_data(stock, horizon_days, fold_config)
        
        X_train = ml_data['X_train']
        y_train = ml_data['y_train_reg']
        X_val = ml_data['X_val']
        y_val = ml_data['y_val_reg']
        X_test = ml_data['X_test']
        y_test = ml_data['y_test_reg']
        
        logger.info(f"  Data: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}, Features={X_train.shape[1]}")
        
        # Prepare dataloaders
        train_loader, val_loader, test_loader = prepare_dataloaders(
            X_train, y_train, X_val, y_val, X_test, y_test, batch_size=BATCH_SIZE
        )
        
        input_dim = X_train.shape[1]
        
        # Create model
        if model_type == 'ann':
            model = AdvancedANN(
                input_dim=input_dim,
                hidden_layers=[512, 256, 128, 64, 32],
                dropout_rate=0.3,
                use_attention=True,
                use_residual=True,
                use_ensemble=True
            )
            trainer = AdvancedANNTrainer(model, config, logger, device=device)
            
        elif model_type == 'qinn':
            model = QINN(
                input_dim=input_dim,
                n_qubits=min(8, max(4, input_dim // 8)),
                n_quantum_layers=2,
                n_classical_layers=2,
                classical_hidden_dim=128,
                encoding_type='angle',
                ansatz='hardware_efficient',
                n_circuits=2,
                dropout_rate=0.3
            )
            trainer = QINNTrainer(model, config, logger, device=device)
        
        else:
            raise ValueError(f"Unknown model: {model_type}")
        
        # Train model
        logger.info(f"  Training {model_type.upper()}...")
        start_time = time.time()
        
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=50,  # Reduced for faster experiments
            early_stopping_patience=10
        )
        
        train_time = time.time() - start_time
        logger.info(f"  Training completed in {format_time(train_time)}")
        
        # Get predictions
        predictions = trainer.predict(test_loader)
        y_pred = predictions['regression_predictions']
        y_pred_cls = predictions['classification_predictions']
        y_test_cls = (y_test > 0).astype(int)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        
        logger.info(f"  Results: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}, Acc={metrics['directional_accuracy']:.1f}%")
        
        # Get model summary
        model_summary = trainer.get_model_summary()
        
        return {
            'status': 'success',
            'experiment_id': experiment_id,
            'stock': stock,
            'horizon': horizon_name,
            'horizon_days': horizon_days,
            'test': test_name,
            'model': model_type,
            'metrics': metrics,
            'training_time': train_time,
            'best_epoch': history.get('val_loss', []).index(min(history.get('val_loss', [1]))) if history.get('val_loss') else 0,
            'final_train_loss': history['train_loss'][-1] if history.get('train_loss') else None,
            'final_val_loss': history['val_loss'][-1] if history.get('val_loss') else None,
            'model_params': model_summary.get('total_parameters', 0),
            'data_info': {
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'n_features': input_dim
            }
        }
        
    except Exception as e:
        logger.error(f"  Error in {experiment_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            'status': 'failed',
            'experiment_id': experiment_id,
            'stock': stock,
            'horizon': horizon_name,
            'test': test_name,
            'model': model_type,
            'error': str(e)
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main experiment execution"""
    
    print_header("ANN vs QINN STOCK PREDICTION EXPERIMENT", char='=')
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Stocks: {len(STOCKS)}")
    print(f"Tests: {len(TEST_CONFIGS)}")
    print(f"Horizons: {len(HORIZONS)}")
    print(f"Models: {len(MODELS)}")
    print(f"Total Experiments: {len(STOCKS) * len(TEST_CONFIGS) * len(HORIZONS) * len(MODELS)}\n")
    
    # Setup
    logger = setup_logging(CONFIG)
    logger.info("="*80)
    logger.info("EXPERIMENT STARTED")
    logger.info("="*80)
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Initialize processor
    print_header("LOADING AND PROCESSING DATA")
    processor = DataProcessor(config=CONFIG, logger=logger)
    
    stock_data = processor.load_all_data()
    logger.info(f"Loaded {len(stock_data)} stocks")
    
    processed_stocks = processor.process_all_data()
    logger.info(f"Processed {len(processed_stocks)} stocks")
    
    # Filter available stocks
    available_stocks = [s for s in STOCKS if s in processed_stocks]
    logger.info(f"Available for experiments: {len(available_stocks)} stocks")
    
    if len(available_stocks) < len(STOCKS):
        missing = [s for s in STOCKS if s not in processed_stocks]
        logger.warning(f"Missing stocks: {missing}")
    
    # Run experiments
    print_header("RUNNING EXPERIMENTS")
    
    all_results = []
    total_experiments = len(available_stocks) * len(TEST_CONFIGS) * len(HORIZONS) * len(MODELS)
    current_experiment = 0
    experiment_start_time = time.time()
    
    for test_name, test_config in TEST_CONFIGS.items():
        print_header(f"TEST: {test_config['name']}", char='-')
        print(f"{test_config['description']}\n")
        
        for stock in available_stocks:
            print(f"\n📊 Stock: {stock}")
            
            for horizon_name, horizon_days in HORIZONS.items():
                print(f"  ⏱️  Horizon: {horizon_name} ({horizon_days} days)")
                
                for model_type in MODELS:
                    current_experiment += 1
                    print(f"    🤖 {model_type.upper()} [{current_experiment}/{total_experiments}]", end=' ')
                    
                    result = run_single_experiment(
                        stock=stock,
                        horizon_name=horizon_name,
                        horizon_days=horizon_days,
                        test_name=test_name,
                        fold_config=test_config['fold_config'],
                        model_type=model_type,
                        processor=processor,
                        processed_stocks=processed_stocks,
                        config=CONFIG,
                        logger=logger,
                        device=DEVICE
                    )
                    
                    all_results.append(result)
                    
                    # Progress update
                    elapsed = time.time() - experiment_start_time
                    avg_time = elapsed / current_experiment
                    remaining = avg_time * (total_experiments - current_experiment)
                    
                    if result['status'] == 'success':
                        print(f"✅ RMSE: {result['metrics']['rmse']:.4f} | Time: {format_time(result['training_time'])}")
                    else:
                        print(f"❌ Failed")
                    
                    print(f"       Progress: {current_experiment}/{total_experiments} | Remaining: ~{format_time(remaining)}")
    
    # Save results
    print_header("SAVING RESULTS")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f"results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✅ Results saved to: {results_file}")
    
    # Generate summary
    print_header("EXPERIMENT SUMMARY")
    
    total_time = time.time() - experiment_start_time
    successful = sum(1 for r in all_results if r['status'] == 'success')
    failed = len(all_results) - successful
    
    print(f"Total Experiments: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(successful/len(all_results)*100):.1f}%")
    print(f"Total Time: {format_time(total_time)}")
    print(f"Average Time: {format_time(total_time/len(all_results))}")
    
    # Best results by model
    print("\n📈 Best Results by Model:")
    for model_type in MODELS:
        model_results = [r for r in all_results if r['status'] == 'success' and r['model'] == model_type]
        if model_results:
            best = max(model_results, key=lambda x: x['metrics']['r2'])
            print(f"\n  {model_type.upper()}:")
            print(f"    Best R²: {best['metrics']['r2']:.4f}")
            print(f"    Stock: {best['stock']}, Horizon: {best['horizon']}, Test: {best['test']}")
            print(f"    RMSE: {best['metrics']['rmse']:.4f}, Dir.Acc: {best['metrics']['directional_accuracy']:.1f}%")
    
    print_header("EXPERIMENT COMPLETED")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results: {results_file}\n")
    
    logger.info("="*80)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)