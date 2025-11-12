"""
Adjusted Experiment V2 - Enhanced ANN vs QINN Stock Prediction
Targeting 70%+ directional accuracy with improved feature engineering
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Set environment variable to fix encoding issues
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================

# Mini experiment configuration
STOCKS = ['TEAM', 'TSLA', 'UNH']

# Simplified horizons
HORIZONS = {
    '1d': 1,
    '15d': 15,
    '1mo': 21
}

# 4-year training configuration
TEST_CONFIG = {
    'name': '4-Year Recent Data (2019-2024)',
    'fold_config': ('2019', '2023', '2024'),
    'description': '4 years training (2019-2022) → val 2023 → test 2024'
}

MODELS = ['ann', 'qinn']

# Training configuration for better convergence
BATCH_SIZE = 32
MAX_EPOCHS = 150  # Increased epochs
EARLY_STOPPING_PATIENCE = 25  # More patience
LEARNING_RATE = 0.0005  # Lower learning rate for stability

# Device configuration
def get_device():
    """Get available device with GPU preference and CPU fallback."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("GPU not available, using CPU")
    return device

DEVICE = get_device()

# ============================================================================
# ADVANCED FEATURE ENGINEERING V2
# ============================================================================

class AdvancedTechnicalFeatures:
    """Enhanced technical indicators for improved prediction."""
    
    @staticmethod
    def add_all_features(df):
        """Add comprehensive technical features."""
        result = df.copy()
        
        # Ensure we have OHLCV columns
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            raise ValueError("Missing required OHLCV columns")
        
        # 1. Price-based features
        result['returns'] = result['Close'].pct_change()
        result['log_returns'] = np.log(result['Close'] / result['Close'].shift(1))
        result['intraday_spread'] = (result['High'] - result['Low']) / result['Close']
        result['overnight_gap'] = (result['Open'] - result['Close'].shift(1)) / result['Close'].shift(1)
        
        # 2. Moving averages and ratios
        for period in [5, 10, 20, 50]:
            result[f'sma_{period}'] = result['Close'].rolling(period).mean()
            result[f'ema_{period}'] = result['Close'].ewm(span=period, adjust=False).mean()
            result[f'price_to_sma_{period}'] = result['Close'] / result[f'sma_{period}']
            result[f'volume_sma_{period}'] = result['Volume'].rolling(period).mean()
        
        # 3. Volatility features
        for period in [5, 10, 20]:
            result[f'volatility_{period}'] = result['returns'].rolling(period).std()
            result[f'volatility_ratio_{period}'] = result[f'volatility_{period}'] / result['returns'].rolling(50).std()
        
        # 4. RSI (Relative Strength Index)
        for period in [7, 14, 21]:
            result[f'rsi_{period}'] = AdvancedTechnicalFeatures.calculate_rsi(result['Close'], period)
        
        # 5. MACD
        exp1 = result['Close'].ewm(span=12, adjust=False).mean()
        exp2 = result['Close'].ewm(span=26, adjust=False).mean()
        result['macd'] = exp1 - exp2
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        # 6. Bollinger Bands
        for period in [20]:
            middle_band = result['Close'].rolling(period).mean()
            std_dev = result['Close'].rolling(period).std()
            result[f'bb_upper_{period}'] = middle_band + (std_dev * 2)
            result[f'bb_lower_{period}'] = middle_band - (std_dev * 2)
            result[f'bb_width_{period}'] = (result[f'bb_upper_{period}'] - result[f'bb_lower_{period}']) / middle_band
            result[f'bb_position_{period}'] = (result['Close'] - result[f'bb_lower_{period}']) / (result[f'bb_upper_{period}'] - result[f'bb_lower_{period}'])
        
        # 7. Stochastic Oscillator
        for period in [14]:
            low_min = result['Low'].rolling(period).min()
            high_max = result['High'].rolling(period).max()
            result[f'stochastic_{period}'] = 100 * (result['Close'] - low_min) / (high_max - low_min + 0.0001)
        
        # 8. ATR (Average True Range)
        high_low = result['High'] - result['Low']
        high_close = np.abs(result['High'] - result['Close'].shift())
        low_close = np.abs(result['Low'] - result['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result['atr_14'] = true_range.rolling(14).mean()
        result['atr_ratio'] = result['atr_14'] / result['Close']
        
        # 9. Volume indicators
        result['volume_ratio'] = result['Volume'] / result['Volume'].rolling(10).mean()
        result['price_volume'] = result['Close'] * result['Volume']
        result['volume_trend'] = result['Volume'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # 10. Momentum indicators
        for period in [5, 10, 20]:
            result[f'momentum_{period}'] = result['Close'] - result['Close'].shift(period)
            result[f'roc_{period}'] = (result['Close'] - result['Close'].shift(period)) / result['Close'].shift(period)
        
        # 11. Support and Resistance
        for period in [20, 50]:
            result[f'resistance_{period}'] = result['High'].rolling(period).max()
            result[f'support_{period}'] = result['Low'].rolling(period).min()
            result[f'sr_ratio_{period}'] = (result['Close'] - result[f'support_{period}']) / (result[f'resistance_{period}'] - result[f'support_{period}'] + 0.0001)
        
        # 12. Pattern recognition features
        result['higher_high'] = ((result['High'] > result['High'].shift(1)) & 
                                 (result['High'].shift(1) > result['High'].shift(2))).astype(int)
        result['lower_low'] = ((result['Low'] < result['Low'].shift(1)) & 
                               (result['Low'].shift(1) < result['Low'].shift(2))).astype(int)
        result['inside_day'] = ((result['High'] < result['High'].shift(1)) & 
                                (result['Low'] > result['Low'].shift(1))).astype(int)
        
        # 13. Advanced features
        result['vwap'] = (result['Close'] * result['Volume']).cumsum() / result['Volume'].cumsum()
        result['price_to_vwap'] = result['Close'] / result['vwap']
        
        # 14. Lagged features (crucial for time series)
        for lag in [1, 2, 3, 5, 10]:
            result[f'returns_lag_{lag}'] = result['returns'].shift(lag)
            result[f'volume_lag_{lag}'] = result['Volume'].shift(lag) / result['Volume'].rolling(20).mean()
        
        # 15. Rolling statistical features
        for period in [10, 20]:
            result[f'skew_{period}'] = result['returns'].rolling(period).skew()
            result[f'kurtosis_{period}'] = result['returns'].rolling(period).kurt()
        
        # Clean up NaN and inf values
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return result
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 0.0001)
        rsi = 100 - (100 / (1 + rs))
        return rsi

# ============================================================================
# IMPROVED DATA PROCESSOR
# ============================================================================

class ImprovedDataProcessor:
    """Enhanced data processor with better feature engineering."""
    
    def __init__(self, logger):
        self.logger = logger
        self.feature_engineer = AdvancedTechnicalFeatures()
        
    def load_stock_data(self, stock_symbol):
        """Load stock data from CSV."""
        try:
            # support both `TEAM.csv` and `TEAM_data.csv` naming conventions
            candidates = [f'data/{stock_symbol}.csv', f'data/{stock_symbol}_data.csv', f'data/{stock_symbol.upper()}.csv']
            file_path = None
            for c in candidates:
                if os.path.exists(c):
                    file_path = c
                    break
            if file_path is None:
                raise FileNotFoundError(f"No CSV found for {stock_symbol} in data/ (looked for: {candidates})")

            df = pd.read_csv(file_path)
            
            # Ensure datetime index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            # normalize column names
            df.columns = df.columns.str.strip()
            # fix common lowercase variants
            cols_lower = {c.lower(): c for c in df.columns}
            for req in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if req not in df.columns and req.lower() in cols_lower:
                    df.rename(columns={cols_lower[req.lower()]: req}, inplace=True)

            # Coerce critical columns to numeric (handle commas, $ etc.)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(',', '').str.replace('$', '').str.strip()
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Fill small gaps after coercion
            df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(method='ffill').fillna(method='bfill')
            
            # Required columns
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required):
                raise ValueError(f"Missing required columns for {stock_symbol}")
            
            self.logger.info(f"Loaded {stock_symbol}: {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {stock_symbol}: {e}")
            raise
    
    def process_stock_features(self, df, stock_symbol):
        """Process stock data with technical indicators."""
        try:
            # Add all technical features
            processed = self.feature_engineer.add_all_features(df)
            
            # Add macro features if available
            processed = self.add_macro_features(processed)
            
            self.logger.info(f"Processed {stock_symbol}: {len(processed.columns)} features")
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing {stock_symbol}: {e}")
            raise
    
    def add_macro_features(self, df):
        """Add macro economic features if available."""
        try:
            macro_file = 'data/macro_data.csv'
            if os.path.exists(macro_file):
                macro_df = pd.read_csv(macro_file)
                if 'Date' in macro_df.columns:
                    macro_df['Date'] = pd.to_datetime(macro_df['Date'])
                    macro_df.set_index('Date', inplace=True)
                
                # Merge macro features
                df = df.merge(macro_df, left_index=True, right_index=True, how='left')
                df.fillna(method='ffill', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Could not add macro features: {e}")
            return df
    
    def prepare_ml_data(self, stock_symbol, horizon, fold_config=('2019', '2023', '2024')):
        """Prepare data for machine learning with proper train/val/test split."""
        # Load and process data
        raw_data = self.load_stock_data(stock_symbol)
        processed_data = self.process_stock_features(raw_data, stock_symbol)

        # Feature selection (remove obvious non-predictive columns and keep numeric types only)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        candidate_cols = [col for col in processed_data.columns if col not in exclude_cols]
        # Keep only numeric columns
        feature_cols = [col for col in candidate_cols if pd.api.types.is_numeric_dtype(processed_data[col])]

        # Ensure datetime index
        if not isinstance(processed_data.index, pd.DatetimeIndex):
            try:
                processed_data.index = pd.to_datetime(processed_data.index)
            except Exception:
                raise ValueError("Index could not be converted to datetime. Ensure CSV has a Date column or index is datetime.")

        # Create year-based masks for splits (we will compute targets per-split to avoid leakage)
        try:
            train_start = int(fold_config[0])
            val_year = int(fold_config[1])
            test_year = int(fold_config[2])
        except Exception:
            train_start = 2019
            val_year = 2023
            test_year = 2024

        train_mask = (processed_data.index.year >= train_start) & (processed_data.index.year <= (val_year - 1))
        val_mask = processed_data.index.year == val_year
        test_mask = processed_data.index.year == test_year

        train_df = processed_data[train_mask].copy()
        val_df = processed_data[val_mask].copy()
        test_df = processed_data[test_mask].copy()

        # Helper to compute forward target on a subset without using outside data
        def _compute_targets(sub_df):
            sub = sub_df.copy()
            future_price = sub['Close'].shift(-horizon)
            sub[f'target_{horizon}d'] = future_price / sub['Close'] - 1
            # drop last rows without forward prices
            if len(sub) > horizon:
                sub = sub.iloc[:-horizon]
            return sub

        train_df = _compute_targets(train_df)
        val_df = _compute_targets(val_df)
        test_df = _compute_targets(test_df)

        # If split becomes empty after dropping tail rows, fallback to temporal quantile split
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            self.logger.warning("Year-based split produced empty partitions after target trimming; falling back to temporal 60/20/20 split")
            n = len(processed_data)
            if n < 3:
                raise ValueError("Not enough data after processing to create train/val/test splits")
            train_end = int(n * 0.6)
            val_end = int(n * 0.8)
            train_df = processed_data.iloc[:train_end].copy()
            val_df = processed_data.iloc[train_end:val_end].copy()
            test_df = processed_data.iloc[val_end:].copy()
            train_df = _compute_targets(train_df)
            val_df = _compute_targets(val_df)
            test_df = _compute_targets(test_df)

        # Now build X/y for each split using only data generated inside that split
        X_train = train_df[feature_cols].values
        y_train = train_df[f'target_{horizon}d'].values
        X_val = val_df[feature_cols].values
        y_val = val_df[f'target_{horizon}d'].values
        X_test = test_df[feature_cols].values
        y_test = test_df[f'target_{horizon}d'].values

        # Build index lists and close price arrays from the per-split DataFrames
        train_index = train_df.index
        val_index = val_df.index
        test_index = test_df.index

        train_close = train_df['Close'].values
        val_close = val_df['Close'].values
        test_close = test_df['Close'].values

        # If any split is empty after trimming for forward targets, fallback to temporal quantile split
        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            self.logger.warning("Year-based split produced empty partitions after target trimming; falling back to temporal 60/20/20 split")
            n = len(processed_data)
            if n < 3:
                raise ValueError("Not enough data after processing to create train/val/test splits")
            train_end = int(n * 0.6)
            val_end = int(n * 0.8)

            full_train = processed_data.iloc[:train_end].copy()
            full_val = processed_data.iloc[train_end:val_end].copy()
            full_test = processed_data.iloc[val_end:].copy()

            # compute targets on these fallback partitions
            def _ct(sub):
                sub2 = sub.copy()
                sub2[f'target_{horizon}d'] = (sub2['Close'].shift(-horizon) / sub2['Close']) - 1
                if len(sub2) > horizon:
                    sub2 = sub2.iloc[:-horizon]
                return sub2

            full_train = _ct(full_train)
            full_val = _ct(full_val)
            full_test = _ct(full_test)

            X_train = full_train[feature_cols].values
            y_train = full_train[f'target_{horizon}d'].values
            X_val = full_val[feature_cols].values
            y_val = full_val[f'target_{horizon}d'].values
            X_test = full_test[feature_cols].values
            y_test = full_test[f'target_{horizon}d'].values

            train_index = full_train.index
            val_index = full_val.index
            test_index = full_test.index

            train_close = full_train['Close'].values
            val_close = full_val['Close'].values
            test_close = full_test['Close'].values

        self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feature_cols,
            'train_index': list(pd.to_datetime(train_index).strftime('%Y-%m-%d')),
            'val_index': list(pd.to_datetime(val_index).strftime('%Y-%m-%d')),
            'test_index': list(pd.to_datetime(test_index).strftime('%Y-%m-%d')),
            'train_close': list(train_close),
            'val_close': list(val_close),
            'test_close': list(test_close)
        }

# ============================================================================
# OPTIMIZED MODEL CONFIGURATIONS
# ============================================================================

def create_optimized_ann_model(input_dim):
    """Create optimized ANN model."""
    from models.ann_model import AdvancedANN
    
    return AdvancedANN(
        input_dim=input_dim,
        hidden_layers=[512, 256, 128, 64, 32, 16],
        dropout_rate=0.25,  # Reduced dropout
        use_attention=True,
        use_residual=True,
        use_ensemble=False  # Disable ensemble for stability
    )

def create_optimized_qinn_model(input_dim):
    """Create optimized QINN model."""
    from models.qinn_model import QINN
    
    return QINN(
        input_dim=input_dim,
        n_qubits=4,  # Reduced for speed
        n_quantum_layers=2,
        n_classical_layers=3,
        classical_hidden_dim=256,
        encoding_type='angle',
        ansatz='hardware_efficient',
        n_circuits=1,
        dropout_rate=0.25
    )

# ============================================================================
# FINANCIAL METRICS
# ============================================================================

class TradingMetrics:
    """Calculate trading performance metrics."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
        """Annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = returns - (risk_free_rate / 252)
        return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(returns):
        """Maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    @staticmethod
    def calculate_win_rate(returns):
        """Percentage of winning trades."""
        return (returns > 0).mean() * 100

# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_improved_experiment(stock, horizon_name, horizon_days, model_type, processor, logger):
    """Run improved single experiment."""
    
    experiment_id = f"{stock}_{horizon_name}_{model_type}"
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting: {experiment_id}")
    logger.info(f"{'='*60}")
    
    results = {
        'experiment_id': experiment_id,
        'stock': stock,
        'horizon': horizon_name,
        'horizon_days': horizon_days,
        'model': model_type
    }
    
    try:
        # Prepare data (use global TEST_CONFIG fold_config for consistent splits)
        ml_data = processor.prepare_ml_data(stock, horizon_days, fold_config=TEST_CONFIG.get('fold_config'))

        X_train = ml_data['X_train']
        y_train = ml_data['y_train']
        X_val = ml_data['X_val']
        y_val = ml_data['y_val']
        X_test = ml_data['X_test']
        y_test = ml_data['y_test']
        
        # Important: Use StandardScaler instead of RobustScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Optional: Apply PCA for dimensionality reduction
        if X_train.shape[1] > 100:
            pca = PCA(n_components=0.95)  # Keep 95% variance
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)
            X_test = pca.transform(X_test)
            logger.info(f"Applied PCA: {ml_data['X_train'].shape[1]} -> {X_train.shape[1]} features")
        
        logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Create classification targets
        y_train_cls = (y_train > 0).astype(int)
        y_val_cls = (y_val > 0).astype(int)
        y_test_cls = (y_test > 0).astype(int)
        
        # Create balanced weights for classification
        pos_weight = len(y_train_cls) / (2 * np.sum(y_train_cls))
        neg_weight = len(y_train_cls) / (2 * (len(y_train_cls) - np.sum(y_train_cls)))
        sample_weights = np.where(y_train_cls == 1, pos_weight, neg_weight)
        
        # Create data loaders
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
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        input_dim = X_train.shape[1]
        
        # Create and train model
        if model_type == 'ann':
            model = create_optimized_ann_model(input_dim)
            from models.ann_model import AdvancedANNTrainer
            trainer = AdvancedANNTrainer(model, None, logger, device=DEVICE)
        else:
            model = create_optimized_qinn_model(input_dim)
            from models.qinn_model import QINNTrainer
            trainer = QINNTrainer(model, None, logger, device=DEVICE)
        
        # Setup custom optimizer with lower learning rate
        trainer.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-5
        )
        trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        logger.info(f"Training {model_type.upper()} model...")
        start_time = time.time()
        
        # Train with more epochs and patience
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=MAX_EPOCHS,
            early_stopping_patience=EARLY_STOPPING_PATIENCE
        )
        
        train_time = time.time() - start_time
        
        # Get predictions
        predictions = trainer.predict(test_loader)
        y_pred = predictions['regression_predictions']
        y_pred_cls = predictions['classification_predictions']
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Classification metrics
        accuracy = accuracy_score(y_test_cls, y_pred_cls) * 100
        
        # Directional accuracy
        direction_correct = ((y_pred > 0) == (y_test > 0)).mean() * 100
        
        # Financial metrics
        trading_returns = np.where(y_pred_cls == 1, y_test, -y_test)
        sharpe = TradingMetrics.calculate_sharpe_ratio(trading_returns)
        max_dd = TradingMetrics.calculate_max_drawdown(trading_returns)
        win_rate = TradingMetrics.calculate_win_rate(trading_returns)
        
        # Store results
        results['status'] = 'success'
        results['metrics'] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'accuracy': accuracy,
            'directional_accuracy': direction_correct,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate
        }
        results['training_time'] = train_time
        results['epochs_trained'] = len(history.get('train_loss', []))
        
        logger.info(f"Results for {model_type.upper()}:")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  R2: {r2:.4f}")
        logger.info(f"  Accuracy: {accuracy:.2f}%")
        logger.info(f"  Directional Accuracy: {direction_correct:.2f}%")
        logger.info(f"  Sharpe Ratio: {sharpe:.3f}")
        logger.info(f"  Win Rate: {win_rate:.2f}%")
        logger.info(f"  Training Time: {train_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error in {experiment_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        results['status'] = 'failed'
        results['error'] = str(e)
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print(" IMPROVED ANN vs QINN STOCK PREDICTION EXPERIMENT ".center(80))
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Stocks: {', '.join(STOCKS)}")
    print(f"  Horizons: {', '.join(HORIZONS.keys())}")
    print(f"  Training: {TEST_CONFIG['description']}")
    print(f"  Device: {DEVICE}")
    print(f"  Target: 70%+ directional accuracy")
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('experiment.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('ImprovedExperiment')
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Initialize processor
    processor = ImprovedDataProcessor(logger)
    
    print("\n" + "="*80)
    print(" RUNNING EXPERIMENTS ".center(80))
    print("="*80)
    
    all_results = []
    total_experiments = len(STOCKS) * len(HORIZONS) * len(MODELS)
    # Prepare results file early so we can write incremental progress
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f"improved_results_{timestamp}.json"
    # create empty file
    with open(results_file, 'w') as f:
        json.dump([], f)
    current = 0
    start_time = time.time()
    
    for stock in STOCKS:
        print(f"\nStock: {stock}")
        
        for horizon_name, horizon_days in HORIZONS.items():
            print(f"  Horizon: {horizon_name} ({horizon_days} days)")
            
            for model_type in MODELS:
                current += 1
                print(f"    Model: {model_type.upper()} [{current}/{total_experiments}]")
                
                result = run_improved_experiment(
                    stock=stock,
                    horizon_name=horizon_name,
                    horizon_days=horizon_days,
                    model_type=model_type,
                    processor=processor,
                    logger=logger
                )
                
                all_results.append(result)

                # Incremental save after every experiment (ensures partial results are preserved)
                try:
                    with open(results_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                except Exception as e:
                    logger.warning(f"Could not save incremental results: {e}")
                
                # Progress update
                if result['status'] == 'success':
                    acc = result['metrics']['directional_accuracy']
                    sharpe = result['metrics']['sharpe_ratio']
                    print(f"      Accuracy: {acc:.1f}%")
                    print(f"      Sharpe: {sharpe:.2f}")
                else:
                    print(f"      Failed: {result.get('error', 'Unknown error')}")
    
    # Save results
    print("\n" + "="*80)
    print(" SAVING RESULTS ".center(80))
    print("="*80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f"improved_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    # Generate summary
    print("\n" + "="*80)
    print(" EXPERIMENT SUMMARY ".center(80))
    print("="*80)
    
    successful = [r for r in all_results if r['status'] == 'success']
    
    if successful:
        # Model comparison
        for model_type in MODELS:
            model_results = [r for r in successful if r['model'] == model_type]
            if model_results:
                avg_acc = np.mean([r['metrics']['directional_accuracy'] for r in model_results])
                avg_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in model_results])
                best_acc = max([r['metrics']['directional_accuracy'] for r in model_results])
                
                print(f"\n{model_type.upper()}:")
                print(f"  Average Accuracy: {avg_acc:.2f}%")
                print(f"  Best Accuracy: {best_acc:.2f}%")
                print(f"  Average Sharpe: {avg_sharpe:.3f}")
    
    total_time = time.time() - start_time
    print(f"\nTotal Time: {total_time/60:.2f} minutes")
    print("\n" + "="*80)
    print(" EXPERIMENT COMPLETE ".center(80))
    print("="*80)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)