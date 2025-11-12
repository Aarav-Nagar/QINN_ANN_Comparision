"""
Robust data loading and processing for ANN vs QINN stock prediction research.

This module handles all aspects of data loading, validation, cleaning, and preparation
for multi-horizon stock prediction with walk-forward validation.

INSTRUCTIONS: This is PART 1 of 2. Copy this entire code, then continue with Part 2.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import glob
from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import joblib

warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Comprehensive data processing class for stock prediction research.
    
    Handles data loading, validation, cleaning, feature engineering,
    and preparation for machine learning models.
    """
    
    def __init__(self, config, logger: logging.Logger):
        """
        Initialize DataProcessor with configuration and logging.
        
        Args:
            config: ModelConfig instance with all parameters
            logger: Configured logger instance
        """
        self.config = config
        self.logger = logger
        self.scalers = {}
        self.feature_selectors = {}
        self.outlier_detectors = {}
        
        # Initialize data containers
        self.stock_data = {}
        self.macro_data = None
        self.processed_data = {}
        
        self.logger.info("DataProcessor initialized")

    def scan_data_directory(self) -> Dict[str, List[str]]:
        """
        Scan data directory for available files and validate structure.
        
        Returns:
            Dict containing categorized file listings
        """
        self.logger.info(f"Scanning data directory: {self.config.DATA_DIR}")
        
        if not os.path.exists(self.config.DATA_DIR):
            self.logger.error(f"Data directory not found: {self.config.DATA_DIR}")
            raise FileNotFoundError(f"Data directory not found: {self.config.DATA_DIR}")
        
        file_scan = {
            'stock_files': [],
            'macro_files': [],
            'missing_stocks': [],
            'extra_files': []
        }
        
        # Look for stock data files
        for stock in self.config.STOCKS:
            possible_patterns = [
                f"{stock}.csv",
                f"{stock}_data.csv", 
                f"{stock.lower()}.csv",
                f"stock_{stock}.csv"
            ]
            
            found = False
            for pattern in possible_patterns:
                file_path = os.path.join(self.config.DATA_DIR, pattern)
                if os.path.exists(file_path):
                    file_scan['stock_files'].append(file_path)
                    found = True
                    break
            
            if not found:
                file_scan['missing_stocks'].append(stock)
        
        # Look for macro data files
        macro_patterns = ['macro*.csv', 'fred*.csv', 'economic*.csv']
        for pattern in macro_patterns:
            macro_files = glob.glob(os.path.join(self.config.DATA_DIR, pattern))
            file_scan['macro_files'].extend(macro_files)
        
        # Log findings
        self.logger.info(f"Found {len(file_scan['stock_files'])} stock data files")
        self.logger.info(f"Found {len(file_scan['macro_files'])} macro data files")
        
        if file_scan['missing_stocks']:
            self.logger.warning(f"Missing stock data for: {file_scan['missing_stocks']}")
        
        return file_scan

    def load_stock_data(self, stock_symbol: str) -> Optional[pd.DataFrame]:
        """
        Load and validate individual stock data with error handling.
        
        Args:
            stock_symbol: Stock ticker symbol
            
        Returns:
            DataFrame with stock data or None if loading fails
        """
        try:
            # Try multiple file patterns
            possible_files = [
                os.path.join(self.config.DATA_DIR, f"{stock_symbol}.csv"),
                os.path.join(self.config.DATA_DIR, f"{stock_symbol}_data.csv"),
                os.path.join(self.config.DATA_DIR, f"{stock_symbol.lower()}.csv")
            ]
            
            df = None
            for file_path in possible_files:
                if os.path.exists(file_path):
                    self.logger.debug(f"Loading {stock_symbol} from {file_path}")
                    
                    # Read CSV - skip the ticker and date label rows
                    df = pd.read_csv(
                        file_path,
                        skiprows=[1, 2]  # Skip "Ticker, NVDA..." and "Date, NaN..." rows
                    )
                    
                    # The first column "Price" actually contains dates
                    if 'Price' in df.columns:
                        df['Date'] = pd.to_datetime(df['Price'], errors='coerce')
                        df = df.set_index('Date')
                        df = df.drop('Price', axis=1, errors='ignore')
                        df.index.name = 'Date'
                    else:
                        # Fallback: try to find date column
                        date_col = None
                        for col in ['Date', 'date', 'DATE', 'Unnamed: 0']:
                            if col in df.columns:
                                date_col = col
                                break
                        
                        if date_col:
                            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                            df = df.set_index(date_col)
                            df.index.name = 'Date'
                        else:
                            # Last resort: try to parse index as dates
                            df.index = pd.to_datetime(df.index, errors='coerce')
                    
                    break
            
            if df is None:
                self.logger.warning(f"No data file found for {stock_symbol}")
                return None
            
            # Remove any rows where the date couldn't be parsed (NaT)
            df = df[df.index.notna()]
            
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    df = df[df.index.notna()]
                except:
                    self.logger.error(f"Cannot convert index to datetime for {stock_symbol}")
                    return None
            
            # Validate required columns
            required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.error(f"Missing columns in {stock_symbol}: {missing_cols}")
                return None
            
            # Convert columns to numeric (in case they're strings)
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Validate data quality
            if len(df) < 252:  # Less than 1 year of data
                self.logger.warning(f"Insufficient data for {stock_symbol}: {len(df)} rows")
                return None
            
            # Check for excessive missing values
            missing_pct = df[required_cols].isnull().sum().max() / len(df)
            if missing_pct > 0.05:  # More than 5% missing
                self.logger.warning(f"High missing data in {stock_symbol}: {missing_pct:.2%}")
            
            # Sort by date and ensure no duplicates
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]
            
            # Forward fill missing values (common in stock data)
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            self.logger.info(f"Successfully loaded {stock_symbol}: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {stock_symbol}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def load_macro_data(self) -> Optional[pd.DataFrame]:
        """
        Load macroeconomic data with multiple source handling.
        
        Returns:
            DataFrame with macro indicators or None if loading fails
        """
        try:
            # Look for macro data files
            macro_patterns = ['macro*.csv', 'fred*.csv', 'economic*.csv']
            macro_files = []
            
            for pattern in macro_patterns:
                files = glob.glob(os.path.join(self.config.DATA_DIR, pattern))
                macro_files.extend(files)
            
            if not macro_files:
                self.logger.warning("No macroeconomic data files found")
                return self._create_minimal_macro_data()
            
            # Load and combine macro data
            macro_dfs = []
            for file_path in macro_files:
                try:
                    df = pd.read_csv(file_path)
                    
                    # Handle date column
                    date_col = None
                    for col in ['Date', 'date', 'DATE', 'Unnamed: 0']:
                        if col in df.columns:
                            date_col = col
                            break
                    
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col])
                        df = df.set_index(date_col)
                    else:
                        df.index = pd.to_datetime(df.index)
                    
                    # Ensure datetime index
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    
                    macro_dfs.append(df)
                    self.logger.debug(f"Loaded macro data from {file_path}")
                except Exception as e:
                    self.logger.warning(f"Could not load macro file {file_path}: {e}")
            
            if not macro_dfs:
                return self._create_minimal_macro_data()
            
            # Combine all macro data
            macro_data = pd.concat(macro_dfs, axis=1)
            macro_data = macro_data.sort_index()
            
            # Remove duplicate columns
            macro_data = macro_data.loc[:, ~macro_data.columns.duplicated()]
            
            # Forward fill missing values
            macro_data = macro_data.fillna(method='ffill').fillna(method='bfill')
            
            self.logger.info(f"Loaded macro data: {len(macro_data)} rows, {len(macro_data.columns)} columns")
            return macro_data
            
        except Exception as e:
            self.logger.error(f"Error loading macro data: {str(e)}")
            return self._create_minimal_macro_data()

    def _create_minimal_macro_data(self) -> pd.DataFrame:
        """Create minimal macro data with time features if real data unavailable."""
        
        self.logger.info("Creating minimal macro data with time features")
        
        # Create date range from 2015 to 2025
        date_range = pd.date_range(start='2015-01-01', end='2025-12-31', freq='D')
        
        macro_data = pd.DataFrame(index=date_range)
        
        # Add time-based features
        macro_data['Month'] = macro_data.index.month
        macro_data['Quarter'] = macro_data.index.quarter
        macro_data['DayOfYear'] = macro_data.index.dayofyear
        macro_data['Weekday'] = macro_data.index.weekday
        macro_data['Year'] = macro_data.index.year
        
        # Add cyclical encodings
        macro_data['Month_sin'] = np.sin(2 * np.pi * macro_data['Month'] / 12)
        macro_data['Month_cos'] = np.cos(2 * np.pi * macro_data['Month'] / 12)
        macro_data['Weekday_sin'] = np.sin(2 * np.pi * macro_data['Weekday'] / 7)
        macro_data['Weekday_cos'] = np.cos(2 * np.pi * macro_data['Weekday'] / 7)
        
        return macro_data

    def create_features_and_targets(self, df: pd.DataFrame, stock_symbol: str) -> pd.DataFrame:
        """
        Create features and prediction targets for multiple horizons.
        
        Args:
            df: Stock price DataFrame
            stock_symbol: Stock symbol for logging
            
        Returns:
            DataFrame with features and targets
        """
        try:
            # Create features (technical indicators etc.) and targets.
            # NOTE: target creation that uses future prices MUST be done on a
            # per-split basis (train / test) to avoid look-ahead leakage. This
            # method will continue to compute technical features; targets are
            # computed later when preparing ML data for each split.
            result_df = df.copy()

            # Basic technical indicators and features are added elsewhere via
            # add_technical_indicators(), but keep this as a safe fallback for
            # downstream code that expects targets here. We will not compute
            # forward-looking targets here to avoid leakage.

            # For backward compatibility, return the dataframe unchanged here
            # (targets will be computed per-split in prepare_ml_data).
            self.logger.debug(f"Prepared features for {stock_symbol}: {len(result_df)} rows (targets computed per-split)")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error creating features/targets for {stock_symbol}: {str(e)}")
            return df
# PART 2 OF 2 - APPEND THIS TO THE END OF PART 1
# This continues the DataProcessor class

    def prepare_train_test_splits(self, df: pd.DataFrame, fold_config: Tuple[str, str, str]) -> Dict:
        """
        Create train/test splits for walk-forward validation.
        
        Args:
            df: DataFrame with features and targets
            fold_config: Tuple of (train_start, train_end, test_year)
            
        Returns:
            Dictionary with train/test data splits
        """
        try:
            train_start, train_end, test_year = fold_config
            
            # Create date filters
            train_mask = (df.index.year >= int(train_start)) & (df.index.year <= int(train_end))
            test_mask = df.index.year == int(test_year)
            
            train_data = df[train_mask].copy()
            test_data = df[test_mask].copy()
            
            if len(train_data) == 0:
                raise ValueError(f"No training data for period {train_start}-{train_end}")
            
            if len(test_data) == 0:
                raise ValueError(f"No test data for year {test_year}")
            
            return {
                'train': train_data,
                'test': test_data,
                'train_period': f"{train_start}-{train_end}",
                'test_period': test_year
            }
            
        except Exception as e:
            self.logger.error(f"Error creating train/test split: {str(e)}")
            raise

    def scale_features(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                      method: str = 'robust') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features using various scaling methods."""
        try:
            # Identify feature columns (exclude targets)
            target_cols = [col for col in train_data.columns if col.startswith('target_') or col.startswith('direction_')]
            feature_cols = [col for col in train_data.columns if col not in target_cols]
            
            train_scaled = train_data.copy()
            test_scaled = test_data.copy()
            
            if method == 'robust':
                scaler = RobustScaler()
            elif method == 'quantile':
                scaler = QuantileTransformer(output_distribution='normal', random_state=self.config.RANDOM_SEED)
            elif method == 'power':
                scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            # Fit scaler on training data and transform both sets
            train_scaled[feature_cols] = scaler.fit_transform(train_data[feature_cols])
            test_scaled[feature_cols] = scaler.transform(test_data[feature_cols])
            
            return train_scaled, test_scaled
            
        except Exception as e:
            self.logger.error(f"Error scaling features: {str(e)}")
            return train_data, test_data

    def prepare_ml_data(self, stock_symbol: str, horizon: int, fold_config: Tuple[str, str, str]) -> Dict:
        """
        Prepare final ML-ready data for a specific stock, horizon, and fold.
        
        Args:
            stock_symbol: Stock ticker symbol
            horizon: Prediction horizon in days
            fold_config: Tuple of (train_start, train_end, test_year)
            
        Returns:
            Dictionary with ML-ready data splits
        """
        try:
            if stock_symbol not in self.processed_data:
                raise ValueError(f"Processed data not available for {stock_symbol}")
            
            df = self.processed_data[stock_symbol].copy()

            # Create train/test split
            splits = self.prepare_train_test_splits(df, fold_config)

            # Compute forward-looking targets on each split separately to avoid
            # look-ahead leakage (do NOT compute targets on the full dataset).
            train_df = splits['train'].copy()
            test_df = splits['test'].copy()

            # Helper: compute targets for a dataframe subset using only data
            # within that subset
            def _compute_targets_for_subset(sub_df: pd.DataFrame) -> pd.DataFrame:
                sub = sub_df.copy()
                for h in self.config.PREDICTION_HORIZONS:
                    future_price = sub['Close'].shift(-h)
                    future_return = (future_price / sub['Close']) - 1
                    sub[f'target_{h}d'] = future_return
                    sub[f'direction_{h}d'] = (future_return > 0).astype(int)
                    sub[f'target_{h}d_ema3'] = sub[f'target_{h}d'].ewm(span=3).mean()
                    try:
                        sub[f'quintile_{h}d'] = pd.qcut(sub[f'target_{h}d'].rank(method='first'), 5, labels=False) + 1
                    except Exception:
                        sub[f'quintile_{h}d'] = (sub[f'target_{h}d'] > 0).astype(int)
                    realized_vol = sub['Close'].pct_change().rolling(window=max(5, h)).std() * np.sqrt(252)
                    sub[f'sharpe_{h}d'] = sub[f'target_{h}d'] / (realized_vol + 1e-8)
                    lookback = min(21, max(1, int(self.config.LOOKBACK_WINDOW // 3)))
                    prior_mom = sub['Close'].pct_change(periods=lookback)
                    sub[f'mom_cont_{h}d'] = (prior_mom > 0).astype(int)
                # Drop last rows that lack forward prices inside this subset
                max_h = max(self.config.PREDICTION_HORIZONS)
                if len(sub) > max_h:
                    sub = sub.iloc[:-max_h]
                return sub

            train_df = _compute_targets_for_subset(train_df)
            test_df = _compute_targets_for_subset(test_df)

            # If any split is too small (<= max horizon) so all targets become
            # NaN after forward shift, fallback to temporal 60/20/20 split.
            max_h = max(self.config.PREDICTION_HORIZONS)
            if len(train_df) <= max_h or len(test_df) <= max_h:
                self.logger.warning("Year-based split produced empty partitions after target trimming; falling back to temporal 60/20/20 split")
                full = df.copy()
                n = len(full)
                if n < 3:
                    raise ValueError("Not enough data after processing to create train/val/test splits")
                train_end = int(n * 0.6)
                val_end = int(n * 0.8)

                full_train = full.iloc[:train_end].copy()
                full_val = full.iloc[train_end:val_end].copy()
                full_test = full.iloc[val_end:].copy()

                # compute targets on these fallback partitions
                def _ct(sub):
                    sub2 = sub.copy()
                    for h in self.config.PREDICTION_HORIZONS:
                        future_price = sub2['Close'].shift(-h)
                        sub2[f'target_{h}d'] = (future_price / sub2['Close']) - 1
                        sub2[f'direction_{h}d'] = (sub2[f'target_{h}d'] > 0).astype(int)
                    max_h = max(self.config.PREDICTION_HORIZONS)
                    if len(sub2) > max_h:
                        sub2 = sub2.iloc[:-max_h]
                    return sub2

                train_df = _ct(full_train)
                val_df = _ct(full_val)
                test_df = _ct(full_test)

                # Replace train_df with concatenation of train+val for downstream
                # scaling which expects a single train split to fit on. We'll
                # create train_df as full training portion and keep val separate.
                train_df = train_df
            else:
                # create a validation split from the tail of train_df (80/20)
                val_cut = int(len(train_df) * 0.8)
                val_df = train_df.iloc[val_cut:].copy()
                train_df = train_df.iloc[:val_cut].copy()

            # Scale features (fit on training split only). Fit scaler on train_df
            # and transform both val and test (val_df may be defined above).
            # If val_df is not defined (rare), create an 80/20 split from train_df.
            if 'val_df' not in locals():
                val_cut = int(len(train_df) * 0.8)
                val_df = train_df.iloc[val_cut:].copy()
                train_df = train_df.iloc[:val_cut].copy()

            train_scaled, val_scaled = self.scale_features(train_df, val_df, method='robust')
            # scale test using scaler fit on train
            _, test_scaled = self.scale_features(train_df, test_df, method='robust')
            
            # Select target columns
            target_col = f'target_{horizon}d'
            direction_col = f'direction_{horizon}d'
            
            if target_col not in train_scaled.columns:
                raise ValueError(f"Target column {target_col} not found")
            
            # Identify target columns and remove any inadvertent target-derived
            # columns from the feature set. Some derived columns (e.g. "quintile_15d",
            # "sharpe_15d", "mom_cont_15d") are computed from forward returns
            # and therefore leak information. We will exclude any column that
            # contains the horizon token "_{h}d" for any configured horizon
            # unless it is explicitly a target or direction column.
            target_cols = [col for col in train_scaled.columns if col.startswith('target_') or col.startswith('direction_')]

            # detect suspicious target-derived columns and exclude them
            leak_like = set()
            for h in self.config.PREDICTION_HORIZONS:
                token = f'_{h}d'
                for col in train_scaled.columns:
                    # if token appears in column name but it's not the actual target/direction
                    if token in col and not (col.startswith('target_') or col.startswith('direction_')):
                        leak_like.add(col)

            # Final feature columns: everything that's not a target nor a detected leak
            feature_cols = [col for col in train_scaled.columns if col not in target_cols and col not in leak_like]

            if leak_like:
                self.logger.warning(f"Excluding {len(leak_like)} target-derived columns that would leak future info: {sorted(list(leak_like))}")
            
            # Prepare final X, y splits
            X_train = train_scaled[feature_cols].values
            y_train_reg = train_scaled[target_col].values
            y_train_cls = train_scaled[direction_col].values
            
            X_test = test_scaled[feature_cols].values
            y_test_reg = test_scaled[target_col].values
            y_test_cls = test_scaled[direction_col].values
            
            # Remove any NaN values that might remain
            train_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train_reg) | np.isnan(y_train_cls))
            test_mask = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test_reg) | np.isnan(y_test_cls))
            
            X_train = X_train[train_mask]
            y_train_reg = y_train_reg[train_mask]
            y_train_cls = y_train_cls[train_mask]
            
            X_test = X_test[test_mask]
            y_test_reg = y_test_reg[test_mask]
            y_test_cls = y_test_cls[test_mask]
            
            # Split training data into train and validation (80/20)
            from sklearn.model_selection import train_test_split

            X_train_final, X_val, y_train_reg_final, y_val_reg, y_train_cls_final, y_val_cls = train_test_split(
                X_train, y_train_reg, y_train_cls,
                test_size=0.2,
                random_state=42,
                shuffle=False  # Keep time series order
            )

            ml_data = {
                'X_train': X_train_final,
                'y_train_reg': y_train_reg_final,
                'y_train_cls': y_train_cls_final,
                'X_val': X_val,
                'y_val_reg': y_val_reg,
                'y_val_cls': y_val_cls,
                'X_test': X_test,
                'y_test_reg': y_test_reg,
                'y_test_cls': y_test_cls,
                'feature_names': feature_cols,
                'n_features': len(feature_cols),
                'train_samples': len(X_train_final),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'fold_info': {
                    'train_period': splits['train_period'],
                    'test_period': splits['test_period']
                }
            }

            self.logger.debug(
                f"ML data prepared for {stock_symbol} {horizon}d: "
                f"train={len(X_train_final)}, val={len(X_val)}, test={len(X_test)}, features={len(feature_cols)}"
            )

            return ml_data
            
        except Exception as e:
            self.logger.error(f"Error preparing ML data for {stock_symbol} {horizon}d: {str(e)}")
            raise

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all stock and macro data with comprehensive error handling."""
        self.logger.info("Starting comprehensive data loading process")
        
        # Scan directory first
        file_scan = self.scan_data_directory()
        
        # Load macro data
        self.macro_data = self.load_macro_data()
        
        # Load stock data
        loaded_stocks = {}
        failed_stocks = []
        
        for stock in self.config.STOCKS:
            try:
                stock_df = self.load_stock_data(stock)
                if stock_df is not None:
                    loaded_stocks[stock] = stock_df
                else:
                    failed_stocks.append(stock)
                    
            except Exception as e:
                self.logger.error(f"Failed to load {stock}: {str(e)}")
                failed_stocks.append(stock)
        
        self.logger.info(f"Successfully loaded {len(loaded_stocks)} stocks")
        if failed_stocks:
            self.logger.warning(f"Failed to load: {failed_stocks}")
        
        self.stock_data = loaded_stocks
        return loaded_stocks

    def process_single_stock(self, stock_symbol: str) -> Optional[pd.DataFrame]:
        """Process a single stock with feature engineering and target creation."""
        try:
            if stock_symbol not in self.stock_data:
                self.logger.error(f"Stock data not loaded for {stock_symbol}")
                return None
            
            df = self.stock_data[stock_symbol].copy()
            
            # Merge with macro data if available
            if self.macro_data is not None:
                # Align dates and merge
                df = df.merge(self.macro_data, left_index=True, right_index=True, how='left')
                df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Create features and targets
            df = self.create_features_and_targets(df, stock_symbol)
            
            # Remove any rows with NaN targets
            target_cols = [col for col in df.columns if col.startswith('target_')]
            df = df.dropna(subset=target_cols)
            
            if len(df) < 500:  # Minimum data requirement
                self.logger.warning(f"Insufficient processed data for {stock_symbol}: {len(df)} rows")
                return None
            
            self.logger.info(f"Successfully processed {stock_symbol}: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing {stock_symbol}: {str(e)}")
            return None

    def process_all_data(self) -> Dict[str, pd.DataFrame]:
        """Process all loaded stock data for machine learning."""
        self.logger.info("Processing all stock data")
        
        if not self.stock_data:
            self.logger.error("No stock data loaded. Call load_all_data() first.")
            return {}
        
        processed_data = {}
        
        for stock in self.stock_data.keys():
            processed_df = self.process_single_stock(stock)
            if processed_df is not None:
                processed_data[stock] = processed_df
        
        self.logger.info(f"Successfully processed {len(processed_data)} stocks")
        self.processed_data = processed_data
        return processed_data

    def get_data_summary(self) -> Dict:
        """Get comprehensive summary of loaded and processed data."""
        summary = {
            'stocks_loaded': len(self.stock_data),
            'stocks_processed': len(self.processed_data),
            'macro_data_available': self.macro_data is not None,
            'stock_details': {}
        }
        
        for stock, df in self.processed_data.items():
            # Safely get date range
            try:
                if isinstance(df.index, pd.DatetimeIndex):
                    date_range = (df.index.min().strftime('%Y-%m-%d'), 
                                  df.index.max().strftime('%Y-%m-%d'))
                else:
                    date_range = (str(df.index.min()), str(df.index.max()))
            except:
                date_range = ('Unknown', 'Unknown')
            
            stock_summary = {
                'rows': len(df),
                'columns': len(df.columns),
                'date_range': date_range,
                'missing_data': int(df.isnull().sum().sum()),
                'features': len([col for col in df.columns if not col.startswith(('target_', 'direction_'))])
            }
            summary['stock_details'][stock] = stock_summary
        
        if self.macro_data is not None:
            summary['macro_features'] = len(self.macro_data.columns)
        
        return summary

    # ------------------ Additional utilities for Stage 1+2 ------------------
    def add_technical_indicators(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """Add multi-timeframe technical indicators to dataframe."""
        if windows is None:
            windows = [5, 10, 20, 50, 100]

        df2 = df.copy()
        close = df2['Close']

        # Basic moving averages and ratios
        for w in windows:
            df2[f'ma_{w}'] = close.rolling(w).mean()
            df2[f'ema_{w}'] = close.ewm(span=w, adjust=False).mean()
            df2[f'ma_ratio_{w}'] = close / (df2[f'ma_{w}'] + 1e-9)

        # Volatility features
        returns = close.pct_change()
        for w in [5, 10, 20, 60]:
            df2[f'ret_std_{w}'] = returns.rolling(w).std()
            df2[f'ret_var_{w}'] = returns.rolling(w).var()
            df2[f'ret_sq_mean_{w}'] = (returns ** 2).rolling(w).mean()

        # Momentum and oscillator features
        delta = close.diff()
        for w in [5, 10, 14, 21]:
            df2[f'rsi_{w}'] = self._rsi(close, window=w)
            df2[f'mom_{w}'] = close.pct_change(periods=w)

        # ATR, Bollinger Bands, VWAP
        df2['hl_range'] = df2['High'] - df2['Low']
        df2['tr'] = np.maximum.reduce([
            (df2['High'] - df2['Low']).abs(),
            (df2['High'] - df2['Close'].shift(1)).abs(),
            (df2['Low'] - df2['Close'].shift(1)).abs()
        ])
        df2['atr_14'] = df2['tr'].rolling(14).mean()

        df2['typical_price'] = (df2['High'] + df2['Low'] + df2['Close']) / 3.0
        df2['vwap'] = (df2['typical_price'] * df2['Volume']).cumsum() / (df2['Volume'].cumsum() + 1e-9)

        # Calendar features
        df2['dow'] = df2.index.weekday
        df2['month'] = df2.index.month
        df2['dow_sin'] = np.sin(2 * np.pi * df2['dow'] / 7)
        df2['dow_cos'] = np.cos(2 * np.pi * df2['dow'] / 7)
        df2['month_sin'] = np.sin(2 * np.pi * df2['month'] / 12)
        df2['month_cos'] = np.cos(2 * np.pi * df2['month'] / 12)

        # Volume-based microstructure proxies
        df2['vol_change'] = df2['Volume'].pct_change()
        df2['vol_rolling_mean_20'] = df2['Volume'].rolling(20).mean()
        df2['vol_over_avg'] = df2['Volume'] / (df2['vol_rolling_mean_20'] + 1e-9)

        # Fill remaining NaNs
        df2 = df2.fillna(method='ffill').fillna(method='bfill')
        return df2

    def _rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.ewm(alpha=1/window, adjust=False).mean()
        ma_down = down.ewm(alpha=1/window, adjust=False).mean()
        rs = ma_up / (ma_down + 1e-9)
        return 100 - (100 / (1 + rs))

    def walk_forward_splits(self, df: pd.DataFrame, n_splits: int = 3, train_window_years: int = 4, test_window_years: int =1) -> List[Dict]:
        """Generate walk-forward fold definitions (returns list of dicts with train/test ranges)."""
        folds = []
        years = sorted(list(set(df.index.year)))
        min_year = min(years)
        max_year = max(years)
        start = min_year
        while start + train_window_years + test_window_years - 1 <= max_year and len(folds) < n_splits:
            train_start = start
            train_end = start + train_window_years - 1
            test_start = train_end + 1
            test_end = train_end + test_window_years
            folds.append({
                'train_start': str(train_start),
                'train_end': str(train_end),
                'test_start': str(test_start),
                'test_end': str(test_end)
            })
            start += test_window_years  # rolling forward
        return folds

    def apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE class balancing if imbalanced-learn is available; otherwise return inputs."""
        try:
            from imblearn.over_sampling import SMOTE
        except Exception:
            self.logger.warning('imblearn not installed; skipping SMOTE')
            return X, y

        try:
            sm = SMOTE(random_state=self.config.RANDOM_SEED)
            X_res, y_res = sm.fit_resample(X, y)
            return X_res, y_res
        except Exception as e:
            self.logger.warning(f'SMOTE failed: {e}')
            return X, y

    def feature_selection_mutual_info(self, X: np.ndarray, y: np.ndarray, k: int = 30) -> List[int]:
        """Select top-k features by mutual information (returns indices)."""
        try:
            mi = mutual_info_regression(X, y)
            topk = np.argsort(mi)[-k:][::-1]
            return topk.tolist()
        except Exception as e:
            self.logger.warning(f'Mutual information selection failed: {e}')
            return list(range(X.shape[1]))

    def shap_feature_importance(self, model, X: np.ndarray, feature_names: List[str]) -> List[Tuple[str, float]]:
        """Compute SHAP values if SHAP is installed; otherwise return empty list."""
        try:
            import shap
        except Exception:
            self.logger.warning('shap not installed; skipping SHAP feature importance')
            return []

        try:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            importances = np.abs(shap_values.values).mean(axis=0)
            pairs = sorted(list(zip(feature_names, importances)), key=lambda x: x[1], reverse=True)
            return pairs
        except Exception as e:
            self.logger.warning(f'SHAP failed: {e}')
            return []

    def boruta_feature_selection_placeholder(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
        """Placeholder wrapper for Boruta (boruta_py). Returns all features if Boruta not installed."""
        try:
            from boruta import BorutaPy
            import xgboost as xgb
        except Exception:
            self.logger.warning('boruta or xgboost not installed; skipping Boruta selection')
            return feature_names

        try:
            rf = xgb.XGBRegressor(n_estimators=100, random_state=self.config.RANDOM_SEED)
            boruta = BorutaPy(rf, n_estimators='auto', random_state=self.config.RANDOM_SEED)
            boruta.fit(X, y)
            selected = [f for f, s in zip(feature_names, boruta.support_) if s]
            return selected
        except Exception as e:
            self.logger.warning(f'Boruta failed: {e}')
            return feature_names


if __name__ == "__main__":
    # Test data processor
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from utils.config import CONFIG, setup_logging
    
    logger = setup_logging(CONFIG)
    processor = DataProcessor(CONFIG, logger)
    
    # Test data loading
    loaded_data = processor.load_all_data()
    print(f"Loaded {len(loaded_data)} stocks")
    
    # Test data processing
    processed_data = processor.process_all_data()
    print(f"Processed {len(processed_data)} stocks")
    
    # Get summary
    summary = processor.get_data_summary()
    print("\nData Summary:")
    print(f"Stocks loaded: {summary['stocks_loaded']}")
    print(f"Stocks processed: {summary['stocks_processed']}")