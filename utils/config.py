"""
Configuration management for ANN vs QINN stock prediction research project.

This module centralizes all configuration parameters for reproducible research.
"""

import os
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import logging

@dataclass
class ModelConfig:
    """Configuration for model architectures and training."""
    
    # Random seeds for reproducibility
    RANDOM_SEED: int = 42
    NUMPY_SEED: int = 42
    TORCH_SEED: int = 42
    
    # Device configuration (CPU only due to RTX 5060 compatibility issues)
    DEVICE: str = 'cpu'
    
    # Data configuration
    STOCKS: List[str] = None
    PREDICTION_HORIZONS: List[int] = None  # Days: 1, 15, 21, 126
    
    # Walk-forward validation folds
    VALIDATION_FOLDS: List[Tuple[str, str, str]] = None
    
    # Performance targets
    MIN_DIRECTIONAL_ACCURACY: float = 0.95
    MAX_RMSE_1DAY: float = 0.02
    MIN_R_SQUARED: float = 0.3
    MIN_SHARPE_RATIO: float = 1.0
    STATISTICAL_SIGNIFICANCE: float = 0.01
    
    # Training configuration
    MAX_EPOCHS: int = 200
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.001
    PATIENCE: int = 25
    MIN_DELTA: float = 1e-5
    
    # Memory and time constraints
    MAX_MEMORY_GB: float = 8.0
    MAX_TRAINING_TIME_HOURS: float = 6.0
    MAX_INDIVIDUAL_TRAINING_MINUTES: float = 15.0
    
    # ANN Architecture Configuration
    ANN_HIDDEN_LAYERS: List[int] = None
    ANN_DROPOUT_RATE: float = 0.3
    ANN_L2_REGULARIZATION: float = 1e-4
    ANN_ATTENTION_HEADS: int = 8
    ANN_RESIDUAL_CONNECTIONS: bool = True
    
    # QINN Configuration  
    QINN_NUM_QUBITS: int = 8  # Limited for classical simulation
    QINN_NUM_LAYERS: int = 3
    QINN_ENTANGLEMENT: str = 'circular'
    QINN_ENCODING: str = 'amplitude'  # or 'angle'
    
    # Feature engineering
    TECHNICAL_INDICATORS: List[str] = None
    LOOKBACK_WINDOW: int = 60
    
    # Directories
    DATA_DIR: str = 'data/'
    RESULTS_DIR: str = 'results/'
    MODELS_DIR: str = 'models/'
    LOGS_DIR: str = 'logs/'
    
    def __post_init__(self):
        """Initialize default values and create directories."""
        
        # Set default stock list
        if self.STOCKS is None:
            self.STOCKS = [
                'PLTR', 'TEAM', 'NVDA', 'MS', 'C', 'IONQ', 'RGTI', 'TSLA', 
                'NKTR', 'PFE', 'UNH', 'CVX', 'XOM', 'LYFT', 'TRIP', 
                'BRK-B', 'QTUM', 'PSKY', 'AVAV', 'LULU', 'BAC'
            ]
        
        # Set prediction horizons
        if self.PREDICTION_HORIZONS is None:
            self.PREDICTION_HORIZONS = [1, 15, 21, 63]  # 1 day, 15 days, 1 month, 6 months
        
        # Set validation folds (train_start, train_end, test_year)
        if self.VALIDATION_FOLDS is None:
            self.VALIDATION_FOLDS = [
                ('2015', '2017', '2018'),
                ('2015', '2020', '2021'), 
                ('2015', '2024', '2025'),
                ('2020', '2024', '2025')
            ]
        
        # Set ANN architecture
        if self.ANN_HIDDEN_LAYERS is None:
            self.ANN_HIDDEN_LAYERS = [512, 256, 128, 64, 32, 16]
        
        # Set technical indicators
        if self.TECHNICAL_INDICATORS is None:
            self.TECHNICAL_INDICATORS = [
                'RSI', 'EMA_12', 'EMA_26', 'ATR', 'OBV', 'MA_ratio',
                'MACD', 'MACD_signal', 'Bollinger_upper', 'Bollinger_lower',
                'Williams_R', 'CCI', 'Stochastic_K', 'Stochastic_D',
                'ADX', 'Aroon_up', 'Aroon_down', 'MFI', 'TRIX',
                'Ultimate_Oscillator', 'Ichimoku_tenkan', 'Ichimoku_kijun'
            ]
        
        # Create directories
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True) 
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.RESULTS_DIR, 'training_logs'), exist_ok=True)

    def set_random_seeds(self) -> None:
        """Set all random seeds for reproducibility."""
        random.seed(self.RANDOM_SEED)
        np.random.seed(self.NUMPY_SEED) 
        torch.manual_seed(self.TORCH_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.TORCH_SEED)
            torch.cuda.manual_seed_all(self.TORCH_SEED)

    def get_optimizer_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration for different optimizers."""
        return {
            'adamw': {
                'lr': self.LEARNING_RATE,
                'weight_decay': self.ANN_L2_REGULARIZATION,
                'betas': (0.9, 0.999),
                'eps': 1e-8
            },
            'radam': {
                'lr': self.LEARNING_RATE, 
                'weight_decay': self.ANN_L2_REGULARIZATION,
                'betas': (0.9, 0.999)
            },
            'lamb': {
                'lr': self.LEARNING_RATE,
                'weight_decay': self.ANN_L2_REGULARIZATION,
                'betas': (0.9, 0.999)
            }
        }

    def get_scheduler_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration for learning rate schedulers."""
        return {
            'cosine_annealing': {
                'T_max': self.MAX_EPOCHS,
                'eta_min': self.LEARNING_RATE * 0.01
            },
            'reduce_on_plateau': {
                'mode': 'min',
                'factor': 0.5,
                'patience': 10,
                'min_lr': self.LEARNING_RATE * 0.001
            },
            'warm_restarts': {
                'T_0': 10,
                'T_mult': 2,
                'eta_min': self.LEARNING_RATE * 0.01
            }
        }

def setup_logging(config: ModelConfig) -> logging.Logger:
    """Setup comprehensive logging configuration."""
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatters
    formatter = logging.Formatter(log_format)
    
    # Setup main logger
    logger = logging.getLogger('QINN_Research')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler for all logs
    file_handler = logging.FileHandler(
        os.path.join(config.LOGS_DIR, 'experiment.log')
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.FileHandler(
        os.path.join(config.LOGS_DIR, 'errors.log')
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    return logger

def validate_environment(config: ModelConfig, logger: logging.Logger) -> bool:
    """Validate that the environment meets all requirements."""
    
    logger.info("Validating environment requirements...")
    
    # Check PyTorch installation
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Device set to: {config.DEVICE}")
    except ImportError:
        logger.error("PyTorch not installed!")
        return False
    
    # Check quantum computing libraries
    try:
        import pennylane as qml
        logger.info(f"PennyLane version: {qml.__version__}")
    except ImportError:
        logger.warning("PennyLane not installed - QINN model will not work")
        return False
    
    # Check data directory
    if not os.path.exists(config.DATA_DIR):
        logger.error(f"Data directory not found: {config.DATA_DIR}")
        return False
    
    # Check for stock data files
    expected_files = [f"{stock}.csv" for stock in config.STOCKS[:5]]  # Check first 5
    missing_files = []
    
    for file in expected_files:
        if not os.path.exists(os.path.join(config.DATA_DIR, file)):
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Some stock data files missing: {missing_files}")
    
    # Memory check (rough estimate)
    try:
        import psutil
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        logger.info(f"Available memory: {available_memory:.2f} GB")
        
        if available_memory < config.MAX_MEMORY_GB:
            logger.warning(f"Available memory ({available_memory:.2f} GB) less than required ({config.MAX_MEMORY_GB} GB)")
    except ImportError:
        logger.warning("psutil not installed - cannot check memory")
    
    logger.info("Environment validation completed")
    return True

# Global configuration instance
CONFIG = ModelConfig()

if __name__ == "__main__":
    # Test configuration setup
    config = ModelConfig()
    config.set_random_seeds()
    
    logger = setup_logging(config)
    logger.info("Configuration initialized successfully")
    
    # Validate environment
    is_valid = validate_environment(config, logger)
    print(f"Environment validation: {'PASSED' if is_valid else 'FAILED'}")
    
    print(f"Stocks to analyze: {len(config.STOCKS)}")
    print(f"Prediction horizons: {config.PREDICTION_HORIZONS}")
    print(f"Validation folds: {len(config.VALIDATION_FOLDS)}")
    print(f"Total experiments: {len(config.STOCKS) * len(config.PREDICTION_HORIZONS) * len(config.VALIDATION_FOLDS)}")