"""
Advanced feature engineering for stock prediction research.

This module implements sophisticated technical indicators and market microstructure
features to maximize predictive power for both ANN and QINN models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
try:
    import talib
except ImportError:
    talib = None
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering class for stock market prediction.
    
    Implements 50+ technical indicators, market microstructure features,
    cross-asset correlations, and volatility surface features.
    """
    
    def __init__(self, config, logger: logging.Logger):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config: ModelConfig instance
            logger: Configured logger instance
        """
        self.config = config
        self.logger = logger
        self.lookback_window = config.LOOKBACK_WINDOW
        
        self.logger.info("AdvancedFeatureEngineer initialized")

    def add_basic_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic technical indicators that are commonly available.
        
        Args:
            df: Stock price DataFrame with OHLCV data
            
        Returns:
            DataFrame with basic technical indicators added
        """
        try:
            result_df = df.copy()
            
            # Price-based features
            result_df['Returns'] = df['Close'].pct_change()
            result_df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            result_df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
            result_df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
            
            # Moving averages
            for period in [5, 10, 20, 50, 100, 200]:
                result_df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
                result_df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
                result_df[f'Price_SMA_Ratio_{period}'] = df['Close'] / result_df[f'SMA_{period}']
            
            # Volatility features
            for period in [10, 20, 30]:
                result_df[f'Volatility_{period}'] = df['Returns'].rolling(period).std()
                result_df[f'High_Low_Range_{period}'] = (df['High'].rolling(period).max() - 
                                                       df['Low'].rolling(period).min()) / df['Close']
            
            # Volume features
            result_df['Volume_SMA_10'] = df['Volume'].rolling(10).mean()
            result_df['Volume_Ratio'] = df['Volume'] / result_df['Volume_SMA_10']
            result_df['Price_Volume'] = df['Close'] * df['Volume']
            
            # RSI and momentum
            result_df['RSI_14'] = self._calculate_rsi(df['Close'], 14)
            result_df['RSI_21'] = self._calculate_rsi(df['Close'], 21)
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            result_df['MACD'] = ema_12 - ema_26
            result_df['MACD_Signal'] = result_df['MACD'].ewm(span=9).mean()
            result_df['MACD_Histogram'] = result_df['MACD'] - result_df['MACD_Signal']
            
            self.logger.debug("Added basic technical indicators")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error adding basic technical indicators: {str(e)}")
            return df

    def add_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sophisticated technical indicators using TA-Lib when available.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with advanced technical indicators
        """
        try:
            result_df = df.copy()
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            sma_20 = df['Close'].rolling(bb_period).mean()
            bb_std_dev = df['Close'].rolling(bb_period).std()
            result_df['BB_Upper'] = sma_20 + (bb_std * bb_std_dev)
            result_df['BB_Lower'] = sma_20 - (bb_std * bb_std_dev)
            result_df['BB_Width'] = (result_df['BB_Upper'] - result_df['BB_Lower']) / sma_20
            result_df['BB_Position'] = (df['Close'] - result_df['BB_Lower']) / (result_df['BB_Upper'] - result_df['BB_Lower'])
            
            # Stochastic Oscillator
            low_min = df['Low'].rolling(14).min()
            high_max = df['High'].rolling(14).max()
            result_df['Stochastic_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
            result_df['Stochastic_D'] = result_df['Stochastic_K'].rolling(3).mean()
            
            # Williams %R
            result_df['Williams_R'] = -100 * (high_max - df['Close']) / (high_max - low_min)
            
            # Commodity Channel Index (CCI)
            tp = (df['High'] + df['Low'] + df['Close']) / 3  # Typical price
            cci_period = 20
            sma_tp = tp.rolling(cci_period).mean()
            mad = tp.rolling(cci_period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            result_df['CCI'] = (tp - sma_tp) / (0.015 * mad)
            
            # Average True Range (ATR)
            high_low = df['High'] - df['Low']
            high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
            low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            result_df['ATR_14'] = true_range.rolling(14).mean()
            result_df['ATR_Ratio'] = result_df['ATR_14'] / df['Close']
            
            # Average Directional Index (ADX)
            result_df = self._calculate_adx(result_df, df, period=14)
            
            # Aroon Oscillator
            aroon_period = 25
            aroon_up = df['High'].rolling(aroon_period).apply(
                lambda x: (aroon_period - x.argmax()) / aroon_period * 100
            )
            aroon_down = df['Low'].rolling(aroon_period).apply(
                lambda x: (aroon_period - x.argmin()) / aroon_period * 100
            )
            result_df['Aroon_Up'] = aroon_up
            result_df['Aroon_Down'] = aroon_down
            result_df['Aroon_Oscillator'] = aroon_up - aroon_down
            
            # Money Flow Index (MFI)
            result_df = self._calculate_mfi(result_df, df, period=14)
            
            # On-Balance Volume (OBV)
            obv = [0]
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv.append(obv[-1] + df['Volume'].iloc[i])
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv.append(obv[-1] - df['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            result_df['OBV'] = obv
            result_df['OBV_SMA'] = result_df['OBV'].rolling(10).mean()
            
            # Ichimoku Cloud components
            result_df = self._calculate_ichimoku(result_df, df)
            
            # TRIX
            ema1 = df['Close'].ewm(span=14).mean()
            ema2 = ema1.ewm(span=14).mean()
            ema3 = ema2.ewm(span=14).mean()
            result_df['TRIX'] = ema3.pct_change() * 10000
            
            # Ultimate Oscillator
            result_df = self._calculate_ultimate_oscillator(result_df, df)
            
            self.logger.debug("Added advanced technical indicators")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error adding advanced technical indicators: {str(e)}")
            return df

    def add_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure and order flow features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with microstructure features
        """
        try:
            result_df = df.copy()
            
            # Price impact and liquidity proxies
            result_df['Amihud_Illiquidity'] = np.abs(df['Returns']) / (df['Volume'] * df['Close'])
            result_df['Volume_Price_Trend'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
            
            # Intraday patterns
            result_df['Open_Close_Ratio'] = df['Open'] / df['Close']
            result_df['High_Close_Ratio'] = df['High'] / df['Close']
            result_df['Low_Close_Ratio'] = df['Low'] / df['Close']
            
            # Volume-weighted features
            result_df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
            result_df['Price_VWAP_Ratio'] = df['Close'] / result_df['VWAP']
            
            # Price acceleration and momentum
            result_df['Price_Acceleration'] = df['Close'].diff(2)
            result_df['Volume_Acceleration'] = df['Volume'].diff(2)
            
            # Tick-based features (approximations)
            result_df['Up_Down_Ratio'] = (df['Close'] > df['Open']).astype(int).rolling(20).mean()
            result_df['High_Low_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            
            # Volatility clustering features
            returns_sq = df['Returns'] ** 2
            result_df['Volatility_Clustering'] = returns_sq.rolling(10).mean()
            result_df['GARCH_Proxy'] = returns_sq.ewm(alpha=0.1).mean()
            
            # Order flow approximations
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            result_df['Money_Flow_Ratio'] = money_flow.rolling(20).mean() / money_flow.rolling(20).std()
            
            self.logger.debug("Added market microstructure features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error adding microstructure features: {str(e)}")
            return df

    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical and mathematical features.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with statistical features
        """
        try:
            result_df = df.copy()
            
            # Higher moments of returns
            for period in [10, 20, 50]:
                returns = df['Close'].pct_change()
                result_df[f'Skewness_{period}'] = returns.rolling(period).skew()
                result_df[f'Kurtosis_{period}'] = returns.rolling(period).kurt()
                
            # Entropy and information theory features
            result_df['Price_Entropy'] = df['Close'].rolling(20).apply(self._calculate_entropy)
            result_df['Volume_Entropy'] = df['Volume'].rolling(20).apply(self._calculate_entropy)
            
            # Hurst exponent (fractal dimension proxy)
            result_df['Hurst_Exponent'] = df['Close'].rolling(50).apply(self._calculate_hurst_exponent)
            
            # Autocorrelation features
            for lag in [1, 5, 10]:
                result_df[f'Price_Autocorr_{lag}'] = df['Close'].rolling(30).apply(
                    lambda x: x.autocorr(lag=lag)
                )
                result_df[f'Returns_Autocorr_{lag}'] = df['Returns'].rolling(30).apply(
                    lambda x: x.autocorr(lag=lag)
                )
            
            # Trend strength and persistence
            result_df['Trend_Strength'] = df['Close'].rolling(20).apply(
                lambda x: stats.linregress(range(len(x)), x)[2] ** 2  # R-squared
            )
            
            # Support and resistance levels
            result_df['Resistance_Distance'] = df['Close'] / df['High'].rolling(50).max()
            result_df['Support_Distance'] = df['Close'] / df['Low'].rolling(50).min()
            
            # Price patterns and cycles
            result_df = self._add_fourier_features(result_df, df)
            
            self.logger.debug("Added statistical features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error adding statistical features: {str(e)}")
            return df

    def add_cross_asset_features(self, df: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Add cross-asset correlation and relative strength features.
        
        Args:
            df: Individual stock DataFrame
            market_data: Dictionary of other assets' data for correlation
            
        Returns:
            DataFrame with cross-asset features
        """
        try:
            result_df = df.copy()
            
            if not market_data:
                self.logger.warning("No market data provided for cross-asset features")
                return result_df
            
            # Market beta calculation
            stock_returns = df['Close'].pct_change()
            
            for asset_name, asset_df in market_data.items():
                if len(asset_df) == 0:
                    continue
                    
                try:
                    # Align dates
                    asset_returns = asset_df['Close'].pct_change()
                    aligned_stock, aligned_asset = stock_returns.align(asset_returns, join='inner')
                    
                    if len(aligned_stock) < 30:  # Need sufficient data
                        continue
                    
                    # Rolling correlations
                    for period in [30, 60, 120]:
                        correlation = aligned_stock.rolling(period).corr(aligned_asset)
                        result_df[f'Corr_{asset_name}_{period}'] = correlation
                    
                    # Rolling beta
                    beta = aligned_stock.rolling(60).cov(aligned_asset) / aligned_asset.rolling(60).var()
                    result_df[f'Beta_{asset_name}'] = beta
                    
                    # Relative strength
                    relative_performance = (aligned_stock.rolling(20).mean() / 
                                          aligned_asset.rolling(20).mean())
                    result_df[f'RelativeStrength_{asset_name}'] = relative_performance
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating cross-asset features for {asset_name}: {e}")
                    continue
            
            self.logger.debug("Added cross-asset features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error adding cross-asset features: {str(e)}")
            return df

    def add_volatility_surface_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility surface and options-implied features (approximations).
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with volatility surface features
        """
        try:
            result_df = df.copy()
            
            # Historical volatility term structure
            for period in [5, 10, 20, 30, 60, 120]:
                returns = df['Close'].pct_change()
                hist_vol = returns.rolling(period).std() * np.sqrt(252)  # Annualized
                result_df[f'HistVol_{period}d'] = hist_vol
            
            # Volatility skew approximation
            result_df['Vol_Skew'] = (result_df['HistVol_30d'] - result_df['HistVol_60d']) / result_df['HistVol_60d']
            
            # Volatility of volatility
            for period in [20, 30]:
                vol_series = result_df[f'HistVol_{period}d']
                result_df[f'VolOfVol_{period}d'] = vol_series.rolling(20).std()
            
            # Volatility mean reversion
            long_vol = result_df['HistVol_60d']
            short_vol = result_df['HistVol_10d']
            result_df['Vol_MeanReversion'] = (long_vol - short_vol) / long_vol
            
            # Garman-Klass volatility estimator
            result_df['GK_Volatility'] = np.sqrt(
                0.5 * np.log(df['High'] / df['Low']) ** 2 - 
                (2 * np.log(2) - 1) * np.log(df['Close'] / df['Open']) ** 2
            )
            
            # Yang-Zhang volatility estimator
            overnight_ret = np.log(df['Open'] / df['Close'].shift(1))
            open_close_ret = np.log(df['Close'] / df['Open'])
            high_open = np.log(df['High'] / df['Open'])
            low_open = np.log(df['Low'] / df['Open'])
            
            result_df['YZ_Volatility'] = np.sqrt(
                overnight_ret.rolling(20).var() + 
                0.5 * (high_open * (high_open - open_close_ret) + 
                       low_open * (low_open - open_close_ret)).rolling(20).mean()
            )
            
            self.logger.debug("Added volatility surface features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error adding volatility surface features: {str(e)}")
            return df

    def engineer_all_features(self, df: pd.DataFrame, market_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply all feature engineering techniques to create comprehensive feature set.
        
        Args:
            df: Stock price DataFrame with OHLCV data
            market_data: Optional dictionary of market data for cross-asset features
            
        Returns:
            DataFrame with all engineered features
        """
        try:
            self.logger.info("Starting comprehensive feature engineering")
            
            # Start with basic price data validation
            if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                raise ValueError("Missing required OHLCV columns")
            
            # Initialize with basic features
            result_df = self.add_basic_technical_indicators(df)
            
            # Add advanced technical indicators
            result_df = self.add_advanced_technical_indicators(result_df)
            
            # Add market microstructure features
            result_df = self.add_market_microstructure_features(result_df)
            
            # Add statistical features
            result_df = self.add_statistical_features(result_df)
            
            # Add cross-asset features if market data available
            if market_data:
                result_df = self.add_cross_asset_features(result_df, market_data)
            
            # Add volatility surface features
            result_df = self.add_volatility_surface_features(result_df)
            
            # Clean up infinite and NaN values
            result_df = self._clean_features(result_df)
            
            # Log feature engineering results
            original_features = len(df.columns)
            new_features = len(result_df.columns) - original_features
            
            self.logger.info(f"Feature engineering completed: {new_features} new features created")
            self.logger.info(f"Total features: {len(result_df.columns)}")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive feature engineering: {str(e)}")
            return df

    # Helper methods for complex calculations
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta).where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_adx(self, result_df: pd.DataFrame, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index."""
        try:
            # True Range
            high_low = df['High'] - df['Low']
            high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
            low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            
            # Directional Movement
            up_move = df['High'] - df['High'].shift(1)
            down_move = df['Low'].shift(1) - df['Low']
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smoothed averages
            atr = pd.Series(true_range).rolling(period).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)
            
            # ADX calculation
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            result_df['ADX'] = adx
            result_df['Plus_DI'] = plus_di
            result_df['Minus_DI'] = minus_di
            
            return result_df
            
        except Exception as e:
            self.logger.warning(f"Error calculating ADX: {e}")
            return result_df
    
    def _calculate_mfi(self, result_df: pd.DataFrame, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Money Flow Index."""
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
            
            money_flow_ratio = positive_flow / negative_flow
            mfi = 100 - (100 / (1 + money_flow_ratio))
            
            result_df['MFI'] = mfi
            return result_df
            
        except Exception as e:
            self.logger.warning(f"Error calculating MFI: {e}")
            return result_df
    
    def _calculate_ichimoku(self, result_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud components."""
        try:
            # Tenkan-sen (Conversion Line)
            tenkan_high = df['High'].rolling(9).max()
            tenkan_low = df['Low'].rolling(9).min()
            result_df['Ichimoku_Tenkan'] = (tenkan_high + tenkan_low) / 2
            
            # Kijun-sen (Base Line)
            kijun_high = df['High'].rolling(26).max()
            kijun_low = df['Low'].rolling(26).min()
            result_df['Ichimoku_Kijun'] = (kijun_high + kijun_low) / 2
            
            # Senkou Span A
            result_df['Ichimoku_SpanA'] = ((result_df['Ichimoku_Tenkan'] + result_df['Ichimoku_Kijun']) / 2).shift(26)
            
            # Senkou Span B
            span_b_high = df['High'].rolling(52).max()
            span_b_low = df['Low'].rolling(52).min()
            result_df['Ichimoku_SpanB'] = ((span_b_high + span_b_low) / 2).shift(26)
            
            return result_df
            
        except Exception as e:
            self.logger.warning(f"Error calculating Ichimoku: {e}")
            return result_df
    
    def _calculate_ultimate_oscillator(self, result_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ultimate Oscillator."""
        try:
            # True Range and Buying Pressure
            high_low = df['High'] - df['Low']
            high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
            low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            
            buying_pressure = df['Close'] - np.minimum(df['Low'], df['Close'].shift(1))
            
            # Calculate for three periods
            periods = [7, 14, 28]
            bp_sums = []
            tr_sums = []
            
            for period in periods:
                bp_sum = buying_pressure.rolling(period).sum()
                tr_sum = true_range.rolling(period).sum()
                bp_sums.append(bp_sum)
                tr_sums.append(tr_sum)
            
            # Ultimate Oscillator formula
            uo = 100 * (
                (4 * bp_sums[0] / tr_sums[0]) +
                (2 * bp_sums[1] / tr_sums[1]) +
                (bp_sums[2] / tr_sums[2])
            ) / 7
            
            result_df['Ultimate_Oscillator'] = uo
            return result_df
            
        except Exception as e:
            self.logger.warning(f"Error calculating Ultimate Oscillator: {e}")
            return result_df
    
    def _calculate_entropy(self, x: pd.Series) -> float:
        """Calculate Shannon entropy of price series."""
        try:
            if len(x) < 2:
                return np.nan
            
            # Discretize the data into bins
            hist, _ = np.histogram(x.dropna(), bins=min(10, len(x)//2))
            hist = hist[hist > 0]  # Remove zero counts
            
            if len(hist) == 0:
                return 0
            
            # Calculate probabilities and entropy
            probs = hist / hist.sum()
            entropy = -np.sum(probs * np.log2(probs))
            
            return entropy
            
        except Exception:
            return np.nan
    
    def _calculate_hurst_exponent(self, x: pd.Series) -> float:
        """Calculate Hurst exponent (simplified version)."""
        try:
            if len(x) < 10:
                return np.nan
            
            x = x.dropna().values
            if len(x) < 10:
                return np.nan
            
            # Calculate log returns
            returns = np.diff(np.log(x))
            
            # Calculate R/S statistic for different lags
            lags = range(2, min(20, len(returns)//4))
            rs_values = []
            
            for lag in lags:
                # Divide into non-overlapping periods
                n_periods = len(returns) // lag
                if n_periods < 2:
                    continue
                
                rs_period = []
                for i in range(n_periods):
                    period_returns = returns[i*lag:(i+1)*lag]
                    if len(period_returns) < lag:
                        continue
                    
                    mean_return = np.mean(period_returns)
                    cumulative_deviation = np.cumsum(period_returns - mean_return)
                    
                    r = np.max(cumulative_deviation) - np.min(cumulative_deviation)
                    s = np.std(period_returns)
                    
                    if s > 0:
                        rs_period.append(r / s)
                
                if rs_period:
                    rs_values.append(np.mean(rs_period))
            
            if len(rs_values) < 3:
                return 0.5  # Default for insufficient data
            
            # Linear regression to find Hurst exponent
            log_lags = np.log(list(lags[:len(rs_values)]))
            log_rs = np.log(rs_values)
            
            hurst = np.polyfit(log_lags, log_rs, 1)[0]
            
            # Clip to reasonable range
            return np.clip(hurst, 0, 1)
            
        except Exception:
            return 0.5  # Default value
    
    def _add_fourier_features(self, result_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fourier transform features for cycle detection."""
        try:
            prices = df['Close'].dropna().values
            if len(prices) < 50:
                return result_df
            
            # Apply FFT to detect dominant cycles
            fft = np.fft.fft(prices[-50:])  # Use last 50 periods
            frequencies = np.fft.fftfreq(50)
            
            # Find dominant frequencies (excluding DC component)
            dominant_freq_idx = np.argsort(np.abs(fft[1:]))[-3:] + 1  # Top 3 frequencies
            
            for i, freq_idx in enumerate(dominant_freq_idx):
                if freq_idx < len(frequencies) and frequencies[freq_idx] != 0:
                    period = 1 / abs(frequencies[freq_idx])
                    if 2 <= period <= 25:  # Reasonable cycle periods
                        cycle_feature = np.sin(2 * np.pi * np.arange(len(df)) / period)
                        result_df[f'Cycle_{i+1}'] = cycle_feature
            
            return result_df
            
        except Exception as e:
            self.logger.warning(f"Error adding Fourier features: {e}")
            return result_df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean infinite and NaN values from features."""
        try:
            # Replace infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill and backward fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # If still NaN, fill with 0 (for features like correlations)
            df = df.fillna(0)
            
            # Remove any columns that are all NaN or all the same value
            for col in df.columns:
                if df[col].nunique() <= 1:
                    self.logger.warning(f"Dropping constant column: {col}")
                    df = df.drop(columns=[col])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning features: {str(e)}")
            return df


if __name__ == "__main__":
    # Test feature engineering
    from utils.config import CONFIG, setup_logging
    
    logger = setup_logging(CONFIG)
    engineer = AdvancedFeatureEngineer(CONFIG, logger)
    
    # Create sample data for testing
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'High': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + 1,
        'Low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) - 1,
        'Close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Test feature engineering
    print("Testing feature engineering...")
    try:
        engineered_data = engineer.engineer_all_features(sample_data)
        print(f"Original features: {len(sample_data.columns)}")
        print(f"Engineered features: {len(engineered_data.columns)}")
        print(f"Feature names sample: {list(engineered_data.columns[:10])}")
        print(f"Data shape: {engineered_data.shape}")
        print(f"Missing values: {engineered_data.isnull().sum().sum()}")
        print("Feature engineering test completed successfully!")
        
    except Exception as e:
        print(f"Feature engineering test failed: {e}")
