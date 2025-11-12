"""
Comprehensive evaluation metrics for stock prediction research.

This module implements all performance metrics required for comparing ANN vs QINN
models including directional accuracy, financial metrics, and statistical tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings('ignore')

class DirectionalAccuracyMetrics:
    """
    Specialized metrics for directional accuracy in financial prediction.
    
    Focuses on the critical 95%+ hit ratio requirement and related metrics.
    """
    
    @staticmethod
    def hit_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate hit ratio (directional accuracy).
        
        Args:
            y_true: True direction (0/1 or -1/1)
            y_pred: Predicted direction (0/1 or -1/1)
            
        Returns:
            Hit ratio as percentage (0-100)
        """
        return accuracy_score(y_true, y_pred) * 100
    
    @staticmethod
    def up_capture_ratio(y_true_returns: np.ndarray, y_pred_direction: np.ndarray) -> float:
        """
        Calculate percentage of upward movements correctly predicted.
        
        Args:
            y_true_returns: True returns
            y_pred_direction: Predicted directions (0 for down, 1 for up)
            
        Returns:
            Up capture ratio as percentage
        """
        true_up = (y_true_returns > 0)
        if true_up.sum() == 0:
            return 0.0
        
        correct_up_predictions = (true_up & (y_pred_direction == 1)).sum()
        return (correct_up_predictions / true_up.sum()) * 100
    
    @staticmethod
    def down_capture_ratio(y_true_returns: np.ndarray, y_pred_direction: np.ndarray) -> float:
        """
        Calculate percentage of downward movements correctly predicted.
        
        Args:
            y_true_returns: True returns
            y_pred_direction: Predicted directions (0 for down, 1 for up)
            
        Returns:
            Down capture ratio as percentage
        """
        true_down = (y_true_returns <= 0)
        if true_down.sum() == 0:
            return 0.0
        
        correct_down_predictions = (true_down & (y_pred_direction == 0)).sum()
        return (correct_down_predictions / true_down.sum()) * 100
    
    @staticmethod
    def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate balanced accuracy (average of sensitivity and specificity).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Balanced accuracy as percentage
        """
        from sklearn.metrics import balanced_accuracy_score
        return balanced_accuracy_score(y_true, y_pred) * 100
    
    @staticmethod
    def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Matthews Correlation Coefficient.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            MCC value (-1 to 1)
        """
        from sklearn.metrics import matthews_corrcoef
        return matthews_corrcoef(y_true, y_pred)

class RegressionMetrics:
    """
    Comprehensive regression metrics for return prediction.
    """
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Square Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        mask = y_true != 0
        if mask.sum() == 0:
            return float('inf')
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if mask.sum() == 0:
            return 0.0
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    @staticmethod
    def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Median Absolute Error."""
        from sklearn.metrics import median_absolute_error
        return median_absolute_error(y_true, y_pred)
    
    @staticmethod
    def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Explained Variance Score."""
        from sklearn.metrics import explained_variance_score
        return explained_variance_score(y_true, y_pred)
    
    @staticmethod
    def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Maximum residual error."""
        from sklearn.metrics import max_error
        return max_error(y_true, y_pred)

class FinancialMetrics:
    """
    Financial performance metrics for trading strategy evaluation.
    """
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            returns: Strategy returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)
    
    @staticmethod
    def calmar_ratio(returns: np.ndarray) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        
        Args:
            returns: Strategy returns
            
        Returns:
            Calmar ratio
        """
        if len(returns) == 0:
            return 0.0
        
        cumulative_returns = np.cumprod(1 + returns)
        max_drawdown = FinancialMetrics.max_drawdown(returns)
        
        if max_drawdown == 0:
            return 0.0
        
        annual_return = (cumulative_returns[-1] ** (252 / len(returns))) - 1
        return annual_return / abs(max_drawdown)
    
    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            returns: Strategy returns
            
        Returns:
            Maximum drawdown (negative value)
        """
        if len(returns) == 0:
            return 0.0
        
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.min(drawdown)
    
    @staticmethod
    def sortino_ratio(returns: np.ndarray, target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (excess return / downside deviation).
        
        Args:
            returns: Strategy returns
            target_return: Target return (daily)
            
        Returns:
            Annualized Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0
        
        return (np.mean(excess_returns) / downside_deviation) * np.sqrt(252)
    
    @staticmethod
    def value_at_risk(returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk at given confidence level.
        
        Args:
            returns: Strategy returns
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            VaR value (negative for losses)
        """
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def conditional_value_at_risk(returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns: Strategy returns
            confidence_level: Confidence level
            
        Returns:
            CVaR value
        """
        if len(returns) == 0:
            return 0.0
        
        var = FinancialMetrics.value_at_risk(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    @staticmethod
    def information_ratio(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """
        Calculate Information Ratio.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Annualized Information Ratio
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
            return 0.0
        
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return (np.mean(excess_returns) / tracking_error) * np.sqrt(252)
    
    @staticmethod
    def alpha_beta(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> Tuple[float, float]:
        """
        Calculate portfolio alpha and beta relative to benchmark.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Tuple of (alpha, beta)
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return 0.0, 1.0
        
        # Remove any NaN values
        mask = ~(np.isnan(portfolio_returns) | np.isnan(benchmark_returns))
        portfolio_clean = portfolio_returns[mask]
        benchmark_clean = benchmark_returns[mask]
        
        if len(portfolio_clean) < 2:
            return 0.0, 1.0
        
        # Linear regression: portfolio_returns = alpha + beta * benchmark_returns
        covariance = np.cov(portfolio_clean, benchmark_clean)[0, 1]
        benchmark_variance = np.var(benchmark_clean)
        
        if benchmark_variance == 0:
            return np.mean(portfolio_clean), 0.0
        
        beta = covariance / benchmark_variance
        alpha = np.mean(portfolio_clean) - beta * np.mean(benchmark_clean)
        
        # Annualize alpha
        alpha_annualized = alpha * 252
        
        return alpha_annualized, beta

class ModelComparisonMetrics:
    """
    Metrics specifically for comparing ANN vs QINN model performance.
    """
    
    @staticmethod
    def calculate_all_metrics(
        y_true_reg: np.ndarray,
        y_pred_reg: np.ndarray,
        y_true_cls: np.ndarray,
        y_pred_cls: np.ndarray,
        y_pred_probs: Optional[np.ndarray] = None,
        returns_for_trading: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for model evaluation.
        
        Args:
            y_true_reg: True regression targets (returns)
            y_pred_reg: Predicted returns
            y_true_cls: True classification labels (directions)
            y_pred_cls: Predicted directions
            y_pred_probs: Predicted probabilities (optional)
            returns_for_trading: Returns for trading simulation (optional)
            
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}
        
        # Directional Accuracy Metrics
        metrics['hit_ratio'] = DirectionalAccuracyMetrics.hit_ratio(y_true_cls, y_pred_cls)
        metrics['balanced_accuracy'] = DirectionalAccuracyMetrics.balanced_accuracy(y_true_cls, y_pred_cls)
        metrics['up_capture_ratio'] = DirectionalAccuracyMetrics.up_capture_ratio(y_true_reg, y_pred_cls)
        metrics['down_capture_ratio'] = DirectionalAccuracyMetrics.down_capture_ratio(y_true_reg, y_pred_cls)
        metrics['mcc'] = DirectionalAccuracyMetrics.mcc(y_true_cls, y_pred_cls)
        
        # Standard Classification Metrics
        metrics['precision'] = precision_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0) * 100
        metrics['recall'] = recall_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0) * 100
        metrics['f1_score'] = f1_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0) * 100
        
        # Regression Metrics
        metrics['rmse'] = RegressionMetrics.rmse(y_true_reg, y_pred_reg)
        metrics['mae'] = mean_absolute_error(y_true_reg, y_pred_reg)
        metrics['r2_score'] = r2_score(y_true_reg, y_pred_reg)
        metrics['mape'] = RegressionMetrics.mape(y_true_reg, y_pred_reg)
        metrics['smape'] = RegressionMetrics.smape(y_true_reg, y_pred_reg)
        metrics['explained_variance'] = RegressionMetrics.explained_variance_score(y_true_reg, y_pred_reg)
        
        # Correlation Metrics
        metrics['pearson_correlation'], _ = pearsonr(y_true_reg, y_pred_reg)
        metrics['spearman_correlation'], _ = spearmanr(y_true_reg, y_pred_reg)
        
        # Financial Metrics (if trading returns provided)
        if returns_for_trading is not None and len(returns_for_trading) > 0:
            # Create simple trading strategy based on predictions
            trading_returns = np.where(y_pred_cls == 1, returns_for_trading, -returns_for_trading)
            
            metrics['sharpe_ratio'] = FinancialMetrics.sharpe_ratio(trading_returns)
            metrics['calmar_ratio'] = FinancialMetrics.calmar_ratio(trading_returns)
            metrics['max_drawdown'] = FinancialMetrics.max_drawdown(trading_returns)
            metrics['sortino_ratio'] = FinancialMetrics.sortino_ratio(trading_returns)
            metrics['var_95'] = FinancialMetrics.value_at_risk(trading_returns, 0.05)
            metrics['cvar_95'] = FinancialMetrics.conditional_value_at_risk(trading_returns, 0.05)
            
            # Cumulative return
            metrics['cumulative_return'] = (np.prod(1 + trading_returns) - 1) * 100
            
            # Win rate
            metrics['win_rate'] = (trading_returns > 0).mean() * 100
        
        # Probability-based metrics (if probabilities provided)
        if y_pred_probs is not None:
            from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
            
            try:
                metrics['log_loss'] = log_loss(y_true_cls, y_pred_probs)
                
                if len(np.unique(y_true_cls)) == 2:  # Binary classification
                    metrics['auc_roc'] = roc_auc_score(y_true_cls, y_pred_probs[:, 1])
                    metrics['brier_score'] = brier_score_loss(y_true_cls, y_pred_probs[:, 1])
                
            except Exception as e:
                # Skip probability metrics if they fail
                pass
        
        # Clean up any NaN or infinite values
        for key, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                metrics[key] = 0.0
        
        return metrics
    
    @staticmethod
    def create_performance_summary(metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Create a performance summary with key highlights.
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Performance summary dictionary
        """
        summary = {
            'primary_metrics': {
                'hit_ratio': metrics.get('hit_ratio', 0),
                'r2_score': metrics.get('r2_score', 0),
                'rmse': metrics.get('rmse', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0)
            },
            
            'classification_performance': {
                'accuracy': metrics.get('hit_ratio', 0),
                'balanced_accuracy': metrics.get('balanced_accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'mcc': metrics.get('mcc', 0)
            },
            
            'regression_performance': {
                'r2_score': metrics.get('r2_score', 0),
                'rmse': metrics.get('rmse', 0),
                'mae': metrics.get('mae', 0),
                'mape': metrics.get('mape', 0),
                'explained_variance': metrics.get('explained_variance', 0)
            },
            
            'financial_performance': {
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'calmar_ratio': metrics.get('calmar_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'cumulative_return': metrics.get('cumulative_return', 0),
                'win_rate': metrics.get('win_rate', 0)
            },
            
            # Performance flags
            'meets_accuracy_target': metrics.get('hit_ratio', 0) >= 95.0,
            'meets_r2_target': metrics.get('r2_score', 0) >= 0.3,
            'meets_rmse_target': metrics.get('rmse', float('inf')) <= 0.02,
            'meets_sharpe_target': metrics.get('sharpe_ratio', 0) >= 1.0
        }
        
        # Overall performance score (weighted combination)
        summary['overall_score'] = (
            metrics.get('hit_ratio', 0) * 0.3 +
            metrics.get('r2_score', 0) * 100 * 0.2 +
            (100 - min(metrics.get('rmse', 0) * 1000, 100)) * 0.2 +
            min(max(metrics.get('sharpe_ratio', 0), 0), 5) * 20 * 0.3
        )
        
        # Success flags
        summary['all_targets_met'] = all([
            summary['meets_accuracy_target'],
            summary['meets_r2_target'], 
            summary['meets_rmse_target'],
            summary['meets_sharpe_target']
        ])
        
        return summary


class MetricsCalculator:
    """
    Main class for coordinating all metric calculations and comparisons.
    """
    
    def __init__(self, config, logger: logging.Logger):
        """
        Initialize metrics calculator.
        
        Args:
            config: Model configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        self.results_history = []
        
    def evaluate_model_predictions(
        self,
        model_name: str,
        stock_symbol: str,
        horizon: int,
        fold_info: Dict,
        predictions: Dict[str, np.ndarray],
        true_values: Dict[str, np.ndarray],
        additional_info: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of model predictions.
        
        Args:
            model_name: Name of the model (ANN/QINN)
            stock_symbol: Stock ticker symbol
            horizon: Prediction horizon in days
            fold_info: Information about the validation fold
            predictions: Dictionary with model predictions
            true_values: Dictionary with true values
            additional_info: Additional information (optional)
            
        Returns:
            Complete evaluation results
        """
        
        self.logger.info(f"Evaluating {model_name} predictions for {stock_symbol} {horizon}d")
        
        # Extract predictions and true values
        y_true_reg = true_values['regression']
        y_pred_reg = predictions['regression_predictions']
        y_true_cls = true_values['classification']
        y_pred_cls = predictions['classification_predictions']
        
        # Optional arrays
        y_pred_probs = predictions.get('classification_probabilities')
        returns_for_trading = true_values.get('returns_for_trading', y_true_reg)
        
        # Calculate all metrics
        metrics = ModelComparisonMetrics.calculate_all_metrics(
            y_true_reg, y_pred_reg, y_true_cls, y_pred_cls,
            y_pred_probs, returns_for_trading
        )
        
        # Create performance summary
        summary = ModelComparisonMetrics.create_performance_summary(metrics)
        
        # Compile complete results
        results = {
            'model_name': model_name,
            'stock_symbol': stock_symbol,
            'horizon_days': horizon,
            'fold_info': fold_info,
            'sample_sizes': {
                'total_predictions': len(y_pred_reg),
                'positive_predictions': (y_pred_cls == 1).sum(),
                'negative_predictions': (y_pred_cls == 0).sum()
            },
            'metrics': metrics,
            'performance_summary': summary,
            'additional_info': additional_info or {}
        }
        
        # Store in history
        self.results_history.append(results)
        
        # Log key results
        self.logger.info(
            f"{model_name} Results - Hit Ratio: {metrics['hit_ratio']:.2f}%, "
            f"R²: {metrics['r2_score']:.4f}, RMSE: {metrics['rmse']:.6f}, "
            f"Sharpe: {metrics['sharpe_ratio']:.3f}"
        )
        
        return results
    
    def compare_models(
        self,
        ann_results: Dict[str, Any],
        qinn_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Direct comparison between ANN and QINN results.
        
        Args:
            ann_results: ANN evaluation results
            qinn_results: QINN evaluation results
            
        Returns:
            Comparison results
        """
        
        # Ensure both results are for the same stock/horizon/fold
        if (ann_results['stock_symbol'] != qinn_results['stock_symbol'] or
            ann_results['horizon_days'] != qinn_results['horizon_days']):
            raise ValueError("Cannot compare results for different stocks/horizons")
        
        ann_metrics = ann_results['metrics']
        qinn_metrics = qinn_results['metrics']
        
        # Calculate improvements (QINN vs ANN)
        improvements = {}
        for metric_name in ann_metrics.keys():
            if metric_name in qinn_metrics:
                ann_value = ann_metrics[metric_name]
                qinn_value = qinn_metrics[metric_name]
                
                if ann_value != 0:
                    improvement = ((qinn_value - ann_value) / abs(ann_value)) * 100
                else:
                    improvement = 0.0 if qinn_value == 0 else float('inf')
                
                improvements[f"{metric_name}_improvement"] = improvement
        
        # Determine winner for key metrics
        winners = {
            'hit_ratio_winner': 'QINN' if qinn_metrics['hit_ratio'] > ann_metrics['hit_ratio'] else 'ANN',
            'r2_winner': 'QINN' if qinn_metrics['r2_score'] > ann_metrics['r2_score'] else 'ANN',
            'rmse_winner': 'QINN' if qinn_metrics['rmse'] < ann_metrics['rmse'] else 'ANN',
            'sharpe_winner': 'QINN' if qinn_metrics['sharpe_ratio'] > ann_metrics['sharpe_ratio'] else 'ANN'
        }
        
        # Overall comparison
        ann_score = ann_results['performance_summary']['overall_score']
        qinn_score = qinn_results['performance_summary']['overall_score']
        
        comparison_results = {
            'stock_symbol': ann_results['stock_symbol'],
            'horizon_days': ann_results['horizon_days'],
            'fold_info': ann_results['fold_info'],
            
            'ann_summary': ann_results['performance_summary'],
            'qinn_summary': qinn_results['performance_summary'],
            
            'improvements': improvements,
            'winners': winners,
            
            'overall_winner': 'QINN' if qinn_score > ann_score else 'ANN',
            'overall_improvement': ((qinn_score - ann_score) / max(abs(ann_score), 0.001)) * 100,
            
            'both_meet_targets': (
                ann_results['performance_summary']['all_targets_met'] and
                qinn_results['performance_summary']['all_targets_met']
            ),
            
            'quantum_advantage': qinn_score > ann_score and qinn_metrics['hit_ratio'] >= 95.0
        }
        
        self.logger.info(
            f"Model Comparison for {ann_results['stock_symbol']} {ann_results['horizon_days']}d: "
            f"Winner = {comparison_results['overall_winner']}, "
            f"Improvement = {comparison_results['overall_improvement']:.2f}%"
        )
        
        return comparison_results
    
    def get_aggregate_results(self, model_name: str) -> Dict[str, Any]:
        """
        Get aggregated results across all experiments for a model.
        
        Args:
            model_name: Name of the model (ANN/QINN)
            
        Returns:
            Aggregated results
        """
        
        model_results = [r for r in self.results_history if r['model_name'] == model_name]
        
        if not model_results:
            return {'error': f'No results found for model {model_name}'}
        
        # Aggregate metrics
        all_metrics = {}
        for metric_name in model_results[0]['metrics'].keys():
            values = [r['metrics'][metric_name] for r in model_results 
                     if not np.isnan(r['metrics'][metric_name])]
            if values:
                all_metrics[f"{metric_name}_mean"] = np.mean(values)
                all_metrics[f"{metric_name}_std"] = np.std(values)
                all_metrics[f"{metric_name}_median"] = np.median(values)
                all_metrics[f"{metric_name}_min"] = np.min(values)
                all_metrics[f"{metric_name}_max"] = np.max(values)
        
        # Count targets met
        targets_met = {
            'accuracy_target_met': sum(1 for r in model_results if r['performance_summary']['meets_accuracy_target']),
            'r2_target_met': sum(1 for r in model_results if r['performance_summary']['meets_r2_target']),
            'rmse_target_met': sum(1 for r in model_results if r['performance_summary']['meets_rmse_target']),
            'sharpe_target_met': sum(1 for r in model_results if r['performance_summary']['meets_sharpe_target']),
            'all_targets_met': sum(1 for r in model_results if r['performance_summary']['all_targets_met'])
        }
        
        # Success rates
        total_experiments = len(model_results)
        success_rates = {k: (v / total_experiments) * 100 for k, v in targets_met.items()}
        
        return {
            'model_name': model_name,
            'total_experiments': total_experiments,
            'aggregated_metrics': all_metrics,
            'targets_met_counts': targets_met,
            'success_rates': success_rates,
            'overall_success_rate': success_rates['all_targets_met']
        }
    
    def export_results_to_csv(self, filename: str = None) -> str:
        """
        Export all results to CSV format.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to the created CSV file
        """
        
        if filename is None:
            filename = f"model_evaluation_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Flatten results for CSV export
        flattened_results = []
        
        for result in self.results_history:
            flat_result = {
                'model_name': result['model_name'],
                'stock_symbol': result['stock_symbol'], 
                'horizon_days': result['horizon_days'],
                'total_predictions': result['sample_sizes']['total_predictions'],
                'positive_predictions': result['sample_sizes']['positive_predictions'],
                'negative_predictions': result['sample_sizes']['negative_predictions']
            }
            
            # Add all metrics
            for metric_name, metric_value in result['metrics'].items():
                flat_result[metric_name] = metric_value
            
            # Add performance flags
            summary = result['performance_summary']
            flat_result['meets_accuracy_target'] = summary['meets_accuracy_target']
            flat_result['meets_r2_target'] = summary['meets_r2_target']
            flat_result['meets_rmse_target'] = summary['meets_rmse_target']
            flat_result['meets_sharpe_target'] = summary['meets_sharpe_target']
            flat_result['all_targets_met'] = summary['all_targets_met']
            flat_result['overall_score'] = summary['overall_score']
            
            flattened_results.append(flat_result)
        
        # Create DataFrame and save
        df = pd.DataFrame(flattened_results)
        
        filepath = f"{self.config.RESULTS_DIR}/{filename}"
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Results exported to {filepath}")
        return filepath


if __name__ == "__main__":
    print("Metrics module test")
    from utils.config import CONFIG, setup_logging
    
    logger = setup_logging(CONFIG)
    calculator = MetricsCalculator(CONFIG, logger)
    print("MetricsCalculator initialized successfully")