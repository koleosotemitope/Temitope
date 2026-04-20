"""
Utilities and Helper Functions for EUR/JPY LSTM Predictor
"""

import os
import json
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCache:
    """Simple caching mechanism for data"""
    
    def __init__(self, cache_dir='cache', ttl=3600):
        self.cache_dir = cache_dir
        self.ttl = ttl  # Time to live in seconds
        os.makedirs(cache_dir, exist_ok=True)
    
    def get(self, key):
        """Get cached data if not expired"""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        
        if not os.path.exists(cache_file):
            return None
        
        # Check if cache is expired
        file_time = os.path.getmtime(cache_file)
        current_time = datetime.now().timestamp()
        
        if current_time - file_time > self.ttl:
            os.remove(cache_file)
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None
    
    def set(self, key, value):
        """Cache data"""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def clear(self, key=None):
        """Clear cache"""
        if key:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)
        else:
            # Clear all cache
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))

class MetricsTracker:
    """Track model and application metrics"""
    
    def __init__(self, metrics_file='metrics.json'):
        self.metrics_file = metrics_file
        self.metrics = self._load_metrics()
    
    def _load_metrics(self):
        """Load metrics from file"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_metrics(self):
        """Save metrics to file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def record_prediction(self, current_price, predicted_price, actual_price=None):
        """Record a prediction"""
        if 'predictions' not in self.metrics:
            self.metrics['predictions'] = []
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'error': float(predicted_price - current_price) if predicted_price else None
        }
        
        if actual_price is not None:
            entry['actual_price'] = float(actual_price)
            entry['accuracy'] = 1 if (predicted_price > current_price) == (actual_price > current_price) else 0
        
        self.metrics['predictions'].append(entry)
        self.save_metrics()
    
    def get_prediction_accuracy(self, last_n=None):
        """Get prediction accuracy"""
        if 'predictions' not in self.metrics:
            return 0.0
        
        predictions = self.metrics['predictions']
        if last_n:
            predictions = predictions[-last_n:]
        
        if not predictions:
            return 0.0
        
        accurate = sum(1 for p in predictions if 'accuracy' in p and p['accuracy'] == 1)
        return (accurate / len(predictions)) * 100
    
    def get_average_error(self, last_n=None):
        """Get average prediction error"""
        if 'predictions' not in self.metrics:
            return 0.0
        
        predictions = self.metrics['predictions']
        if last_n:
            predictions = predictions[-last_n:]
        
        errors = [abs(p['error']) for p in predictions if 'error' in p and p['error'] is not None]
        
        if not errors:
            return 0.0
        
        return np.mean(errors)

class TechnicalIndicators:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_sma(prices, period=20):
        """Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(prices, period=20):
        """Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Relative Strength Index"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    @staticmethod
    def calculate_bbands(prices, period=20, std_dev=2):
        """Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return upper, sma, lower
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

class PredictionEvaluator:
    """Evaluate prediction quality"""
    
    @staticmethod
    def calculate_mape(actual, predicted):
        """Mean Absolute Percentage Error"""
        return np.mean(np.abs((actual - predicted) / actual)) * 100
    
    @staticmethod
    def calculate_rmse(actual, predicted):
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    @staticmethod
    def calculate_mae(actual, predicted):
        """Mean Absolute Error"""
        return np.mean(np.abs(actual - predicted))
    
    @staticmethod
    def calculate_directional_accuracy(actual, predicted):
        """Calculate directional accuracy (% of correct up/down predictions)"""
        if len(actual) < 2 or len(predicted) < 2:
            return 0.0
        
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        
        correct = np.sum(actual_direction == predicted_direction)
        return (correct / len(actual_direction)) * 100
    
    @staticmethod
    def generate_report(actual, predicted):
        """Generate comprehensive evaluation report"""
        return {
            'RMSE': PredictionEvaluator.calculate_rmse(actual, predicted),
            'MAE': PredictionEvaluator.calculate_mae(actual, predicted),
            'MAPE': PredictionEvaluator.calculate_mape(actual, predicted),
            'Directional_Accuracy': PredictionEvaluator.calculate_directional_accuracy(actual, predicted)
        }

class DataNormalizer:
    """Normalize and denormalize data"""
    
    @staticmethod
    def normalize_minmax(data, feature_range=(0, 1)):
        """Min-Max normalization"""
        scaler = MinMaxScaler(feature_range=feature_range)
        return scaler.fit_transform(data.reshape(-1, 1)), scaler
    
    @staticmethod
    def normalize_zscore(data):
        """Z-score normalization"""
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std, (mean, std)
    
    @staticmethod
    def denormalize_minmax(normalized_data, scaler):
        """Inverse min-max normalization"""
        return scaler.inverse_transform(normalized_data)
    
    @staticmethod
    def denormalize_zscore(normalized_data, params):
        """Inverse z-score normalization"""
        mean, std = params
        return normalized_data * std + mean

def create_logger(name, log_file=None):
    """Create a logger instance"""
    logger = logging.getLogger(name)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    return logger

def ensure_directories():
    """Ensure all necessary directories exist"""
    directories = [
        'models',
        'data',
        'logs',
        'cache',
        'templates',
        'static'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory: {directory}")

if __name__ == "__main__":
    # Test utilities
    ensure_directories()
    print("✓ All directories created successfully")
    
    # Test cache
    cache = DataCache()
    cache.set('test', {'value': 123})
    print(f"✓ Cache test: {cache.get('test')}")
    
    # Test metrics
    tracker = MetricsTracker()
    tracker.record_prediction(128.5, 129.0)
    print(f"✓ Metrics recorded successfully")
