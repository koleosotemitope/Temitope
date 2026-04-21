"""
Configuration file for EUR/JPY LSTM Predictor
"""

# Data Configuration
DATA_CONFIG = {
    'ticker': 'EURJPY=X',
    'historical_period': '10y',  # 1mo, 3mo, 6mo, 1y, 5y, 10y, max
    'data_dir': 'data',
    # Path to locally downloaded CSV file (required in local-only data mode)
    # Supported format: Investing.com weekly export (columns: Date, Price, Open, High, Low, Vol., Change %)
    'local_csv_path': r'C:\Users\koleo\Downloads\mlops dataset\EUR_JPY Historical Data.csv',
}

# Model Configuration
MODEL_CONFIG = {
    'lookback': 60,  # Number of days to look back for prediction
    'lstm_units': [50, 50, 50],  # Units in each LSTM layer
    'dropout_rate': 0.2,
    'dense_units': 25,
    'model_name': 'lstm_eurjpy',
    'model_dir': 'models',
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 32,
    'validation_split': 0.2,
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'loss_function': 'mse',
    'metrics': ['mae'],
}

# Flask Configuration
FLASK_CONFIG = {
    'debug': True,
    'host': '0.0.0.0',
    'port': 5000,
    'threaded': True,
}

# API Configuration
API_CONFIG = {
    'max_prediction_days': 30,
    'update_interval': 3600,  # seconds (1 hour)
    'cache_enabled': True,
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'normalization': 'minmax',  # minmax or standard
    'train_test_split': 0.8,
    'shuffle_data': False,  # Important: Time series shouldn't be shuffled
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'forecast_days': 14,
    'confidence_threshold': 0.5,
    'ensemble_enabled': False,
    'use_moving_average': True,
    'ma_periods': [5, 10, 20],
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'logs/app.log',
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'use_technical_indicators': False,
    'indicators': ['RSI', 'MACD', 'BB'],  # Relative Strength Index, MACD, Bollinger Bands
    'calculate_returns': True,
    'calculate_volatility': True,
}
