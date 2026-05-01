import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import pickle
import matplotlib.pyplot as plt
import itertools
import warnings
from data_scraper import EURJPYDataScraper
from config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG

class LSTMPredictor:
    """LSTM model for EUR/JPY price prediction"""
    
    def __init__(
        self,
        model_dir='models',
        lookback=60,
        lstm_units=None,
        dropout_rate=0.2,
        dense_units=25,
        learning_rate=0.001,
    ):
        self.model_dir = model_dir
        self.lookback = lookback
        self.lstm_units = lstm_units or [64, 32]
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.model = None
        self.target_scaler = None
        self.feature_columns = []
        os.makedirs(model_dir, exist_ok=True)
        
    def build_model(self, input_shape):
        """Build LSTM neural network"""
        print(
            f"Building LSTM model with input shape {input_shape}, "
            f"units={self.lstm_units}, dropout={self.dropout_rate}, dense={self.dense_units}, lr={self.learning_rate}"
        )

        layers = []
        for i, units in enumerate(self.lstm_units):
            is_last = i == len(self.lstm_units) - 1
            if i == 0:
                layers.append(LSTM(units, activation='tanh', return_sequences=not is_last, input_shape=input_shape))
            else:
                layers.append(LSTM(units, activation='tanh', return_sequences=not is_last))
            layers.append(Dropout(self.dropout_rate))

        layers.extend([
            Dense(self.dense_units, activation='relu'),
            Dense(1)
        ])

        self.model = Sequential(layers)

        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae'])
        print(self.model.summary())
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=1):
        """Train the LSTM model"""
        print(f"\nTraining LSTM model for {epochs} epochs...")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
        ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        print(f"\n--- Model Performance ---")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"MSE: {mse:.6f}")
        
        return rmse, mae, mse
    
    def save_model(self, name='lstm_eurjpy'):
        """Save trained model"""
        model_path = os.path.join(self.model_dir, f'{name}.h5')
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save lookback parameter
        params = {
            'model_type': 'lstm',
            'lookback': self.lookback,
            'feature_columns': self.feature_columns,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'dense_units': self.dense_units,
            'learning_rate': self.learning_rate,
        }
        params_path = os.path.join(self.model_dir, f'{name}_params.pkl')
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)
        
        return model_path
    
    def load_model(self, name='lstm_eurjpy'):
        """Load trained model"""
        from tensorflow.keras.models import load_model
        model_path = os.path.join(self.model_dir, f'{name}.h5')
        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        params_path = os.path.join(self.model_dir, f'{name}_params.pkl')
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
            self.lookback = params['lookback']
            self.feature_columns = params.get('feature_columns', [])
            self.lstm_units = params.get('lstm_units', self.lstm_units)
            self.dropout_rate = params.get('dropout_rate', self.dropout_rate)
            self.dense_units = params.get('dense_units', self.dense_units)
            self.learning_rate = params.get('learning_rate', self.learning_rate)
        
        return self.model
    
    def predict(self, sequence, scaler):
        """Make prediction for next price"""
        sequence = sequence.reshape(1, sequence.shape[0], 1)
        prediction = self.model.predict(sequence, verbose=0)
        # Inverse transform to get actual price
        actual_price = scaler.inverse_transform(prediction)
        return actual_price[0][0]
    
    def plot_history(self, history, save_path='models/training_history.png'):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")


def _build_lgbm_frame(feature_data, lag_count=30):
    """Build a tabular supervised frame for one-step-ahead regression."""
    frame = feature_data.copy()

    for lag in range(1, lag_count + 1):
        frame[f'close_lag_{lag}'] = frame['Close'].shift(lag)

    returns = frame['Close'].pct_change()
    for lag in range(1, 6):
        frame[f'return_lag_{lag}'] = returns.shift(lag)

    for window in [3, 7, 14]:
        frame[f'roll_mean_{window}'] = frame['Close'].rolling(window).mean().shift(1)
        frame[f'roll_std_{window}'] = frame['Close'].rolling(window).std().shift(1)

    frame['target_next_return'] = frame['Close'].shift(-1) / frame['Close'] - 1.0
    frame = frame.dropna()

    feature_cols = [
        f'close_lag_{lag}' for lag in range(1, lag_count + 1)
    ] + [
        f'return_lag_{lag}' for lag in range(1, 6)
    ] + [
        'roll_mean_3', 'roll_std_3',
        'roll_mean_7', 'roll_std_7',
        'roll_mean_14', 'roll_std_14',
        'EMA10', 'EMA20', 'EMA20_SLOPE', 'MACD_HIST', 'ADX', 'TREND_WEAKENING',
    ]

    X = frame[feature_cols]
    y = frame['target_next_return']
    return X, y, feature_cols


class LightGBMPredictor:
    """Gradient-boosted tree forecaster for EUR/JPY prediction."""

    def __init__(self, model_dir='models', lag_count=30):
        self.model_dir = model_dir
        self.lag_count = lag_count
        self.model = None
        self.feature_columns = []
        os.makedirs(model_dir, exist_ok=True)

    def tune_and_train(self, X_train, y_train, tuning_trials=20):
        """Tune hyperparameters with time-series CV and fit final model."""
        search_space = {
            'n_estimators': [200, 400, 600, 800],
            'learning_rate': [0.01, 0.03, 0.05, 0.08],
            'num_leaves': [15, 31, 63, 127],
            'max_depth': [-1, 4, 6, 8, 10],
            'min_child_samples': [10, 20, 30, 50],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0.0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.0, 0.1, 0.5, 1.0],
        }

        tscv = TimeSeriesSplit(n_splits=3)
        base_model = LGBMRegressor(objective='regression', random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=search_space,
            n_iter=tuning_trials,
            scoring='neg_mean_squared_error',
            cv=tscv,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

        print(f"\nRunning LightGBM tuning ({tuning_trials} trials)...")
        search.fit(X_train, y_train)
        self.model = search.best_estimator_

        cv_results = pd.DataFrame(search.cv_results_)
        cv_results = cv_results.sort_values('rank_test_score')
        cv_out = cv_results[[
            'rank_test_score',
            'mean_test_score',
            'std_test_score',
            'params',
        ]]
        cv_out.to_csv(os.path.join(self.model_dir, 'tuning_results.csv'), index=False)

        print(f"Best LightGBM params: {search.best_params_}")
        return search.best_params_

    def evaluate(self, X_test, y_test):
        pred_returns = self.model.predict(X_test)
        last_close = X_test['close_lag_1'].values
        pred_close = last_close * (1.0 + pred_returns)
        true_close = last_close * (1.0 + y_test.values)

        mse = mean_squared_error(true_close, pred_close)
        mae = mean_absolute_error(true_close, pred_close)
        rmse = np.sqrt(mse)

        print("\n--- Model Performance ---")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"MSE: {mse:.6f}")
        return rmse, mae, mse

    def save_model(self, name='lstm_eurjpy', best_params=None):
        model_path = os.path.join(self.model_dir, f'{name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        params = {
            'model_type': 'lightgbm',
            'target_mode': 'next_return',
            'lag_count': self.lag_count,
            'feature_columns': self.feature_columns,
            'best_params': best_params or {},
        }
        params_path = os.path.join(self.model_dir, f'{name}_params.pkl')
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)

        print(f"Model saved to {model_path}")
        print(f"Model params saved to {params_path}")


def train_eurjpy_model():
    """Main training pipeline using LSTM as the forecaster."""
    np.random.seed(42)

    scraper = EURJPYDataScraper()
    local_path = DATA_CONFIG.get('local_csv_path')
    data = scraper.fetch_data(period=DATA_CONFIG.get('historical_period', '10y'), local_path=local_path)
    if data is None or data.empty:
        print('Failed to fetch data')
        return

    feature_data = scraper.engineer_features(data)
    if feature_data is None or feature_data.empty:
        print('Feature engineering failed')
        return

    lookback = int(MODEL_CONFIG.get('lookback', 60))
    lstm_units = MODEL_CONFIG.get('lstm_units', [50, 50])
    dropout_rate = float(MODEL_CONFIG.get('dropout_rate', 0.2))
    dense_units = int(MODEL_CONFIG.get('dense_units', 25))
    learning_rate = float(TRAINING_CONFIG.get('learning_rate', 0.001))
    epochs = int(TRAINING_CONFIG.get('epochs', 50))
    batch_size = int(TRAINING_CONFIG.get('batch_size', 32))
    model_name = MODEL_CONFIG.get('model_name', 'lstm_eurjpy')
    model_dir = MODEL_CONFIG.get('model_dir', 'models')

    # Scale features
    scraper.fit_feature_pipeline(feature_data)
    scaled = scraper.transform_feature_frame(feature_data)

    # Build supervised sequences: X = (lookback, n_features), y = next Close (column 0)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i, 0])  # column 0 is 'Close'
    X, y = np.array(X), np.array(y)

    # Time-based split: 80% train, 10% val, 10% test
    n = len(X)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Build and train LSTM
    predictor = LSTMPredictor(
        model_dir=model_dir,
        lookback=lookback,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        learning_rate=learning_rate,
    )
    predictor.feature_columns = scraper.feature_columns
    predictor.target_scaler = scraper.target_scaler

    input_shape = (lookback, X_train.shape[2])
    predictor.build_model(input_shape)
    history = predictor.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
    predictor.evaluate(X_test, y_test)
    predictor.save_model(name=model_name)
    predictor.plot_history(history)

    print("\nTraining complete!")

if __name__ == "__main__":
    train_eurjpy_model()
