import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import pickle
import matplotlib.pyplot as plt
from data_scraper import EURJPYDataScraper
from config import DATA_CONFIG

class LSTMPredictor:
    """LSTM model for EUR/JPY price prediction"""
    
    def __init__(self, model_dir='models', lookback=60):
        self.model_dir = model_dir
        self.lookback = lookback
        self.model = None
        self.target_scaler = None
        self.feature_columns = []
        os.makedirs(model_dir, exist_ok=True)
        
    def build_model(self, input_shape):
        """Build LSTM neural network"""
        print(f"Building LSTM model with input shape {input_shape}...")
        
        self.model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        print(self.model.summary())
        return self.model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Train the LSTM model"""
        print(f"\nTraining LSTM model for {epochs} epochs...")
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
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
            'lookback': self.lookback,
            'feature_columns': self.feature_columns,
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

def train_eurjpy_model():
    """Main training pipeline"""
    # Scrape data
    scraper = EURJPYDataScraper()
    local_path = DATA_CONFIG.get('local_csv_path')
    data = scraper.fetch_data(period=DATA_CONFIG.get('historical_period', '10y'), local_path=local_path)
    
    if data is None or data.empty:
        print("Failed to fetch data")
        return
    
    # Prepare data
    try:
        X_train, X_test, y_train, y_test, scaled_prices, feature_data = scraper.prepare_data(data, lookback=60)
    except ValueError as e:
        print(f"Data preparation failed: {e}")
        return
    
    # Build and train model
    predictor = LSTMPredictor(lookback=60)
    predictor.target_scaler = scraper.target_scaler
    predictor.feature_columns = scraper.feature_columns
    
    model = predictor.build_model((X_train.shape[1], X_train.shape[2]))
    history = predictor.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    
    # Evaluate
    predictor.evaluate(X_test, y_test)
    
    # Save model
    predictor.save_model('lstm_eurjpy')
    predictor.plot_history(history)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    train_eurjpy_model()
