from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import os
from data_scraper import EURJPYDataScraper
from config import DATA_CONFIG, PREDICTION_CONFIG
from sklearn.preprocessing import MinMaxScaler
import traceback

app = Flask(__name__)

# Global variables for model
model = None
scaler = None
lookback = 60
latest_data = None

def initialize_model():
    """Load trained model and initialize components"""
    global model, scaler, lookback, latest_data
    
    try:
        # Load model
        model_path = 'models/lstm_eurjpy.h5'
        if os.path.exists(model_path):
            model = load_model(model_path)
            print("Model loaded successfully")
        else:
            print(f"Model not found at {model_path}. Please train the model first.")
            return False
        
        # Load parameters
        params_path = 'models/lstm_eurjpy_params.pkl'
        if os.path.exists(params_path):
            with open(params_path, 'rb') as f:
                params = pickle.load(f)
                lookback = params['lookback']
        
        # Initialize scaler
        scraper = EURJPYDataScraper()
        scaler = scraper.scaler
        
        # Fetch latest data
        update_latest_data()
        
        return True
    except Exception as e:
        print(f"Error initializing model: {e}")
        traceback.print_exc()
        return False

def update_latest_data():
    """Update latest EUR/JPY data from local CSV"""
    global latest_data
    try:
        scraper = EURJPYDataScraper()
        data = scraper.fetch_data(
            period=DATA_CONFIG.get('historical_period', '10y'),
            local_path=DATA_CONFIG.get('local_csv_path')
        )
        latest_data = data
        return data
    except Exception as e:
        print(f"Error updating data: {e}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/current-price')
def get_current_price():
    """Get current EUR/JPY price"""
    try:
        scraper = EURJPYDataScraper()
        price = scraper.get_latest_price()
        
        if latest_data is not None:
            historical = latest_data['Close'].tail(30).tolist()
            dates = latest_data.index.tail(30).strftime('%Y-%m-%d').tolist()
        else:
            historical = []
            dates = []
        
        return jsonify({
            'success': True,
            'current_price': float(price) if price else 0,
            'historical': historical,
            'dates': dates
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict EUR/JPY prices for week 1 and week 2 ahead"""
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
        
        # Get latest data from local CSV
        scraper = EURJPYDataScraper()
        data = scraper.fetch_data(
            period=DATA_CONFIG.get('historical_period', '10y'),
            local_path=DATA_CONFIG.get('local_csv_path')
        )
        if data is None or data.empty:
            return jsonify({'success': False, 'error': 'Could not load local price data'}), 500
        prices = data['Close'].values.reshape(-1, 1)
        
        # Normalize prices
        scaled_prices = scaler.fit_transform(prices)
        
        # Prepare sequence and forecast horizon (always keep at least 2 for UI)
        horizon_weeks = int(PREDICTION_CONFIG.get('forecast_weeks', 2))
        horizon_weeks = max(2, horizon_weeks)
        sequence = scaled_prices[-lookback:, 0].copy()

        # Iterative multi-step forecast: each predicted step feeds the next step
        predicted_scaled_values = []
        for _ in range(horizon_weeks):
            step_pred = model.predict(sequence.reshape(1, lookback, 1), verbose=0)[0][0]
            predicted_scaled_values.append(step_pred)
            sequence = np.append(sequence[1:], step_pred)

        predicted_prices = scaler.inverse_transform(np.array(predicted_scaled_values).reshape(-1, 1)).flatten()
        week1_price = float(predicted_prices[0])
        week2_price = float(predicted_prices[1])
        
        current_price = prices[-1][0]
        week1_change = ((week1_price - current_price) / current_price) * 100
        week2_change = ((week2_price - current_price) / current_price) * 100

        last_date = pd.to_datetime(data.index[-1])
        week1_date = (last_date + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        week2_date = (last_date + pd.Timedelta(days=14)).strftime('%Y-%m-%d')
        
        return jsonify({
            'success': True,
            'current_price': float(current_price),
            'week1_date': week1_date,
            'week2_date': week2_date,
            'week1_predicted_price': week1_price,
            'week2_predicted_price': week2_price,
            'predicted_price': week2_price,
            'prediction_horizon_weeks': 2,
            'week1_change': float(week1_change),
            'week2_change': float(week2_change),
            'change': float(week2_change),
            'change_direction': 'up' if week2_change > 0 else 'down'
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model-info')
def get_model_info():
    """Get model information"""
    try:
        return jsonify({
            'success': True,
            'model_type': 'LSTM (Long Short-Term Memory)',
            'lookback_days': lookback,
            'layers': 'LSTM(50) -> Dropout -> LSTM(50) -> Dropout -> LSTM(50) -> Dense',
            'optimization': 'Adam (lr=0.001)',
            'loss_function': 'Mean Squared Error',
            'training_data': '10 years of EUR/JPY weekly data',
            'data_source': 'Local CSV file',
            'prediction_horizon_weeks': int(PREDICTION_CONFIG.get('forecast_weeks', 2))
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/historical-data')
def get_historical_data():
    """Get historical data for chart"""
    try:
        days = request.args.get('days', 90, type=int)
        scraper = EURJPYDataScraper()
        data = scraper.fetch_data(
            period=DATA_CONFIG.get('historical_period', '10y'),
            local_path=DATA_CONFIG.get('local_csv_path')
        )
        if data is None or data.empty:
            return jsonify({'success': False, 'error': 'Could not load local historical data'}), 500
        data = data.tail(days)
        
        result = {
            'dates': data.index.strftime('%Y-%m-%d').tolist(),
            'open': data['Open'].tolist(),
            'high': data['High'].tolist(),
            'low': data['Low'].tolist(),
            'close': data['Close'].tolist(),
            'volume': data['Volume'].tolist()
        }
        
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    print("Initializing application...")
    if initialize_model():
        print("Application ready!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize model. Please train the model first using lstm_model.py")
