from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import os
from data_scraper import EURJPYDataScraper
from config import DATA_CONFIG, PREDICTION_CONFIG
import traceback

app = Flask(__name__)

# Global variables for model
model = None
lookback = 60
latest_data = None
feature_columns = []


def get_trend_summary(feature_data):
    """Summarize current trend strength and whether it is weakening."""
    if feature_data is None or feature_data.empty:
        return {
            'trend_strength': 'unknown',
            'trend_signal': 'Unavailable',
            'adx': None,
            'macd_hist': None,
            'trend_weakening': False,
        }

    last_row = feature_data.iloc[-1]
    adx_value = float(last_row['ADX']) if pd.notna(last_row['ADX']) else None
    macd_hist_value = float(last_row['MACD_HIST']) if pd.notna(last_row['MACD_HIST']) else None
    weakening = bool(last_row['TREND_WEAKENING'])

    if adx_value is None:
        strength = 'unknown'
    elif adx_value >= 25:
        strength = 'strong'
    elif adx_value >= 20:
        strength = 'moderate'
    else:
        strength = 'weak'

    if weakening:
        signal = 'Trend strength is weakening'
    elif strength == 'strong':
        signal = 'Trend remains strong'
    elif strength == 'moderate':
        signal = 'Trend is steady'
    else:
        signal = 'Trend is weak or sideways'

    return {
        'trend_strength': strength,
        'trend_signal': signal,
        'adx': adx_value,
        'macd_hist': macd_hist_value,
        'trend_weakening': weakening,
    }

def initialize_model():
    """Load trained model and initialize components"""
    global model, lookback, latest_data, feature_columns
    
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
                feature_columns = params.get('feature_columns', [])
        
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
        source_data = latest_data if latest_data is not None else scraper.fetch_data(
            period=DATA_CONFIG.get('historical_period', '10y'),
            local_path=DATA_CONFIG.get('local_csv_path')
        )
        feature_data = scraper.engineer_features(source_data) if source_data is not None else pd.DataFrame()
        trend_summary = get_trend_summary(feature_data)
        
        if source_data is not None and not source_data.empty:
            historical = source_data['Close'].tail(30).tolist()
            dates = source_data.index.tail(30).strftime('%Y-%m-%d').tolist()
        else:
            historical = []
            dates = []
        
        return jsonify({
            'success': True,
            'current_price': float(price) if price else 0,
            'historical': historical,
            'dates': dates,
            **trend_summary,
        })
    except Exception as e:
        print(f"Current price error: {e}")
        traceback.print_exc()
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
        feature_data = scraper.engineer_features(data)
        if feature_data is None or feature_data.empty:
            return jsonify({'success': False, 'error': 'Could not engineer prediction features'}), 500
        if len(feature_data) <= lookback:
            return jsonify({'success': False, 'error': 'Not enough feature rows for prediction'}), 500

        scraper.fit_feature_pipeline(feature_data)
        scaled_features = scraper.transform_feature_frame(feature_data)
        
        # Prepare sequence and forecast horizon (always keep at least 2 for UI)
        horizon_weeks = int(PREDICTION_CONFIG.get('forecast_weeks', 2))
        horizon_weeks = max(2, horizon_weeks)
        sequence = scaled_features[-lookback:].copy()

        # Iterative multi-step forecast: each predicted step feeds the next step
        predicted_scaled_values = []
        ema10 = feature_data['EMA10'].copy()
        ema20 = feature_data['EMA20'].copy()
        adx_series = feature_data['ADX'].copy()
        macd_hist_series = feature_data['MACD_HIST'].copy()
        trend_weakening_series = feature_data['TREND_WEAKENING'].copy()
        last_date = pd.to_datetime(feature_data.index[-1])

        for _ in range(horizon_weeks):
            step_pred = model.predict(sequence.reshape(1, lookback, sequence.shape[1]), verbose=0)[0][0]
            predicted_scaled_values.append(step_pred)

            predicted_close = scraper.inverse_transform_close([step_pred])[0]
            next_date = last_date + pd.Timedelta(days=7)

            new_close_series = pd.concat([feature_data['Close'], pd.Series([predicted_close], index=[next_date])])
            new_ema10 = new_close_series.ewm(span=10, adjust=False).mean().iloc[-1]
            new_ema20 = new_close_series.ewm(span=20, adjust=False).mean().iloc[-1]
            new_ema20_slope = new_ema20 - ema20.iloc[-1]

            ema12 = new_close_series.ewm(span=12, adjust=False).mean().iloc[-1]
            ema26 = new_close_series.ewm(span=26, adjust=False).mean().iloc[-1]
            macd_value = ema12 - ema26
            macd_series = new_close_series.ewm(span=12, adjust=False).mean() - new_close_series.ewm(span=26, adjust=False).mean()
            macd_signal = macd_series.ewm(span=9, adjust=False).mean().iloc[-1]
            new_macd_hist = macd_value - macd_signal

            new_adx = adx_series.iloc[-1] if not adx_series.empty else 20.0
            weakening = int(
                len(adx_series) >= 2
                and new_adx < adx_series.iloc[-1]
                and adx_series.iloc[-1] < adx_series.iloc[-2]
                and new_adx > 20
                and new_macd_hist < macd_hist_series.iloc[-1]
            )

            new_row = pd.DataFrame(
                [{
                    'Close': predicted_close,
                    'EMA10': new_ema10,
                    'EMA20': new_ema20,
                    'EMA20_SLOPE': new_ema20_slope,
                    'MACD_HIST': new_macd_hist,
                    'ADX': new_adx,
                    'TREND_WEAKENING': weakening,
                }]
            )
            scaled_new_row = scraper.transform_feature_frame(new_row)[0]
            sequence = np.vstack([sequence[1:], scaled_new_row])

            feature_data = pd.concat([
                feature_data,
                pd.DataFrame(
                    [{
                        'Open': predicted_close,
                        'High': predicted_close,
                        'Low': predicted_close,
                        'Close': predicted_close,
                        'Volume': 0,
                        'EMA10': new_ema10,
                        'EMA20': new_ema20,
                        'EMA20_SLOPE': new_ema20_slope,
                        'MACD_HIST': new_macd_hist,
                        'ADX': new_adx,
                        'TREND_WEAKENING': weakening,
                    }],
                    index=[next_date],
                ),
            ])
            ema10 = pd.concat([ema10, pd.Series([new_ema10], index=[next_date])])
            ema20 = pd.concat([ema20, pd.Series([new_ema20], index=[next_date])])
            adx_series = pd.concat([adx_series, pd.Series([new_adx], index=[next_date])])
            macd_hist_series = pd.concat([macd_hist_series, pd.Series([new_macd_hist], index=[next_date])])
            trend_weakening_series = pd.concat([trend_weakening_series, pd.Series([weakening], index=[next_date])])
            last_date = next_date

        predicted_prices = scraper.inverse_transform_close(predicted_scaled_values)
        week1_price = float(predicted_prices[0])
        week2_price = float(predicted_prices[1])
        
        current_price = float(data['Close'].iloc[-1])
        week1_change = ((week1_price - current_price) / current_price) * 100
        week2_change = ((week2_price - current_price) / current_price) * 100

        base_date = pd.to_datetime(data.index[-1])
        week1_date = (base_date + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        week2_date = (base_date + pd.Timedelta(days=14)).strftime('%Y-%m-%d')
        trend_summary = get_trend_summary(scraper.engineer_features(data))
        
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
            'change_direction': 'up' if week2_change > 0 else 'down',
            **trend_summary,
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
            'prediction_horizon_weeks': int(PREDICTION_CONFIG.get('forecast_weeks', 2)),
            'engineered_features': ', '.join(feature_columns) if feature_columns else 'Close only',
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
