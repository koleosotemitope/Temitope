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


def detect_current_trend_weakening(feature_data):
    """Detect if CURRENT trend is weakening with detailed risk metrics.
    
    Returns: {
        'is_weakening': True|False,
        'risk_level': 'critical'|'high'|'moderate'|'low'|'none',
        'risk_score': 0-100,
        'reasons': [list of specific indicators],
        'adx_trend': 'declining'|'flat'|'rising',
        'macd_trend': 'declining'|'flat'|'rising',
        'ema_signal': 'bearish'|'neutral'|'bullish',
    }
    """
    if feature_data is None or len(feature_data) < 5:
        return {
            'is_weakening': False,
            'risk_level': 'none',
            'risk_score': 0,
            'reasons': [],
            'adx_trend': 'unknown',
            'macd_trend': 'unknown',
            'ema_signal': 'unknown',
        }
    
    # Extract last 5 values to detect trends
    adx_values = feature_data['ADX'].tail(5).astype(float).values
    macd_values = feature_data['MACD_HIST'].tail(5).astype(float).values
    ema20_values = feature_data['EMA20'].tail(5).astype(float).values
    ema20_slope = feature_data['EMA20_SLOPE'].tail(5).astype(float).values
    close_prices = feature_data['Close'].tail(5).astype(float).values
    
    current_adx = adx_values[-1]
    current_macd = macd_values[-1]
    current_ema20 = ema20_values[-1]
    current_price = close_prices[-1]
    
    reasons = []
    risk_score = 0
    
    # ADX Analysis (Trend Strength Decline)
    adx_trend = 'flat'
    if adx_values[-1] < adx_values[-2] < adx_values[-3]:
        adx_trend = 'declining'
        risk_score += 25
        reasons.append(f"ADX declining (now {current_adx:.1f}, was {adx_values[-3]:.1f})")
    elif adx_values[-1] > adx_values[-2]:
        adx_trend = 'rising'
        risk_score -= 10
    
    # MACD Histogram Analysis (Momentum Decline)
    macd_trend = 'flat'
    if macd_values[-1] < macd_values[-2] < macd_values[-3]:
        macd_trend = 'declining'
        risk_score += 20
        reasons.append(f"MACD momentum declining ({macd_values[-1]:.6f})")
    elif macd_values[-1] > macd_values[-2]:
        macd_trend = 'rising'
        risk_score -= 5
    
    # EMA20 Slope Analysis (Price-MA Divergence)
    ema_signal = 'neutral'
    avg_slope = np.mean(ema20_slope[-3:])
    if avg_slope < -0.5:  # Steep decline in EMA slope
        ema_signal = 'bearish'
        risk_score += 15
        reasons.append(f"EMA20 slope turning negative (divergence warning)")
    elif avg_slope > 0.5:
        ema_signal = 'bullish'
        risk_score -= 5
    
    # Price vs EMA20 Analysis (Support Break)
    price_distance = current_price - current_ema20
    if price_distance < 0 and abs(price_distance) > 0.5:  # Price breaks below EMA20
        risk_score += 15
        reasons.append(f"Price breaks below EMA20 (distance: {price_distance:.4f})")
    
    # ADX Level Check (Already in weak trend territory)
    if current_adx < 20:
        risk_score += 10
        reasons.append(f"ADX below 20 (weak trend, {current_adx:.1f})")
    elif current_adx < 25 and adx_trend == 'declining':
        risk_score += 5
    
    # Consolidation Risk (Low volatility before crash)
    atr_range = np.max(close_prices[-3:]) - np.min(close_prices[-3:])
    if atr_range < 0.2:  # Very tight consolidation
        risk_score += 10
        reasons.append("Market in tight consolidation (breakout imminent)")
    
    # Normalize risk score
    risk_score = max(0, min(100, risk_score))
    
    # Determine risk level
    if risk_score >= 70:
        risk_level = 'critical'
        is_weakening = True
    elif risk_score >= 50:
        risk_level = 'high'
        is_weakening = True
    elif risk_score >= 35:
        risk_level = 'moderate'
        is_weakening = True
    elif risk_score >= 20:
        risk_level = 'low'
        is_weakening = False
    else:
        risk_level = 'none'
        is_weakening = False
    
    return {
        'is_weakening': is_weakening,
        'risk_level': risk_level,
        'risk_score': int(risk_score),
        'reasons': reasons,
        'adx_trend': adx_trend,
        'macd_trend': macd_trend,
        'ema_signal': ema_signal,
    }

def get_trend_summary(feature_data):
    """Summarize current trend strength and whether it is weakening."""
    if feature_data is None or feature_data.empty:
        return {
            'trend_strength': 'unknown',
            'trend_signal': 'Unavailable',
            'adx': None,
            'macd_hist': None,
            'trend_weakening': False,
            'weakening_risk': detect_current_trend_weakening(None),
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

    weakening_risk = detect_current_trend_weakening(feature_data)

    return {
        'trend_strength': strength,
        'trend_signal': signal,
        'adx': adx_value,
        'macd_hist': macd_hist_value,
        'trend_weakening': weakening,
        'weakening_risk': weakening_risk,
    }

def _classify_outlook_from_row(row):
    """Classify trend outlook from a forecasted feature row."""
    adx_value = float(row['ADX']) if pd.notna(row['ADX']) else 20.0
    macd_hist = float(row['MACD_HIST']) if pd.notna(row['MACD_HIST']) else 0.0
    weakening = bool(row['TREND_WEAKENING'])

    if weakening:
        return 'likely_weakening'
    if adx_value >= 25 and macd_hist > 0:
        return 'strengthening'
    return 'stable'


def analyze_trend_weakening_outlook(forecast_feature_data, checkpoint_days=(7, 14)):
    """Analyze whether trend is likely to weaken at forecast checkpoints."""
    if forecast_feature_data is None or forecast_feature_data.empty:
        return {
            'day7_outlook': 'unknown',
            'day14_outlook': 'unknown',
            'warning': None,
        }

    day7_index = min(checkpoint_days[0] - 1, len(forecast_feature_data) - 1)
    day14_index = min(checkpoint_days[1] - 1, len(forecast_feature_data) - 1)
    day7_outlook = _classify_outlook_from_row(forecast_feature_data.iloc[day7_index])
    day14_outlook = _classify_outlook_from_row(forecast_feature_data.iloc[day14_index])

    if day14_outlook == 'likely_weakening' or day7_outlook == 'likely_weakening':
        warning = '⚠️ ALERT: Trend shows signs of weakening over the next 14 days'
    else:
        warning = None

    return {
        'day7_outlook': day7_outlook,
        'day14_outlook': day14_outlook,
        'warning': warning,
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
            recent_data = source_data.tail(30)
            historical = recent_data['Close'].tolist()
            dates = recent_data.index.strftime('%Y-%m-%d').tolist()
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
    """Predict EUR/JPY prices for the next 14 days."""
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
        horizon_days = int(PREDICTION_CONFIG.get('forecast_days', PREDICTION_CONFIG.get('forecast_weeks', 2)))
        horizon_days = max(2, horizon_days)
        sequence = scaled_features[-lookback:].copy()

        # Iterative multi-step forecast: each predicted step feeds the next step
        predicted_scaled_values = []
        ema10 = feature_data['EMA10'].copy()
        ema20 = feature_data['EMA20'].copy()
        adx_series = feature_data['ADX'].copy()
        macd_hist_series = feature_data['MACD_HIST'].copy()
        trend_weakening_series = feature_data['TREND_WEAKENING'].copy()
        last_date = pd.to_datetime(feature_data.index[-1])

        for _ in range(horizon_days):
            step_pred = model.predict(sequence.reshape(1, lookback, sequence.shape[1]), verbose=0)[0][0]
            predicted_scaled_values.append(step_pred)

            predicted_close = scraper.inverse_transform_close([step_pred])[0]
            next_date = last_date + pd.Timedelta(days=1)

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
        forecast_dates = [(pd.to_datetime(data.index[-1]) + pd.Timedelta(days=step + 1)).strftime('%Y-%m-%d') for step in range(horizon_days)]
        forecast_prices = [float(price) for price in predicted_prices]
        forecast_feature_data = feature_data.tail(horizon_days).copy()

        day7_index = min(6, len(forecast_prices) - 1)
        day14_index = min(13, len(forecast_prices) - 1)
        day7_price = forecast_prices[day7_index]
        day14_price = forecast_prices[day14_index]
        
        current_price = float(data['Close'].iloc[-1])
        day7_change = ((day7_price - current_price) / current_price) * 100
        day14_change = ((day14_price - current_price) / current_price) * 100

        day7_date = forecast_dates[day7_index]
        day14_date = forecast_dates[day14_index]
        trend_summary = get_trend_summary(scraper.engineer_features(data))
        trend_outlook = analyze_trend_weakening_outlook(forecast_feature_data, checkpoint_days=(7, 14))
        
        return jsonify({
            'success': True,
            'current_price': float(current_price),
            'forecast_dates': forecast_dates,
            'forecast_prices': forecast_prices,
            'day7_date': day7_date,
            'day14_date': day14_date,
            'day7_predicted_price': day7_price,
            'day14_predicted_price': day14_price,
            'predicted_price': day14_price,
            'prediction_horizon_days': horizon_days,
            'day7_change': float(day7_change),
            'day14_change': float(day14_change),
            'change': float(day14_change),
            'change_direction': 'up' if day14_change > 0 else 'down',
            'day7_outlook': trend_outlook['day7_outlook'],
            'day14_outlook': trend_outlook['day14_outlook'],
            'trend_warning': trend_outlook['warning'],
            # Compatibility aliases mapped to 1-week and 2-week checkpoints
            'week1_date': day7_date,
            'week2_date': day14_date,
            'week1_predicted_price': day7_price,
            'week2_predicted_price': day14_price,
            'prediction_horizon_weeks': 2,
            'week1_change': float(day7_change),
            'week2_change': float(day14_change),
            'week1_outlook': trend_outlook['day7_outlook'],
            'week2_outlook': trend_outlook['day14_outlook'],
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
            'training_data': '10 years of EUR/JPY daily data',
            'data_source': 'Local CSV file',
            'prediction_horizon_days': int(PREDICTION_CONFIG.get('forecast_days', PREDICTION_CONFIG.get('forecast_weeks', 2))),
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
