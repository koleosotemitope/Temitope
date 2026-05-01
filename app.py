from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import os
from data_scraper import EURJPYDataScraper
from config import DATA_CONFIG, PREDICTION_CONFIG, MODEL_CONFIG
import traceback

app = Flask(__name__)

# Global variables for model
model = None
model_type = 'unknown'
lookback = 60
lag_count = 30
target_mode = 'next_return'
arima_order = None
model_family = 'arima'
sarima_seasonal_order = None
latest_data = None
feature_columns = []


def _build_lgbm_feature_row(history_data, engineered_data, lag_count_value):
    """Build one-step feature row for LightGBM iterative forecasting."""
    close_series = history_data['Close'].astype(float)
    if len(close_series) < lag_count_value + 14:
        raise ValueError('Not enough history for LightGBM feature generation.')

    latest_features = engineered_data.iloc[-1]
    feature_row = {}

    for lag in range(1, lag_count_value + 1):
        feature_row[f'close_lag_{lag}'] = float(close_series.iloc[-lag])

    returns = close_series.pct_change().fillna(0.0)
    for lag in range(1, 6):
        feature_row[f'return_lag_{lag}'] = float(returns.iloc[-lag])

    for window in [3, 7, 14]:
        window_slice = close_series.iloc[-window:]
        feature_row[f'roll_mean_{window}'] = float(window_slice.mean())
        feature_row[f'roll_std_{window}'] = float(window_slice.std()) if len(window_slice) > 1 else 0.0

    for col in ['EMA10', 'EMA20', 'EMA20_SLOPE', 'MACD_HIST', 'ADX', 'TREND_WEAKENING']:
        feature_row[col] = float(latest_features[col])

    return pd.DataFrame([feature_row])


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
    global model, model_type, lookback, lag_count, target_mode, arima_order, model_family, sarima_seasonal_order, latest_data, feature_columns
    
    try:
        model_name = MODEL_CONFIG.get('model_name', 'lstm_eurjpy')
        model_path_h5 = os.path.join('models', f'{model_name}.h5')
        model_path_pkl = os.path.join('models', f'{model_name}.pkl')
        params_path = os.path.join('models', f'{model_name}_params.pkl')

        if os.path.exists(params_path):
            with open(params_path, 'rb') as f:
                params = pickle.load(f)
                model_type = params.get('model_type', 'lstm')
                lookback = params.get('lookback', lookback)
                lag_count = int(params.get('lag_count', lag_count))
                target_mode = params.get('target_mode', target_mode)
                arima_order = params.get('arima_order', arima_order)
                model_family = params.get('model_family', model_family)
                sarima_seasonal_order = params.get('sarima_seasonal_order', sarima_seasonal_order)
                feature_columns = params.get('feature_columns', [])
        else:
            model_type = 'lstm'

        if model_type in ['lightgbm', 'arima'] and os.path.exists(model_path_pkl):
            with open(model_path_pkl, 'rb') as f:
                model = pickle.load(f)
            print(f"{model_type.upper()} model loaded successfully")
        elif os.path.exists(model_path_h5):
            from tensorflow.keras.models import load_model

            model = load_model(model_path_h5)
            model_type = 'lstm'
            print('LSTM model loaded successfully')
        else:
            print(f"Model not found at {model_path_pkl} or {model_path_h5}. Please train the model first.")
            return False
        
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
        if model_type == 'lstm' and len(feature_data) <= lookback:
            return jsonify({'success': False, 'error': 'Not enough feature rows for prediction'}), 500
        
        # Prepare sequence and forecast horizon (always keep at least 2 for UI)
        horizon_days = int(PREDICTION_CONFIG.get('forecast_days', PREDICTION_CONFIG.get('forecast_weeks', 2)))
        horizon_days = max(2, horizon_days)
        history_data = data.copy()
        last_date = pd.to_datetime(history_data.index[-1])
        predicted_prices = []

        if model_type == 'arima':
            forecast = model.get_forecast(steps=horizon_days)
            predicted_prices = [float(v) for v in forecast.predicted_mean.tolist()]

            for predicted_close in predicted_prices:
                next_date = last_date + pd.Timedelta(days=1)
                history_data = pd.concat([
                    history_data,
                    pd.DataFrame(
                        [{
                            'Open': predicted_close,
                            'High': predicted_close,
                            'Low': predicted_close,
                            'Close': predicted_close,
                            'Volume': 0,
                        }],
                        index=[next_date],
                    ),
                ])
                last_date = next_date
        elif model_type == 'lightgbm':
            for _ in range(horizon_days):
                engineered = scraper.engineer_features(history_data)
                if engineered is None or engineered.empty:
                    return jsonify({'success': False, 'error': 'Could not build LightGBM feature frame'}), 500

                feature_row = _build_lgbm_feature_row(history_data, engineered, lag_count)
                if feature_columns:
                    feature_row = feature_row.reindex(columns=feature_columns, fill_value=0.0)

                raw_pred = float(model.predict(feature_row)[0])
                if target_mode == 'next_return':
                    raw_pred = float(np.clip(raw_pred, -0.05, 0.05))
                    last_close = float(history_data['Close'].iloc[-1])
                    predicted_close = last_close * (1.0 + raw_pred)
                else:
                    predicted_close = raw_pred
                predicted_prices.append(predicted_close)

                next_date = last_date + pd.Timedelta(days=1)
                history_data = pd.concat([
                    history_data,
                    pd.DataFrame(
                        [{
                            'Open': predicted_close,
                            'High': predicted_close,
                            'Low': predicted_close,
                            'Close': predicted_close,
                            'Volume': 0,
                        }],
                        index=[next_date],
                    ),
                ])
                last_date = next_date
        else:
            scraper.fit_feature_pipeline(feature_data)
            scaled_features = scraper.transform_feature_frame(feature_data)
            sequence = scaled_features[-lookback:].copy()
            predicted_scaled_values = []

            for _ in range(horizon_days):
                step_pred = model.predict(sequence.reshape(1, lookback, sequence.shape[1]), verbose=0)[0][0]
                predicted_scaled_values.append(step_pred)

                predicted_close = scraper.inverse_transform_close([step_pred])[0]
                next_date = last_date + pd.Timedelta(days=1)

                new_row = pd.DataFrame(
                    [{
                        'Close': predicted_close,
                        'EMA10': predicted_close,
                        'EMA20': predicted_close,
                        'EMA20_SLOPE': 0.0,
                        'MACD_HIST': 0.0,
                        'ADX': 20.0,
                        'TREND_WEAKENING': 0,
                    }]
                )
                scaled_new_row = scraper.transform_feature_frame(new_row)[0]
                sequence = np.vstack([sequence[1:], scaled_new_row])
                last_date = next_date

            predicted_prices = scraper.inverse_transform_close(predicted_scaled_values)

        forecast_dates = [(pd.to_datetime(data.index[-1]) + pd.Timedelta(days=step + 1)).strftime('%Y-%m-%d') for step in range(horizon_days)]
        forecast_prices = [float(price) for price in predicted_prices]
        forecast_feature_data = scraper.engineer_features(history_data).tail(horizon_days).copy()

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

@app.route('/api/prophet-predict', methods=['POST'])
def prophet_predict():
    """Predict EUR/JPY prices for the next 14 days using Facebook Prophet."""
    try:
        from prophet import Prophet

        scraper = EURJPYDataScraper()
        data = scraper.fetch_data(
            period=DATA_CONFIG.get('historical_period', '10y'),
            local_path=DATA_CONFIG.get('local_csv_path')
        )
        if data is None or data.empty:
            return jsonify({'success': False, 'error': 'Could not load local price data'}), 500

        # Prophet requires columns 'ds' (datetime) and 'y' (value)
        df_prophet = data[['Close']].reset_index()
        df_prophet.columns = ['ds', 'y']
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)

        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        m.fit(df_prophet)

        horizon_days = int(PREDICTION_CONFIG.get('forecast_days', 14))
        future = m.make_future_dataframe(periods=horizon_days)
        forecast = m.predict(future)

        future_forecast = forecast.tail(horizon_days)
        forecast_dates = future_forecast['ds'].dt.strftime('%Y-%m-%d').tolist()
        forecast_prices = [round(float(v), 4) for v in future_forecast['yhat'].tolist()]
        forecast_lower  = [round(float(v), 4) for v in future_forecast['yhat_lower'].tolist()]
        forecast_upper  = [round(float(v), 4) for v in future_forecast['yhat_upper'].tolist()]

        current_price = float(data['Close'].iloc[-1])
        day7_index  = min(6,  len(forecast_prices) - 1)
        day14_index = min(13, len(forecast_prices) - 1)
        day7_price  = forecast_prices[day7_index]
        day14_price = forecast_prices[day14_index]
        day7_change  = ((day7_price  - current_price) / current_price) * 100
        day14_change = ((day14_price - current_price) / current_price) * 100

        return jsonify({
            'success': True,
            'current_price': current_price,
            'forecast_dates': forecast_dates,
            'forecast_prices': forecast_prices,
            'forecast_lower': forecast_lower,
            'forecast_upper': forecast_upper,
            'day7_date': forecast_dates[day7_index],
            'day14_date': forecast_dates[day14_index],
            'day7_predicted_price': day7_price,
            'day14_predicted_price': day14_price,
            'day7_change': round(float(day7_change), 2),
            'day14_change': round(float(day14_change), 2),
            'model': 'Facebook Prophet',
        })
    except Exception as e:
        print(f"Prophet prediction error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model-info')
def get_model_info():
    """Get model information"""
    try:
        return jsonify({
            'success': True,
            'model_type': 'ARIMA' if model_type == 'arima' else ('LightGBM Regressor' if model_type == 'lightgbm' else 'LSTM (Long Short-Term Memory)'),
            'model_family': model_family if model_type == 'arima' else None,
            'lookback_days': lookback,
            'lag_count': lag_count if model_type == 'lightgbm' else None,
            'arima_order': arima_order if model_type == 'arima' else None,
            'sarima_seasonal_order': sarima_seasonal_order if model_type == 'arima' else None,
            'layers': 'ARIMA statistical model' if model_type == 'arima' else ('Gradient boosting trees' if model_type == 'lightgbm' else 'LSTM recurrent layers with dense head'),
            'optimization': ('AIC tuning + holdout RMSE model selection' if model_type == 'arima' else ('RandomizedSearchCV + TimeSeriesSplit' if model_type == 'lightgbm' else 'Adam (lr=0.001)')),
            'loss_function': 'MSE (regression)',
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
