# EUR/JPY LSTM Price Predictor

A machine learning project that uses LSTM (Long Short-Term Memory) neural networks to predict EUR/JPY currency pair prices and provides a web interface for real-time predictions.

## Project Structure

```
MLOPS_Demo/
├── app.py                    # Flask web application
├── lstm_model.py             # LSTM model training script
├── data_scraper.py           # Data scraping and preparation
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── models/                   # Trained models directory
│   ├── lstm_eurjpy.h5        # Trained LSTM model
│   ├── lstm_eurjpy_params.pkl # Model parameters
│   └── training_history.png   # Training visualization
├── data/                     # Data directory
│   └── eurjpy_historical.csv  # Historical price data
├── templates/                # HTML templates
│   └── index.html             # Web interface
└── static/                   # Static files (CSS, JS)
```

## Features

✨ **Core Features:**
- **LSTM Neural Network**: 3-layer LSTM with dropout for EUR/JPY price prediction
- **Real-time Data**: Fetches current prices from Yahoo Finance
- **Web Interface**: Beautiful, responsive dashboard built with Flask and Chart.js
- **Historical Analysis**: 30-day, 90-day, 180-day, and 1-year historical data visualization
- **Prediction Model**: Trained on 5 years of EUR/JPY historical data
- **REST API**: Clean API endpoints for price data and predictions

## Installation

### 1. Create and activate virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the LSTM Model

Before running the web application, you need to train the LSTM model:

```bash
python lstm_model.py
```

This will:
- Download 5 years of EUR/JPY historical data
- Preprocess and normalize the data
- Train the LSTM model (50 epochs)
- Save the model to `models/lstm_eurjpy.h5`
- Generate training history visualization

**Expected output:**
```
Fetching EUR/JPY data for 5y...
Preparing data with lookback=60...
Building LSTM model with input shape (60, 1)...
Training LSTM model for 50 epochs...
Epoch 1/50
...
Training complete!
```

⏱️ **Training time:** 5-10 minutes (depends on your system)

### Step 2: Run the Web Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

**Expected output:**
```
Initializing application...
Model loaded successfully
Application ready!
 * Running on http://0.0.0.0:5000
```

### Step 3: Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

## API Endpoints

### Get Current Price
```
GET /api/current-price
```
Returns current EUR/JPY price and 30-day historical data.

**Response:**
```json
{
  "success": true,
  "current_price": 128.45,
  "historical": [128.1, 128.2, ...],
  "dates": ["2024-03-20", "2024-03-21", ...]
}
```

### Make Prediction
```
POST /api/predict
```
Predicts the next EUR/JPY price.

**Response:**
```json
{
  "success": true,
  "current_price": 128.45,
  "predicted_price": 129.12,
  "change": 0.52,
  "change_direction": "up"
}
```

### Get Historical Data
```
GET /api/historical-data?days=90
```
Gets historical data for specified number of days.

**Parameters:**
- `days`: Number of days (default: 90)

**Response:**
```json
{
  "success": true,
  "data": {
    "dates": ["2023-12-20", ...],
    "open": [128.0, ...],
    "high": [128.5, ...],
    "low": [127.5, ...],
    "close": [128.2, ...],
    "volume": [1000000, ...]
  }
}
```

### Get Model Information
```
GET /api/model-info
```
Returns details about the trained LSTM model.

## Model Architecture

The LSTM model consists of:

1. **Input Layer**: 60 time steps (days)
2. **LSTM Layer 1**: 50 units + Dropout(0.2)
3. **LSTM Layer 2**: 50 units + Dropout(0.2)
4. **LSTM Layer 3**: 50 units + Dropout(0.2)
5. **Dense Layer**: 25 units (ReLU activation)
6. **Output Layer**: 1 unit (Linear activation)

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss Function: Mean Squared Error
- Batch Size: 32
- Epochs: 50
- Train/Test Split: 80/20

## Deployment

### Local Development
```bash
python app.py
```

### Production with Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t eurjpy-predictor .
docker run -p 5000:5000 eurjpy-predictor
```

### Azure App Service Deployment

1. Create an Azure App Service with Python runtime
2. Deploy using Azure CLI or Git integration
3. Set up environment variables for production
4. Configure app settings in Azure portal

## Troubleshooting

### Model Not Found Error
**Solution:** Run `python lstm_model.py` first to train the model.

### TensorFlow/CUDA Issues
**Solution:** Install CPU-only TensorFlow:
```bash
pip install tensorflow-cpu
```

### Data Download Fails
**Solution:** Check internet connection and ensure Yahoo Finance is accessible.

### Port 5000 Already in Use
**Solution:** Use a different port:
```bash
python -c "import app; app.app.run(port=5001)"
```

## Performance Metrics

After training, the model typically achieves:
- **RMSE**: ~0.5-1.0 JPY
- **MAE**: ~0.3-0.7 JPY
- **Accuracy**: 50-65% directional accuracy

Note: Forex prediction is inherently difficult; use predictions as one input among many for trading decisions.

## Data Sources

- **Historical Data**: Yahoo Finance
- **Currency Pair**: EUR/JPY (EURJPY=X)
- **Data Period**: 5 years of daily data
- **Updates**: Real-time via yfinance API

## Technologies Used

- **Backend**: Flask (Python)
- **ML/DL**: TensorFlow/Keras, scikit-learn
- **Data**: pandas, numpy, yfinance
- **Frontend**: HTML5, CSS3, Chart.js
- **Deployment**: Flask, Gunicorn, Docker

## Future Enhancements

- [ ] Add more technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Implement ensemble models
- [ ] Add backtesting framework
- [ ] Create mobile app interface
- [ ] Add email alerts for predictions
- [ ] Store historical predictions for accuracy tracking
- [ ] Add multiple currency pairs
- [ ] Implement real-time WebSocket updates

## Risk Disclaimer

This prediction tool is for educational purposes only. Do not rely solely on AI predictions for financial decisions. Always conduct your own research and consult with a financial advisor before making investment decisions.

## License

MIT License - feel free to use for educational and commercial purposes.

## Author

Created as an MLOps demonstration project.

## Contact & Support

For issues or questions, please refer to the code documentation or create an issue in the repository.

---

**Last Updated**: April 2026
