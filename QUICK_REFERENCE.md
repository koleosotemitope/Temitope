# EUR/JPY LSTM Predictor - Quick Reference Guide

## 🚀 Common Commands

### Setup & Installation
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows CMD)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements_new.txt
```

### Training & Running

```bash
# Train LSTM model (5-10 minutes)
python lstm_model.py

# Run web application
python app.py

# Run with specific port
python -c "import app; app.app.run(port=5001)"

# Run in production
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Commands

```bash
# Build Docker image
docker build -t eurjpy-predictor .

# Run Docker container
docker run -p 5000:5000 eurjpy-predictor

# Using Docker Compose
docker-compose up -d
docker-compose down
docker-compose logs -f

# Check container status
docker ps
docker stats eurjpy-predictor

# Remove containers and images
docker stop eurjpy-predictor
docker rm eurjpy-predictor
docker rmi eurjpy-predictor
```

### Testing

```bash
# Run all tests
python test_app.py --all

# Run unit tests only
python test_app.py --unit

# Run integration test
python test_app.py --integration

# Run performance benchmark
python test_app.py --benchmark

# Run with verbose output
python -m pytest test_app.py -v
```

### Monitoring & Debugging

```bash
# Check application logs
tail -f logs/app.log

# Monitor system resources
docker stats eurjpy-predictor

# Check if server is running
curl http://localhost:5000

# Test API endpoints
curl http://localhost:5000/api/current-price
curl http://localhost:5000/api/model-info

# View Flask debug info
python -c "import app; app.app.config['DEBUG'] = True; app.app.run()"
```

---

## 📁 File Quick Reference

| File | Purpose | Run Command |
|------|---------|------------|
| `lstm_model.py` | Train LSTM | `python lstm_model.py` |
| `data_scraper.py` | Fetch data | `python data_scraper.py` |
| `app.py` | Web server | `python app.py` |
| `test_app.py` | Run tests | `python test_app.py --all` |
| `setup.py` | Interactive setup | `python setup.py` |
| `config.py` | Configuration | Edit directly |
| `utils.py` | Helper tools | Import in code |

---

## 🌐 Web Interface URLs

| Endpoint | Description | Method |
|----------|-------------|--------|
| `/` | Main dashboard | GET |
| `/api/current-price` | Get current price | GET |
| `/api/predict` | Make prediction | POST |
| `/api/historical-data` | Get history | GET |
| `/api/model-info` | Model info | GET |
| `/health` | Health check | GET |

---

## 📊 Working with Data

### View Current Data
```python
import yfinance as yf
data = yf.download('EURJPY=X', period='1mo')
print(data.tail())
```

### Train with Different Period
```python
from data_scraper import EURJPYDataScraper
scraper = EURJPYDataScraper()
data = scraper.fetch_data(period='1y')  # 1mo, 3mo, 6mo, 1y, 5y
X_train, X_test, y_train, y_test, scaled = scraper.prepare_data(data)
```

### Use Cached Data
```python
# Skip downloading if data exists
import os
if os.path.exists('data/eurjpy_historical.csv'):
    data = pd.read_csv('data/eurjpy_historical.csv', index_col=0, parse_dates=True)
```

---

## 🧠 Model Customization

### Change Model Architecture
```python
# In lstm_model.py
model = Sequential([
    LSTM(100, activation='relu', return_sequences=True, input_shape=(60, 1)),
    Dropout(0.3),
    LSTM(100, activation='relu'),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dense(1)
])
```

### Adjust Training Parameters
```python
# In config.py
TRAINING_CONFIG = {
    'epochs': 100,          # More epochs
    'batch_size': 16,       # Smaller batches
    'learning_rate': 0.0005, # Lower learning rate
}
```

### Change Lookback Period
```python
# In config.py
MODEL_CONFIG = {
    'lookback': 120,  # Use 4 months instead of 2
}
```

---

## 🔧 Troubleshooting

### Issue: "Model not found"
```bash
# Solution: Train the model first
python lstm_model.py
```

### Issue: "Port 5000 already in use"
```bash
# Solution 1: Change port in app.py
# Edit app.py last line: app.run(port=5001)

# Solution 2: Kill process using port
# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux:
lsof -i :5000
kill -9 <PID>
```

### Issue: "No module named 'tensorflow'"
```bash
# Solution: Install TensorFlow
pip install tensorflow==2.13.0

# Or CPU-only version
pip install tensorflow-cpu
```

### Issue: "CUDA out of memory"
```bash
# Solution: Use CPU instead of GPU
# Set environment variable:
# Windows: set CUDA_VISIBLE_DEVICES=-1
# Linux: export CUDA_VISIBLE_DEVICES=-1
# Then run: python app.py
```

### Issue: "Data download failed"
```bash
# Solution: Check internet connection and try again
# Or use local data if available
import pandas as pd
data = pd.read_csv('data/eurjpy_historical.csv', index_col=0, parse_dates=True)
```

---

## 🌍 Deployment Checklists

### Before Deployment
- [ ] Run all tests: `python test_app.py --all`
- [ ] Check model exists: `ls models/lstm_eurjpy.h5`
- [ ] Verify data: `ls data/eurjpy_historical.csv`
- [ ] Update `config.py` for production
- [ ] Review `requirements_new.txt`
- [ ] Check logs are working

### Docker Deployment
- [ ] Build image: `docker build -t eurjpy-predictor .`
- [ ] Test locally: `docker run -p 5000:5000 eurjpy-predictor`
- [ ] Push to registry: `docker push your-registry/eurjpy-predictor`
- [ ] Deploy with compose: `docker-compose up -d`

### Cloud Deployment (Azure)
- [ ] Create resource group
- [ ] Create App Service Plan
- [ ] Create Web App
- [ ] Configure startup command
- [ ] Set environment variables
- [ ] Deploy code
- [ ] Verify endpoint

---

## 📈 Performance Tips

### Improve Training Speed
```python
# Use GPU (if available)
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# Reduce lookback period
lookback = 30  # Instead of 60

# Use smaller batch size
batch_size = 16  # Instead of 32
```

### Improve API Response Time
```python
# Enable caching in app.py
from utils import DataCache
cache = DataCache(ttl=300)  # 5 minute TTL

# Reduce model size
# Train on less data or use quantization
```

### Improve Memory Usage
```python
# Load model only once
# Use model optimization
# Implement garbage collection
import gc
gc.collect()
```

---

## 📊 Monitoring Checklist

### Daily Tasks
- [ ] Check application is running
- [ ] Verify predictions are being made
- [ ] Review error logs
- [ ] Monitor disk space

### Weekly Tasks
- [ ] Analyze prediction accuracy
- [ ] Review performance metrics
- [ ] Update data
- [ ] Check security logs

### Monthly Tasks
- [ ] Retrain model if needed
- [ ] Optimize performance
- [ ] Update dependencies
- [ ] Backup data and models

---

## 🔐 Security Checklist

- [ ] Use environment variables for secrets
- [ ] Enable HTTPS in production
- [ ] Implement rate limiting
- [ ] Add authentication if needed
- [ ] Validate all inputs
- [ ] Use secure headers
- [ ] Keep dependencies updated
- [ ] Regular security audits

---

## 📚 Documentation References

| Document | Content |
|----------|---------|
| `README.md` | Full documentation |
| `DEPLOYMENT.md` | Deployment guide |
| `PROJECT_SUMMARY.md` | Project overview |
| `config.py` | Configuration options |
| `utils.py` | Helper functions |

---

## 💡 Pro Tips

1. **Use `.env` file for secrets**
   ```bash
   # Create .env file
   FLASK_SECRET_KEY=your_secret_key
   DATABASE_URL=your_db_url
   ```

2. **Monitor with Gunicorn workers**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app
   ```

3. **Enable logging in production**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

4. **Use health check endpoint**
   ```bash
   curl http://localhost:5000/health
   ```

5. **Backup models regularly**
   ```bash
   tar -czf backup_models.tar.gz models/
   ```

---

**Last Updated:** April 2026  
**Version:** 1.0  
**Status:** Production Ready
