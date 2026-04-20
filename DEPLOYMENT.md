# EUR/JPY LSTM Predictor - Deployment Guide

## Table of Contents
1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Azure App Service](#azure-app-service)
4. [AWS Deployment](#aws-deployment)
5. [Performance Optimization](#performance-optimization)
6. [Monitoring & Logging](#monitoring--logging)

---

## Local Development

### Quick Start (Windows)

1. **Open PowerShell and navigate to project directory:**
```powershell
cd "C:\Users\koleo\Downloads\to be submited\tope\MLOPS_Demo"
```

2. **Activate virtual environment:**
```powershell
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies:**
```powershell
pip install -r requirements_new.txt
```

4. **Train the model (first time only):**
```powershell
python lstm_model.py
```

5. **Run the application:**
```powershell
python app.py
```

6. **Access the web interface:**
Open browser and go to: `http://localhost:5000`

### Troubleshooting

**If you get "cannot be loaded because running scripts is disabled":**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**If port 5000 is already in use:**
```python
# Edit app.py and change the port:
app.run(debug=True, port=5001)
```

---

## Docker Deployment

### Docker Basics

**Build the image:**
```bash
docker build -t eurjpy-predictor:latest .
```

**Run the container:**
```bash
docker run -d \
  --name eurjpy-predictor \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  eurjpy-predictor:latest
```

**View logs:**
```bash
docker logs -f eurjpy-predictor
```

**Stop the container:**
```bash
docker stop eurjpy-predictor
docker rm eurjpy-predictor
```

### Docker Compose (Recommended)

**Start services:**
```bash
docker-compose up -d
```

**View logs:**
```bash
docker-compose logs -f eurjpy-predictor
```

**Stop services:**
```bash
docker-compose down
```

**Rebuild images:**
```bash
docker-compose up -d --build
```

---

## Azure App Service

### Prerequisites
- Azure account
- Azure CLI installed
- Git configured

### Deployment Steps

1. **Create Resource Group:**
```bash
az group create \
  --name eurjpy-rg \
  --location eastus
```

2. **Create App Service Plan:**
```bash
az appservice plan create \
  --name eurjpy-plan \
  --resource-group eurjpy-rg \
  --sku B2 \
  --is-linux
```

3. **Create Web App:**
```bash
az webapp create \
  --resource-group eurjpy-rg \
  --plan eurjpy-plan \
  --name eurjpy-predictor \
  --runtime "PYTHON:3.10"
```

4. **Configure Startup Command:**
```bash
az webapp config set \
  --resource-group eurjpy-rg \
  --name eurjpy-predictor \
  --startup-file "gunicorn -w 4 -b 0.0.0.0:8000 app:app"
```

5. **Deploy from Git:**
```bash
# If not already a git repo
git init
git add .
git commit -m "Initial commit"

# Create Azure remote
az webapp deployment source config-zip \
  --resource-group eurjpy-rg \
  --name eurjpy-predictor \
  --src deployment.zip
```

### Or use Azure DevOps Pipeline

Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Azure

on: [push]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Deploy to Azure App Service
        uses: azure/webapps-deploy@v2
        with:
          app-name: eurjpy-predictor
          publish-profile: ${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE }}
```

---

## AWS Deployment

### Using EC2 + RDS

1. **Launch EC2 Instance:**
- Ubuntu 20.04 LTS
- t3.medium (recommended)
- Security group: Allow ports 80, 443, 22, 5000

2. **Connect and Setup:**
```bash
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-venv git

# Clone repository
git clone your-repo-url
cd MLOPS_Demo

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_new.txt
```

3. **Train Model:**
```bash
python lstm_model.py
```

4. **Run with Gunicorn:**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using AWS Elastic Beanstalk

1. **Create .ebextensions/python.config:**
```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: app:app
  aws:elasticbeanstalk:application:environment:
    PYTHONPATH: /var/app/current:$PYTHONPATH
```

2. **Deploy:**
```bash
eb create eurjpy-env
eb deploy
```

### Using AWS Lambda (Serverless)

⚠️ **Note:** Due to large TensorFlow dependencies and long training time, Lambda is not recommended for this project. Use EC2 or Elastic Beanstalk instead.

---

## Performance Optimization

### 1. Model Optimization

**Quantization (Reduce model size):**
```python
# In lstm_model.py
import tensorflow_lite as tflite

converter = tflite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

**Model Pruning:**
```python
import tensorflow_model_optimization as tfmot

pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(...)
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)
```

### 2. Caching Strategy

Add Redis caching to `app.py`:
```python
import redis
cache = redis.Redis(host='localhost', port=6379)

@app.route('/api/current-price')
def get_current_price():
    cached = cache.get('eurjpy_price')
    if cached:
        return json.loads(cached)
    # ... fetch data ...
    cache.setex('eurjpy_price', 300, json.dumps(result))  # 5 min cache
```

### 3. Database Optimization

Use PostgreSQL for storing predictions:
```python
# requirements.txt: add psycopg2-binary
import psycopg2

def save_prediction(current, predicted, change):
    conn = psycopg2.connect("dbname=eurjpy user=postgres")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions VALUES (%s, %s, %s)",
        (current, predicted, change)
    )
    conn.commit()
```

### 4. Load Balancing

With AWS ALB:
```bash
# Create target group
aws elbv2 create-target-group --name eurjpy-tg

# Create ALB
aws elbv2 create-load-balancer --name eurjpy-alb
```

---

## Monitoring & Logging

### Application Logging

Update `app.py`:
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    
    file_handler = RotatingFileHandler('logs/eurjpy.log', maxBytes=10240000, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
```

### Azure Monitor

```bash
# Create Application Insights
az monitor app-insights component create \
  --app eurjpy-insights \
  --location eastus \
  --resource-group eurjpy-rg
```

### CloudWatch (AWS)

```bash
# View logs
aws logs tail /aws/elasticbeanstalk/eurjpy-env/var/log/eb-engine.log
```

### Docker Container Monitoring

```bash
# Check container stats
docker stats eurjpy-predictor

# View container logs
docker logs eurjpy-predictor
```

---

## Scaling Strategies

### Horizontal Scaling (Multiple Instances)
- Use load balancer (AWS ALB, Azure Load Balancer)
- Run multiple Docker containers behind nginx
- Deploy to Kubernetes for advanced orchestration

### Vertical Scaling (Larger Instance)
- Upgrade VM/EC2 size
- Increase memory and CPU
- Use GPU instances for model training

### Caching & CDN
- Implement Redis for API response caching
- Use CloudFront (AWS) or Azure CDN for static assets
- Cache historical data locally

---

## Security Considerations

1. **Environment Variables:**
```bash
# Use .env file (not in git)
FLASK_SECRET_KEY=your_secret_key
DATABASE_URL=postgresql://...
```

2. **HTTPS/SSL:**
- Use Let's Encrypt for free certificates
- Configure in nginx/Apache
- Enable HSTS headers

3. **API Authentication:**
```python
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    # Verify credentials
    return True

@app.route('/api/predict')
@auth.login_required
def predict():
    # Protected endpoint
```

4. **Rate Limiting:**
```python
from flask_limiter import Limiter
limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/api/predict')
@limiter.limit("10 per minute")
def predict():
    pass
```

---

## Health Checks & Alerting

### Health Check Endpoint

Add to `app.py`:
```python
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now()
    })
```

### Azure Alerts

```bash
az monitor metrics alert create \
  --name eurjpy-high-latency \
  --resource-group eurjpy-rg \
  --scopes /subscriptions/.../eurjpy-predictor \
  --condition "avg response_time > 2000"
```

---

## Rollback Procedures

### Docker Rollback
```bash
# Tag previous working image
docker tag eurjpy-predictor:v1.0 eurjpy-predictor:latest
docker run eurjpy-predictor:latest
```

### Azure Rollback
```bash
# Swap deployment slots
az webapp deployment slot swap \
  --name eurjpy-predictor \
  --resource-group eurjpy-rg \
  --slot staging
```

---

## Cost Optimization

- **Azure**: Use Spot VMs for non-critical workloads
- **AWS**: Use Reserved Instances for predictable usage
- **Docker**: Use multi-stage builds to reduce image size
- **Data**: Limit API calls with caching (300s TTL)
- **Storage**: Archive old predictions to cold storage

---

For more information, refer to the main [README.md](README.md).
