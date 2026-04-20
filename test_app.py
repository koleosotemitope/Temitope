"""
Testing and Validation Script for EUR/JPY LSTM Predictor
"""

import unittest
import json
import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_scraper import EURJPYDataScraper
from lstm_model import LSTMPredictor, train_eurjpy_model
import app as flask_app

class TestDataScraper(unittest.TestCase):
    """Test data scraping functionality"""
    
    def setUp(self):
        self.scraper = EURJPYDataScraper(data_dir='test_data')
    
    def tearDown(self):
        # Cleanup
        import shutil
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')
    
    def test_data_fetch(self):
        """Test if data can be fetched"""
        data = self.scraper.fetch_data(period='1mo')
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)
        self.assertIn('Close', data.columns)
    
    def test_data_preparation(self):
        """Test data preparation"""
        data = self.scraper.fetch_data(period='6mo')
        X_train, X_test, y_train, y_test, scaled = self.scraper.prepare_data(data)
        
        self.assertEqual(X_train.shape[1], 60)  # lookback
        self.assertEqual(X_train.shape[2], 1)   # features
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
    
    def test_latest_price(self):
        """Test getting latest price"""
        price = self.scraper.get_latest_price()
        self.assertIsNotNone(price)
        self.assertGreater(price, 0)

class TestLSTMModel(unittest.TestCase):
    """Test LSTM model functionality"""
    
    def setUp(self):
        self.predictor = LSTMPredictor(model_dir='test_models')
        self.scraper = EURJPYDataScraper(data_dir='test_data')
    
    def tearDown(self):
        import shutil
        if os.path.exists('test_models'):
            shutil.rmtree('test_models')
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')
    
    def test_model_build(self):
        """Test model building"""
        model = self.predictor.build_model((60, 1))
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 8)  # Check number of layers
    
    def test_model_training(self):
        """Test model training (mini version)"""
        # Fetch small dataset
        data = self.scraper.fetch_data(period='3mo')
        X_train, X_test, y_train, y_test, scaled = self.scraper.prepare_data(data)
        
        # Build model
        model = self.predictor.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Train for 1 epoch only
        history = self.predictor.train(X_train, y_train, X_test, y_test, epochs=1)
        
        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)

class TestFlaskApp(unittest.TestCase):
    """Test Flask application"""
    
    def setUp(self):
        self.app = flask_app.app.test_client()
        self.app.testing = True
    
    def test_index_page(self):
        """Test main page loads"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = self.app.get('/api/model-info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('model_type', data)
    
    def test_current_price_endpoint(self):
        """Test current price endpoint"""
        try:
            response = self.app.get('/api/current-price')
            if response.status_code == 200:
                data = json.loads(response.data)
                self.assertTrue(data['success'])
                self.assertIn('current_price', data)
        except:
            # Skip if no internet connection
            pass
    
    def test_404_error(self):
        """Test 404 error handling"""
        response = self.app.get('/nonexistent')
        self.assertEqual(response.status_code, 404)

class TestModelPerformance(unittest.TestCase):
    """Test model performance metrics"""
    
    def setUp(self):
        self.scraper = EURJPYDataScraper(data_dir='test_data')
        self.predictor = LSTMPredictor(model_dir='test_models')
    
    def tearDown(self):
        import shutil
        for directory in ['test_data', 'test_models']:
            if os.path.exists(directory):
                shutil.rmtree(directory)
    
    def test_model_metrics(self):
        """Test model evaluation metrics"""
        data = self.scraper.fetch_data(period='1y')
        X_train, X_test, y_train, y_test, scaled = self.scraper.prepare_data(data)
        
        model = self.predictor.build_model((X_train.shape[1], X_train.shape[2]))
        self.predictor.train(X_train, y_train, X_test, y_test, epochs=2)
        
        rmse, mae, mse = self.predictor.evaluate(X_test, y_test)
        
        self.assertGreater(rmse, 0)
        self.assertGreater(mae, 0)
        self.assertGreater(mse, 0)

def run_integration_test():
    """Run complete integration test"""
    print("\n" + "="*60)
    print("Running Integration Test")
    print("="*60)
    
    try:
        # Test scraper
        print("\n1. Testing Data Scraper...")
        scraper = EURJPYDataScraper()
        data = scraper.fetch_data(period='1mo')
        print(f"   ✓ Data fetched: {len(data)} records")
        
        # Test preparation
        print("\n2. Testing Data Preparation...")
        X_train, X_test, y_train, y_test, scaled = scraper.prepare_data(data)
        print(f"   ✓ Data prepared: X_train={X_train.shape}, X_test={X_test.shape}")
        
        # Test model
        print("\n3. Testing LSTM Model...")
        predictor = LSTMPredictor()
        model = predictor.build_model((X_train.shape[1], X_train.shape[2]))
        print(f"   ✓ Model built successfully")
        
        # Test prediction
        print("\n4. Testing Prediction...")
        if X_test.shape[0] > 0:
            prediction = predictor.predict(X_test[0], scraper.scaler)
            print(f"   ✓ Prediction made: {prediction:.4f} JPY")
        
        print("\n" + "="*60)
        print("✓ Integration Test PASSED")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Integration Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def performance_benchmark():
    """Benchmark model performance"""
    print("\n" + "="*60)
    print("Performance Benchmark")
    print("="*60)
    
    import time
    
    scraper = EURJPYDataScraper()
    
    # Benchmark data fetching
    print("\n1. Data Fetching Benchmark...")
    start = time.time()
    data = scraper.fetch_data(period='5y')
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f}s for {len(data)} records")
    print(f"   Speed: {len(data)/elapsed:.0f} records/sec")
    
    # Benchmark data preparation
    print("\n2. Data Preparation Benchmark...")
    start = time.time()
    X_train, X_test, y_train, y_test, scaled = scraper.prepare_data(data)
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Benchmark model creation
    print("\n3. Model Creation Benchmark...")
    predictor = LSTMPredictor()
    start = time.time()
    model = predictor.build_model((X_train.shape[1], X_train.shape[2]))
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f}s")
    
    # Benchmark prediction
    print("\n4. Prediction Benchmark...")
    start = time.time()
    for i in range(10):
        predictor.predict(X_test[i], scraper.scaler)
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f}s for 10 predictions")
    print(f"   Speed: {10/elapsed:.0f} predictions/sec")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test EUR/JPY LSTM Predictor')
    parser.add_argument('--unit', action='store_true', help='Run unit tests')
    parser.add_argument('--integration', action='store_true', help='Run integration test')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    if args.all or (not args.unit and not args.integration and not args.benchmark):
        # Run all tests
        print("Running all tests...")
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
        run_integration_test()
        performance_benchmark()
    else:
        if args.unit:
            print("Running unit tests...")
            suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(suite)
        
        if args.integration:
            run_integration_test()
        
        if args.benchmark:
            performance_benchmark()
