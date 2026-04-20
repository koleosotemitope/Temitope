import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class EURJPYDataScraper:
    """Loads EUR/JPY historical data from a local CSV file."""

    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    @staticmethod
    def _is_valid_price_data(data):
        return data is not None and not data.empty and 'Close' in data.columns

    def _load_local_csv(self, local_path):
        """
        Load EUR/JPY data from a locally downloaded CSV file.
        Supports Investing.com weekly export format:
          Date, Price, Open, High, Low, Vol., Change %
        """
        if not local_path or not os.path.exists(local_path):
            print(f"Local file not found: {local_path}")
            return pd.DataFrame()

        try:
            data = pd.read_csv(local_path)
            data.columns = [c.strip() for c in data.columns]

            if 'Price' in data.columns and 'Close' not in data.columns:
                data = data.rename(columns={'Price': 'Close'})

            data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
            data = data.set_index('Date').sort_index()

            for col in ['Open', 'High', 'Low', 'Close']:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')

            data['Volume'] = 0
            data = data.dropna(subset=['Close'])
            return data[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            print(f"Local CSV load error: {e}")
            return pd.DataFrame()

    def fetch_data(self, period='10y', local_path=None):
        """
        Fetch EUR/JPY historical data from local CSV only.
        period is kept for compatibility with existing callers.
        """
        print("Fetching EUR/JPY data...")
        try:
            if local_path is None:
                from config import DATA_CONFIG

                local_path = DATA_CONFIG.get('local_csv_path')

            data = self._load_local_csv(local_path)
            if self._is_valid_price_data(data):
                print(f"Data source: Local file ({os.path.basename(local_path)})")

            if not self._is_valid_price_data(data):
                raise ValueError('No valid EUR/JPY price data found in local file.')

            csv_path = os.path.join(self.data_dir, 'eurjpy_historical.csv')
            data.to_csv(csv_path)
            print(f"Data saved to {csv_path}")
            print(f"Data shape: {data.shape}")
            print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")

            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def prepare_data(self, data, lookback=60):
        """
        Prepare data for LSTM model.
        lookback: number of previous time steps to use as input variables
        """
        print(f"\nPreparing data with lookback={lookback}...")

        if data is None or data.empty:
            raise ValueError('Input data is empty. Cannot prepare sequences for LSTM.')
        if 'Close' not in data.columns:
            raise ValueError("Input data does not contain a 'Close' column.")
        if len(data) <= lookback:
            raise ValueError(
                f"Not enough rows ({len(data)}) for lookback={lookback}. Need at least {lookback + 1} rows."
            )

        prices = data['Close'].values.reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices)

        X, y = [], []
        for i in range(len(scaled_prices) - lookback):
            X.append(scaled_prices[i : i + lookback, 0])
            y.append(scaled_prices[i + lookback, 0])

        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        print(f"X shape: {X.shape}, y shape: {y.shape}")

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        return X_train, X_test, y_train, y_test, scaled_prices

    def get_latest_price(self):
        """Get the latest EUR/JPY price from local CSV."""
        try:
            from config import DATA_CONFIG

            local_path = DATA_CONFIG.get('local_csv_path')
            data = self._load_local_csv(local_path)
            if self._is_valid_price_data(data):
                return float(data['Close'].iloc[-1])
            return None
        except Exception:
            return None


if __name__ == '__main__':
    from config import DATA_CONFIG

    scraper = EURJPYDataScraper()
    local_path = DATA_CONFIG.get('local_csv_path')
    data = scraper.fetch_data(period=DATA_CONFIG.get('historical_period', '10y'), local_path=local_path)
    if data is not None:
        X_train, X_test, y_train, y_test, scaled_prices = scraper.prepare_data(data)
