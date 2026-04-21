import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class EURJPYDataScraper:
    """Loads EUR/JPY historical data from a local CSV file."""

    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = [
            'Close',
            'EMA10',
            'EMA20',
            'EMA20_SLOPE',
            'MACD_HIST',
            'ADX',
            'TREND_WEAKENING',
        ]

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

    @staticmethod
    def _calculate_adx(data, period=14):
        high = data['High']
        low = data['Low']
        close = data['Close']

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr_components = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)
        atr = true_range.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr.replace(0, np.nan))
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        return dx.rolling(window=period).mean()

    def engineer_features(self, data):
        """Create technical features used by the model and dashboard."""
        if data is None or data.empty:
            return pd.DataFrame()

        feature_data = data.copy().sort_index()
        feature_data['EMA10'] = feature_data['Close'].ewm(span=10, adjust=False).mean()
        feature_data['EMA20'] = feature_data['Close'].ewm(span=20, adjust=False).mean()
        feature_data['EMA20_SLOPE'] = feature_data['EMA20'].diff()

        ema12 = feature_data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = feature_data['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        feature_data['MACD_HIST'] = macd - macd_signal

        feature_data['ADX'] = self._calculate_adx(feature_data)
        feature_data['TREND_WEAKENING'] = (
            (feature_data['ADX'] < feature_data['ADX'].shift(1))
            & (feature_data['ADX'].shift(1) < feature_data['ADX'].shift(2))
            & (feature_data['ADX'] > 20)
            & (feature_data['MACD_HIST'] < feature_data['MACD_HIST'].shift(1))
        ).astype(int)

        feature_data = feature_data.dropna(subset=self.feature_columns)
        return feature_data

    def fit_feature_pipeline(self, feature_data):
        self.feature_scaler.fit(feature_data[self.feature_columns])
        self.target_scaler.fit(feature_data[['Close']])

    def transform_feature_frame(self, feature_data):
        return self.feature_scaler.transform(feature_data[self.feature_columns])

    def inverse_transform_close(self, scaled_close):
        values = np.array(scaled_close).reshape(-1, 1)
        return self.target_scaler.inverse_transform(values).flatten()

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
        feature_data = self.engineer_features(data)
        if len(feature_data) <= lookback:
            raise ValueError(
                f"Not enough rows ({len(feature_data)}) for lookback={lookback}. Need at least {lookback + 1} rows."
            )

        self.fit_feature_pipeline(feature_data)
        scaled_features = self.transform_feature_frame(feature_data)
        scaled_targets = self.target_scaler.transform(feature_data[['Close']])

        X, y = [], []
        for i in range(len(scaled_features) - lookback):
            X.append(scaled_features[i : i + lookback])
            y.append(scaled_targets[i + lookback, 0])

        X = np.array(X)
        y = np.array(y)

        print(f"X shape: {X.shape}, y shape: {y.shape}")

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        return X_train, X_test, y_train, y_test, scaled_targets, feature_data

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
        X_train, X_test, y_train, y_test, scaled_prices, feature_data = scraper.prepare_data(data)
