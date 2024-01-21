
import ccxt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def main():
    # Connect to Binance
    exchange = ccxt.binance({
        'rateLimit': 1200,  # Binance rate limit
        'enableRateLimit': True
    })

    # Get all symbols that end with 'USDT'
    markets = exchange.load_markets()
    symbols = [market for market in markets if market.endswith('USDT')]

    # Function to fetch OHLCV data for a symbol
    def fetch_ohlcv_data(symbol, timeframe='1d'):
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    # Fetch and store data for each symbol
    all_data = []
    for symbol in symbols:
        try:
            data = fetch_ohlcv_data(symbol)
            all_data.append(data['close'])  # Example: focusing on 'close' prices
            print(f'Data for {symbol} saved.')
        except Exception as e:
            print(f'Failed to fetch data for {symbol}: {e}')

    # Combine and preprocess data
    combined_data = pd.concat(all_data, axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(combined_data)

    # Create a dataset for training (example function, adjust as needed)
    def create_dataset(dataset, time_step=60):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step)]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    # Prepare data for the model
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model (using a small number of epochs for demonstration)
    model.fit(X, y, batch_size=1, epochs=1)

    # Save the model
    model.save('model.h5')
    print("Model saved as model.h5")

if __name__ == '__main__':
    main()
