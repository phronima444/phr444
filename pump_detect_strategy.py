import pandas as pd
import time

def detect_pump(exchange, symbol, time_interval='5s'):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=time_interval, limit=1)
        ohlcv_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        current_price = ohlcv_df['close'].iloc[-1]
        previous_price = ohlcv_df['open'].iloc[-1]
        volume = ohlcv_df['volume'].iloc[-1]

        price_change = ((current_price - previous_price) / previous_price) * 100

        if price_change >= 1.5:
            return symbol, price_change, volume

    except Exception as e:
        print(e)

    return None

def main():
    exchange = Exchange()
    while True:
        symbols = exchange.load_markets().keys()
        for symbol in symbols:
            pump_detection = detect_pump(exchange, symbol)
            if pump_detection is not None:
                print('Detected pump:', pump_detection)
        time.sleep(5)

if __name__ == '__main__':
    main()
