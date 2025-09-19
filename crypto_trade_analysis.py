# crypto_trade_analysis.py

import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import ta

# ------------------------------
# Fetch Binance Data
# ------------------------------
def fetch_binance(symbol="BTC/USDT", timeframe="1h", limit=200):
    try:
        binance = ccxt.binance()
        ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception:
        return None

# ------------------------------
# Fetch Yahoo Finance Data
# ------------------------------
def fetch_yfinance(symbol="BTC-USD", period="30d", interval="1h"):
    df = yf.download(symbol, period=period, interval=interval)
    if df.empty:
        return None
    df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
    return df

# ------------------------------
# Compute Indicators
# ------------------------------
def compute_indicators(df):
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['close']).macd()
    df['SignalLine'] = ta.trend.MACD(df['close']).macd_signal()
    df['ATR'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14
    ).average_true_range()
    df['VolSpike'] = np.where(
        df['volume'] > df['volume'].rolling(20).mean() * 1.5, 1, 0
    )
    return df

# ------------------------------
# Generate Signals
# ------------------------------
def generate_signals(df):
    df['Signal'] = np.where(
        (df['EMA9'] > df['EMA21']) & (df['RSI'] < 70) & (df['VolSpike'] == 1),
        "Buy",
        np.where(
            (df['EMA9'] < df['EMA21']) & (df['RSI'] > 30) & (df['VolSpike'] == 1),
            "Short",
            "Hold",
        ),
    )

    df['Entry'] = df['close']
    df['SL'] = np.where(
        df['Signal'] == "Buy",
        df['close'] - df['ATR'],
        np.where(df['Signal'] == "Short", df['close'] + df['ATR'], np.nan),
    )
    df['TP'] = np.where(
        df['Signal'] == "Buy",
        df['close'] + 2 * df['ATR'],
        np.where(df['Signal'] == "Short", df['close'] - 2 * df['ATR'], np.nan),
    )

    df['TradeType'] = np.where(df['ATR'] > df['close'] * 0.01, "Futures", "Spot")

    return df
