# crypto_trade_analysis.py

import pandas as pd
import numpy as np
import ta


# ==============================
# Compute Technical Indicators
# ==============================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes OHLCV dataframe and adds indicators:
    RSI, MACD, EMA20, SMA50, Bollinger Bands
    """

    # ✅ Make sure column names match
    if "Close" not in df.columns:
        raise ValueError(f"Expected 'Close' in dataframe, but got: {df.columns}")

    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # EMA / SMA
    df["EMA20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["SMA50"] = ta.trend.SMAIndicator(df["Close"], window=50).sma_indicator()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()

    return df


# ==============================
# Generate Buy/Sell Signals
# ==============================
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses indicators to generate Buy / Sell signals
    """

    signals = []

    for i in range(len(df)):
        if df["RSI"].iloc[i] < 30 and df["MACD"].iloc[i] > df["MACD_Signal"].iloc[i]:
            signals.append("BUY")
        elif df["RSI"].iloc[i] > 70 and df["MACD"].iloc[i] < df["MACD_Signal"].iloc[i]:
            signals.append("SELL")
        else:
            signals.append("HOLD")

    df["Signal"] = signals
    return df
