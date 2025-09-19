# crypto_trade_dashboard.py

import streamlit as st
import pandas as pd
import yfinance as yf
import ccxt
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime

# ------------------------------
# Streamlit page config
# ------------------------------
st.set_page_config(page_title="CryptoSage Ultra Pro", layout="wide")
st.title("💹 CryptoSage Ultra Pro - PRO")
st.markdown("Spot & Futures Crypto Analysis | Buy/Short Signals")

# ------------------------------
# Function to fetch crypto data
# ------------------------------
def get_crypto_data(symbol, interval='1d', limit=100):
    try:
        binance = ccxt.binance()
        ohlcv = binance.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        # Fallback to Yahoo Finance if Binance fails
        yf_symbol = symbol.replace("/", "-")
        df = yf.download(yf_symbol, period='3mo', interval='1d')
        df = df[['Open','High','Low','Close','Volume']]
        return df

# ------------------------------
# Top 10 coins to analyze
# ------------------------------
coins = ["BTC/USDT","ETH/USDT","BNB/USDT","XRP/USDT","ADA/USDT",
         "SOL/USDT","DOGE/USDT","DOT/USDT","MATIC/USDT","LTC/USDT"]

st.subheader("Top 10 Coins Analysis")
analysis_results = []

for coin in coins:
    df = get_crypto_data(coin)
    
    # Simple signal example: Close > Open = BUY, else SHORT
    last_row = df.iloc[-1]
    signal = "BUY" if last_row['Close'] > last_row['Open'] else "SHORT"
    analysis_results.append({
        "Coin": coin,
        "Last Close": last_row['Close'],
        "Signal": signal
    })

st.table(pd.DataFrame(analysis_results))

# ------------------------------
# Choose a coin for charts
# ------------------------------
st.subheader("Detailed Coin Chart")
selected_coin = st.selectbox("Select a coin for chart", coins)
df_chart = get_crypto_data(selected_coin)

# ------------------------------
# Candlestick Chart
# ------------------------------
st.markdown("### Candlestick Chart")
try:
    mpf.plot(
        df_chart,
        type='candle',
        volume=True,
        style='yahoo',
        title=f'{selected_coin} Candlestick Chart',
        show_nontrading=False
    )
except Exception as e:
    st.error(f"Error displaying candlestick chart: {e}")

# ------------------------------
# Close Price Line Chart
# ------------------------------
st.markdown("### Close Price Chart")
try:
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df_chart['Close'], label='Close Price', color='blue')  # 1D series
    ax.set_title(f'{selected_coin} Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error displaying line chart: {e}")
