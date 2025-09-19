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
        if df.empty:
            return pd.DataFrame()  # return empty DataFrame
        df = df[['Open','High','Low','Close','Volume']]
        return df

# ------------------------------
# Top 10 coins to analyze
# ------------------------------
coins = ["BTC/USDT","ETH/USDT","BNB/USDT","XRP/USDT","ADA/USDT",
         "SOL/USDT","DOGE/USDT","DOT/USDT","MATIC/USDT","LTC/USDT"]

st.subheader("Top 10 Coins Analysis with Top 3 Highlights")
analysis_results = []

for coin in coins:
    df = get_crypto_data(coin)
    
    if df.empty:
        analysis_results.append({
            "Coin": coin,
            "Last Close": "No data",
            "Signal": "N/A",
            "Change (%)": "N/A",
            "DataFrame": None
        })
        continue
    
    last_row = df.iloc[-1]
    signal = "BUY" if last_row['Close'] > last_row['Open'] else "SHORT"
    prev_close = df.iloc[-2]['Close'] if len(df) > 1 else last_row['Open']
    change_percent = ((last_row['Close'] - prev_close) / prev_close) * 100
    
    analysis_results.append({
        "Coin": coin,
        "Last Close": round(last_row['Close'], 4),
        "Signal": signal,
        "Change (%)": round(change_percent, 2),
        "DataFrame": df
    })

df_analysis = pd.DataFrame(analysis_results)

# Filter out coins with no data
df_analysis_valid = df_analysis[df_analysis["Change (%)"] != "N/A"]

# Highlight Top 3 coins
top3 = df_analysis_valid.sort_values(by="Change (%)", ascending=False).head(3)

st.table(df_analysis.drop(columns=["DataFrame"]))
st.markdown("### 🔥 Top 3 Coins to Watch Now")
st.table(top3.drop(columns=["DataFrame"]))

# ------------------------------
# Automatically show charts for Top 3 coins
# ------------------------------
st.subheader("Charts for Top 3 Coins")

for idx, row in top3.iterrows():
    coin = row['Coin']
    df_chart = row['DataFrame']
    
    if df_chart is None or df_chart.empty:
        st.error(f"No data available for {coin}")
        continue
    
    st.markdown(f"### {coin} Candlestick Chart")
    try:
        mpf.plot(
            df_chart,
            type='candle',
            volume=True,
            style='yahoo',
            title=f'{coin} Candlestick Chart',
            show_nontrading=False
        )
    except Exception as e:
        st.error(f"Error displaying candlestick chart for {coin}: {e}")
    
    st.markdown(f"### {coin} Close Price Chart")
    try:
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df_chart['Close'], label='Close Price', color='blue')
        ax.set_title(f'{coin} Close Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error displaying line chart for {coin}: {e}")
