import streamlit as st
import pandas as pd
import mplfinance as mpf
import requests

from crypto_trade_analysis import fetch_binance, fetch_yfinance, compute_indicators, generate_signals

st.set_page_config(page_title="Crypto Trade Dashboard", layout="wide")
st.title("💹 Crypto Trade Dashboard - Automated Real-Time Signals")
st.markdown("Top 20 Coins + Custom Coin | Buy/Short Signals | Entry, SL, TP Recommendations")

# ------------------------------
# Settings
# ------------------------------
timeframe = st.sidebar.selectbox("Select Timeframe", ["1h","4h","1d"], index=0)

# ------------------------------
# Top 20 Coins
# ------------------------------
top_coins_binance = [
    "BTC/USDT","ETH/USDT","BNB/USDT","XRP/USDT","ADA/USDT",
    "SOL/USDT","DOGE/USDT","LTC/USDT","MATIC/USDT","DOT/USDT",
    "AVAX/USDT","SHIB/USDT","TRX/USDT","LINK/USDT","UNI/USDT",
    "ATOM/USDT","ETC/USDT","XLM/USDT","VET/USDT","FIL/USDT"
]

# ------------------------------
# Fetch Crypto News
# ------------------------------
def fetch_crypto_news():
    try:
        url = "https://cryptopanic.com/api/v1/posts/?auth_token=demo&public=true"
        response = requests.get(url)
        data = response.json()
        news_list = []
        for post in data.get("results", [])[:5]:
            title = post.get("title")
            coin = post.get("currencies")[0]["code"] if post.get("currencies") else "GENERAL"
            news_list.append(f"{coin}: {title}")
        return news_list
    except Exception:
        return ["No news available"]

# ------------------------------
# Plot Candlestick Chart
# ------------------------------
def plot_candlestick(df, symbol="BTC/USDT"):
    mc = mpf.make_marketcolors(up='g', down='r', edge='i', wick='i', volume='in')
    s = mpf.make_mpf_style(marketcolors=mc)
    buy_signals = df[df['Signal']=="Buy"]
    short_signals = df[df['Signal']=="Short"]
    apds = []
    if not buy_signals.empty:
        apds.append(mpf.make_addplot(buy_signals['close'], type='scatter', markersize=100, marker='^', color='g'))
    if not short_signals.empty:
        apds.append(mpf.make_addplot(short_signals['close'], type='scatter', markersize=100, marker='v', color='r'))
    fig, axlist = mpf.plot(df, type='candle', style=s, addplot=apds, returnfig=True, volume=True, figsize=(12,6))
    return fig

# ------------------------------
# Analyze Top Coins
# ------------------------------
results = []

for coin in top_coins_binance:
    df = fetch_binance(coin, timeframe=timeframe)
    if df is None:
        yf_symbol = coin.replace("/USDT","-USD")
        df = fetch_yfinance(yf_symbol, period="30d", interval=timeframe)
    if df is None or df.empty:
        continue
    
    df = compute_indicators(df)
    df = generate_signals(df)

    last_signal = df['Signal'].iloc[-1]
    last_price = df['Entry'].iloc[-1]
    last_sl = df['SL'].iloc[-1]
    last_tp = df['TP'].iloc[-1]
    trade_type = df['TradeType'].iloc[-1]
    
    results.append([coin, last_price, last_signal, trade_type, last_sl, last_tp])
    
    st.subheader(f"Candlestick Chart: {coin}")
    fig = plot_candlestick(df, coin)
    st.pyplot(fig)

# Display table
results_df = pd.DataFrame(results, columns=["Coin","Entry","Signal","TradeType","SL","TP"])
results_df['Rank'] = results_df['Signal'].map({"Buy":1,"Short":2,"Hold":3})
results_df.sort_values('Rank', inplace=True)
st.subheader("Top Coins & Trade Recommendations")
st.dataframe(results_df[['Coin','Entry','Signal','TradeType','SL','TP']])

# ------------------------------
# Custom Coin Search
# ------------------------------
st.sidebar.subheader("Custom Coin Analysis")
custom_coin = st.sidebar.text_input("Enter Coin Symbol (e.g., BTC/USDT)")

if custom_coin:
    df = fetch_binance(custom_coin, timeframe=timeframe)
    if df is None:
        yf_symbol = custom_coin.replace("/USDT","-USD")
        df = fetch_yfinance(yf_symbol, period="30d", interval=timeframe)
    if df is None or df.empty:
        st.warning(f"Data not found for {custom_coin}")
    else:
        df = compute_indicators(df)
        df = generate_signals(df)

        last_signal = df['Signal'].iloc[-1]
        last_price = df['Entry'].iloc[-1]
        last_sl = df['SL'].iloc[-1]
        last_tp = df['TP'].iloc[-1]
        trade_type = df['TradeType'].iloc[-1]

        st.subheader(f"Custom Coin Chart: {custom_coin}")
        fig = plot_candlestick(df, custom_coin)
        st.pyplot(fig)

        st.write(f"*Entry:* {last_price}")
        st.write(f"*Signal:* {last_signal}")
        st.write(f"*Trade Type:* {trade_type}")
        st.write(f"*Stop-Loss (SL):* {last_sl}")
        st.write(f"*Take-Profit (TP):* {last_tp}")

# ------------------------------
# Latest Crypto News
# ------------------------------
st.subheader("Latest Crypto News")
news = fetch_crypto_news()
for item in news:
    st.write(f"- {item}")
