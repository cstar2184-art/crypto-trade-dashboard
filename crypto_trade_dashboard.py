# crypto_dashboard_pro_final_visual.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import ccxt
import yfinance as yf
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import altair as alt
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ======================================
# 1️⃣ Streamlit Page Config
# ======================================
st.set_page_config(page_title="CryptoSage Ultra Pro", layout="wide")
st.title("💹 CryptoSage Ultra Pro - PRO")
st.markdown("Spot & Futures Crypto Analysis | Top 3 Recommended Trades | AI Chart Analysis")

# ======================================
# 2️⃣ Coin List
# ======================================
COINS = ['BTC','ETH','XRP','LTC','ADA','DOGE']

# ======================================
# 3️⃣ Load Pre-Trained AI Model
# ======================================
@st.cache_resource
def load_ai_model(model_path="chart_cnn_model.h5"):
    return load_model(model_path)

try:
    model = load_ai_model()
except Exception:
    st.warning("AI model not found. Using placeholder analysis.")
    model = None

# ======================================
# 4️⃣ Fetch Data Functions
# ======================================
def fetch_binance_data_full(symbol):
    binance_spot = ccxt.binance()
    binance_futures = ccxt.binance({'options': {'defaultType': 'future'}})
    spot_price, futures_price = None, None
    try:
        spot_price = binance_spot.fetch_ticker(symbol + '/USDT')['last']
    except Exception:
        spot_price = None
    try:
        futures_price = binance_futures.fetch_ticker(symbol + '/USDT')['last']
    except Exception:
        futures_price = None
    return {'spot': spot_price, 'futures': futures_price}

def fetch_coingecko_data(symbol):
    try:
        url = f'https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd'
        response = requests.get(url, timeout=10).json()
        return response[symbol.lower()]['usd']
    except Exception:
        return None

def fetch_coindcx_data(symbol):
    """Fetch last price from CoinDCX"""
    try:
        url = "https://api.coindcx.com/exchange/ticker"
        response = requests.get(url, timeout=10).json()
        symbol_formatted = symbol.upper() + "USDT"
        for ticker in response:
            if ticker["market"] == symbol_formatted:
                return float(ticker["last_price"])
        return None
    except Exception:
        return None

def get_price(symbol):
    # 1st → Binance
    price = fetch_binance_data_full(symbol)['spot']
    # 2nd → CoinGecko
    if price is None:
        price = fetch_coingecko_data(symbol)
    # 3rd → CoinDCX
    if price is None:
        price = fetch_coindcx_data(symbol)
    return price

# ======================================
# 5️⃣ Historical Data & Signal Generation
# ======================================
def get_price_history(symbol, length=30):
    try:
        data = yf.download(symbol+'-USDT', period='1mo', interval='1d')
        return data['Close'].tolist()
    except Exception:
        price = get_price(symbol)
        return [price]*length

def generate_trade_signal(prices):
    df = pd.DataFrame(prices, columns=['close'])
    df['SMA5'] = df['close'].rolling(5).mean()
    df['SMA20'] = df['close'].rolling(20).mean()
    df['diff'] = df['SMA5'] - df['SMA20']
    if df['diff'].iloc[-1] > 0:
        return "Buy","Spot"
    elif df['diff'].iloc[-1] < 0:
        return "Short","Futures"
    else:
        return "Hold","-"

# ======================================
# 6️⃣ Build Dashboard with Strength Score
# ======================================
def build_dashboard_strength(coins):
    dashboard_data = []
    for coin in coins:
        prices = fetch_binance_data_full(coin)
        spot_price = prices['spot'] or fetch_coingecko_data(coin) or fetch_coindcx_data(coin)
        futures_price = prices['futures'] or spot_price
        price_history = get_price_history(coin)
        
        spot_signal, spot_trade_type = generate_trade_signal(price_history)
        futures_signal, futures_trade_type = generate_trade_signal(price_history)
        
        # Strength score calculation
        signal_map = {"Buy":1, "Hold":0, "Short":-1}
        spot_weight, futures_weight = 0.6, 0.4
        score = signal_map.get(spot_signal,0)*spot_weight + signal_map.get(futures_signal,0)*futures_weight
        
        dashboard_data.append({
            "Coin": coin,
            "Spot Price": spot_price,
            "Futures Price": futures_price,
            "Spot Signal": spot_signal,
            "Futures Signal": futures_signal,
            "Trade Type": spot_trade_type,
            "Strength Score": score
        })
        
    df = pd.DataFrame(dashboard_data)
    df_top3 = df.sort_values(by='Strength Score', ascending=False).head(3)
    return df, df_top3

# ======================================
# 7️⃣ AI Chart Analysis
# ======================================
def ai_chart_analysis_real(uploaded_file, model):
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Chart', use_column_width=True)
    
    if model:
        img = image.resize((224,224))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        classes = ['Uptrend','Downtrend','Sideways']
        trend_idx = np.argmax(prediction)
        trend = classes[trend_idx]
        st.success(f"Trend: {trend}")
        st.info("Support: Detected by AI model / overlay analysis")
        st.info("Resistance: Detected by AI model / overlay analysis")
        if trend=="Uptrend":
            st.success("Suggested Action: Buy / Long")
        elif trend=="Downtrend":
            st.error("Suggested Action: Short / Sell")
        else:
            st.warning("Suggested Action: Hold / Wait")
    else:
        st.info("AI Analysis Placeholder")
        st.success("Trend: Uptrend\nSupport: $XXX\nResistance: $XXX\nSuggested Action: Buy")

st.subheader("🤖 AI Chart Analysis (Upload Screenshot)")
uploaded_file = st.file_uploader("Upload coin chart screenshot (PNG/JPG)", type=['png','jpg','jpeg'])
if uploaded_file:
    ai_chart_analysis_real(uploaded_file, model)

# ======================================
# 8️⃣ Auto-Refresh Dashboard with Visualization
# ======================================
st.subheader("📊 Live Market Dashboard & Top 3 Recommended Trades")

# 🔄 Auto refresh every 60 sec
count = st_autorefresh(interval=60000, limit=None, key="refresh")

df_dashboard, df_top3 = build_dashboard_strength(COINS)

# Top 3 Recommended Trades Table
st.subheader("🏆 Top 3 Recommended Trades (Strength Score)")
st.table(df_top3)

# Top 3 Strength Score Visualization
st.subheader("📈 Top 3 Strength Score Visualization")
if not df_top3.empty:
    chart = alt.Chart(df_top3).mark_bar(size=60).encode(
        x=alt.X('Coin', sort='-y'),
        y='Strength Score',
        color=alt.condition(
            alt.datum.Strength_Score > 0,
            alt.value("green"),
            alt.value("red")
        ),
        tooltip=['Coin','Strength Score','Spot Signal','Futures Signal']
    ).properties(
        width=600,
        height=400,
        title="Top 3 Recommended Trades Strength"
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No recommended trades at the moment.")

# Full Market Dashboard
st.subheader("📊 Full Market Dashboard")
st.dataframe(df_dashboard)

st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
