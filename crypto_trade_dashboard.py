# crypto_trade_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import altair as alt

# Import utility functions
from crypto_trade_analysis import (
    fetch_binance_data_full,
    fetch_coingecko_data,
    get_price,
    get_price_history,
    generate_trade_signal
)

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="CryptoSage Ultra Pro", layout="wide")
st.title("💹 CryptoSage Ultra Pro - PRO")
st.markdown("Spot & Futures Crypto Analysis | Top 3 Recommended Trades | AI Chart Analysis")

# -----------------------------
# Coin List
# -----------------------------
COINS = ['BTC','ETH','XRP','LTC','ADA','DOGE']

# -----------------------------
# Load AI Model
# -----------------------------
@st.cache_resource
def load_ai_model(model_path="chart_cnn_model.h5"):
    return load_model(model_path)

try:
    model = load_ai_model()
except Exception:
    st.warning("AI model not found. Using placeholder analysis.")
    model = None

# -----------------------------
# AI Chart Analysis
# -----------------------------
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

# -----------------------------
# Dashboard with Strength Score
# -----------------------------
def build_dashboard_strength(coins):
    dashboard_data = []
    for coin in coins:
        prices = fetch_binance_data_full(coin)
        spot_price = prices['spot'] or fetch_coingecko_data(coin)
        futures_price = prices['futures'] or spot_price
        price_history = get_price_history(coin)
        
        spot_signal, spot_trade_type = generate_trade_signal(price_history)
        futures_signal, futures_trade_type = generate_trade_signal(price_history)
        
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

# -----------------------------
# Auto-Refresh Dashboard
# -----------------------------
st.subheader("📊 Live Market Dashboard & Top 3 Recommended Trades")
dashboard_placeholder = st.empty()
REFRESH_INTERVAL = 60  # seconds

while True:
    df_dashboard, df_top3 = build_dashboard_strength(COINS)
    
    with dashboard_placeholder.container():
        # Top 3 Table
        st.subheader("🏆 Top 3 Recommended Trades (Strength Score)")
        st.table(df_top3)
        
        # Top 3 Bar Chart
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
        
        # Full Dashboard
        st.subheader("📊 Full Market Dashboard")
        st.dataframe(df_dashboard)
        
        st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    time.sleep(REFRESH_INTERVAL)
