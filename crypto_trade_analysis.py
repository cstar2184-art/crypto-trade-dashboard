# crypto_trade_analysis.py

import pandas as pd
import yfinance as yf
import requests
import ccxt

# -----------------------------
# Fetch Binance & Futures Data
# -----------------------------
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

# -----------------------------
# Fetch CoinGecko Data
# -----------------------------
def fetch_coingecko_data(symbol):
    try:
        url = f'https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd'
        response = requests.get(url, timeout=10).json()
        return response[symbol.lower()]['usd']
    except Exception:
        return None

# -----------------------------
# Get Current Price (Binance fallback CoinGecko)
# -----------------------------
def get_price(symbol):
    price = fetch_binance_data_full(symbol)['spot']
    if price is None:
        price = fetch_coingecko_data(symbol)
    return price

# -----------------------------
# Get Price History for Signals
# -----------------------------
def get_price_history(symbol, length=30):
    try:
        data = yf.download(symbol+'-USDT', period='1mo', interval='1d')
        return data['Close'].tolist()
    except Exception:
        price = get_price(symbol)
        return [price]*length

# -----------------------------
# Generate Buy/Short/Hold Signal
# -----------------------------
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
