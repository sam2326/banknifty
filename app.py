import yfinance as yf
import streamlit as st
import random
import torch
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas_ta as ta

# Streamlit UI setup
st.set_page_config(page_title="Trading Predictions", layout="wide")
st.title("Trading Predictions")

# Supported Tickers
SUPPORTED_TICKERS = {
    "BankNifty": "^NSEBANK",
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS"
}

# Sidebar Inputs
st.sidebar.title("User Inputs")
ticker_name = st.sidebar.selectbox("Select Ticker", list(SUPPORTED_TICKERS.keys()))
ticker_symbol = SUPPORTED_TICKERS[ticker_name]
expiry_date = st.sidebar.date_input("Select Expiry Date", min_value=datetime.today())
strike_price = st.sidebar.number_input("Enter Strike Price", min_value=0, value=48600)
option_type = st.sidebar.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.sidebar.number_input("Enter Current LTP", min_value=0.0, value=254.60, step=0.05)

# Load FinBERT Model
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone', do_lower_case=False)
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

def get_financial_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs).item()
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[sentiment]

def fetch_ticker_data(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="3mo", interval="1d")
        return data
    except Exception as e:
        st.write(f"Error fetching data: {e}")
        return None

def fetch_sp500_data():
    try:
        sp500 = yf.Ticker("^GSPC")
        data = sp500.history(period="1d", interval="1m")
        return data["Close"].iloc[-1]
    except Exception:
        return None

def get_news_sentiment(ticker_name):
    api_key = "990f863a4f65430a99f9b0cac257f432"
    url = f'https://newsapi.org/v2/everything?q={ticker_name}&apiKey={api_key}'
    try:
        response = requests.get(url)
        data = response.json()
        headlines = [article['title'] for article in data.get('articles', []) if article.get('title')]
        return sum(TextBlob(headline).sentiment.polarity for headline in headlines) / len(headlines) if headlines else 0
    except Exception:
        return 0

def determine_market_trend(data):
    data["5_MA"] = data["Close"].rolling(window=5).mean()
    data["20_MA"] = data["Close"].rolling(window=20).mean()
    return "up" if data["5_MA"].iloc[-1] > data["20_MA"].iloc[-1] else "down"

def predict_ltp(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model.predict([X_test])[0]

# Main Prediction Logic
def predict():
    data = fetch_ticker_data(ticker_symbol)
    if data is None or data.empty:
        st.warning(f"Could not fetch data for {ticker_name}.")
        return

    current_price = data["Close"].iloc[-1]
    st.write(f"Current price for {ticker_name}: {current_price}")

    try:
        india_vix = yf.Ticker("^INDIAVIX").history(period="1d", interval="1m")["Close"].iloc[-1]
    except:
        india_vix = 15.0
    st.write(f"India VIX: {india_vix}")

    sp500_price = fetch_sp500_data()
    st.write(f"Current S&P 500 price: {sp500_price if sp500_price else 'Unavailable'}")

    market_trend = determine_market_trend(data)
    st.write(f"Market Trend: {market_trend}")

    sentiment_score = get_news_sentiment(ticker_name)
    st.write(f"Sentiment Score: {sentiment_score} (Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

    # Feature Engineering for Prediction
    data["MACD"], _, _ = ta.macd(data["Close"])
    data["RSI"] = ta.rsi(data["Close"], length=14)
    data["ATR"] = ta.atr(data["High"], data["Low"], data["Close"])
    data = data.dropna()

    features = ["Close", "MACD", "RSI", "ATR"]
    X = data[features]
    y = data["Close"].shift(-1).dropna()
    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, shuffle=False)
    X_pred = X.iloc[-1].values

    predicted_ltp = predict_ltp(X_train, y_train, X_pred)
    st.write(f"Predicted LTP for next day: {predicted_ltp}")

    stop_loss = round(predicted_ltp * 0.98, 2)
    target_price = round(predicted_ltp * 1.05, 2)

    st.write(f"Stop Loss: {stop_loss}")
    st.write(f"Target Price: {target_price}")

    rrr = round((target_price - predicted_ltp) / (predicted_ltp - stop_loss), 2) if stop_loss else None
    st.write(f"Risk-to-Reward Ratio (RRR): {rrr}")

    if market_trend == "down" and option_type == "Call":
        st.write("Suggestion: Avoid Buying Calls in Downtrend")
    elif market_trend == "up" and option_type == "Put":
        st.write("Suggestion: Avoid Buying Puts in Uptrend")
    elif rrr and rrr > 1:
        st.write("Suggestion: Buy")
    else:
        st.write("Suggestion: Avoid")

if st.button("Get Prediction"):
    predict()
