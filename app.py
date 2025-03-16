import yfinance as yf
import streamlit as st
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
import pandas_ta as ta  

st.set_page_config(page_title="Trading Predictions", layout="wide")
st.title("Trading Predictions")

SUPPORTED_TICKERS = {
    "BankNifty": "^NSEBANK",
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS"
}

st.sidebar.title("User Inputs")
ticker_name = st.sidebar.selectbox("Select Ticker", list(SUPPORTED_TICKERS.keys()))
ticker_symbol = SUPPORTED_TICKERS[ticker_name]
expiry_date = st.sidebar.date_input("Select Expiry Date", min_value=datetime.today())
strike_price = st.sidebar.number_input("Enter Strike Price", min_value=0, value=48600)
option_type = st.sidebar.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.sidebar.number_input("Enter Current LTP", min_value=0.0, value=254.60, step=0.05)

# NewsAPI Key (Taken from your initial code)
NEWS_API_KEY = "990f863a4f65430a99f9b0cac257f432"

def get_news_sentiment(ticker_name):
    url = f'https://newsapi.org/v2/everything?q={ticker_name} OR RBI OR "interest rates" OR "monetary policy"&apiKey={NEWS_API_KEY}'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure request was successful
        data = response.json()

        if 'articles' in data and data['articles']:
            headlines = [article['title'] for article in data['articles'] if article['title']]
            sentiment_score = sum(TextBlob(headline).sentiment.polarity for headline in headlines) / len(headlines)
            return round(sentiment_score, 2)
        else:
            return 0  # Default if no news data
    except requests.exceptions.RequestException:
        return 0  # Default if API fails

def determine_market_trend():
    try:
        data = yf.Ticker(ticker_symbol).history(period="1mo")
        data['5_MA'] = data['Close'].rolling(window=5).mean()
        data['20_MA'] = data['Close'].rolling(window=20).mean()

        macd_result = data.ta.macd(close="Close")
        if macd_result is not None:
            data["MACD"] = macd_result["MACD_12_26_9"]
            data["MACD_signal"] = macd_result["MACDs_12_26_9"]
        else:
            data["MACD"], data["MACD_signal"] = 0, 0

        data["RSI"] = data.ta.rsi(close="Close", length=14)
        if data["RSI"].isnull().all():
            data["RSI"] = 50  

        prev_day_close = data["Close"].iloc[-2]
        current_close = data["Close"].iloc[-1]

        if (data['5_MA'].iloc[-1] > data['20_MA'].iloc[-1]) and (data["MACD"].iloc[-1] > data["MACD_signal"].iloc[-1]) and (data["RSI"].iloc[-1] > 50) and (current_close > prev_day_close):
            return "up"
        elif (data['5_MA'].iloc[-1] < data['20_MA'].iloc[-1]) and (data["MACD"].iloc[-1] < data["MACD_signal"].iloc[-1]) and (data["RSI"].iloc[-1] < 50) and (current_close < prev_day_close):
            return "down"
        else:
            return "neutral"
    except Exception as e:
        st.write(f"Error determining market trend: {e}")
        return "neutral"

def predict_ltp(ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score, market_trend):
    india_vix = india_vix if india_vix is not None else 15.0
    sentiment_score = sentiment_score if isinstance(sentiment_score, (int, float)) else 0.0
    sp500_price = sp500_price if sp500_price is not None else 0.0

    trend_factor = 0.02 if market_trend == "up" else -0.02 if market_trend == "down" else 0
    sentiment_factor = india_vix * 0.1 + sentiment_score * 0.1
    strike_impact = (strike_price - ticker_price) * (-0.02 if market_trend == "down" else 0.01)
    sp500_impact = sp500_price * 0.003
    random_factor = random.uniform(-0.005, 0.01)
    momentum_factor = (ticker_price - ticker_price * 0.99) * (0.05 if market_trend == "up" else -0.05)

    return round(ltp + trend_factor + sentiment_factor + strike_impact + sp500_impact + momentum_factor + (ltp * random_factor), 2)

if st.button("Get Prediction"):
    market_trend = determine_market_trend()
    st.write(f"Market Trend: {market_trend}")

    sentiment_score = get_news_sentiment(ticker_name)
    st.write(f"Sentiment Score: {sentiment_score} (Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

    predicted_ltp = predict_ltp(ltp, 48060, strike_price, 13.33, 5636, sentiment_score, market_trend)
    st.write(f"Predicted LTP: {predicted_ltp}")

    stop_loss_factor = 0.02 if market_trend == "up" else 0.01
    profit_factor = 0.05 if market_trend == "up" else 0.03

    stop_loss = round(predicted_ltp * (1 - stop_loss_factor), 2)
    max_ltp = round(predicted_ltp * (1 + profit_factor), 2)

    st.write(f"Stop Loss: {stop_loss}")
    st.write(f"Target Price: {max_ltp}")

    rrr = round((max_ltp - predicted_ltp) / (predicted_ltp - stop_loss), 2) if stop_loss else None
    st.write(f"Risk-to-Reward Ratio (RRR): {rrr}")

    if predicted_ltp < ltp:
        st.write("❌ **Suggestion: Avoid - Predicted LTP is lower than Current LTP**")
    elif market_trend == "down" and option_type == "Call":
        st.write("❌ **Suggestion: Avoid Buying Calls in Downtrend**")
    elif market_trend == "up" and option_type == "Put":
        st.write("❌ **Suggestion: Avoid Buying Puts in Uptrend**")
    elif sentiment_score < -0.2:
        st.write("❌ **Suggestion: Avoid - Negative News Sentiment Detected**")
    elif rrr and rrr > 1:
        st.write("✅ **Suggestion: Buy**")
    else:
        st.write("❌ **Suggestion: Avoid**")
