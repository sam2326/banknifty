import yfinance as yf
import streamlit as st
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import pandas as pd
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
import time

# Streamlit UI setup
st.title("Enhanced Multi-Index Options Prediction App")
st.write("""
    This app predicts the next day's movement for options based on real-time market data, sentiment analysis, and strike recommendations.
    You can select multiple indices or stocks like **BankNifty**, **Nifty 50**, **Reliance**, and more.
""")

# Supported Tickers
SUPPORTED_TICKERS = {
    "BankNifty": "^NSEBANK",
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS"
}

# Dropdown for selecting ticker
ticker_name = st.selectbox("Select Index/Stock", list(SUPPORTED_TICKERS.keys()))
ticker_symbol = SUPPORTED_TICKERS[ticker_name]

# Input fields
expiry_date = st.date_input("Select Expiry Date", min_value=datetime.today())
strike_price = st.number_input("Enter Strike Price", min_value=0, value=53700)
option_type = st.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.number_input("Enter Current LTP", min_value=0.0, value=765.50, step=0.05)

# Load FinBERT for Sentiment Analysis
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone', do_lower_case=False)
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Function to get financial sentiment using FinBERT
def get_financial_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs).item()
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[sentiment]

# Fetch Ticker Data
def fetch_ticker_data(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="1d", interval="1m")
        current_price = data["Close"].iloc[-1]
        return current_price
    except Exception as e:
        st.write(f"Error fetching data for {ticker}: {e}")
        return None

# Fetch S&P 500 Data
def fetch_sp500_data():
    try:
        sp500 = yf.Ticker("^GSPC")
        data = sp500.history(period="1d", interval="1m")
        return data["Close"].iloc[-1]
    except Exception as e:
        st.write(f"Error fetching S&P 500 data: {e}")
        return None

# News Sentiment Analysis
def get_news_sentiment(ticker_name):
    api_key = "990f863a4f65430a99f9b0cac257f432"  # Your NewsAPI key
    url = f'https://newsapi.org/v2/everything?q={ticker_name} OR RBI OR "interest rates"&apiKey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        headlines = [article['title'] for article in data.get('articles', []) if article.get('title')]
        sentiment_score = sum(TextBlob(headline).sentiment.polarity for headline in headlines) / len(headlines) if headlines else 0
        return sentiment_score
    except requests.exceptions.RequestException as e:
        st.write(f"Error fetching news: {e}")
        return 0

# Predict LTP
def predict_ltp(current_ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score):
    sentiment_factor = india_vix * 0.1 + sentiment_score * 0.05
    strike_impact = (strike_price - ticker_price) * (0.01 if strike_price < ticker_price else -0.01)
    sp500_impact = sp500_price * 0.005
    random_factor = random.uniform(-0.01, 0.02)
    predicted_ltp = current_ltp + sentiment_factor + strike_impact + sp500_impact + (current_ltp * random_factor)
    return round(predicted_ltp, 2)

# Fetch Option Chain Data
def fetch_option_chain(ticker, expiry_date):
    try:
        ticker_obj = yf.Ticker(ticker)
        options = ticker_obj.option_chain(expiry_date)
        return options.calls, options.puts
    except Exception as e:
        st.write(f"Error fetching option chain: {e}")
        return None, None

# Recommend Strikes
def recommend_strikes(calls, puts, current_price):
    calls['Proximity'] = abs(calls['strike'] - current_price)
    puts['Proximity'] = abs(puts['strike'] - current_price)
    recommended_calls = calls.sort_values(by=['Proximity', 'openInterest'], ascending=[True, False]).head(3)
    recommended_puts = puts.sort_values(by=['Proximity', 'openInterest'], ascending=[True, False]).head(3)
    return recommended_calls, recommended_puts

# Auto-Refresh
def auto_refresh():
    while True:
        time.sleep(5)
        st.experimental_rerun()

# Main Logic
if st.button("Get Prediction and Recommended Strikes"):
    ticker_price = fetch_ticker_data(ticker_symbol)
    if ticker_price is None:
        st.warning(f"Could not fetch data for {ticker_name}.")
    else:
        st.write(f"Current price for {ticker_name}: {ticker_price}")

        india_vix = yf.Ticker("^INDIAVIX").history(period="1d", interval="1m")["Close"].iloc[-1]
        sp500_price = fetch_sp500_data()
        sentiment_score = get_news_sentiment(ticker_name)

        st.write(f"India VIX: {india_vix}")
        st.write(f"S&P 500 Price: {sp500_price}")
        st.write(f"Sentiment Score: {sentiment_score}")

        predicted_ltp = predict_ltp(ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score)
        st.write(f"Predicted LTP: {predicted_ltp}")

        stop_loss = round(predicted_ltp * 0.98, 2)
        max_ltp = round(predicted_ltp * 1.02, 2)
        st.write(f"Stop Loss: {stop_loss}")
        st.write(f"Maximum LTP: {max_ltp}")

        if predicted_ltp > ltp:
            st.write("Recommendation: Profit")
        else:
            st.write("Recommendation: Loss")

        # Fetch and Recommend Strikes
        calls, puts = fetch_option_chain(ticker_symbol, expiry_date.strftime("%Y-%m-%d"))
        if calls is not None and puts is not None:
            recommended_calls, recommended_puts = recommend_strikes(calls, puts, ticker_price)
            st.subheader("Recommended Call Strikes")
            st.dataframe(recommended_calls[['strike', 'lastPrice', 'openInterest', 'impliedVolatility']])
            st.subheader("Recommended Put Strikes")
            st.dataframe(recommended_puts[['strike', 'lastPrice', 'openInterest', 'impliedVolatility']])
