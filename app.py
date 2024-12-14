import os
import yfinance as yf
import streamlit as st
import openai
from datetime import datetime
from textblob import TextBlob
import requests

# OpenAI API Key
openai.api_key = "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"

# NewsAPI Key
NEWS_API_KEY = "990f863a4f65430a99f9b0cac257f432"

# Streamlit UI setup
st.set_page_config(page_title="Advanced Trading Predictions", layout="wide")

# Title of the App
st.title("Advanced Trading Predictions with GPT-Powered Sentiment Analysis")

# Supported Tickers
SUPPORTED_TICKERS = {
    "BankNifty": "^NSEBANK",
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS"
}

# Sidebar for Inputs
st.sidebar.title("User Inputs")
ticker_name = st.sidebar.selectbox("Select Ticker", list(SUPPORTED_TICKERS.keys()))
ticker_symbol = SUPPORTED_TICKERS[ticker_name]
expiry_date = st.sidebar.date_input("Select Expiry Date", min_value=datetime.today())
strike_price = st.sidebar.number_input("Enter Strike Price", min_value=0, value=53700)
option_type = st.sidebar.selectbox("Select Option Type", ["Call", "Put"])
ltp = st.sidebar.number_input("Enter Current LTP", min_value=0.0, value=765.50, step=0.05)
risk_percent = st.sidebar.number_input("Enter Risk Percentage (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
profit_percent = st.sidebar.number_input("Enter Profit Percentage (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)

# Function to fetch data for selected ticker
def fetch_ticker_data(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="1d", interval="1m")
        current_price = data["Close"].iloc[-1]
        return current_price, data
    except Exception as e:
        st.write(f"Error fetching data for {ticker}: {e}")
        return None, None

# Function to fetch S&P 500 data
def fetch_sp500_data():
    try:
        sp500 = yf.Ticker("^GSPC")
        data = sp500.history(period="1d", interval="1m")
        return data["Close"].iloc[-1]
    except Exception as e:
        st.write(f"Error fetching S&P 500 data: {e}")
        return None

# Function to perform sentiment analysis using OpenAI GPT
def get_advanced_sentiment(text):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Analyze the sentiment of this financial text: '{text}' and return as Positive, Neutral, or Negative.",
            max_tokens=50,
            temperature=0.5
        )
        sentiment = response.choices[0].text.strip()
        return sentiment
    except Exception as e:
        st.write(f"Error using OpenAI API for sentiment analysis: {e}")
        return "Neutral"

# Function to get sentiment score using TextBlob for fallback
def get_textblob_sentiment(news_headlines):
    sentiment_score = 0
    for headline in news_headlines:
        try:
            sentiment_score += TextBlob(headline).sentiment.polarity
        except Exception as e:
            st.write(f"Error analyzing sentiment for headline: {headline}. Error: {e}")
    
    return round(sentiment_score / len(news_headlines), 2) if news_headlines else 0

# Function to fetch news and perform sentiment analysis
def get_news_sentiment(ticker_name):
    url = f'https://newsapi.org/v2/everything?q={ticker_name} OR RBI OR "interest rates" OR "monetary policy"&apiKey={NEWS_API_KEY}'

    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        data = response.json()

        if 'articles' in data and data['articles']:
            articles = data['articles']
            headlines = [article['title'] for article in articles if article['title']]
            
            # Use OpenAI for advanced sentiment analysis on combined headlines
            combined_headlines = " ".join(headlines)
            sentiment = get_advanced_sentiment(combined_headlines)
            st.write(f"News Sentiment using OpenAI: {sentiment}")
            return sentiment
        else:
            st.write("Warning: No articles found.")
            return "Neutral"
    except requests.exceptions.RequestException as e:
        st.write(f"Error fetching news: {e}")
        return "Neutral"

# Function to determine market trend using moving averages
def determine_market_trend():
    try:
        data = yf.Ticker(ticker_symbol).history(period="1mo")
        data['5_MA'] = data['Close'].rolling(window=5).mean()
        data['20_MA'] = data['Close'].rolling(window=20).mean()
        if data['5_MA'].iloc[-1] > data['20_MA'].iloc[-1]:
            return "up"
        elif data['5_MA'].iloc[-1] < data['20_MA'].iloc[-1]:
            return "down"
        else:
            return "neutral"
    except Exception as e:
        st.write(f"Error determining market trend: {e}")
        return "neutral"

# Function to calculate LTP prediction
def predict_ltp(current_ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score, market_trend):
    trend_factor = 0.02 if market_trend == "up" else -0.02 if market_trend == "down" else 0
    sentiment_factor = india_vix * 0.1 + (1 if sentiment_score == "Positive" else -1 if sentiment_score == "Negative" else 0)
    strike_impact = (strike_price - ticker_price) * (-0.02 if market_trend == "down" else 0.01)
    sp500_impact = sp500_price * 0.003

    predicted_ltp = current_ltp + trend_factor + sentiment_factor + strike_impact + sp500_impact
    return round(predicted_ltp, 2)

# Main prediction logic
def predict():
    ticker_price, ticker_data = fetch_ticker_data(ticker_symbol)
    if ticker_price is None:
        st.warning(f"Could not fetch data for {ticker_name}.")
        return

    st.write(f"Current price for {ticker_name}: {ticker_price}")

    india_vix_ticker = yf.Ticker("^INDIAVIX")
    try:
        india_vix = india_vix_ticker.history(period="1d", interval="1m")["Close"].iloc[-1]
    except:
        india_vix = 15.0
        st.write("Warning: Using default India VIX value.")
    st.write(f"India VIX: {india_vix}")

    sp500_price = fetch_sp500_data()
    if sp500_price is None:
        st.warning("Could not fetch S&P 500 data.")
    else:
        st.write(f"Current S&P 500 price: {sp500_price}")

    market_trend = determine_market_trend()
    st.write(f"Market Trend: {market_trend}")

    sentiment_score = get_news_sentiment(ticker_name)
    predicted_ltp = predict_ltp(ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score, market_trend)
    st.write(f"Predicted LTP for next day: {predicted_ltp}")

if st.button("Get Prediction"):
    predict()
