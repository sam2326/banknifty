import yfinance as yf
import streamlit as st
import openai
import requests
import pandas as pd
import ta
from datetime import datetime

# Set your OpenAI API key
openai.api_key = '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'

# Streamlit UI setup
st.set_page_config(page_title="Trading Predictions", layout="wide")

# Title of the App
st.title("Trading Predictions")

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
        data = ticker_obj.history(period="1mo", interval="1d")
        current_price = data["Close"].iloc[-1]
        return current_price, data
    except Exception as e:
        st.write(f"Error fetching data for {ticker}: {e}")
        return None, None

# Function to calculate technical indicators
def calculate_technical_indicators(data):
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Diff'] = macd.macd_diff()
    return data

# Function to fetch S&P 500 data
def fetch_sp500_data():
    try:
        sp500 = yf.Ticker("^GSPC")
        data = sp500.history(period="1d", interval="1m")
        return data["Close"].iloc[-1]
    except Exception as e:
        st.write(f"Error fetching S&P 500 data: {e}")
        return None

# Function to get news sentiment for a given index/stock
def get_news_sentiment(ticker_name):
    api_key = "990f863a4f65430a99f9b0cac257f432"  # Your NewsAPI key
    url = f'https://newsapi.org/v2/everything?q={ticker_name} OR RBI OR "interest rates" OR "monetary policy"&apiKey={api_key}'

    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        data = response.json()

        if 'articles' in data and data['articles']:
            articles = data['articles']
            headlines = [article['title'] for article in articles if article['title']]
            sentiment_score = get_sentiment_score(headlines)
            return sentiment_score
        else:
            st.write("Warning: No articles found.")
            return 0
    except requests.exceptions.RequestException as e:
        st.write(f"Error fetching news: {e}")
        return 0  # Return 0 if there's an error

# Function to calculate sentiment score from headlines using OpenAI
def get_sentiment_score(news_headlines):
    sentiment_score = 0
    for headline in news_headlines:
        try:
            if isinstance(headline, str) and headline.strip():
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=f"Analyze the sentiment of the following financial news headline: '{headline}'\nSentiment (Positive, Neutral, Negative):",
                    max_tokens=1
                )
                sentiment = response.choices[0].text.strip().lower()
                if sentiment == 'positive':
                    sentiment_score += 1
                elif sentiment == 'negative':
                    sentiment_score -= 1
        except Exception as e:
            st.write(f"Error analyzing sentiment for headline: {headline}. Error: {e}")
    
    return round(sentiment_score / len(news_headlines), 2) if news_headlines else 0

# Function to determine market trend using moving averages
def determine_market_trend(data):
    try:
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

# Function to display sentiment score with timestamp
def display_sentiment_with_time():
    sentiment_score = get_news_sentiment(ticker_name)  # Fetch news sentiment
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get timestamp
    st.write(f"Sentiment Score: {sentiment_score} (Last updated: {timestamp})")
    return sentiment_score  # Return sentiment score for use in the prediction

# Enhanced prediction function
def predict_ltp(current_ltp, ticker_data, strike_price, india_vix, sp500_price, sentiment_score, market_trend):
    # Extract latest technical indicators
    latest_rsi = ticker_data['RSI'].iloc[-1]
    latest_macd = ticker_data['MACD'].iloc[-1]
    latest_macd_signal = ticker_data['MACD_Signal'].iloc[- 
