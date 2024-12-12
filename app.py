import yfinance as yf
import streamlit as st
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
from serpapi import GoogleSearch

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

# Load FinBERT for Sentiment Analysis
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone', do_lower_case=False)
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Function to fetch news headlines using SerpAPI
def fetch_serpapi_news(ticker_name):
    params = {
        "engine": "google_news",
        "q": f"{ticker_name} OR market updates OR interest rates OR inflation",
        "api_key": "3e07a371a85442d6bb9740ebe7b0fbb0dec5330512554095910ffa51c6cee2d2",
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        articles = results.get('articles', [])
        headlines = [article['title'] for article in articles if article.get('title')]
        return headlines
    except Exception as e:
        st.write(f"Error fetching news using SerpAPI: {e}")
        return []

# Function to calculate sentiment score from headlines
def get_sentiment_score(news_headlines):
    sentiment_score = 0
    for headline in news_headlines:
        try:
            if isinstance(headline, str) and headline.strip():
                sentiment_score += TextBlob(headline).sentiment.polarity
        except Exception as e:
            st.write(f"Error analyzing sentiment for headline: {headline}. Error: {e}")
    
    return round(sentiment_score / len(news_headlines), 2) if news_headlines else 0

# Function to fetch Google Trends data using SerpAPI
def fetch_search_trends(ticker_name):
    params = {
        "engine": "google",
        "q": ticker_name,
        "api_key": "3e07a371a85442d6bb9740ebe7b0fbb0dec5330512554095910ffa51c6cee2d2",
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        search_volume = results.get('search_information', {}).get('total_results', 0)
        return search_volume
    except Exception as e:
        st.write(f"Error fetching search trends using SerpAPI: {e}")
        return 0

# Function to fetch competitor insights using SerpAPI
def fetch_competitor_data(ticker_name):
    params = {
        "engine": "google_finance",
        "q": f"{ticker_name} competitors",
        "api_key": "3e07a371a85442d6bb9740ebe7b0fbb0dec5330512554095910ffa51c6cee2d2",
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        competitors = results.get('competitor_stocks', [])
        competitor_prices = {c['symbol']: c['price'] for c in competitors}
        return competitor_prices
    except Exception as e:
        st.write(f"Error fetching competitor data using SerpAPI: {e}")
        return {}

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

# Function to predict LTP for the selected ticker
def predict_ltp(current_ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score, search_volume):
    sentiment_factor = india_vix * 0.1 + sentiment_score * 0.05
    trend_factor = search_volume * 0.00001
    strike_impact = (strike_price - ticker_price) * (0.01 if strike_price < ticker_price else -0.01)
    sp500_impact = sp500_price * 0.005
    random_factor = random.uniform(-0.01, 0.02)
    predicted_ltp = current_ltp + sentiment_factor + trend_factor + strike_impact + sp500_impact + (current_ltp * random_factor)
    return round(predicted_ltp, 2)

# Main logic for prediction
def predict():
    ticker_price, ticker_data = fetch_ticker_data(ticker_symbol)
    if ticker_price is None:
        st.warning(f"Could not fetch data for {ticker_name}.")
        return

    st.write(f"Current price for {ticker_name}: {ticker_price}")

    # Fetch India VIX
    india_vix_ticker = yf.Ticker("^INDIAVIX")
    try:
        india_vix = india_vix_ticker.history(period="1d", interval="1m")["Close"].iloc[-1]
    except:
        india_vix = 15.0  # Default VIX value if fetching fails
        st.write("Warning: Using default India VIX value.")
    st.write(f"India VIX: {india_vix}")

    # Fetch S&P 500 data
    sp500_price = fetch_sp500_data()
    if sp500_price is None:
        st.warning("Could not fetch S&P 500 data.")
    else:
        st.write(f"Current S&P 500 price: {sp500_price}")

    # Fetch news sentiment
    news_headlines = fetch_serpapi_news(ticker_name)
    sentiment_score = get_sentiment_score(news_headlines)
    st.write(f"News Sentiment Score: {sentiment_score}")

    # Fetch search trends
    search_volume = fetch_search_trends(ticker_name)
    st.write(f"Google Search Volume: {search_volume}")

    # Predict LTP
    predicted_ltp = predict_ltp(ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score, search_volume)
    st.write(f"Predicted LTP for next day: {predicted_ltp}")

    # Stop Loss and Max LTP
    stop_loss = round(predicted_ltp * (1 - (risk_percent / 100)), 2)
    max_ltp = round(predicted_ltp * (1 + (profit_percent / 100)), 2)

    st.write(f"Stop Loss: {stop_loss}")
    st.write(f"Target Price: {max_ltp}")

    # Risk-to-Reward Ratio (RRR)
    rrr = round((max_ltp - predicted_ltp) / (predicted_ltp - stop_loss), 2) if stop_loss and max_ltp else None
    st.write(f"Risk-to-Reward Ratio (RRR): {rrr}")

    # Trading suggestion
    if rrr and rrr > 1:
        st.write("Suggestion: Buy")
    else:
        st.write("Suggestion: Avoid")

    # Display competitor data
    competitor_data = fetch_competitor_data(ticker_name)
    if competitor_data:
        st.write("Competitor Performance:")
        for symbol, price in competitor_data.items():
            st.write(f"{symbol}: {price}")

# Add a button to trigger prediction manually
if st.button("Get Prediction"):
    predict()
