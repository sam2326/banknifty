import yfinance as yf
import streamlit as st
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import requests
from textblob import TextBlob
from datetime import datetime, timedelta

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

# Function to get financial sentiment using FinBERT
def get_financial_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs).item()

    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[sentiment]

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

# Function to get Google News sentiment for a given ticker using SerpAPI
def fetch_google_news(ticker_name, serpapi_key):
    url = f'https://serpapi.com/search?q={ticker_name}&tbm=nws&api_key={serpapi_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'news_results' in data:
            articles = data['news_results']
            headlines = [article['title'] for article in articles if article['title']]
            sentiment_score = get_sentiment_score(headlines)
            return sentiment_score
        else:
            st.write("Warning: No Google News articles found.")
            return 0
    except requests.exceptions.RequestException as e:
        st.write(f"Error fetching Google News: {e}")
        return 0

# Function to fetch Google Finance data using SerpAPI
def fetch_google_finance(ticker_name, serpapi_key):
    url = f'https://serpapi.com/search?q={ticker_name}+site:finance.google.com&api_key={serpapi_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'stock_results' in data:
            stock_data = data['stock_results'][0]
            current_price = stock_data['price']
            return current_price
        else:
            st.write("Warning: No Google Finance data found.")
            return None
    except requests.exceptions.RequestException as e:
        st.write(f"Error fetching Google Finance data: {e}")
        return None

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

# Function to predict LTP for the selected ticker
def predict_ltp(current_ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score):
    sentiment_factor = india_vix * 0.1 + sentiment_score * 0.05
    strike_impact = (strike_price - ticker_price) * (0.01 if strike_price < ticker_price else -0.01)
    sp500_impact = sp500_price * 0.005
    random_factor = random.uniform(-0.01, 0.02)
    predicted_ltp = current_ltp + sentiment_factor + strike_impact + sp500_impact + (current_ltp * random_factor)
    return round(predicted_ltp, 2)

# Main logic for prediction
def predict():
    ticker_price, ticker_data = fetch_ticker_data(ticker_symbol)
    if ticker_price is None:
        st.warning(f"Could not fetch data for {ticker_name}.")
    else:
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

    # Fetch Google News sentiment
    serpapi_key = "c4e807b7fe597e3421fba24db713faf8f6242eff6825b335936d662dc5dde10f"  # Your SerpAPI key
    google_news_sentiment_score = fetch_google_news(ticker_name, serpapi_key)
    st.write(f"Google News Sentiment Score: {google_news_sentiment_score}")

    # Fetch Google Finance data
    google_finance_price = fetch_google_finance(ticker_name, serpapi_key)
    if google_finance_price:
        st.write(f"Google Finance Current Price: {google_finance_price}")
    else:
        google_finance_price = ticker_price  # Use Yahoo Finance price if Google Finance data is unavailable

    # Fetch news sentiment for the selected ticker (existing FinBERT sentiment)
    sentiment_score = get_news_sentiment(ticker_name)
    st.write(f"Sentiment Score based on news: {sentiment_score}")

    # Predict LTP using additional data (including Google News sentiment)
    predicted_ltp = predict_ltp(ltp, google_finance_price, strike_price, india_vix, sp500_price, google_news_sentiment_score)
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

# Add a button to trigger prediction manually
if st.button("Get Prediction"):
    predict()
