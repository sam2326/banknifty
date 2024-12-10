import yfinance as yf
import streamlit as st
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import requests
from textblob import TextBlob
from bs4 import BeautifulSoup
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

# Function to scrape Google News for headlines
def scrape_google_news(query):
    """
    Scrapes Google News for the latest headlines based on a query.
    Args:
        query (str): Search query for Google News.
    Returns:
        list: A list of news headlines.
    """
    base_url = "https://news.google.com"
    search_url = f"{base_url}/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    try:
        response = requests.get(search_url)
        response.raise_for_status()  # Ensure the request was successful
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract headlines
        headlines = []
        for item in soup.find_all("a", class_="DY5T1d"):
            headlines.append(item.get_text())

        return headlines[:10]  # Return the top 10 headlines
    except Exception as e:
        st.write(f"Error scraping Google News: {e}")
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

    # Fetch news sentiment using scraped headlines
    headlines = scrape_google_news(ticker_name)
    sentiment_score = get_sentiment_score(headlines)
    st.write(f"Sentiment Score based on news: {sentiment_score}")

    # Predict LTP
    predicted_ltp = predict_ltp(ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score)
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
