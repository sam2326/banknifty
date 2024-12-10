import yfinance as yf
import streamlit as st
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re

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

# Combined News Sentiment from API and Scraping
def get_combined_sentiment(ticker_name):
    # Get headlines from NewsAPI
    newsapi_headlines = get_newsapi_headlines(ticker_name)

    # Get headlines from Google News scraping
    scraped_headlines = scrape_google_news(ticker_name)

    # Combine headlines
    combined_headlines = newsapi_headlines + scraped_headlines

    if not combined_headlines:
        st.write("Warning: No news headlines found for sentiment analysis.")
        return 0  # Return a default neutral sentiment score

    # Calculate sentiment score
    return get_sentiment_score(combined_headlines)

# Fetch headlines using NewsAPI
def get_newsapi_headlines(ticker_name):
    api_key = "990f863a4f65430a99f9b0cac257f432"  # Your NewsAPI key
    url = f'https://newsapi.org/v2/everything?q={ticker_name}&apiKey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        data = response.json()

        if 'articles' in data and data['articles']:
            return [article['title'] for article in data['articles'] if 'title' in article]
        else:
            return []
    except Exception as e:
        st.write(f"Error fetching data from NewsAPI: {e}")
        return []

# Google News Scraping function
def scrape_google_news(ticker_name):
    url = f"https://news.google.com/search?q={ticker_name}&hl=en-IN&gl=IN&ceid=IN%3Aen"
    try:
        # Get the HTML of the page
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract headlines from the page
        headlines = []
        for headline in soup.find_all('h3'):
            text = headline.get_text()
            if text:
                headlines.append(text)

        return headlines
    except Exception as e:
        st.write(f"Error fetching data from Google News: {e}")
        return []

# Calculate Sentiment Score from Headlines
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

    # Fetch combined news sentiment
    sentiment_score = get_combined_sentiment(ticker_name)
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

# Button to trigger prediction
if st.button("Get Prediction"):
    predict()
