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
st.title("Enhanced Multi-Index Options Prediction App")
st.write("""
    This app predicts the next day's movement for options based on real-time market data, including sentiment analysis.
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

# Function to fetch data for selected ticker
def fetch_ticker_data(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period="1d", interval="1m")
        current_price = data["Close"].iloc[-1]
        return current_price
    except Exception as e:
        st.write(f"Error fetching data for {ticker}: {e}")
        return None

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
    url = f'https://newsapi.org/v2/everything?q={ticker_name} OR RBI OR "Reserve Bank of India"&apiKey={api_key}'

    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        data = response.json()

        if 'articles' in data and data['articles']:
            articles = data['articles']
            headlines = [article['title'] for article in articles if article['title']]
            if headlines:
                sentiment_score = get_sentiment_score(headlines)
                return sentiment_score
            else:
                st.write("Warning: No valid headlines found.")
                return 0
        else:
            st.write("Warning: No articles found.")
            return 0
    except requests.exceptions.RequestException as e:
        st.write(f"Error fetching news: {e}")
        return 0  # Return 0 if there's an error

# Function to calculate sentiment score from headlines
def get_sentiment_score(news_headlines):
    sentiment_score = 0
    # Ensure that headlines are valid and not empty
    for headline in news_headlines:
        try:
            # Check if headline is a valid string and perform sentiment analysis
            if isinstance(headline, str) and headline.strip():
                sentiment_score += TextBlob(headline).sentiment.polarity
        except Exception as e:
            st.write(f"Error analyzing sentiment for headline: {headline}. Error: {e}")
    
    # Avoid division by zero if there are no valid headlines
    return sentiment_score / len(news_headlines) if news_headlines else 0

# Function to predict LTP for the selected ticker
def predict_ltp(current_ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score):
    sentiment_factor = india_vix * 0.1 + sentiment_score * 0.05
    strike_impact = (strike_price - ticker_price) * (0.01 if strike_price < ticker_price else -0.01)
    sp500_impact = sp500_price * 0.005
    random_factor = random.uniform(-0.01, 0.02)
    predicted_ltp = current_ltp + sentiment_factor + strike_impact + sp500_impact + (current_ltp * random_factor)
    return round(predicted_ltp, 2)

# Main logic for prediction
if st.button("Get Prediction"):
    ticker_price = fetch_ticker_data(ticker_symbol)
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

        # Fetch news sentiment
        sentiment_score = get_news_sentiment(ticker_name)
        st.write(f"Sentiment Score based on news: {sentiment_score}")

        # Predict LTP
        predicted_ltp = predict_ltp(ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score)
        st.write(f"Predicted LTP for next day: {predicted_ltp}")

        # Stop Loss and Max LTP
        stop_loss = predicted_ltp * 0.98
        max_ltp = predicted_ltp * 1.02
        st.write(f"Stop Loss: {round(stop_loss, 2)}")
        st.write(f"Maximum LTP: {round(max_ltp, 2)}")

        # Recommendation
        if predicted_ltp > ltp:
            st.write("Recommendation: Profit")
            st.write(f"Expected Profit: {round(predicted_ltp - ltp, 2)}")
        else:
            st.write("Recommendation: Loss")
            st.write(f"Expected Loss: {round(ltp - predicted_ltp, 2)}")
