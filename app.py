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

# Function to display sentiment score with timestamp
def display_sentiment_with_time():
    sentiment_score = get_news_sentiment(ticker_name)  # Fetch news sentiment
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get timestamp
    st.write(f"Sentiment Score: {sentiment_score} (Last updated: {timestamp})")
    return sentiment_score  # Return sentiment score for use in the prediction

# Enhanced prediction function
def predict_ltp(current_ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score, market_trend):
    trend_factor = 0.02 if market_trend == "up" else -0.02 if market_trend == "down" else 0
    sentiment_factor = india_vix * 0.1 + sentiment_score * 0.1
    strike_impact = (strike_price - ticker_price) * (-0.02 if market_trend == "down" else 0.01)
    sp500_impact = sp500_price * 0.003
    random_factor = random.uniform(-0.005, 0.01)
    momentum_factor = (ticker_price - ticker_price * 0.99) * (0.05 if market_trend == "up" else -0.05)

    predicted_ltp = current_ltp + trend_factor + sentiment_factor + strike_impact + sp500_impact + momentum_factor + (current_ltp * random_factor)
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

    sentiment_score = display_sentiment_with_time()

    predicted_ltp = predict_ltp(ltp, ticker_price, strike_price, india_vix, sp500_price, sentiment_score, market_trend)
    st.write(f"Predicted LTP for next day: {predicted_ltp}")

    stop_loss_factor = 0.02 if market_trend == "up" else 0.01
    profit_factor = 0.05 if market_trend == "up" else 0.03

    stop_loss = round(predicted_ltp * (1 - stop_loss_factor), 2)
    max_ltp = round(predicted_ltp * (1 + profit_factor), 2)

    st.write(f"Stop Loss: {stop_loss}")
    st.write(f"Target Price: {max_ltp}")

    rrr = round((max_ltp - predicted_ltp) / (predicted_ltp - stop_loss), 2) if stop_loss and max_ltp else None
    st.write(f"Risk-to-Reward Ratio (RRR): {rrr}")

    if rrr and rrr > 1:
        st.write("Suggestion: Buy")
    else:
        st.write("Suggestion: Avoid")

if st.button("Get Prediction"):
    predict()
